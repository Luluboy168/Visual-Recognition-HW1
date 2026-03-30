import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc

from dataset import get_kfold_dataloaders

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def mixup_cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1. - lam)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]

    if np.random.rand() < 0.5:
        mixed_x = lam * x + (1 - lam) * x[index, :]
    else:
        B, C, H, W = x.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return mixed_x, y_a, y_b, lam


def merge_multi(state_dicts):
    """Averages the weights of multiple state dictionaries (Model Stock)."""
    merged_state = {}
    num_models = len(state_dicts)
    for key in state_dicts[0].keys():
        merged_state[key] = sum(
            [state[key] for state in state_dicts]) / num_models
    return merged_state


def plot_and_save_metrics(
        train_losses, val_losses, train_accs, val_accs, save_dir, fold_idx):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Avg Train Loss',
             marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Merged Val Loss',
             marker='o', markersize=4)
    plt.title(f'Loss Curve - Fold {fold_idx}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Avg Train Acc',
             marker='o', markersize=4)
    plt.plot(epochs, val_accs, label='Merged Val Acc',
             marker='o', markersize=4)
    plt.title(f'Accuracy Curve - Fold {fold_idx}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'learning_curve_fold_{fold_idx}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"=> Saved learning curve plot to {plot_path}")


def train(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fold: {args.fold_idx+1}/{args.n_folds}")
    print(f"{args.num_models} models for Model Stock")

    train_loader, val_loader = get_kfold_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        n_folds=args.n_folds,
        fold_idx=args.fold_idx,
        seed=args.seed
    )

    models = []
    optimizers = []
    scalers = []
    schedulers = []

    warmup_epochs = 5
    main_epochs = max(1, args.epochs - warmup_epochs)
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    main_steps = main_epochs * steps_per_epoch

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for i in range(args.num_models):
        m = torch.hub.load(
            'zhanghang1989/ResNeSt', 'resnest200', pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, args.num_classes)
        m.to(device)
        models.append(m)

        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizers.append(opt)
        scalers.append(torch.amp.GradScaler('cuda'))

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_steps)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=main_steps)
        schedulers.append(optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        ))

    merged_model = torch.hub.load(
        'zhanghang1989/ResNeSt', 'resnest200', pretrained=False)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, args.num_classes)
    merged_model.to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(args.epochs):
        epoch_model_losses = []
        epoch_model_accs = []

        for i in range(args.num_models):
            models[i].train()
            set_seed(args.seed + epoch * args.num_models + i)

            running_loss, correct, total = 0.0, 0, 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [Train Model {i+1}]",
                leave=False)
            for images, labels in train_pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizers[i].zero_grad()
                apply_augment = np.random.rand() < args.mixup_prob

                if apply_augment:
                    mixed_images, tgts_a, tgts_b, lam = mixup_cutmix(
                        images, labels)
                    with torch.amp.autocast('cuda'):
                        outputs = models[i](mixed_images)
                        loss = lam * criterion(outputs, tgts_a) + \
                            (1. - lam) * criterion(outputs, tgts_b)
                else:
                    with torch.amp.autocast('cuda'):
                        outputs = models[i](images)
                        loss = criterion(outputs, labels)

                scalers[i].scale(loss).backward()
                scalers[i].unscale_(optimizers[i])
                torch.nn.utils.clip_grad_norm_(
                    models[i].parameters(), max_norm=1.0)

                scale_before = scalers[i].get_scale()
                scalers[i].step(optimizers[i])
                scalers[i].update()

                if scale_before <= scalers[i].get_scale():
                    schedulers[i].step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)

                if apply_augment:
                    correct += predicted.eq(tgts_a).sum().item()
                else:
                    correct += predicted.eq(labels).sum().item()

                train_pbar.set_postfix(
                    {'loss': loss.item(), 'acc': correct/total})

            epoch_model_losses.append(running_loss / len(train_loader.dataset))
            epoch_model_accs.append(correct / total)

        avg_train_loss = sum(epoch_model_losses) / args.num_models
        avg_train_acc = sum(epoch_model_accs) / args.num_models

        model_states = [m.state_dict() for m in models]
        merged_state = merge_multi(model_states)
        merged_model.load_state_dict(merged_state)
        merged_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val Merged]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = merged_model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_pbar.set_postfix(
                    {'loss': loss.item(), 'acc': val_correct/val_total})

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(val_epoch_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(val_epoch_acc)

        print(
            f"Epoch {epoch+1} Summary: \n"
            f"  -> Avg Train Loss: {avg_train_loss:.4f} "
            f"| Avg Train Acc: {avg_train_acc:.4f}\n"
            f"  -> Merged Val Loss: {val_epoch_loss:.4f} "
            f"| Merged Val Acc: {val_epoch_acc:.4f}"
        )

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            save_path = os.path.join(
                args.checkpoint_dir, f'best_model_fold_{args.fold_idx}.pth')
            with open(save_path, 'wb') as f:
                torch.save(merged_state, f)
            print(f"=> Saved new best merged model for Fold "
                  f"{args.fold_idx} to {save_path}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"=> Early stopping counter: {epochs_no_improve} "
                  f"out of {args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\n=> Early stopping triggered for Fold {args.fold_idx}")
            break

        for m in models:
            m.load_state_dict(merged_state)

    plot_and_save_metrics(
        train_losses, val_losses, train_accs, val_accs,
        args.checkpoint_dir, args.fold_idx)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train ResNeSt-200 with Model Stock & K-Fold')

    parser.add_argument(
        '--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument(
        '--num_workers', type=int, default=10,
        help='number of data loading workers')
    parser.add_argument(
        '--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--mixup_prob', type=float, default=0.4,
        help='Probability of applying Mixup/CutMix')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience (epochs)')
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of cross-validation folds')
    parser.add_argument(
        '--num_models', type=int, default=2,
        help='Number of parallel models for Model Stock')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_dir = os.path.join(base_dir, '..', 'data')
    args.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    args.num_classes = 100
    args.img_size = 320

    for current_fold in range(args.n_folds):
        print(f"\n{'='*50}")
        print(f" STARTING TRAINING FOR FOLD {current_fold+1}/{args.n_folds}")
        print(f"{'='*50}\n")

        args.fold_idx = current_fold
        train(args)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
