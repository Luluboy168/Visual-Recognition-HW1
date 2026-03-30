import os
import torch
import torch.nn as nn
import pandas as pd
import gc
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import datasets

from dataset import get_test_dataloader


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_dir = os.path.join(args.data_dir, 'test')
    test_loader = get_test_dataloader(
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    ensemble_probs = None
    all_img_names = []
    zoom_size = int(args.img_size * 1.14)

    for fold in range(args.n_folds):
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f'best_model_fold_{fold}.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Missing trained model for Fold {fold} at {checkpoint_path}")

        print(f"\nLoading architecture for Fold {fold}...")
        model = torch.hub.load(
            'zhanghang1989/ResNeSt', 'resnest200', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        model.to(device)

        print(f"Loading weights from {os.path.basename(checkpoint_path)}...")
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        fold_probs = []
        fold_img_names = []

        with torch.no_grad():
            for images, img_names in tqdm(
                    test_loader, desc=f"Fold {fold} (4-Pass TTA)"):
                images = images.to(device, non_blocking=True)

                logits_orig = model(images)
                logits_flipped = model(TF.hflip(images))
                images_zoomed = TF.resize(
                    images, [zoom_size, zoom_size],
                    interpolation=InterpolationMode.BILINEAR)
                images_zoomed = TF.center_crop(
                    images_zoomed, [args.img_size, args.img_size])
                logits_zoomed = model(images_zoomed)

                logits_zoomed_flipped = model(TF.hflip(images_zoomed))

                avg_logits = (logits_orig + logits_flipped +
                              logits_zoomed + logits_zoomed_flipped) / 4.0
                probs = torch.softmax(avg_logits, dim=1)

                fold_probs.append(probs.cpu())
                fold_img_names.extend(img_names)

        fold_probs = torch.cat(fold_probs, dim=0)

        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_img_names = fold_img_names
        else:
            ensemble_probs += fold_probs

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_probs /= args.n_folds
    _, final_predictions = ensemble_probs.max(1)
    train_dir = os.path.join(args.data_dir, 'train')
    temp_dataset = datasets.ImageFolder(train_dir)
    idx_to_class = {v: k for k, v in temp_dataset.class_to_idx.items()}

    predictions = []
    for item_name, pred_idx in zip(all_img_names, final_predictions.numpy()):
        img_base_name = os.path.splitext(item_name)[0]
        actual_class_label = idx_to_class[pred_idx]
        predictions.append({
            'image_name': img_base_name,
            'pred_label': actual_class_label
        })

    df = pd.DataFrame(predictions)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'prediction.csv')
    df.to_csv(out_path, index=False)
    print(f"\nFinal ensemble prediction saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Ensemble Inference for K-Fold ResNeSt-200')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='number of data loading workers')
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of trained folds to ensemble')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_dir = os.path.join(base_dir, '..', 'data')
    args.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    args.num_classes = 100
    args.img_size = 320

    inference(args)
