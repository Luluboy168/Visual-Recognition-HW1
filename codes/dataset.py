import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import (
    InterpolationMode,
    RandAugment,
    RandomErasing,
)


def get_transforms(img_size=320):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        RandomErasing(p=0.1)
    ])

    resize_size = int(img_size * 1.14)
    val_transform = transforms.Compose([
        transforms.Resize(
            resize_size, interpolation=InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_transform


class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_kfold_indices(num_samples, n_folds, fold_idx, seed=42):
    np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    val_idx = folds[fold_idx]
    train_idx = np.concatenate(
        [folds[i] for i in range(n_folds) if i != fold_idx])

    return train_idx, val_idx


def get_kfold_dataloaders(
        data_dir, batch_size=32, num_workers=4,
        img_size=320, n_folds=5, fold_idx=0, seed=42):
    train_dir = os.path.join(data_dir, 'train')

    base_dataset = datasets.ImageFolder(train_dir)

    train_idx, val_idx = get_kfold_indices(
        len(base_dataset), n_folds, fold_idx, seed)

    train_subset = Subset(base_dataset, train_idx)
    val_subset = Subset(base_dataset, val_idx)

    train_transform, val_transform = get_transforms(img_size)

    train_data = TransformWrapper(train_subset, transform=train_transform)
    val_data = TransformWrapper(val_subset, transform=val_transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_test_dataloader(test_dir, batch_size=32, num_workers=4, img_size=320):
    _, test_transform = get_transforms(img_size)
    test_dataset = TestDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return test_loader
