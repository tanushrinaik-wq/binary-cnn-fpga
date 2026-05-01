from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["rock", "paper", "scissors"]


@dataclass
class DatasetConfig:
    data_root: Path
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    auto_download: bool = True


class TransformSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class NumpyImageDataset(Dataset):
    def __init__(self, images: List[np.ndarray], labels: List[int], transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx].astype(np.uint8), mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[idx])


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def _is_imagefolder_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    entries = [child.name.lower() for child in path.iterdir() if child.is_dir()]
    return all(class_name in entries for class_name in CLASS_NAMES)


def _maybe_load_tfds_dataset():
    try:
        import tensorflow_datasets as tfds
    except Exception:
        return None

    try:
        train_ds = tfds.load("rock_paper_scissors", split="train", as_supervised=True)
        test_ds = tfds.load("rock_paper_scissors", split="test", as_supervised=True)
    except Exception:
        return None

    images: List[np.ndarray] = []
    labels: List[int] = []
    for image, label in tfds.as_numpy(train_ds.concatenate(test_ds)):
        images.append(image)
        labels.append(int(label))
    return images, labels


def _resolve_dataset_root(root: Path, auto_download: bool = True) -> Path:
    candidates = [
        root,
        root / "rockpaperscissors",
        root / "rps",
        root / "Rock-Paper-Scissors",
    ]
    for candidate in candidates:
        if _is_imagefolder_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset directory under {root}. "
        "Expected an ImageFolder layout with class folders rock/paper/scissors."
    )


def load_imagefolder_dataset(root: Path) -> datasets.ImageFolder:
    dataset_root = _resolve_dataset_root(root, auto_download=False)
    ds = datasets.ImageFolder(dataset_root)
    classes = [name.lower() for name in ds.classes]
    if sorted(classes) != sorted(CLASS_NAMES):
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {ds.classes}")
    return ds


def split_indices(length: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(length * train_ratio)
    n_val = int(length * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def create_dataloaders(config: DatasetConfig) -> Dict[str, DataLoader]:
    train_tf, eval_tf = build_transforms(config.image_size)
    tfds_data = _maybe_load_tfds_dataset() if config.auto_download else None
    if tfds_data is not None:
        images, labels = tfds_data
        train_idx, val_idx, test_idx = split_indices(
            len(labels),
            config.seed,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )
        splits = {
            "train": NumpyImageDataset([images[i] for i in train_idx], [labels[i] for i in train_idx], train_tf),
            "val": NumpyImageDataset([images[i] for i in val_idx], [labels[i] for i in val_idx], eval_tf),
            "test": NumpyImageDataset([images[i] for i in test_idx], [labels[i] for i in test_idx], eval_tf),
        }
    else:
        base_dataset = load_imagefolder_dataset(config.data_root)
        train_idx, val_idx, test_idx = split_indices(
            len(base_dataset),
            config.seed,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )
        splits = {
            "train": TransformSubset(base_dataset, train_idx, train_tf),
            "val": TransformSubset(base_dataset, val_idx, eval_tf),
            "test": TransformSubset(base_dataset, test_idx, eval_tf),
        }

    loaders = {
        name: DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(config.num_workers > 0),
        )
        for name, ds in splits.items()
    }
    return loaders


def export_split_manifest(config: DatasetConfig, output_path: Path) -> None:
    tfds_data = _maybe_load_tfds_dataset() if config.auto_download else None
    if tfds_data is not None:
        _, labels = tfds_data
        train_idx, val_idx, test_idx = split_indices(
            len(labels),
            config.seed,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )
        manifest = {
            "seed": config.seed,
            "source": "tfds:rock_paper_scissors",
            "classes": CLASS_NAMES,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }
    else:
        base_dataset = load_imagefolder_dataset(config.data_root)
        train_idx, val_idx, test_idx = split_indices(
            len(base_dataset),
            config.seed,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )
        manifest = {
            "seed": config.seed,
            "source": "imagefolder",
            "classes": base_dataset.classes,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
