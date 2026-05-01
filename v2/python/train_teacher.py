from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import DatasetConfig, create_dataloaders, export_split_manifest
from models import TeacherMobileNetV2


def run_epoch(model, loader, criterion, optimizer, device, scaler):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += images.size(0)
    return total_loss / total_items, total_correct / total_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--out-dir", type=Path, default=Path("../artifacts"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(data_root=args.data_root, batch_size=args.batch_size)
    export_split_manifest(cfg, args.out_dir / "dataset_split.json")
    loaders = create_dataloaders(cfg)

    model = TeacherMobileNetV2().to(device)
    for param in model.model.features.parameters():
        param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    log_path = args.out_dir / "teacher_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, scaler)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, device, scaler)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.out_dir / "teacher_mobilenetv2.pth")


if __name__ == "__main__":
    main()
