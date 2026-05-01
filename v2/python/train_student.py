from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import DatasetConfig, create_dataloaders
from models import BCNNStudent, TeacherMobileNetV2


def kd_loss(student_logits, teacher_logits, labels, temperature: float):
    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return 0.5 * ce + 0.5 * kd


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += images.size(0)
    return total_loss / total_items, total_correct / total_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--artifacts", type=Path, default=Path("../artifacts"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(data_root=args.data_root, batch_size=args.batch_size)
    loaders = create_dataloaders(cfg)

    student = BCNNStudent().to(device)
    teacher = TeacherMobileNetV2(pretrained=False).to(device)
    teacher.load_state_dict(torch.load(args.artifacts / "teacher_mobilenetv2.pth", map_location=device))
    teacher.eval()

    optimizer = Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best_acc = 0.0

    with (args.artifacts / "student_fp32_log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "teacher_val_acc"])
        teacher_val_loss, teacher_val_acc = evaluate(teacher, loaders["val"], device)
        for epoch in range(1, args.epochs + 1):
            student.train()
            total_loss = 0.0
            total_items = 0
            for images, labels in tqdm(loaders["train"], leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.no_grad():
                    with autocast(enabled=(device.type == "cuda")):
                        teacher_logits = teacher(images)
                with autocast(enabled=(device.type == "cuda")):
                    student_logits = student(images)
                    loss = kd_loss(student_logits, teacher_logits, labels, args.temperature)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * images.size(0)
                total_items += images.size(0)
            scheduler.step()
            train_loss = total_loss / total_items
            val_loss, val_acc = evaluate(student, loaders["val"], device)
            writer.writerow([epoch, train_loss, val_loss, val_acc, teacher_val_acc])
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(student.state_dict(), args.artifacts / "student_fp32.pth")


if __name__ == "__main__":
    main()
