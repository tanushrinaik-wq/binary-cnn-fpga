from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from dataset import CLASS_NAMES, DatasetConfig, create_dataloaders
from models import BCNNBinarized, BCNNStudent, TeacherMobileNetV2, count_parameters, model_size_mb


def timed_inference(model, sample, device, runs=100, warmup=10):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(runs):
                _ = model(sample)
            ender.record()
            torch.cuda.synchronize()
            return starter.elapsed_time(ender) / runs
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample)
        end = time.perf_counter()
        return (end - start) * 1000.0 / runs


def collect_metrics(model, loader, device):
    model.eval()
    preds: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
    acc = 100.0 * np.mean(np.array(preds) == np.array(labels_all))
    return acc, preds, labels_all


def plot_confusion(labels, preds, title, output):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def bop_estimate() -> float:
    return float(
        32 * 32 * 32 * 9
        + 32 * 32 * 64 * 32
        + 16 * 16 * 64 * 9
        + 16 * 16 * 128 * 64
        + 8 * 8 * 128 * 9
        + 8 * 8 * 256 * 128
    ) / 1e6


def mac_estimate_teacher() -> float:
    return 300.0


def mac_estimate_student() -> float:
    conv1 = 32 * 32 * 32 * (3 * 3 * 3)
    pw2 = 32 * 32 * 64 * 32
    pw3 = 16 * 16 * 128 * 64
    pw4 = 8 * 8 * 256 * 128
    fc = 256 * 3
    return (conv1 + pw2 + pw3 + pw4 + fc) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--artifacts", type=Path, default=Path("../artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("../reports"))
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    loaders = create_dataloaders(DatasetConfig(data_root=args.data_root))
    sample_batch = next(iter(loaders["test"]))[0][:32]

    models_info = []
    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if torch.cuda.is_available() else None

    for name, model in [
        ("Float Teacher", TeacherMobileNetV2(pretrained=False)),
        ("Float Student", BCNNStudent()),
        ("Binarized Student", BCNNBinarized()),
    ]:
        ckpt = {
            "Float Teacher": "teacher_mobilenetv2.pth",
            "Float Student": "student_fp32.pth",
            "Binarized Student": "student_binarized.pth",
        }[name]
        model.load_state_dict(torch.load(args.artifacts / ckpt, map_location="cpu"))
        acc, preds, labels = collect_metrics(model.to(cpu), loaders["test"], cpu)
        cpu_ms = timed_inference(model.to(cpu), sample_batch, cpu)
        gpu_ms = timed_inference(model.to(gpu), sample_batch, gpu) if gpu is not None else float("nan")
        confusion_name = {
            "Float Teacher": "confusion_matrix_teacher.png",
            "Float Student": "confusion_matrix_student_fp32.png",
            "Binarized Student": "confusion_matrix_student_binarized.png",
        }[name]
        plot_confusion(labels, preds, name, args.report_dir / confusion_name)
        models_info.append(
            {
                "Metric": name,
                "Test accuracy (%)": acc,
                "Top-1 error (%)": 100.0 - acc,
                "Model size (MB)": model_size_mb(model),
                "Param count": count_parameters(model),
                "Inference time CPU (ms)": cpu_ms,
                "Inference time GPU (ms)": gpu_ms,
                "MACs (millions)": mac_estimate_teacher() if name == "Float Teacher" else mac_estimate_student(),
                "BOPs (binary ops, M)": bop_estimate() if name == "Binarized Student" else 0.0,
            }
        )

    df = pd.DataFrame(models_info)
    df.to_csv(args.report_dir / "comparison_report.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(df["Metric"], df["Test accuracy (%)"])
    plt.xticks(rotation=15)
    plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.bar(df["Metric"], df["Inference time CPU (ms)"])
    plt.xticks(rotation=15)
    plt.title("CPU Inference")
    plt.tight_layout()
    plt.savefig(args.report_dir / "comparison_report.png")


if __name__ == "__main__":
    main()
