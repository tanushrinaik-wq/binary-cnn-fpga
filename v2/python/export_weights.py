from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from dataset import DatasetConfig, create_dataloaders
from models import BCNNBinarized


def q88(value: np.ndarray) -> np.ndarray:
    return np.clip(np.round(value * 256.0), -32768, 32767).astype(np.int16)


def int8_sym(value: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(value))
    scale = 1.0 if scale == 0 else scale / 127.0
    quant = np.clip(np.round(value / scale), -127, 127).astype(np.int8)
    return quant


def pack_bits(bits: np.ndarray, pack_width: int = 32) -> List[int]:
    bits = bits.astype(np.uint8).flatten()
    pad = (-len(bits)) % pack_width
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    words = []
    for idx in range(0, len(bits), pack_width):
        word = 0
        chunk = bits[idx : idx + pack_width]
        for bit in chunk:
            word = (word << 1) | int(bit)
        words.append(word)
    return words


def write_hex(words: Iterable[int], output: Path, width_bits: int):
    digits = max(1, (width_bits + 3) // 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for word in words:
            mask = (1 << width_bits) - 1
            f.write(f"{(int(word) & mask):0{digits}X}\n")


def write_mif(words: Iterable[int], output: Path, width_bits: int):
    words = list(words)
    digits = max(1, (width_bits + 3) // 4)
    with output.open("w") as f:
        f.write(f"WIDTH = {width_bits};\n")
        f.write(f"DEPTH = {len(words)};\n")
        f.write("ADDRESS_RADIX = HEX;\n")
        f.write("DATA_RADIX = HEX;\n")
        f.write("CONTENT BEGIN\n")
        for idx, word in enumerate(words):
            f.write(f"{idx:X} : {(int(word) & ((1 << width_bits) - 1)):0{digits}X};\n")
        f.write("END;\n")


def fold_bn_threshold(bn: torch.nn.BatchNorm2d) -> np.ndarray:
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    sigma = np.sqrt(var + bn.eps)
    threshold = -beta * sigma / gamma + mean
    return q88(threshold)


def export_conv1(model: BCNNBinarized, out_dir: Path, summary: Dict):
    weight = model.features.conv1.weight.detach().cpu().numpy()
    quant = int8_sym(weight)
    words = quant.reshape(-1).astype(np.int8).astype(np.uint8)
    write_hex(words, out_dir / "conv1_weights.hex", 8)
    write_mif(words, out_dir / "conv1_weights.bin", 8)
    threshold = fold_bn_threshold(model.features.bn1)
    write_hex(threshold.astype(np.uint16), out_dir / "bn1_threshold.hex", 16)
    write_mif(threshold.astype(np.uint16), out_dir / "bn1_threshold.bin", 16)
    summary["conv1"] = {"shape": list(weight.shape), "words": int(words.size), "width_bits": 8}


def export_binary_conv(weight: torch.Tensor, out_prefix: Path, alpha_axis: int, summary_key: str, summary: Dict):
    weight_np = weight.detach().cpu().numpy()
    sign_bits = (weight_np >= 0).astype(np.uint8)
    packed = pack_bits(sign_bits)
    write_hex(packed, out_prefix.with_name(out_prefix.name + "_weights_packed.hex"), 32)
    write_mif(packed, out_prefix.with_name(out_prefix.name + "_weights_packed.bin"), 32)
    alpha = np.mean(np.abs(weight_np).reshape(weight_np.shape[0], -1), axis=1)
    alpha_q = q88(alpha)
    write_hex(alpha_q.astype(np.uint16), out_prefix.with_name(out_prefix.name + "_alpha.hex"), 16)
    write_mif(alpha_q.astype(np.uint16), out_prefix.with_name(out_prefix.name + "_alpha.bin"), 16)
    summary[summary_key] = {
        "shape": list(weight_np.shape),
        "packed_words": len(packed),
        "alpha_words": int(alpha_q.size),
    }


def export_fc(model: BCNNBinarized, out_dir: Path, summary: Dict):
    weight = q88(model.fc.weight.detach().cpu().numpy())
    bias = q88(model.fc.bias.detach().cpu().numpy())
    write_hex(weight.astype(np.uint16).reshape(-1), out_dir / "fc_weights.hex", 16)
    write_hex(bias.astype(np.uint16), out_dir / "fc_bias.hex", 16)
    summary["fc"] = {"shape": list(weight.shape), "bias_shape": list(bias.shape)}


def export_model_summary(model: BCNNBinarized, out_dir: Path, summary: Dict):
    output = out_dir / "model_summary.json"
    output.write_text(json.dumps(summary, indent=2))


def export_weight_stats(model: BCNNBinarized, out_dir: Path):
    lines = []
    for name, tensor in model.state_dict().items():
        if tensor.dtype.is_floating_point:
            arr = tensor.detach().cpu().numpy()
            lines.append(
                f"{name}: min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f} std={arr.std():.6f}"
            )
    (out_dir / "weight_stats.txt").write_text("\n".join(lines))


def tensor_to_packed_hex(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().numpy()
    bits = (array >= 0).astype(np.uint8)
    return "".join(f"{word:08X}" for word in pack_bits(bits))


def generate_test_vectors(model: BCNNBinarized, loaders, out_dir: Path, count: int = 10):
    images_hex = []
    labels = []
    golden = {"test_images": []}
    seen = 0
    for images, batch_labels in loaders["test"]:
        for image, label in zip(images, batch_labels):
            if seen >= count:
                break
            image_batch = image.unsqueeze(0)
            feats = model.features.forward_features(image_batch)
            logits = model(image_batch)
            np_img = image.detach().cpu().numpy()
            de_norm = np_img.copy()
            for c, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
                de_norm[c] = (de_norm[c] * std + mean) * 255.0
            de_norm = np.clip(np.round(de_norm), 0, 255).astype(np.uint8)
            flat = de_norm.transpose(1, 2, 0).reshape(-1)
            images_hex.extend(flat.tolist())
            labels.append(int(label))
            golden["test_images"].append(
                {
                    "index": seen,
                    "label": int(label),
                    "pixel_data_hex": "".join(f"{byte:02X}" for byte in flat.tolist()),
                    "conv1_output_expected": np.clip(np.round(feats["conv1"].detach().cpu().numpy()), -128, 127)
                    .astype(np.int8)
                    .reshape(-1)
                    .tolist(),
                    "bconv2_dw_output_expected": tensor_to_packed_hex(feats["bconv2_dw"]),
                    "bconv2_pw_output_expected": tensor_to_packed_hex(feats["bconv2_pw"]),
                    "bconv3_dw_output_expected": tensor_to_packed_hex(feats["bconv3_dw"]),
                    "bconv3_pw_output_expected": tensor_to_packed_hex(feats["bconv3_pw"]),
                    "bconv4_dw_output_expected": tensor_to_packed_hex(feats["bconv4_dw"]),
                    "bconv4_pw_output_expected": tensor_to_packed_hex(feats["bconv4_pw"]),
                    "fc_output_expected": q88(logits.detach().cpu().numpy()).reshape(-1).tolist(),
                    "class_prediction": int(logits.argmax(dim=1).item()),
                }
            )
            seen += 1
        if seen >= count:
            break
    write_hex(images_hex, out_dir / "test_images.hex", 8)
    write_hex(labels, out_dir / "test_labels.hex", 8)
    (out_dir / "golden_reference.json").write_text(json.dumps(golden, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("../artifacts"))
    parser.add_argument("--weights-out", type=Path, default=Path("../weights"))
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    args = parser.parse_args()

    device = torch.device("cpu")
    model = BCNNBinarized().to(device)
    model.load_state_dict(torch.load(args.artifacts / "student_binarized.pth", map_location=device))
    model.eval()

    args.weights_out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict] = {}
    export_conv1(model, args.weights_out, summary)
    export_binary_conv(model.features.bconv2_dw.weight, args.weights_out / "bconv2_dw", 0, "bconv2_dw", summary)
    export_binary_conv(model.features.bconv2_pw.weight, args.weights_out / "bconv2_pw", 0, "bconv2_pw", summary)
    export_binary_conv(model.features.bconv3_dw.weight, args.weights_out / "bconv3_dw", 0, "bconv3_dw", summary)
    export_binary_conv(model.features.bconv3_pw.weight, args.weights_out / "bconv3_pw", 0, "bconv3_pw", summary)
    export_binary_conv(model.features.bconv4_dw.weight, args.weights_out / "bconv4_dw", 0, "bconv4_dw", summary)
    export_binary_conv(model.features.bconv4_pw.weight, args.weights_out / "bconv4_pw", 0, "bconv4_pw", summary)
    for bn_name, bn_module in [
        ("bconv2", model.features.bn2),
        ("bconv3", model.features.bn3),
        ("bconv4", model.features.bn4),
    ]:
        threshold = fold_bn_threshold(bn_module)
        write_hex(threshold.astype(np.uint16), args.weights_out / f"{bn_name}_bn_threshold.hex", 16)
        write_mif(threshold.astype(np.uint16), args.weights_out / f"{bn_name}_bn_threshold.bin", 16)
        summary[f"{bn_name}_bn_threshold"] = {"channels": int(threshold.size)}
    export_fc(model, args.weights_out, summary)
    export_model_summary(model, args.weights_out, summary)
    export_weight_stats(model, args.weights_out)

    loaders = create_dataloaders(DatasetConfig(data_root=args.data_root))
    generate_test_vectors(model, loaders, args.weights_out)


if __name__ == "__main__":
    main()
