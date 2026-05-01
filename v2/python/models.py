from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def hard_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= 1).to(grad_output.dtype)
        return grad_output * mask


class BinarizeActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SignSTE.apply(x)


class BinarizedConv2d(nn.Conv2d):
    def _binarized_weight(self) -> torch.Tensor:
        w = self.weight
        alpha = w.abs().flatten(1).mean(dim=1, keepdim=True)
        alpha = alpha.view(-1, 1, 1, 1)
        w_sign = SignSTE.apply(w)
        return w_sign * alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self._binarized_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BinarizedLinear(nn.Linear):
    def _binarized_weight(self) -> torch.Tensor:
        w = self.weight
        alpha = w.abs().mean(dim=1, keepdim=True)
        return SignSTE.apply(w) * alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._binarized_weight(), self.bias)


class ConvBnSign(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, groups: int = 1):
        super().__init__()
        self.conv = BinarizedConv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = BinarizeActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FPConvBnSign(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = BinarizeActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StudentFeatures(nn.Module):
    def __init__(self, binarized: bool):
        super().__init__()
        block = ConvBnSign if binarized else None
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.sign1 = BinarizeActivation()
        if binarized:
            dw2 = BinarizedConv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
            pw2 = BinarizedConv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
            dw3 = BinarizedConv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
            pw3 = BinarizedConv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
            dw4 = BinarizedConv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
            pw4 = BinarizedConv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            dw2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
            pw2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
            dw3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
            pw3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
            dw4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
            pw4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bconv2_dw = dw2
        self.bconv2_pw = pw2
        self.bn2 = nn.BatchNorm2d(64)
        self.sign2 = BinarizeActivation()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bconv3_dw = dw3
        self.bconv3_pw = pw3
        self.bn3 = nn.BatchNorm2d(128)
        self.sign3 = BinarizeActivation()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bconv4_dw = dw4
        self.bconv4_pw = pw4
        self.bn4 = nn.BatchNorm2d(256)
        self.sign4 = BinarizeActivation()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = {}
        x = self.conv1(x)
        feats["conv1"] = x
        x = self.sign1(self.bn1(x))
        feats["bn1_sign"] = x
        x = self.bconv2_dw(x)
        feats["bconv2_dw"] = x
        x = self.bconv2_pw(x)
        feats["bconv2_pw"] = x
        x = self.sign2(self.bn2(x))
        feats["bn2_sign"] = x
        x = self.pool1(x)
        feats["maxpool1"] = x
        x = self.bconv3_dw(x)
        feats["bconv3_dw"] = x
        x = self.bconv3_pw(x)
        feats["bconv3_pw"] = x
        x = self.sign3(self.bn3(x))
        feats["bn3_sign"] = x
        x = self.pool2(x)
        feats["maxpool2"] = x
        x = self.bconv4_dw(x)
        feats["bconv4_dw"] = x
        x = self.bconv4_pw(x)
        feats["bconv4_pw"] = x
        x = self.sign4(self.bn4(x))
        feats["bn4_sign"] = x
        x = self.gap(x)
        feats["gap"] = x
        return feats


class BCNNStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = StudentFeatures(binarized=False)
        self.fc = nn.Linear(256, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features.forward_features(x)
        gap = feats["gap"].flatten(1)
        return self.fc(gap)


class BCNNBinarized(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = StudentFeatures(binarized=True)
        self.fc = BinarizedLinear(256, 3)

    def load_from_fp32(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys while loading fp32 checkpoint: {unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features.forward_features(x)
        gap = feats["gap"].flatten(1)
        return self.fc(gap)


class TeacherMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)

