# Project Description

## Title

EdgeProbe: Binary CNN Acceleration on a Constrained FPGA

## Objective

Build a compact edge-vision pipeline that:

1. trains a binary-friendly image classifier in software
2. exports hardware-friendly weights and thresholds
3. runs inference in RTL using XNOR and popcount instead of conventional multipliers

## Problem setup

- Input image size: `32 x 32`, grayscale
- Input transport: SPI byte stream
- Task: binary classification, `rock` vs `not-rock`
- FPGA target: Intel Cyclone IV `EP4CE115F29C7`

## ML flow

- Dataset source: Rock-Paper-Scissors from TensorFlow Datasets
- Teacher: MobileNetV2-based classifier
- Student: small BCNN trained with Straight-Through Estimator (STE)
- Distillation: teacher soft labels guide the student during training
- Export: `.mif` and `.hex` files for the RTL pipeline

## RTL flow

1. Receive bytes over SPI.
2. Threshold pixels into 1-bit activations.
3. Build `3 x 3` windows with line buffers.
4. Run binary convolution using XNOR plus popcount.
5. Pool and classify the resulting feature maps.
6. Emit a final 1-bit class decision.
