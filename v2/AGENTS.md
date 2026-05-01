# BCNN ImageNet Upgrade: Full RTL + Python Pipeline Implementation

## ROLE & OBJECTIVE

You are an expert FPGA/RTL engineer and ML engineer. Your task is to:

1. Audit all existing Verilog files for a toy BCNN (Binary Convolutional Neural Network)
2. Design and implement a NEW, larger RTL architecture that is compatible with ImageNet-pretrained models
3. Build a complete Python pipeline: train → binarize → export weights → simulate in ModelSim
4. Produce a rigorous comparison report: float baseline vs. binarized model (accuracy, inference time, resource usage)

Everything must be internally consistent — the Python pipeline's weight export format must exactly match the RTL's memory initialization format. No mismatches allowed.

---

## PART 0 — INVENTORY EXISTING VERILOG FILES

Before writing a single line of new code, do the following:

- List every .v and .vh file in the project directory
- For each file, extract: module name, port list, parameters, memory sizes, data widths
- Identify the current datapath width (1-bit weights, multi-bit activations, etc.)
- Identify how weights are currently loaded (hardcoded, .bin files, $readmemh, ROM modules)
- Identify the current architecture: layer order, filter sizes, number of filters, pooling type, output format
- Note any ModelSim testbenches (.v or .do files)
- Document any Quartus project files (.qpf, .qsf, .sdc)
- Produce a concise architecture summary table before proceeding

KNOWN BASELINE ARCHITECTURE (verify against actual files):
Input: 32x32 grayscale (1 channel)
Layer 1: Conv2D(1→8, 3x3) → BatchNorm → ReLU
Layer 2: Conv2D(8→16, 3x3) → BatchNorm → ReLU
Layer 3: GlobalAveragePooling2D
Layer 4: Dense(16→2) → Softmax
Task: Binary classification (Rock vs. Paper vs. Scissors, treated as binary)
Target HW: Cyclone IV EP4CE115F29C7
Tools: Quartus II 13.0sp1, ModelSim-Altera 10.1d

---

## PART 1 — NEW ARCHITECTURE DESIGN

### 1.1 Constraints

- Device: Cyclone IV EP4CE115F29C7
  - Logic Elements: ~114,480 LEs
  - Embedded Memory: ~3,888 Kbits (M9K blocks)
  - DSP blocks: 532
  - Max internal clock target: 50 MHz (be conservative)
- Input: 64x64 RGB (3 channels) — upsized to leverage ImageNet pretraining features
- Task: Rock-Paper-Scissors classification (3 classes) — NOT binary; treat as 3-class softmax
- Binarization scheme: XNOR-Net style (±1 binarized weights + inputs, XNOR+popcount replaces MAC)
- The Python model MUST use the same architecture as the RTL — no divergence

### 1.2 Recommended New Architecture

Design a BCNN inspired by MobileNet-style depthwise separable convolutions, but fully binarized.
Use knowledge distillation from a pretrained MobileNetV2 (ImageNet) teacher to train the student BCNN.
The RTL must implement the STUDENT architecture only — not the teacher.

STUDENT ARCHITECTURE (implement exactly this in both Python and RTL):

| Name           | Type                          | In Ch | Out Ch | Kernel | Stride | Input HW | Output HW |
| -------------- | ----------------------------- | ----- | ------ | ------ | ------ | -------- | --------- |
| conv1          | Standard Conv (NOT binarized) | 3     | 32     | 3x3    | 2      | 64x64    | 32x32     |
| bn1 + hardtanh | BatchNorm + sign binarize     | 32    | 32     | -      | -      | 32x32    | 32x32     |
| bconv2_dw      | Binarized Depthwise Conv      | 32    | 32     | 3x3    | 1      | 32x32    | 32x32     |
| bconv2_pw      | Binarized Pointwise Conv      | 32    | 64     | 1x1    | 1      | 32x32    | 32x32     |
| bn2 + hardtanh | BatchNorm + sign binarize     | 64    | 64     | -      | -      | 32x32    | 32x32     |
| maxpool1       | MaxPool 2x2, stride 2         | 64    | 64     | -      | -      | 32x32    | 16x16     |
| bconv3_dw      | Binarized Depthwise Conv      | 64    | 64     | 3x3    | 1      | 16x16    | 16x16     |
| bconv3_pw      | Binarized Pointwise Conv      | 64    | 128    | 1x1    | 1      | 16x16    | 16x16     |
| bn3 + hardtanh | BatchNorm + sign binarize     | 128   | 128    | -      | -      | 16x16    | 16x16     |
| maxpool2       | MaxPool 2x2, stride 2         | 128   | 128    | -      | -      | 16x16    | 8x8       |
| bconv4_dw      | Binarized Depthwise Conv      | 128   | 128    | 3x3    | 1      | 8x8      | 8x8       |
| bconv4_pw      | Binarized Pointwise Conv      | 128   | 256    | 1x1    | 1      | 8x8      | 8x8       |
| bn4 + hardtanh | BatchNorm + sign binarize     | 256   | 256    | -      | -      | 8x8      | 8x8       |
| gap            | GlobalAveragePooling2D        | 256   | 256    | -      | -      | 8x8      | 1x1       |
| fc             | Dense(256→3)                  | 256   | 3      | -      | -      | -        | -         |
| softmax        | Softmax (Python only)         | -     | -      | -      | -      | -        | -         |

NOTES:

- conv1 uses FULL PRECISION (int8 or float32 in Python; int8 in RTL) — first layer is NOT binarized
- All bconv layers use XNOR-Net: weights ∈ {+1,-1}, activations ∈ {+1,-1}, replaced by XNOR+popcount
- BatchNorm parameters (gamma, beta, mean, variance) are folded into thresholds for RTL
- Depthwise conv: each filter operates on a single channel independently
- Pointwise conv: 1x1 conv across all channels
- The fc layer uses full-precision accumulated popcount outputs scaled by learned alpha

### 1.3 XNOR-Net Binarization Rules

- Weight binarization: W_b = sign(W_real), alpha = mean(|W_real|) per output channel
- Activation binarization: A_b = sign(A_real) applied after BatchNorm
- XNOR operation: out = popcount(XNOR(W_b, A_b)) \* 2 - N where N = total input bits
- BN folding for threshold: threshold_i = -beta_i \* sigma_i / gamma_i + mean_i
  (if pre-BN activation > threshold → binarized output = +1, else -1)
- Alpha scaling (per-layer weight scale): applied at accumulation stage, folded into BN for RTL simplicity

---

## PART 2 — PYTHON PIPELINE

### 2.1 Environment

Python 3.9+, PyTorch 2.x, torchvision, numpy, matplotlib, pandas, tqdm
Install: pip install torch torchvision numpy matplotlib pandas tqdm

### 2.2 Dataset Preparation

- Dataset: Rock Paper Scissors (download via torchvision or Kaggle)
  URL: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
  Fallback: use torchvision ImageFolder with local path
- Preprocessing:
  - Resize to 64x64 (LANCZOS)
  - RGB (3 channels)
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet stats)
  - Augmentation (training only): RandomHorizontalFlip, ColorJitter(0.2,0.2,0.2)
- Split: 80% train, 10% val, 10% test (fixed seed=42)
- Batch size: 32

### 2.3 Teacher Model (Knowledge Distillation Source)

- Load MobileNetV2 pretrained on ImageNet (torchvision.models.mobilenet_v2(pretrained=True))
- Replace final classifier: nn.Linear(1280, 3)
- Fine-tune for 10 epochs on RPS dataset (learning rate 1e-4, Adam)
- Save as teacher_mobilenetv2.pth
- Log: train loss, val loss, val accuracy per epoch

### 2.4 Student BCNN — Full Precision Training with Distillation

Implement the exact student architecture from Part 1.2 as a PyTorch nn.Module called BCNNStudent.

- Training:
  - Temperature T=4 for KD soft labels
  - Loss = 0.5 _ CE(student_logits, hard_labels) + 0.5 _ KL(student_soft, teacher_soft) \* T^2
  - Optimizer: Adam, lr=1e-3, weight_decay=1e-4
  - Scheduler: CosineAnnealingLR, T_max=50
  - Epochs: 50
  - Save best checkpoint by val accuracy: student_fp32.pth
- Log per epoch: train_loss, val_loss, val_acc, teacher_val_acc

### 2.5 Binarization

Implement a BinarizeLinear and BinarizeConv2d module:

- Forward pass: binarize weights on the fly (STE for gradients)
- Weight binarization: sign(weight), alpha computed per output channel
- Activation binarization: applied via a BinarizeActivation module (sign, STE backward)

Fine-tune the binarized student (BCNNBinarized) for 30 more epochs:

- Start from student_fp32.pth weights
- Use same loss (CE only, no distillation in binarized fine-tune — teacher too different)
- lr=5e-4, CosineAnnealingLR
- Save: student_binarized.pth
- Log: train_loss, val_loss, val_acc

### 2.6 Weight Export to Hex/Bin Files

After binarization fine-tune converges, export ALL parameters for RTL use.
For each layer, export the following files:

FILE NAMING CONVENTION: {layer*name}*{param_type}.{ext}

conv1_weights.hex — int8 quantized weights, row-major [Out, In, H, W], 2's complement hex
conv1_weights.bin — Quartus .bin
format for M9K ROM initialization

For all binarized conv layers (bconv\*):
{layer}\_weights_packed.hex — weights packed as 32-bit words (32 binary weights per word, MSB first)
{layer}\_weights_packed.bin

{layer}\_alpha.hex — per-output-channel alpha scale (Q8.8 fixed point, 16-bit)
{layer}\_bn_threshold.hex — per-channel BN threshold (Q8.8 fixed point, 16-bit)
{layer}\_bn_threshold.bin

fc_weights.hex — full-precision fc weights, Q8.8 fixed-point (16-bit), [3, 256]
fc_bias.hex — fc biases, Q8.8 fixed-point (16-bit), [3]

Also export:
model_summary.json — all layer names, shapes, parameter counts, memory usage
weight_stats.txt — min/max/mean/std for each layer before/after binarization

MIF FORMAT (Quartus-compatible):
WIDTH = <bits_per_word>;
DEPTH = <num_words>;
ADDRESS_RADIX = HEX;
DATA_RADIX = HEX;
CONTENT BEGIN
0 : <hex_data>;
...
END;

HEX FORMAT: plain text, one word per line, no prefix (not 0x...), uppercase

### 2.7 Comparison & Reporting

Compute and log ALL of the following in a results table (CSV + printed):

| Metric                  | Float Teacher | Float Student | Binarized Student |
| ----------------------- | ------------- | ------------- | ----------------- |
| Test accuracy (%)       |               |               |
| Top-1 error (%)         |               |               |
| Model size (MB)         |               |               |
| Param count             |               |               |
| Inference time CPU (ms) |               |               |
| Inference time GPU (ms) |               |               |
| MACs (millions)         |               |               |
| BOPs (binary ops, M)    |               |               | ← binarized only  |

- Inference time: average over 100 forward passes on a batch of 32, warm-up 10 passes
- Use torch.cuda.Event for GPU timing, time.perf_counter for CPU timing
- BOPs: count XNOR+popcount ops for each binarized layer
- MACs: use thop or torchinfo library

Save: comparison_report.csv, comparison_report.png (bar charts for accuracy and inference time)

Also generate confusion matrices for all 3 models (matplotlib, save as PNG).

---

## PART 3 — RTL IMPLEMENTATION

### 3.1 File Structure

Organize RTL as follows:

rtl/
top/
bcnn_top.v — top-level, connects all layers, drives AXI-lite or simple handshake
layers/
conv1_layer.v — int8 standard conv (not binarized)
bconv_dw_layer.v — parameterized binarized depthwise conv
bconv_pw_layer.v — parameterized binarized pointwise conv
bn_threshold.v — batchnorm folded as threshold comparator
maxpool_layer.v — 2x2 maxpool, parameterized
gap_layer.v — global average pool, parameterized
fc_layer.v — full-precision fc with fixed-point multiply-accumulate
mem/
weight_rom.v — parameterized M9K ROM wrapper
fifo_buffer.v — simple synchronous FIFO for inter-layer buffering
common/
xnor_popcount.v — parameterized XNOR + popcount unit (core primitive)
fixed_point_mult.v — Q8.8 fixed-point multiplier
sigmoid_lut.v — optional LUT-based sigmoid/softmax approx
tb/
tb_bcnn_top.v — full system testbench
tb_xnor_popcount.v — unit testbench for XNOR-popcount
tb_conv1_layer.v — unit testbench for conv1
scripts/
run_sim.do — ModelSim do-file for full simulation
run_unit_tests.do — ModelSim do-file for unit tests
compile_quartus.tcl — Quartus II TCL compilation script

### 3.2 RTL Design Principles

- All modules: synchronous reset (active-high rst), posedge clk
- Use parameters for filter counts, input dims, data widths — do NOT hardcode
- Handshake protocol: valid/ready (simple, not full AXI4-Stream — Quartus 13 compatibility)
- Data representation:
  - Binarized weights/activations: 1-bit packed into 32-bit words (parameterized PACK_WIDTH=32)
  - int8 activations for conv1: signed 8-bit, 2's complement
  - Fixed-point for BN thresholds and fc: Q8.8 (16-bit signed)
  - Popcount outputs: ceil(log2(N))+1 bits where N = number of bits being counted

### 3.3 xnor_popcount.v — Core Primitive

Implement a parameterized XNOR-popcount unit:
parameter N = 32; // number of bits to XNOR and count
input [N-1:0] a; // binarized activation
input [N-1:0] b; // binarized weight
output [$clog2(N):0] out; // popcount result

Implementation: use a tree-of-adders (not loop) for synthesis efficiency.
Provide a 3-stage pipelined version (xnor_popcount_pipe.v) for use in conv layers.

### 3.4 bconv_dw_layer.v — Binarized Depthwise Conv

Parameters:
IN_CHANNELS, KERNEL_H=3, KERNEL_W=3, IMG_H, IMG_W, PACK_WIDTH=32
Behavior:

- Load kernel weights from weight_rom (one ROM per channel)
- Slide 3x3 window across spatial dims (line buffer approach, not full frame buffer)
- For each position: compute XNOR-popcount of 9-bit window vs 9-bit kernel
- Apply BN threshold from bn_threshold ROM → output 1-bit binarized result
- Stream outputs with valid signal

Line buffer: implement as 2 line buffers (rows) of shift registers, not full RAM if possible.
If line buffers exceed 1024 bits, use M9K RAM.

### 3.5 bconv_pw_layer.v — Binarized Pointwise Conv

Parameters:
IN_CHANNELS, OUT_CHANNELS, IMG_H, IMG_W, PACK_WIDTH=32
Behavior:

- For each spatial position: accumulate XNOR-popcount over all IN_CHANNELS
- Weights: [OUT_CHANNELS, IN_CHANNELS] stored in ROM, packed as PACK_WIDTH-bit words
- Output 1-bit binarized result after BN threshold

### 3.6 conv1_layer.v — First Layer (Full Precision Int8)

Parameters:
IN_CHANNELS=3, OUT_CHANNELS=32, KERNEL_H=3, KERNEL_W=3, IMG_H=64, IMG_W=64, STRIDE=2
Behavior:

- int8 × int8 multiply (use DSP blocks): 18-bit signed accumulator
- Weights loaded from conv1_weights ROM
- Output: int8 (clip/saturate after accumulation)
- After conv: apply BN threshold → 1-bit binarized activation (hardtanh ≈ sign)

### 3.7 weight_rom.v — Parameterized ROM Wrapper

parameter DATA_WIDTH = 32;
parameter DEPTH = 1024;
parameter INIT_FILE = ""; // path to .bin
file

Implement as inferred M9K ROM (synchronous read, 1-cycle latency).
Use $readmemh for simulation, .bin
file for Quartus synthesis.
Include both:
`ifdef SIMULATION
      initial $readmemh(INIT_FILE_HEX, mem);
    `else
// Quartus: use altsyncram with mif
`endif

### 3.8 bcnn_top.v — Top Level

- Interface:
  input clk, rst
  input [7:0] pixel_in — one pixel per clock (R, G, B interleaved)
  input pixel_valid
  output [1:0] class_out — 0=Rock, 1=Paper, 2=Scissors
  output result_valid

- State machine: IDLE → LOAD_IMAGE → CONV1 → BCONV2 → ... → FC → OUTPUT
- Implement using generate blocks and instantiation of layer modules
- Drive each layer's valid/ready from state machine
- Gate clocks or use enable signals for power saving (optional but note if skipped)

### 3.9 tb_bcnn_top.v — Full System Testbench

- Load test images from hex file: test_images.hex (generated by Python pipeline)
  Format: one image per block, 64×64×3 bytes, 8-bit unsigned, row-major, RGB interleaved
- Feed pixels at 1 per clock cycle (pixel_valid = 1 every clock)
- Compare class_out against expected labels from test_labels.hex
- Report: total images, correct, accuracy (printed via $display)
- Dump VCD: $dumpfile("bcnn_tb.vcd"); $dumpvars(0, tb_bcnn_top);
- Add assertion: if result_valid not seen within 64*64*3 + 10000 clocks after last pixel, flag TIMEOUT

### 3.10 run_sim.do — ModelSim Script

vlib work
vmap work work
vlog ../rtl/common/xnor_popcount.v
vlog ../rtl/common/fixed_point_mult.v
vlog ../rtl/mem/weight_rom.v
vlog ../rtl/mem/fifo_buffer.v
vlog ../rtl/layers/conv1_layer.v
vlog ../rtl/layers/bconv_dw_layer.v
vlog ../rtl/layers/bconv_pw_layer.v
vlog ../rtl/layers/bn_threshold.v
vlog ../rtl/layers/maxpool_layer.v
vlog ../rtl/layers/gap_layer.v
vlog ../rtl/layers/fc_layer.v
vlog ../rtl/top/bcnn_top.v
vlog ../rtl/tb/tb_bcnn_top.v
vsim -t 1ns -voptargs="+acc" tb_bcnn_top
add wave -radix hex /tb_bcnn_top/\*
run -all
quit

---

## PART 4 — ALIGNMENT & VERIFICATION

### 4.1 Golden Reference Check

The Python pipeline must output a GOLDEN REFERENCE FILE for RTL verification:
golden_reference.json:
{
"test_images": [
{
"index": 0,
"label": 1,
"pixel_data_hex": "...", // flat hex of 64*64*3 bytes
"conv1_output_expected": [...], // int8 values after conv1
"bconv2_dw_output_expected": [...], // 1-bit packed hex after bconv2_dw
...
"fc_output_expected": [...], // Q8.8 fixed-point
"class_prediction": 1
}
]
}

The RTL testbench must compare against this file at each layer boundary (use $readmemh to load and compare layer by layer).

### 4.2 Numerical Alignment Rules

- ALL fixed-point arithmetic in RTL must match Python's behavior to within rounding error
- Python pipeline must use the SAME quantization scheme as RTL:
  - int8 for conv1: scale = max(|W|) / 127, zero_point = 0
  - Q8.8 for BN thresholds and fc weights
- If using Q8.8: Python must quantize as round(val \* 256).clip(-32768, 32767).astype(np.int16)
- BN threshold folding: compute thresholds in Python and verify sign match against full-precision BN output
- Test: run same 10 test images through Python (quantized inference) and RTL simulation. Accuracy must match exactly on those 10 images.

### 4.3 Memory Budget Check

After RTL is complete, compute and report:
Layer | Weight bits | ROM words | M9K blocks (9Kbit each)
---------------------|-------------|-----------|------------------------
conv1 | 3*32*3*3*8 | |
bconv2_dw | 32*1*9 | |
bconv2_pw | 32*64 | |
bconv3_dw | 64*1*9 | |
bconv3_pw | 64*128 | |
bconv4_dw | 128*1*9 | |
bconv4_pw | 128*256 | |
fc | 256*3\*16 | |
BN thresholds (all) | | |
TOTAL | | | Must be < 432 (total available)

If total M9K usage exceeds 432 blocks, flag and suggest: reduce OUT_CHANNELS of bconv4_pw to 128.

---

## PART 5 — DELIVERABLES CHECKLIST

Python pipeline files:
[ ] dataset.py — dataset loading, augmentation, dataloaders
[ ] models.py — BCNNStudent, BCNNBinarized, TeacherMobileNetV2
[ ] train_teacher.py — train and save teacher model
[ ] train_student.py — full-precision student training with distillation
[ ] binarize.py — binarization modules (STE), fine-tuning loop
[ ] export_weights.py — weight export to .hex and .bin

[ ] compare.py — inference time, accuracy, BOPs comparison + plots
[ ] run_pipeline.sh — single script to run entire pipeline end-to-end
[ ] requirements.txt — pinned dependencies

RTL files:
[ ] rtl/common/xnor_popcount.v
[ ] rtl/common/xnor_popcount_pipe.v
[ ] rtl/common/fixed_point_mult.v
[ ] rtl/mem/weight_rom.v
[ ] rtl/mem/fifo_buffer.v
[ ] rtl/layers/conv1_layer.v
[ ] rtl/layers/bconv_dw_layer.v
[ ] rtl/layers/bconv_pw_layer.v
[ ] rtl/layers/bn_threshold.v
[ ] rtl/layers/maxpool_layer.v
[ ] rtl/layers/gap_layer.v
[ ] rtl/layers/fc_layer.v
[ ] rtl/top/bcnn_top.v
[ ] rtl/tb/tb_bcnn_top.v
[ ] rtl/tb/tb_xnor_popcount.v
[ ] rtl/scripts/run_sim.do
[ ] rtl/scripts/run_unit_tests.do
[ ] rtl/scripts/compile_quartus.tcl

Exported weight files (generated by export_weights.py):
[ ] weights/conv1_weights.hex + .bin

[ ] weights/bconv*\_weights_packed.hex + .bin
(one per bconv layer)
[ ] weights/bconv*\_bn_threshold.hex + .bin

[ ] weights/fc_weights.hex + fc_bias.hex
[ ] weights/model_summary.json
[ ] test_images.hex + test_labels.hex (for RTL testbench)
[ ] golden_reference.json

Reports:
[ ] comparison_report.csv
[ ] comparison_report.png
[ ] memory_budget.txt
[ ] confusion_matrix_teacher.png
[ ] confusion_matrix_student_fp32.png
[ ] confusion_matrix_student_binarized.png

---

## IMPLEMENTATION ORDER (follow this sequence)

1. Inventory existing Verilog files → produce summary
2. Write models.py (BCNNStudent + BCNNBinarized + Teacher)
3. Write dataset.py
4. Train teacher, train student (full-precision) — verify student val accuracy > 80%
5. Binarize and fine-tune — verify val accuracy > 70% (acceptable degradation)
6. Write export_weights.py and generate all .hex/.bin
   files
7. Write xnor_popcount.v + unit testbench → simulate in ModelSim → verify correctness
8. Write conv1_layer.v → unit testbench → verify against Python golden reference
9. Write bconv_dw_layer.v + bconv_pw_layer.v → verify
10. Write remaining layer modules
11. Write bcnn_top.v + full system testbench
12. Run full simulation in ModelSim → compare accuracy against Python
13. Write compare.py → generate comparison report
14. Compile in Quartus → check timing/resource reports
15. Final deliverables check

---

## CRITICAL CONSTRAINTS (never violate)

- Quartus II 13.0 compatibility: do NOT use SystemVerilog syntax. Verilog 2001 only.
  No: logic, always_ff, always_comb, typedef, enum, packed structs, ++ operator
  Yes: reg, wire, always @(posedge clk), parameter, generate/endgenerate, $clog2
- $clog2 may not be supported in Quartus 13 — use a custom clog2 function or hardcode widths
- All .bin
  files must use Quartus II .bin
  format exactly (see Part 2.6)
- ModelSim-Altera 10.1d: use vlog (not vcom), .do scripts, -voptargs="+acc"
- Weight files must be regenerated fresh if the architecture changes — never reuse stale exports
- Python quantized inference (int8/Q8.8) must numerically match RTL within ±1 LSB
- Do not use floating-point in RTL (no $itor, real, etc.)
- If any layer's M9K usage alone exceeds 200 blocks, split into ping-pong double-buffered approach
