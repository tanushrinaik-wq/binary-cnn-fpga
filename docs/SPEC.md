This document is designed to guide a high-level AI coding agent through the end-to-end implementation of an RTL-based Binary CNN (BCNN) Accelerator. Since physical hardware verification is a secondary priority, the focus is on Cycle-Accurate RTL Simulation and Resource-Performance Benchmarking using Quartus II 13.0.
Project Specification: BCNN Edge Vision Accelerator
Target Hardware: Intel Cyclone IV (EP4CE115)
Development Environment: Quartus II 13.0 / ModelSim-Altera
Input Source: Arduino Nicla Vision (Simulated via SPI)
Core Logic: Binarized Weights/Activations, XNOR-Popcount Arithmetic

1. System Architecture Requirement
   The agent must implement a three-stage pipeline:
   SPI Slave Receiver: 4-wire SPI (MOSI, MISO, SCK, CS). Must handle a stream of 1-bit or 8-bit image data and buffer it into an internal Line Buffer or RAM.
   BCNN Compute Engine: \* Binarization Layer: Convert input to ±1 (mapped to 0 and 1).
   XNOR-Popcount Core: A parallel array of XNOR gates followed by a combinatorial adder tree (Popcount) to calculate the dot product of binarized weights and activations.
   Thresholding/Activation: Implement a hard-threshold function to determine the output bit for the next layer.
   Controller (FSM): Orchestrate data flow from the SPI buffer through the layers and output the final classification index.
2. Phase-Wise Tasks for the AI Agent
   Phase A: Data Preparation (Python Script)
   Weight Extraction: Take a pre-trained CNN (Keras/PyTorch) for a task like defect detection.
   Binarization: Convert weights to 1-bit (Sign function).
   Memory Initialization: Generate .mif or .hex files for Quartus ROMs containing these binary weights.
   Image-to-Hex: Convert sample images into a hex format that the Verilog testbench can read using $readmemh.
Phase B: RTL Development (Verilog)
Module - spi_slave.v: Handle 20MHz+ SPI clock synchronization.
Module - bcnn_layer.v: Create a parameterized module where the number of filters and kernel size (e.g., 3x3) can be defined.
Module - popcount.v: Implement an efficient balanced tree of adders to minimize the combinational path and maximize $F_{max}$.
   Top Module - accelerator*top.v: Wire the SPI input, memory controllers, and the BCNN compute engine.
   Phase C: Simulation & Validation (ModelSim)
   Testbench Construction: * Instantiate the accelerator*top.
   Simulate the Arduino Nicla Vision by toggling the SPI signals at 20MHz.
   Load input image data from the Python-generated hex files.
   Performance Metrics: * Implement a latency_counter in the testbench.
   Log the exact number of clock cycles from the first SPI bit to the valid classification signal.
   Calculate Total Inference Time = $Cycles \times (1 / Clock\ Frequency)$.
3. Benchmarking & Reporting Requirements
   The agent must produce the following outputs for review:
   Quartus Compilation Report:
   Logic Element (LE) Usage: Prove the design fits in a Cyclone IV.
   $F_{max}$ Analysis: Report the maximum possible clock frequency.
   ModelSim Waveform (VCD): A screenshot/dump showing the input stream and the classification result appearing on the output pins.
   Comparative Analysis (The "FPGA vs. MCU" Argument): \* Calculate the Theoretical GOPs (Giga-Operations per Second) of the FPGA design.
   Compare this against a baseline MCU execution (approximate cycles for sequential XNOR-popcount on an ARM Cortex-M7).
4. Specific Implementation Constraints
   No Multipliers: Ensure no \* operators are used in the compute core; everything must be XNOR and bit-addition.
   Synchronous Design: All logic must be clocked on the rising edge of the global FPGA clock. Use a dual-clock FIFO if the SPI domain is asynchronous.
   Memory Mapping: Use Internal Block RAM (M4K blocks) for activations to avoid off-chip latency.

=======================================================================================================================================

BCNN Edge Vision Accelerator — Phase A & B Roadmap
Here's a granular step-by-step breakdown of everything you need to do, in order.

Phase A: Data Preparation (Jupyter Notebook)
Step 1 — Define Your Task & Dataset
Pick a simple binary or multi-class classification task that's realistic for edge vision. Defect detection (good vs. defective) is ideal because it's binary and the dataset is manageable. You can use a public dataset like MVTec AD, or generate synthetic "defect" images. The key constraint is that your images should be small — 28×28 or 32×32 grayscale — because the FPGA's internal BRAM is limited on a Cyclone IV.
Step 2 — Build & Train a Small CNN in Keras
Design a CNN that is intentionally tiny — you're going to binarize it, so accuracy will take a hit. A good starting architecture is:

Input: 32×32×1 (grayscale)
Conv2D (8 filters, 3×3) → BatchNorm → ReLU
Conv2D (16 filters, 3×3) → BatchNorm → ReLU
GlobalAveragePooling
Dense → Softmax (or Sigmoid for binary)

Train it with standard float32 weights first to establish a full-precision baseline accuracy. Save this model. This baseline is what you'll compare against after binarization.
Step 3 — Binarize the Weights (Sign Function)
After training, extract each convolutional layer's weight tensors. Apply the sign function: any weight ≥ 0 maps to +1 (represented as binary 1), any weight < 0 maps to -1 (represented as binary 0). Do not binarize the bias or BatchNorm parameters — those get folded into the threshold constants at the next step.
Step 4 — Fold BatchNorm into Thresholds
This is a critical step that's often skipped in naive implementations. The BatchNorm parameters (γ, β, running mean, running variance) can be algebraically combined with the layer's bias to produce a single integer threshold per output channel. After your XNOR-popcount gives you a dot product sum, you compare it against this precomputed threshold rather than running a full BatchNorm at inference time. Compute these thresholds in the notebook and store them as a list of integers.
Step 5 — Generate .mif Files for Quartus ROMs
For each convolutional layer, flatten the binarized weight tensor into a 1D bitstream. Format it as a Quartus Memory Initialization File (.mif). The structure of a .mif is straightforward — it has a header declaring depth (number of words) and width (bits per word), followed by address-value pairs in hex or binary. One .mif per layer is the cleanest approach. Also generate a separate .mif for your threshold constants.
Step 6 — Convert Test Images to Hex for the Testbench
Pick 5–10 representative test images (both classes). Convert each to grayscale, resize to your input resolution, flatten to a 1D array of pixel values, and write them out as a hex file where each line is one byte. These files will be loaded in ModelSim using $readmemh. Keep a ground-truth label file alongside them so your testbench can verify output correctness automatically.
Step 7 — Document Accuracy Numbers
In the notebook, run inference using your software-simulated binarized model (manually applying sign to weights and thresholding the popcount outputs) on your test set. Record this accuracy. It will be lower than the float baseline — that's expected. This number is your reference point; if your RTL produces different results, there's a bug.

Phase B: RTL Development (Verilog)
Step 1 — Set Up Your Quartus Project
Create a new Quartus II 13.0 project targeting the EP4CE115F29C7 device. Set your top-level module name, organize your directory with subfolders for rtl/, sim/, mem/ (for your .mif files), and tb/. Add all .mif files to the project so Quartus can initialize the ROMs during synthesis.
Step 2 — Implement spi_slave.v
The SPI slave needs to handle a 20MHz SPI clock coming from the Arduino Nicla Vision while your FPGA runs on a much faster internal clock (say 50MHz or 100MHz). Because these are in different clock domains, you must synchronize the SCK signal into the FPGA clock domain using a 2-stage flip-flop synchronizer. Detect the rising edge of SCK by comparing the current and previous synchronized values. On each detected rising edge, shift MOSI into a shift register. Once you've collected 8 bits (or 1 bit, depending on your protocol choice), assert a byte_valid flag and push the data into a small FIFO. The FIFO output feeds into your compute engine in the FPGA clock domain. CS (chip select, active low) should reset the shift register and signal start/end of a frame.
Step 3 — Implement the Dual-Clock FIFO
This bridges the SPI clock domain and the FPGA clock domain. You can use Quartus's built-in IP (ALTFIFO with independent clocks) or write a simple Gray-code pointer based async FIFO yourself. The write side is clocked by the synchronized SPI clock, the read side by your main FPGA clock. Size it to hold at least one full row of your input image to prevent underflow during compute.
Step 4 — Implement popcount.v
This is the arithmetic heart of the design. The module takes two N-bit vectors (binarized input activations and binarized weights) and counts the number of positions where they agree (XNOR), then returns that count. Build this as a balanced binary adder tree: first XNOR all bit pairs to get N bits, then add pairs of those bits to get N/2 two-bit sums, then add pairs of those to get N/4 three-bit sums, and so on until you reach a single log2(N)-bit result. No \* operators anywhere — only ^~ (XNOR) and + on small bit-width values. Make this module fully parameterized by N so you can reuse it for different kernel sizes.
Step 5 — Implement bcnn_layer.v
This module wraps the popcount into a full convolutional layer. It receives a flattened kernel-sized window of binarized activations from a line buffer, fetches the corresponding binarized weights from ROM, runs popcount for each filter, compares the result against the precomputed threshold (fetched from a threshold ROM), and outputs a 1-bit result per filter. Parameterize it by: number of input channels, number of filters, and kernel size. The line buffer logic (storing previous rows to form the sliding 3×3 window) lives here or in a dedicated line_buffer.v module.
Step 6 — Implement the Line Buffer
For a 3×3 convolution on a streaming input, you need to store the previous two rows. Implement this as two shift registers (or two inferred BRAMs if rows are wide enough) that hold one full row each. As new pixels arrive from the FIFO, they shift through. Once enough rows are buffered, the 3×3 window can be assembled from the tail of row N-2, the tail of row N-1, and the current incoming row N pixels.
Step 7 — Implement the Binarization Input Stage
The first thing that happens to raw pixel data coming out of the FIFO is binarization. This is trivial: compare each 8-bit pixel against a threshold (typically 128, or the mean of the batch). If the pixel is above threshold, output 1; otherwise output 0. This single-bit value then feeds into the line buffer.
Step 8 — Implement accelerator_top.v
Wire everything together. The port list should match what the Nicla Vision will drive: MOSI, MISO, SCK, CS, and then your output classification bits plus a valid signal. Internally instantiate: the SPI slave, the async FIFO, the input binarization logic, the line buffer, Layer 1 BCNN, Layer 2 BCNN, the global average pooling (which in binarized form is just a popcount across spatial positions), and the final dense layer. Connect them in the pipeline order.
Step 9 — Implement the FSM Controller
The FSM coordinates everything. It has roughly these states:

IDLE: Waiting for CS to go low
RECEIVE: Clocking in bytes from SPI, filling the line buffer
COMPUTE_L1: Trigger Layer 1 computation once the first 3×3 window is ready
COMPUTE_L2: Feed L1 outputs into Layer 2
POOLING: Accumulate spatial outputs for global average pooling
CLASSIFY: Compare pooled outputs, find argmax, assert output
OUTPUT: Hold the classification result on output pins, assert valid

Transitions between states are driven by counters tracking how many pixels/windows have been processed.
Step 10 — Assign Pins and Set Timing Constraints
In Quartus, open the Pin Planner and assign your SPI signals and output pins to physical FPGA pins on the EP4CE115 development board. In the TimeQuest Timing Analyzer, create two clock constraints: one for your internal PLL-generated clock and one for the incoming SPI SCK (treated as a separate clock domain). Set a false path between the two clock domains (they're bridged by the async FIFO). This is what allows Quartus to report a clean Fmax for your main logic.

Ordering & Dependencies Summary
Notebook Step 1-2 (Train model)
→ Notebook Step 3-4 (Binarize + fold thresholds)
→ Notebook Step 5 (Generate .mif files) ← needed before RTL Step 5
→ Notebook Step 6 (Generate image hex) ← needed before Testbench (Phase C)
→ Notebook Step 7 (Software baseline acc) ← needed for final comparison

RTL Step 2 (SPI slave) → Step 3 (FIFO) → Step 7 (Binarize) → Step 6 (Line buffer)
→ Step 4 (Popcount) → Step 5 (BCNN layer) → Step 8 (Top) → Step 9 (FSM) → Step 10 (Constraints)
The notebook and the RTL can proceed in parallel once you've locked down your model architecture and image dimensions, since those determine the parameters for every Verilog module.
