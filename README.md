# Edge AI Course - https://www.samy101.com/edge-ai-26/

## Team Members:

- 27230 Tanushri Naik(tanushrinaik@iisc.ac.in), MTech in RAS 1st year(RBCCPS)
- 26514 Mehuli Chatterjee(mehulic@iisc.ac.in), MTech in RAS 1st year(RBCCPS)
- 26020 Rayaan Ghosh(rayaanghosh@iisc.ac.in), MTech in RAS 1st year(RBCCPS)
- 25887 Ayush Sagar(ayushsagar@iisc.ac.in), MTech-CFTI in RAS 1st year(RBCCPS)

# EdgeProbe: Binary CNN FPGA Accelerator

EdgeProbe is an end-to-end edge AI project that trains a small binary convolutional neural network (BCNN) in Python and maps the inference pipeline to Verilog for FPGA simulation. The hardware path replaces multiply-accumulate operations with XNOR and popcount, then performs global pooling and binary classification on streamed image data received over SPI.

The repository currently contains:

- a training and export notebook: `bcnn_ste_kd.ipynb`
- RTL modules for the accelerator in `verilog/`
- project documentation in `docs/`
- a ModelSim helper script: `run_sim.tcl`

## What this project does

The workflow is:

1. Load and preprocess the Rock-Paper-Scissors dataset.
2. Remap the labels to a binary task: `rock` vs `not-rock`.
3. Train a teacher network and a compact BCNN student with STE-based binarization and knowledge distillation.
4. Export binary weights, thresholds, dense-layer coefficients, and test images as `.mif` / `.hex` files.
5. Simulate the RTL accelerator that consumes image bytes over SPI and emits a final class bit.

## External datasets

This project uses an external dataset through TensorFlow Datasets:

- TensorFlow Datasets catalog entry: https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
- Original dataset homepage: http://laurencemoroney.com/rock-paper-scissors-dataset

The notebook does not rely on a separately checked-in cleaned dataset snapshot. Preprocessing is performed deterministically inside `bcnn_ste_kd.ipynb` by:

- resizing images to `32 x 32`
- converting RGB to grayscale
- normalizing to `[0, 1]`
- remapping labels to `rock = 0`, `not-rock = 1`

If you create and version a cleaned dataset snapshot in the future, store it in a dedicated `data/` directory and commit the full derived dataset alongside the code that generated it.

## Project structure

```text
binary-cnn-fpga/
|-- README.md
|-- bcnn_ste_kd.ipynb
|-- run_sim.tcl
|-- docs/
|   |-- README.md
|   |-- RTL Design Doc.md
|   |-- SPEC.md
|   |-- STATUS.md
|   `-- project_description.md
`-- verilog/
    |-- accelerator_top.v
    |-- bcnn_layer.v
    |-- bcnn_layer2.v
    |-- fsm_controller.v
    |-- line_buffer.v
    |-- line_buffer_multi.v
    |-- popcount.v
    |-- pool_classifier.v
    |-- spi_fifo.v
    |-- spi_slave.v
    `-- tb_accelerator.v
```

## Generated artifacts

Running the notebook is expected to create an `output/` directory with generated assets for simulation and synthesis. Based on the notebook, the main exported files are:

- `output/mem/conv1_weights.mif`
- `output/mem/conv2_weights.mif`
- `output/hex/conv1_thresh.hex`
- `output/hex/conv2_thresh.hex`
- `output/hex/dense_w0.hex`
- `output/hex/dense_w1.hex`
- `output/hex/dense_b.hex`
- `output/hex/test_img_*.hex`
- `output/metadata/fixed_point_spec.json`
- `output/metadata/inference_timing.json`
- `output/metadata/rtl_manifest.json`

The checked-in RTL currently loads memory files by bare filename with `$readmemh` / `$readmemb`, so generated files need to be placed in the simulator working directory or the RTL paths need to be updated.

Important current mismatch:

- the notebook exports `conv1_weights.mif` and `conv2_weights.mif`
- the RTL currently reads `conv1_weights.hex` and `conv2_weights.hex`
- the notebook content inspected in this repository does not show generated `conv1_flip.mif` or `conv2_flip.mif` files

As a result, reproducing simulation from a clean checkout currently requires either adapting the export step or updating the RTL file-loading convention.

## Reproducing the project

### 1. Set up the Python environment

Create an environment with the packages used by the notebook:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow tensorflow-datasets numpy scipy scikit-learn matplotlib tqdm
```

Notes:

- `tensorflow` will download additional model weights when `MobileNetV2(weights="imagenet")` is used.
- `tensorflow-datasets` will download the Rock-Paper-Scissors dataset the first time the notebook runs.

### 2. Run the training and export notebook

Open and execute `bcnn_ste_kd.ipynb` from top to bottom.

Expected high-level stages:

1. Load the Rock-Paper-Scissors dataset.
2. Convert it into a binary `rock` / `not-rock` classification problem.
3. Train the teacher model.
4. Train the BCNN student with STE and knowledge distillation.
5. Binarize and export weights, thresholds, dense parameters, and test images.
6. Produce a software-side binary reference for RTL comparison.

### 3. Prepare simulation assets

The testbench currently expects:

- `image.hex`
- `conv1_weights.hex`
- `conv1_thresh.hex`
- `conv1_flip.mif`
- `conv2_weights.hex`
- `conv2_thresh.hex`
- `conv2_flip.mif`
- `dense_w0.hex`
- `dense_w1.hex`
- `dense_b.hex`

After running the notebook:

1. Collect the generated `.hex` and `.mif` files from `output/`.
2. Rename or copy the selected test image to `image.hex` if you keep the current testbench unchanged.
3. Reconcile the current naming mismatch between exported weight `.mif` files and RTL-expected weight `.hex` files.
4. Place all required memory files in the ModelSim working directory, or update the RTL and testbench to reference the exported paths directly.

### 4. Run RTL simulation

The repository includes `run_sim.tcl`, but its `cd` line is still a placeholder and must be updated to your real simulation directory.

Typical flow:

```tcl
vlib work
vmap work work
vlog popcount.v
vlog line_buffer.v
vlog line_buffer_multi.v
vlog spi_slave.v
vlog bcnn_layer.v
vlog bcnn_layer2.v
vlog fsm_controller.v
vlog accelerator_top.v
vlog tb_accelerator.v
vsim -t 1ns work.tb_accelerator
run -all
```

The testbench:

- drives a `50 MHz` system clock
- drives a `20 MHz` SPI clock
- loads `image.hex`
- streams the image into the DUT over SPI
- reports `class_out` and latency when `valid_out` is asserted

### 5. Optional Quartus flow

To take the design beyond simulation:

1. Create a Quartus II 13.0 project for `EP4CE115F29C7`.
2. Add the Verilog modules and generated memory initialization files.
3. Set the top-level module and timing constraints.
4. Compile and review utilization and timing reports.

## Documentation map

- `docs/README.md`: documentation index
- `docs/project_description.md`: concise project overview
- `docs/RTL Design Doc.md`: module-by-module RTL explanation
- `REPORT.md`: course submission report

## Current limitations

- There is no pinned `requirements.txt` yet.
- `run_sim.tcl` still needs its path adjusted before use.
- Simulation assumes a single `image.hex` stimulus unless the testbench is extended.
