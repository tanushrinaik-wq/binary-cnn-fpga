# RTL Design Doc

This document describes the current Verilog implementation of the BCNN accelerator in `verilog/`. It replaces the old `verilog/README.md` and keeps the focus on how the checked-in RTL is organized today.

## Design overview

The RTL implements a streaming binary classifier with these stages:

1. SPI byte reception
2. FIFO buffering
3. fixed-threshold pixel binarization
4. first `3 x 3` binary convolution layer
5. second `3 x 3` binary convolution layer
6. global pooling and dense classification

Current image assumptions:

- input size: `32 x 32`
- input format: grayscale bytes
- classifier output: single-bit class decision

## Module map

| File | Purpose |
| --- | --- |
| `accelerator_top.v` | Integrates SPI, FIFO, line buffers, convolution layers, and classifier |
| `spi_slave.v` | Receives SPI bytes and exposes them in the system clock domain |
| `spi_fifo.v` | Buffers received bytes before pixel processing |
| `line_buffer.v` | Builds `3 x 3` windows for the first convolution layer |
| `bcnn_layer.v` | First binary convolution layer |
| `line_buffer_multi.v` | Builds multi-channel `3 x 3` windows for the second convolution layer |
| `bcnn_layer2.v` | Second binary convolution layer |
| `popcount.v` | Shared XNOR plus popcount primitive |
| `pool_classifier.v` | Global pooling and dense classification |
| `fsm_controller.v` | Frame-level control signals for the classifier |
| `tb_accelerator.v` | End-to-end simulation testbench |

## Top-level dataflow

The effective datapath is:

```text
SPI -> spi_slave -> spi_fifo -> threshold -> line_buffer
    -> bcnn_layer -> line_buffer_multi -> bcnn_layer2
    -> pool_classifier -> class_out
```

`fsm_controller.v` is instantiated by the testbench and supplies `start_frame` and `end_frame` to the top-level design.

## Key implementation details

### `accelerator_top.v`

- Uses `IMG_W = 32`, `IMG_H = 32`
- Thresholds raw pixels using `fifo_out > 8'd127`
- Instantiates:
  - `spi_slave`
  - `spi_fifo`
  - `line_buffer`
  - `bcnn_layer`
  - `line_buffer_multi`
  - `bcnn_layer2`
  - `pool_classifier`

The design currently hardwires pipeline readiness to `1'b1`, so there is no real backpressure path through the compute stages.

### `spi_slave.v`

- Synchronizes `spi_sck`, `spi_mosi`, and `spi_cs_n` into the system clock domain
- Detects SPI edges after synchronization
- Shifts in one byte at a time
- Pulses `byte_valid` when a full byte is assembled

This is a compact receive-only SPI frontend matched to the testbench rather than a general-purpose SPI peripheral.

### `spi_fifo.v`

- Stores received bytes before thresholding
- Helps decouple ingress timing from the compute path
- Uses synchronous read behavior, so downstream logic aligns with a delayed read enable

### `line_buffer.v`

- Accepts 1-bit pixels
- Maintains row history
- Emits one valid `3 x 3` window whenever enough rows and columns have been buffered

### `bcnn_layer.v`

- Implements the first binary convolution stage
- Uses stored binary weights plus threshold comparisons
- Outputs one bit per output channel

This layer is parameterized, but the checked-in design uses it as a `1 -> 8` channel stage.

### `line_buffer_multi.v`

- Consumes the `8` channel outputs from layer 1
- Produces one `3 x 3` window per channel for the second convolution stage

### `bcnn_layer2.v`

- Implements the second binary convolution stage
- Treats each input channel as a separate `3 x 3` popcount and then accumulates across channels
- Produces `16` output feature bits in the current configuration

### `popcount.v`

- Performs XNOR between activation bits and weight bits
- Counts the number of matching positions

This is the core arithmetic block that replaces multiply-accumulate behavior in the convolution path.

### `pool_classifier.v`

- Accumulates binary activations across the spatial map
- Applies dense-layer coefficients loaded from memory files
- Produces a final binary decision through score comparison

### `fsm_controller.v`

- Tracks frame activity and layer-2 progress
- Asserts `start_frame` and `end_frame` for the classifier pipeline

### `tb_accelerator.v`

- Generates a `50 MHz` system clock and `20 MHz` SPI clock
- Loads `image.hex`
- Streams image bytes over SPI
- Reports classification result and latency
- Terminates on completion or watchdog timeout

## Memory file expectations

The RTL expects external initialization files such as:

- `conv1_weights.hex`
- `conv1_thresh.hex`
- `conv1_flip.mif`
- `conv2_weights.hex`
- `conv2_thresh.hex`
- `conv2_flip.mif`
- `dense_w0.hex`
- `dense_w1.hex`
- `dense_b.hex`
- `image.hex`

These files are not currently checked into the repository, so simulation is not turnkey until the notebook export flow is run and the generated files are placed where the simulator can find them.

There is also a current filename mismatch between the notebook and RTL:

- the notebook exports `conv1_weights.mif` and `conv2_weights.mif`
- the RTL reads `conv1_weights.hex` and `conv2_weights.hex`
- the notebook content in this repository does not visibly export `conv1_flip.mif` or `conv2_flip.mif`

That mismatch needs to be resolved before the documented flow becomes reproducible end to end.

## Known constraints

- Input binarization threshold is fixed in RTL.
- File loading depends on the simulator working directory.
- The checked-in simulation flow is not yet fully automated.
- The repository does not currently include synthesized Quartus reports.

## Recommended usage order

1. Run the notebook and generate the memory files.
2. Place the exported artifacts into the simulation working directory.
3. Update `run_sim.tcl` to the correct local path.
4. Compile and run `tb_accelerator.v`.
