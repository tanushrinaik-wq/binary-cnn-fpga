# Legacy RTL Design Guide

This directory contains the original hardware implementation of a small binary convolutional neural network (BCNN) accelerator. The design is written as a streaming inference pipeline around a very small image classifier:

- input: `32 x 32` grayscale image
- ingress protocol: SPI byte stream
- preprocessing: fixed threshold to 1-bit pixels
- feature extractor:
  - layer 1: binary `3 x 3` convolution, `1 -> 8` channels
  - layer 2: binary `3 x 3` convolution, `8 -> 16` channels
- classifier:
  - global average pooling over all `16` channels
  - 2-class dense layer
  - 1-bit output `class_out`

The implementation is designed for Intel/Altera Cyclone IV and was clearly iterated several times. Comments such as `FINAL`, `FIXED`, `SPEC-COMPLIANT`, and `CDC SAFE` indicate that the code reflects multiple bug-fix passes rather than a first-draft architecture.

## Directory overview

| File | Purpose |
| --- | --- |
| `accelerator_top.v` | Top-level integration of SPI, FIFO, window generators, BCNN layers, and classifier |
| `spi_slave.v` | SPI byte receiver with clock-domain synchronization into `sys_clk` |
| `spi_fifo.v` | Single-clock FIFO buffering bytes between SPI and compute pipeline |
| `line_buffer.v` | Builds a `3 x 3` window for the first convolution from a 1-bit pixel stream |
| `bcnn_layer.v` | First binary convolution layer: `1 -> 8` output channels |
| `line_buffer_multi.v` | Builds `3 x 3` windows for all channels for the second layer |
| `bcnn_layer2.v` | Second binary convolution layer: `8 -> 16` output channels |
| `popcount.v` | XNOR + popcount primitive used by both convolution layers |
| `pool_classifier.v` | Global average pooling and 2-class dense classifier |
| `fsm_controller.v` | Generates frame start/end control for pooling/classification |
| `tb_accelerator.v` | Full-system simulation testbench |

## High-level dataflow

The full datapath is:

1. SPI serial input enters `spi_slave`
2. `spi_slave` emits 8-bit bytes on the system clock domain
3. `spi_fifo` buffers those bytes
4. `accelerator_top` thresholds each byte at `127` to produce a 1-bit pixel
5. `line_buffer` converts the serial bitstream into `3 x 3` spatial windows
6. `bcnn_layer` applies 8 binary filters and outputs 8 feature bits per valid window
7. `line_buffer_multi` builds `3 x 3` windows across all 8 channels
8. `bcnn_layer2` applies 16 binary filters and outputs 16 feature bits per valid window
9. `pool_classifier` accumulates each channel over the full feature map
10. Dense weights convert the pooled 16-channel vector into two scores
11. `class_out` is `1` when score 1 is greater than score 0, otherwise `0`

There is no explicit BatchNorm, ReLU, or softmax module in RTL. The network is already reduced to binary comparisons and thresholding by the time it is exported to hardware.

## Architectural model

The design uses binary activations and binary convolution weights:

- image bytes are thresholded to 1 bit:
  - `pixel_bin = (fifo_out > 8'd127)`
- convolution is implemented as:
  - XNOR between activation bits and weight bits
  - popcount of matching bits
  - threshold compare
- optional `flip` bits invert the comparison direction per output channel

This means the mathematical operation is not multiply-accumulate in the usual CNN sense. Instead, it asks:

- how many input bits match the stored binary kernel?
- is that count above or below a learned threshold?

The result is a binary output bit per filter.

## Top-level integration: `accelerator_top.v`

File: [accelerator_top.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\accelerator_top.v)

This is the central integration module. It contains four major subsystems:

1. SPI ingress
2. FIFO buffering
3. Layer-1 binary convolution path
4. Layer-2 binary convolution path and classifier

### Parameters

- `IMG_W = 32`
- `IMG_H = 32`
- `CHANNELS = 8`

Internally it also defines:

- `CH1 = 8`
- `CH2 = 16`

These are the output channel counts of the two BCNN layers.

### SPI stage

`spi_slave` receives the raw SPI signals:

- `spi_sck`
- `spi_mosi`
- `spi_cs_n`

and emits:

- `byte_data[7:0]`
- `byte_valid`
- `frame_active`
- `frame_done`

`frame_active` is just `~cs_s` inside `spi_slave`, so it indicates that chip select is currently asserted. `frame_done` is a one-cycle pulse on the system clock when chip select rises.

### FIFO stage

Bytes from `spi_slave` are written into `spi_fifo`. Read control is:

```verilog
wire pipeline_ready = 1'b1;
wire fifo_rd = (!fifo_empty) && pipeline_ready;
```

Important implication:

- this top level does not support real backpressure
- it documents the assumption that the pipeline always consumes data
- if any downstream block stalls, the current design would need a real `ready` chain

The delayed signal `fifo_rd_d` aligns the FIFO read request with the registered read output of the FIFO. Since `spi_fifo` uses synchronous read memory, the actual byte appears one clock after `read_en`.

### Thresholding stage

`accelerator_top` converts each grayscale byte to a single binary pixel:

```verilog
wire pixel_bin = (fifo_out > 8'd127);
```

This is an irreversible decision. The downstream network never sees multi-bit grayscale data.

### Layer 1

`line_buffer` receives the 1-bit serial pixel stream and emits:

- `valid_out`
- `window_out[8:0]`

Once the first `3 x 3` neighborhood is available, `bcnn_layer` processes it through 8 binary filters and outputs:

- `l1_valid`
- `l1_out[7:0]`

Each bit in `l1_out` is the binary output of one filter.

### Layer 2

`line_buffer_multi` takes the 8-bit-wide stream from layer 1 and builds `3 x 3` windows for every channel. The output bus is:

- `window_l2[CH1*9-1:0]`

With `CH1 = 8`, this is `72` bits total, arranged as 8 groups of 9 bits.

`bcnn_layer2` applies 16 filters over that 8-channel binary window and emits:

- `l2_valid`
- `l2_out[15:0]`

### Classifier

`bcnn_valid` is simply `l2_valid`. The top then instantiates `pool_classifier`, parameterized for:

- `CHANNELS = 16`
- `IMG_WIDTH = IMG_W - 4 = 28`
- `IMG_HEIGHT = IMG_H - 4 = 28`

This reflects the fact that two valid `3 x 3` convolutions on a `32 x 32` image reduce spatial size by 2 twice:

- `32 -> 30 -> 28`

The classifier receives one 16-bit activation vector per valid spatial location and eventually emits:

- `valid_out`
- `class_out`

`class_out` is a binary decision only.

## SPI receiver: `spi_slave.v`

File: [spi_slave.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\spi_slave.v)

This module converts external SPI signaling into byte-wide events in the system clock domain.

### CDC strategy

The design uses two-stage synchronizers for:

- `spi_sck`
- `spi_mosi`
- `spi_cs_n`

This is the most important safety feature in the file. Raw SPI signals are asynchronous to `sys_clk`, so direct sampling would be unsafe. The module then only uses the synchronized versions:

- `sck_s`
- `mosi_s`
- `cs_s`

### Edge detection

The module stores previous values of synchronized `sck_s` and `cs_s`, then derives:

- `sck_rising`
- `cs_rising`
- `cs_falling`

This allows the logic to:

- reset the byte assembler on frame start
- shift one bit on each synchronized SPI clock rising edge
- pulse `frame_done` when chip select deasserts

### Shift logic

When `!cs_s && sck_rising`, the module shifts in one `mosi_s` bit. After `DATA_BITS` bits, it emits:

- `byte_data`
- `byte_valid = 1`

`spi_miso` is tied to zero, so the interface is receive-only.

### Interpretation

This is not a general SPI slave with configurable CPOL/CPHA behavior. It is a minimal byte receiver matched to the testbench and surrounding accelerator.

## FIFO: `spi_fifo.v`

File: [spi_fifo.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\spi_fifo.v)

This module buffers incoming pixel bytes between SPI ingress and the convolution pipeline.

### Why a FIFO is needed

SPI traffic arrives according to the external serial clock, but the compute pipeline wants clean byte-wide input on `sys_clk`. `spi_slave` already moves the interface into the `sys_clk` domain; `spi_fifo` then absorbs small timing mismatches or bursts.

### Memory organization

- default depth: `64`
- width: `8`
- storage: `mem[0:DEPTH-1]`

Pointers use an extra wrap bit:

- `wr_ptr`
- `rd_ptr`

This supports straightforward `full` and `empty` detection.

### Key behavior

- synchronous write
- synchronous read
- registered read output
- sticky overflow flag
- sticky underflow flag

The comments state that depth 64 is enough for two full image rows of `32` bytes. That is a sensible buffering choice for this image size.

### Notes

Because the FIFO uses registered reads, there is a one-cycle delay between `read_en` and the data appearing on `read_data`. `accelerator_top` compensates for this with `fifo_rd_d`.

## First window generator: `line_buffer.v`

File: [line_buffer.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\line_buffer.v)

This module builds `3 x 3` sliding windows from a 1-bit pixel stream.

### Internal structure

- two row buffers:
  - `row_buf1`
  - `row_buf2`
- three 3-bit shift registers:
  - `shift_row0`
  - `shift_row1`
  - `shift_row2`

Conceptually:

- `row_buf1` holds the previous row
- `row_buf2` holds the row before that
- the shift registers hold the last 3 columns for each of the three active rows

### Counters

- `col` tracks the current column
- `row_count` tracks how many rows have been processed

`window_ready` becomes true only after at least 3 rows and 3 columns have been observed.

### Output timing

The output is registered and gated by:

- `window_ready`
- delayed `valid_in`

This means the module only emits a valid `3 x 3` window after the internal row buffers and shift registers contain enough history.

### Practical effect

For a `32 x 32` input image, the module emits `30 x 30 = 900` valid windows.

## Binary conv layer 1: `bcnn_layer.v`

File: [bcnn_layer.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\bcnn_layer.v)

This module performs the first binary convolution stage.

### Interface

Inputs:

- `valid_in`
- `window_in[K*K*IN_CH-1:0]`

Outputs:

- `valid_out`
- `out_bits[OUT_CH-1:0]`

With defaults:

- `K = 3`
- `IN_CH = 1`
- `OUT_CH = 8`

the input window width is 9 bits and the output width is 8 bits.

### Stored parameters

For each output channel, the layer stores:

- a binary weight vector `weights[i]`
- a numeric threshold `thresholds[i]`
- a binary inversion control `flip[i]`

The files loaded are:

- `conv1_weights.hex`
- `conv1_thresh.hex`
- `conv1_flip.mif`

### Core computation

For each filter:

1. `popcount` computes the number of matching bits between `window_in` and `weights[i]`
2. the result is compared against `thresholds[i]`
3. if `flip[i]` is set, the comparison direction is inverted

The output update is:

```verilog
out_bits[j] <= flip[j] ?
    (pc[j] < thresholds[j]) :
    (pc[j] >= thresholds[j]);
```

That lets the exporter encode cases where the learned decision boundary should be interpreted in the opposite sense.

### Timing

`valid_out` is just a registered version of `valid_in`. The actual compare happens in the same cycle for all 8 filters.

## Popcount core: `popcount.v`

File: [popcount.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\popcount.v)

This module computes:

- bitwise XNOR between two vectors
- number of resulting `1` bits

### Structure

1. XNOR stage:
   - `x[i] = ~(a[i] ^ b[i])`
2. adder tree:
   - `level1`
   - `level2`
   - `level3`
3. final reduction loop

This is optimized for small `N`, especially `N = 9`, which is exactly what this design uses for each spatial `3 x 3` patch.

### Important limitation

The comment is correct: this implementation is only partially balanced. For larger bit counts, the final reduction loop becomes a longer combinational chain.

For this legacy design that is acceptable because:

- layer 1 uses `N = 9`
- layer 2 also uses `N = 9` per channel, then adds across channels separately

## Multi-channel window generator: `line_buffer_multi.v`

File: [line_buffer_multi.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\line_buffer_multi.v)

This is the multi-channel analogue of `line_buffer.v`.

### Purpose

After layer 1, each spatial sample is not one bit anymore; it is an 8-bit vector of channel outputs. `line_buffer_multi` builds a `3 x 3` spatial neighborhood for every channel in parallel.

### Internal structure

- row buffers store `CHANNELS` bits per column
- shift-register triplets:
  - `shift0[c]`
  - `shift1[c]`
  - `shift2[c]`

For each channel `i`, the output slice:

```verilog
window_out[i*9 +: 9]
```

contains the `3 x 3` neighborhood for that channel.

### Output width

With default `CHANNELS = 8`, output width is:

- `8 * 9 = 72` bits

### Critical issue

The file ends with:

```verilog
endmodules
```

That is a syntax error. It should be:

```verilog
endmodule
```

So this file, as checked in, is not compile-clean until that typo is fixed.

## Binary conv layer 2: `bcnn_layer2.v`

File: [bcnn_layer2.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\bcnn_layer2.v)

This module performs the second convolution stage over 8 input channels and produces 16 output channels.

### Interface

Inputs:

- `valid_in`
- `window_in[IN_CH*K*K-1:0]`

Outputs:

- `valid_out`
- `out_bits[OUT_CH-1:0]`

Defaults:

- `K = 3`
- `IN_CH = 8`
- `OUT_CH = 16`

So the input window width is:

- `8 * 3 * 3 = 72` bits

### Weight organization

Each output filter stores:

- `weights[f]` of width `IN_CH * K * K = 72`
- `thresholds[f]`
- `flip[f]`

Files loaded:

- `conv2_weights.hex`
- `conv2_thresh.hex`
- `conv2_flip.mif`

### Computation strategy

Unlike layer 1, this layer does not call `popcount` once on the full 72-bit vector. Instead, it does:

1. split the input into 8 channel-local 9-bit windows
2. split the filter the same way
3. run one `popcount` per channel
4. sum the 8 channel-local popcounts
5. compare the total against threshold

This is a more structured approach and keeps the popcount primitive in its efficient `N = 9` regime.

### Flattened indexing fix

The comments mention a “flattened array fix.” That refers to this expression:

```verilog
pc[f*IN_CH + c]
```

instead of trying to use a two-dimensional wire array in a way older Verilog tools may dislike. This is one of the signs that the module was adapted for older Verilog-2001 toolchains.

### Spatial size

The input feature map to this layer is `30 x 30`. A valid `3 x 3` convolution therefore yields `28 x 28` outputs.

## Frame controller: `fsm_controller.v`

File: [fsm_controller.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\fsm_controller.v)

This module does not control the convolution pipeline itself. Instead, it provides frame-level control for the classifier.

### States

- `IDLE`
- `RUNNING`
- `WAIT_CLASS`

### Inputs used

- `frame_active`: from `spi_slave`, indicates chip-select active
- `l2_valid`: counts valid layer-2 outputs
- `classifier_valid`: signals final classification ready

### Outputs

- `start_frame`
- `end_frame`

### Behavior

1. enter `RUNNING` when a frame becomes active
2. count each valid layer-2 output
3. when the final spatial output is seen, pulse `end_frame`
4. wait until classifier result is ready
5. return to `IDLE`

### Window counting

It computes:

```verilog
TOTAL_WINDOWS = (IMG_W - 4) * (IMG_H - 4)
```

For `32 x 32`, that is:

- `(32 - 4) * (32 - 4) = 28 * 28 = 784`

That exactly matches the layer-2 output map size.

## Global average pooling and classifier: `pool_classifier.v`

File: [pool_classifier.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\pool_classifier.v)

This is the final classification stage.

### Inputs

- `start`
- `end_frame`
- `valid_in`
- `data_in[CHANNELS-1:0]`

`data_in` is a 16-bit binary vector, one bit per layer-2 channel.

### GAP accumulation

The module keeps:

```verilog
reg [ACC_WIDTH-1:0] acc [0:CHANNELS-1];
```

Each `acc[i]` counts how many times channel `i` was `1` across the full spatial map.

Because the map is `28 x 28`, each accumulator is effectively computing a sum over 784 binary values.

### Dense classifier weights

The dense layer is stored as:

- `W0[channel]`
- `W1[channel]`
- `B0`
- `B1`

with files:

- `dense_w0.hex`
- `dense_w1.hex`
- `dense_b.hex`

The classifier therefore has only two outputs. It is a binary classifier represented as two scores.

### Two-stage pipeline

The current version uses:

1. registered multiply stage
2. registered sum stage

The DSP-oriented comments matter:

- `prod0[i] <= acc[i] * W0[i]`
- `prod1[i] <= acc[i] * W1[i]`

This avoids a LUT-heavy hand-coded multiplier and encourages Quartus to infer dedicated DSP blocks.

### Final decision

```verilog
class_out <= (score1 > score0);
```

So:

- `0` means class 0 wins
- `1` means class 1 wins

There is no probability output.

## Testbench: `tb_accelerator.v`

File: [tb_accelerator.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\verilog\tb_accelerator.v)

This is a full-chip simulation harness.

### Clocks

- system clock: 20 ns period, so 50 MHz
- SPI clock: 50 ns period, so 20 MHz

### Reset

Active-low reset is asserted for 100 ns.

### Image loading

The testbench loads:

- `image.hex`

into `image_mem`, then serializes every byte onto `spi_mosi` synchronized to the SPI clock.

### DUT structure

The testbench instantiates:

- `accelerator_top`
- `fsm_controller`

The top itself does not instantiate the FSM; the testbench wires them together explicitly.

### Output reporting

When `valid_out` is asserted, the testbench prints:

- final `class_out`
- total latency in cycles
- estimated inference time in microseconds

### Watchdog

There is a 50 ms simulation watchdog to avoid infinite hangs if `valid_out` never appears.

## Expected external memory files

The legacy RTL relies on several memory initialization files:

### Layer 1

- `conv1_weights.hex`
- `conv1_thresh.hex`
- `conv1_flip.mif`

### Layer 2

- `conv2_weights.hex`
- `conv2_thresh.hex`
- `conv2_flip.mif`

### Dense classifier

- `dense_w0.hex`
- `dense_w1.hex`
- `dense_b.hex`

### Testbench input

- `image.hex`

The code assumes these files are available in the simulator or synthesis working directory.

## Latency and throughput intuition

The convolution path is structurally streaming:

- one thresholded pixel in per valid cycle
- one `3 x 3` window out once buffers are primed
- one vector of filter outputs per valid window

The spatial warm-up cost comes from:

- filling the FIFO
- priming the line buffers
- waiting for enough rows/columns to build the first valid window

After that, the pipeline is intended to emit one valid result per cycle through the convolution stages.

The final classifier is not fully streaming per pixel. It accumulates over the whole frame and only emits a decision at the end.

## Known issues and caveats

### 1. `line_buffer_multi.v` syntax error

The file ends with `endmodules` instead of `endmodule`.

### 2. `$clog2` usage

Several modules use `$clog2`. Some older Quartus/Verilog-2001 flows can be sensitive to this depending on settings.

### 3. Hardwired no-backpressure assumption

`accelerator_top.v` sets:

```verilog
wire pipeline_ready = 1'b1;
```

So the design documents backpressure but does not actually implement it.

### 4. Binary input thresholding is fixed

The threshold `> 127` is built into the top-level datapath. Any training/export flow must match that assumption exactly.

### 5. Binary output only

This accelerator is not a 3-class classifier. The output is a single bit.

### 6. File-path dependency

The weight files are loaded by bare filenames rather than centralized ROM wrappers, so simulation and synthesis depend on working-directory setup.

## Summary

This legacy design is a compact, streaming, two-layer BCNN accelerator with:

- SPI byte ingress
- grayscale-to-binary thresholding
- line-buffer-based `3 x 3` window generation
- XNOR + popcount binary convolution
- global average pooling
- 2-class dense classification

Its strongest features are:

- simple pipeline structure
- efficient binary convolution implementation
- explicit CDC handling on SPI input
- compact frame-level control

Its main limitations are:

- binary-only task/output
- hardcoded assumptions about preprocessing
- fragile file-based parameter loading
- lack of real backpressure
- at least one syntax issue in the checked-in sources

If you need to modify or extend this design, the safest order is:

1. fix `line_buffer_multi.v`
2. confirm all weight/init files load correctly
3. verify testbench end-to-end behavior
4. only then change channel counts, image size, or classifier structure
