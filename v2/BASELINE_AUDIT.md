# Baseline RTL Audit

## File inventory

| File | Kind | Notes |
| --- | --- | --- |
| `verilog/accelerator_top.v` | RTL top | SPI ingress, FIFO, 2 BCNN layers, classifier |
| `verilog/bcnn_layer.v` | RTL layer | First binary conv layer |
| `verilog/bcnn_layer2.v` | RTL layer | Second binary conv layer |
| `verilog/line_buffer.v` | RTL utility | 3x3 line buffer, 1-bit single-channel |
| `verilog/line_buffer_multi.v` | RTL utility | 3x3 line buffer, multi-channel; has syntax bug |
| `verilog/popcount.v` | RTL utility | XNOR + popcount core |
| `verilog/pool_classifier.v` | RTL layer | GAP + 2-class dense classifier |
| `verilog/spi_fifo.v` | RTL utility | Single-clock FIFO for SPI ingress |
| `verilog/spi_slave.v` | RTL utility | SPI byte receiver |
| `verilog/fsm_controller.v` | RTL control | Frame lifecycle controller |
| `verilog/tb_accelerator.v` | Testbench | Full-image simulation through SPI |

## Missing project artifacts

- No `.vh` headers found.
- No ModelSim `.do` files found.
- No Quartus `.qpf`, `.qsf`, or `.sdc` files found.
- A root-level `run_sim.tcl` exists, but it is not a ModelSim `.do` script.

## Per-file summary

| File | Module | Parameters | Key ports | Data widths / memories |
| --- | --- | --- | --- | --- |
| `accelerator_top.v` | `accelerator_top` | `IMG_W=32`, `IMG_H=32`, `CHANNELS=8` | SPI I/O, `start_frame`, `end_frame`, `valid_out`, `class_out` | FIFO bytes are 8-bit; binarizes to 1-bit after threshold at 127 |
| `bcnn_layer.v` | `bcnn_layer` | `K=3`, `IN_CH=1`, `OUT_CH=8` | `window_in[K*K*IN_CH-1:0]`, `out_bits[OUT_CH-1:0]` | Weights: `OUT_CH` words of `N=9` bits; thresholds width `clog2(N+1)` |
| `bcnn_layer2.v` | `bcnn_layer2` | `K=3`, `IN_CH=8`, `OUT_CH=16` | `window_in[IN_CH*K*K-1:0]`, `out_bits[OUT_CH-1:0]` | Weights: `OUT_CH` words of `72` bits; per-channel popcounts accumulated across 8 channels |
| `line_buffer.v` | `line_buffer` | `IMG_WIDTH=32` | `pixel_in`, `window_out[8:0]` | Two 1-bit row buffers of length `IMG_WIDTH`; 3 shift registers |
| `line_buffer_multi.v` | `line_buffer_multi` | `IMG_WIDTH=30`, `CHANNELS=8` | `pixel_in[CHANNELS-1:0]`, `window_out[CHANNELS*9-1:0]` | Two row buffers of `CHANNELS` bits by `IMG_WIDTH`; syntax error at file end |
| `popcount.v` | `popcount` | `N=9` | `a[N-1:0]`, `b[N-1:0]`, `count[...]` | Balanced XNOR reduction tree; output width `clog2(N+1)` |
| `pool_classifier.v` | `pool_classifier` | `CHANNELS=16`, `IMG_WIDTH=28`, `IMG_HEIGHT=28` | `data_in[15:0]`, `class_out` | Accumulator per channel over full map; weights and bias are signed 16-bit |
| `spi_fifo.v` | `spi_fifo` | `DATA_WIDTH=8`, `DEPTH=64` | byte write/read interface | 64x8 memory, single clock, registered read |
| `spi_slave.v` | `spi_slave` | `DATA_BITS=8` | raw SPI pins, `byte_data[7:0]` | Synchronizers for `sck`, `mosi`, `cs_n`, byte-wide shift register |
| `fsm_controller.v` | `fsm_controller` | `IMG_W=32`, `IMG_H=32` | frame control and classifier handshake | Tracks `(IMG_W-4)*(IMG_H-4)` windows |
| `tb_accelerator.v` | `tb_accelerator` | localparams only | drives SPI serial image stream | Loads `image.hex`, clocks SPI and system domains |

## Baseline datapath findings

- Input format: 8-bit grayscale pixel stream over SPI.
- First thresholding stage: `pixel_bin = (fifo_out > 8'd127)` converts input to 1-bit before any convolution.
- Convolution math: XNOR + popcount only.
- Activations: 1-bit throughout both conv layers.
- Classifier: global-average accumulation of 1-bit channels, then signed 16-bit dense weights.
- Output: single bit `class_out`, so the deployed task is binary classification.

## Weight-loading findings

| Module | Files loaded | Method |
| --- | --- | --- |
| `bcnn_layer.v` | `conv1_weights.hex`, `conv1_thresh.hex`, `conv1_flip.mif` | `$readmemh` for weights/thresholds, `$readmemb` for flip flags |
| `bcnn_layer2.v` | `conv2_weights.hex`, `conv2_thresh.hex`, `conv2_flip.mif` | `$readmemh` for weights/thresholds, `$readmemb` for flip flags |
| `pool_classifier.v` | `dense_w0.hex`, `dense_w1.hex`, `dense_b.hex` | `$readmemh` |
| `tb_accelerator.v` | `image.hex` | `$readmemh` |

No ROM wrappers or Quartus memory init files are present in the baseline. Memory loading is direct inside modules.

## Baseline architecture

| Stage | Verified implementation |
| --- | --- |
| Input | `32x32x1` grayscale bytes streamed over SPI |
| Preprocess | Hard threshold at 127 to 1-bit |
| Layer 1 | 3x3 binary conv, `1 -> 8` channels |
| Layer 2 | 3x3 binary conv, `8 -> 16` channels |
| Pooling | Global average pool over `28x28` layer-2 map |
| Classifier | Dense `16 -> 2` equivalent scores, compared to one output bit |
| Output | Binary class bit |

## Deviations from the requested baseline description

- The checked-in RTL is not grayscale multi-bit activation followed by BN/ReLU. It binarizes the raw input first, then stays binary.
- There is no explicit BatchNorm module; thresholds and optional `flip` flags fold the decision.
- There is no explicit softmax and no 3-class output.
- The top-level classifier is binary, not 3-class.
- The current design is a 2-layer BCNN, not the 3-conv floating/binarized hybrid described in the prompt.

## Architecture summary table

| Item | Legacy design |
| --- | --- |
| Input size | `32x32x1` |
| Input precision | `uint8` ingress, thresholded to `1b` |
| Layer sequence | `threshold -> binconv(1,8,3x3) -> binconv(8,16,3x3) -> GAP -> dense(16,2)` |
| Weight precision | Binary conv weights, 16-bit dense weights |
| Activation precision | 1-bit for conv pipeline, integer counters for GAP |
| Pooling type | Global average pooling only |
| Output format | 1-bit class select |
| Testbench | `verilog/tb_accelerator.v` |
| Quartus project files | None present |
| ModelSim scripts | None present |

