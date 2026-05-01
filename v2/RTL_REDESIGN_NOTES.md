# RTL Redesign Notes

## Why the redesign was necessary

The first `v2` RTL pass used reference-style behavioral code in the heaviest layers:

- full-frame memories
- nested loops over channels and taps
- large combinational accumulation inside one clocked block

That style is easy to read but unsafe for Quartus elaboration and timing closure on Cyclone IV. `conv1_layer` was the first obvious failure point.

## What changed

The compute-heavy layers were rewritten as folded sequential datapaths:

- [conv1_layer.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\v2\rtl\layers\conv1_layer.v)
  - loads one full image
  - computes one tap per cycle
  - computes one output channel at a time
  - emits one packed output pixel after all 32 channels finish

- [bconv_dw_layer.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\v2\rtl\layers\bconv_dw_layer.v)
  - loads one full feature map
  - computes one depthwise channel at a time
  - computes one kernel bit compare per cycle
  - emits one packed output pixel after all channels finish

- [bconv_pw_layer.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\v2\rtl\layers\bconv_pw_layer.v)
  - loads one full feature map
  - computes one output channel at a time
  - processes one packed 32-bit word of input channels per cycle
  - uses [xnor_popcount.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\v2\rtl\common\xnor_popcount.v) as a bounded primitive

- [fc_layer.v](C:\Users\Rayaan_Ghosh\Desktop\binary-cnn-fpga\v2\rtl\layers\fc_layer.v)
  - captures GAP output once
  - computes one feature multiply-accumulate per cycle
  - evaluates three classes sequentially

## Resulting tradeoff

This redesign reduces elaboration risk substantially, but it also reduces throughput.

- Better synthesis behavior
- Lower combinational depth
- Much higher latency per image

That is intentional for this pass. The priority here is:

1. elaborate cleanly
2. simulate consistently
3. then recover throughput with controlled parallelism

## What is still not “final signoff”

- Timing at `50 MHz` is still not proven until Quartus reports it.
- Numerical equivalence against Python still depends on regenerated weights and golden checks.
- The current architecture is folded/tiled, not a fully overlapped streaming dataflow between all layers.

## Recommended next closure path

1. Run Quartus elaboration on the rewritten RTL.
2. Regenerate weights with the corrected binarized activation path in Python.
3. Generate per-layer golden dumps.
4. Add layer-by-layer RTL self-checks in the testbench.
5. Only after that, increase parallelism where timing and area allow.
