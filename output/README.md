# Output Artifact Index

This directory contains generated artifacts from the notebook export flow.

## Structure

- `hex/` - RTL simulation assets such as weights, thresholds, classifier coefficients, test images, and golden outputs
- `mem/` - Quartus-oriented memory initialization files
- `metadata/` - machine-readable manifests and timing summaries
- `models/` - saved teacher and student Keras models
- `plots/` - exported figures used in the report and presentation

## Key files

### `metadata/`

- `fixed_point_spec.json` - fixed-point format used for the dense layer export
- `inference_timing.json` - CPU timing comparison between float Keras inference and software binary inference
- `rtl_manifest.json` - exported model and threshold metadata aligned to the RTL pipeline

### `hex/`

- `conv*_weights.hex`, `conv*_thresh.hex`, `conv*_flip.mif` - layer parameters used by the Verilog modules
- `dense_w0.hex`, `dense_w1.hex`, `dense_b.hex` - final classifier coefficients
- `test_img_*.hex` - testbench-ready images
- `golden_outputs.txt` - expected labels and predictions for selected testbench images

### `mem/`

- `conv1_weights.mif`
- `conv2_weights.mif`

These are kept separately because the RTL simulation flow in this repository primarily loads the `hex/` assets, while Quartus memory initialization can use the `mif` files.
