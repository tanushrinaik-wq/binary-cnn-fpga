# Deliverable Status

## Implemented source files

- `python/dataset.py`
- `python/models.py`
- `python/train_teacher.py`
- `python/train_student.py`
- `python/binarize.py`
- `python/export_weights.py`
- `python/compare.py`
- `run_pipeline.sh`
- `requirements.txt`
- `rtl/common/xnor_popcount.v`
- `rtl/common/xnor_popcount_pipe.v`
- `rtl/common/fixed_point_mult.v`
- `rtl/mem/weight_rom.v`
- `rtl/mem/fifo_buffer.v`
- `rtl/layers/conv1_layer.v`
- `rtl/layers/bconv_dw_layer.v`
- `rtl/layers/bconv_pw_layer.v`
- `rtl/layers/bn_threshold.v`
- `rtl/layers/maxpool_layer.v`
- `rtl/layers/gap_layer.v`
- `rtl/layers/fc_layer.v`
- `rtl/top/bcnn_top.v`
- `rtl/tb/tb_bcnn_top.v`
- `rtl/tb/tb_xnor_popcount.v`
- `rtl/tb/tb_conv1_layer.v`
- `rtl/scripts/run_sim.do`
- `rtl/scripts/run_unit_tests.do`
- `rtl/scripts/compile_quartus.tcl`
- `reports/memory_budget.txt`
- `BASELINE_AUDIT.md`

## Generated artifacts not produced in this session

These require dataset availability, checkpoint training, and/or external tools:

- `artifacts/teacher_mobilenetv2.pth`
- `artifacts/student_fp32.pth`
- `artifacts/student_binarized.pth`
- `weights/*.hex`
- `weights/*.bin`
- `weights/model_summary.json`
- `weights/weight_stats.txt`
- `weights/test_images.hex`
- `weights/test_labels.hex`
- `weights/golden_reference.json`
- `reports/comparison_report.csv`
- `reports/comparison_report.png`
- `reports/confusion_matrix_teacher.png`
- `reports/confusion_matrix_student_fp32.png`
- `reports/confusion_matrix_student_binarized.png`

## Blocking environment constraints

- No Rock-Paper-Scissors dataset is present under `v2/data`.
- MobileNetV2 pretrained weights may require network access on first use.
- No Verilog compiler or ModelSim binary was available in the shell session.
- Quartus was not available in the shell session.

## Expected generation flow

1. Place the dataset in `v2/data` using ImageFolder layout.
2. Install Python dependencies from `v2/requirements.txt`.
3. Run `v2/run_pipeline.sh`.
4. Run ModelSim with `v2/rtl/scripts/run_unit_tests.do` and `v2/rtl/scripts/run_sim.do`.
5. Run Quartus with `v2/rtl/scripts/compile_quartus.tcl`.
