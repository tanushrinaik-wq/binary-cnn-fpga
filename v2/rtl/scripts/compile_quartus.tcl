project_new bcnn_v2 -overwrite
set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE115F29C7
set_global_assignment -name TOP_LEVEL_ENTITY bcnn_top
set_global_assignment -name VERILOG_FILE ../common/xnor_popcount.v
set_global_assignment -name VERILOG_FILE ../common/xnor_popcount_pipe.v
set_global_assignment -name VERILOG_FILE ../common/fixed_point_mult.v
set_global_assignment -name VERILOG_FILE ../mem/weight_rom.v
set_global_assignment -name VERILOG_FILE ../mem/fifo_buffer.v
set_global_assignment -name VERILOG_FILE ../layers/bn_threshold.v
set_global_assignment -name VERILOG_FILE ../layers/maxpool_layer.v
set_global_assignment -name VERILOG_FILE ../layers/gap_layer.v
set_global_assignment -name VERILOG_FILE ../layers/fc_layer.v
set_global_assignment -name VERILOG_FILE ../layers/conv1_layer.v
set_global_assignment -name VERILOG_FILE ../layers/bconv_dw_layer.v
set_global_assignment -name VERILOG_FILE ../layers/bconv_pw_layer.v
set_global_assignment -name VERILOG_FILE ../top/bcnn_top.v
set_global_assignment -name EDA_SIMULATION_TOOL "ModelSim-Altera (Verilog)"
set_global_assignment -name FMAX_REQUIREMENT "50 MHz"
execute_flow -compile
project_close
