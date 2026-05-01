vlib work
vmap work work
vlog ../common/xnor_popcount.v
vlog ../common/xnor_popcount_pipe.v
vlog ../common/fixed_point_mult.v
vlog ../mem/weight_rom.v
vlog ../mem/fifo_buffer.v
vlog ../layers/bn_threshold.v
vlog ../layers/maxpool_layer.v
vlog ../layers/gap_layer.v
vlog ../layers/fc_layer.v
vlog ../layers/conv1_layer.v
vlog ../layers/bconv_dw_layer.v
vlog ../layers/bconv_pw_layer.v
vlog ../top/bcnn_top.v
vlog ../tb/tb_bcnn_top.v
vsim -t 1ns -voptargs="+acc" tb_bcnn_top
add wave -radix hex /tb_bcnn_top/*
run -all
quit

