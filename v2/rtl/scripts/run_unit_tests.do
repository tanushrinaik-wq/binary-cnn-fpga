vlib work
vmap work work
vlog ../common/xnor_popcount.v
vlog ../layers/conv1_layer.v
vlog ../tb/tb_xnor_popcount.v
vlog ../tb/tb_conv1_layer.v
vsim -t 1ns -voptargs="+acc" tb_xnor_popcount
run -all
vsim -t 1ns -voptargs="+acc" tb_conv1_layer
run -all
quit

