# ============================================================
# run_sim.tcl — paste your actual folder path on the cd line
# ============================================================

cd C:/Users/yourname/bcnn_sim

quietly set StdArithNoWarnings 1
quietly set NumericStdNoWarnings 1

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

add wave -divider "Clock/Reset"
add wave /tb_accelerator/clk
add wave /tb_accelerator/rst_n

add wave -divider "SPI Interface"
add wave /tb_accelerator/spi_sck
add wave /tb_accelerator/spi_mosi
add wave /tb_accelerator/spi_cs_n

add wave -divider "Pipeline"
add wave /tb_accelerator/frame_active
add wave /tb_accelerator/frame_done
add wave /tb_accelerator/l2_valid

add wave -divider "Result"
add wave /tb_accelerator/valid_out
add wave /tb_accelerator/class_out

run -all