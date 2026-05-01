`timescale 1ns / 1ps

module bn_threshold #(parameter DATA_WIDTH = 16)(
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire signed [DATA_WIDTH-1:0] threshold,
    output wire bit_out
);
    assign bit_out = (data_in > threshold);
endmodule

