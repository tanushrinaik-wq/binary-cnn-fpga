`timescale 1ns / 1ps

module fixed_point_mult #(parameter WIDTH = 16, parameter FRAC = 8)(
    input  wire signed [WIDTH-1:0] a,
    input  wire signed [WIDTH-1:0] b,
    output wire signed [WIDTH-1:0] y
);
    wire signed [(2*WIDTH)-1:0] mult_full;
    assign mult_full = a * b;
    assign y = mult_full >>> FRAC;
endmodule

