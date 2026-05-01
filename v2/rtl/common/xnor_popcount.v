`timescale 1ns / 1ps

module xnor_popcount #(parameter N = 32, parameter COUNT_WIDTH = 6)(
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    output reg  [COUNT_WIDTH-1:0] out
);
    integer i;
    reg [N-1:0] xnor_bits;

    always @(*) begin
        xnor_bits = ~(a ^ b);
        out = {COUNT_WIDTH{1'b0}};
        for (i = 0; i < N; i = i + 1)
            out = out + xnor_bits[i];
    end
endmodule

