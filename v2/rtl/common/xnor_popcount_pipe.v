`timescale 1ns / 1ps

module xnor_popcount_pipe #(parameter N = 32, parameter COUNT_WIDTH = 6)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    output reg  valid_out,
    output reg  [COUNT_WIDTH-1:0] out
);
    integer i;
    reg [N-1:0] stage1_xnor;
    reg         stage1_valid;
    reg [COUNT_WIDTH-1:0] stage2_sum;
    reg                   stage2_valid;
    reg [COUNT_WIDTH-1:0] sum_next;

    always @(posedge clk) begin
        if (rst) begin
            stage1_xnor <= {N{1'b0}};
            stage1_valid <= 1'b0;
        end else begin
            stage1_xnor <= ~(a ^ b);
            stage1_valid <= valid_in;
        end
    end

    always @(*) begin
        sum_next = {COUNT_WIDTH{1'b0}};
        for (i = 0; i < N; i = i + 1)
            sum_next = sum_next + stage1_xnor[i];
    end

    always @(posedge clk) begin
        if (rst) begin
            stage2_sum <= {COUNT_WIDTH{1'b0}};
            stage2_valid <= 1'b0;
        end else begin
            stage2_sum <= sum_next;
            stage2_valid <= stage1_valid;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            out <= {COUNT_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            out <= stage2_sum;
            valid_out <= stage2_valid;
        end
    end
endmodule
