`timescale 1ns / 1ps

module fc_layer #(
    parameter IN_FEATURES = 256,
    parameter COUNT_WIDTH = 8,
    parameter FP_WIDTH = 16,
    parameter SCORE_WIDTH = 32
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [IN_FEATURES*COUNT_WIDTH-1:0] data_in,
    output reg  valid_out,
    output reg  [1:0] class_out,
    output reg  signed [SCORE_WIDTH-1:0] score0,
    output reg  signed [SCORE_WIDTH-1:0] score1,
    output reg  signed [SCORE_WIDTH-1:0] score2
);
    reg signed [FP_WIDTH-1:0] weights [0:(3*IN_FEATURES)-1];
    reg signed [FP_WIDTH-1:0] bias [0:2];
    integer i;
    reg signed [SCORE_WIDTH-1:0] tmp0;
    reg signed [SCORE_WIDTH-1:0] tmp1;
    reg signed [SCORE_WIDTH-1:0] tmp2;
    reg signed [COUNT_WIDTH:0] feature;

    initial begin
        $readmemh("../../weights/fc_weights.hex", weights);
        $readmemh("../../weights/fc_bias.hex", bias);
    end

    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 0;
            class_out <= 0;
            score0 <= 0;
            score1 <= 0;
            score2 <= 0;
        end else begin
            valid_out <= 0;
            if (valid_in) begin
                tmp0 = bias[0];
                tmp1 = bias[1];
                tmp2 = bias[2];
                for (i = 0; i < IN_FEATURES; i = i + 1) begin
                    feature = $signed({1'b0, data_in[i*COUNT_WIDTH +: COUNT_WIDTH]});
                    tmp0 = tmp0 + feature * weights[i];
                    tmp1 = tmp1 + feature * weights[IN_FEATURES + i];
                    tmp2 = tmp2 + feature * weights[(2*IN_FEATURES) + i];
                end
                score0 <= tmp0;
                score1 <= tmp1;
                score2 <= tmp2;
                if (tmp0 >= tmp1 && tmp0 >= tmp2)
                    class_out <= 2'd0;
                else if (tmp1 >= tmp2)
                    class_out <= 2'd1;
                else
                    class_out <= 2'd2;
                valid_out <= 1'b1;
            end
        end
    end
endmodule

