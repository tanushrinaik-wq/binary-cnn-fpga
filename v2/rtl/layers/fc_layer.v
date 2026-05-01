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
    reg [IN_FEATURES*COUNT_WIDTH-1:0] data_reg;

    reg busy;
    reg [1:0] class_idx;
    reg [8:0] feature_idx;
    reg signed [SCORE_WIDTH-1:0] acc;

    reg signed [COUNT_WIDTH:0] feature_value;
    reg signed [SCORE_WIDTH-1:0] acc_work;

    initial begin
        $readmemh("../../weights/fc_weights.hex", weights);
        $readmemh("../../weights/fc_bias.hex", bias);
    end

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            class_idx <= 0;
            feature_idx <= 0;
            acc <= 0;
            data_reg <= 0;
            valid_out <= 0;
            class_out <= 0;
            score0 <= 0;
            score1 <= 0;
            score2 <= 0;
        end else begin
            valid_out <= 1'b0;

            if (!busy) begin
                if (valid_in) begin
                    data_reg <= data_in;
                    busy <= 1'b1;
                    class_idx <= 0;
                    feature_idx <= 0;
                    acc <= bias[0];
                end
            end else begin
                feature_value = $signed({1'b0, data_reg[feature_idx*COUNT_WIDTH +: COUNT_WIDTH]});
                acc_work = acc + (feature_value * weights[(class_idx * IN_FEATURES) + feature_idx]);

                if (feature_idx == IN_FEATURES - 1) begin
                    if (class_idx == 0)
                        score0 <= acc_work;
                    else if (class_idx == 1)
                        score1 <= acc_work;
                    else
                        score2 <= acc_work;

                    if (class_idx == 2) begin
                        if (score0 >= score1 && score0 >= acc_work)
                            class_out <= 2'd0;
                        else if (score1 >= acc_work)
                            class_out <= 2'd1;
                        else
                            class_out <= 2'd2;
                        valid_out <= 1'b1;
                        busy <= 1'b0;
                    end else begin
                        class_idx <= class_idx + 1'b1;
                        feature_idx <= 0;
                        acc <= bias[class_idx + 1'b1];
                    end
                end else begin
                    feature_idx <= feature_idx + 1'b1;
                    acc <= acc_work;
                end
            end
        end
    end
endmodule
