`timescale 1ns / 1ps

module bcnn_top(
    input  wire clk,
    input  wire rst,
    input  wire [7:0] pixel_in,
    input  wire pixel_valid,
    output wire [1:0] class_out,
    output wire result_valid
);
    wire conv1_valid;
    wire [31:0] conv1_data;
    wire dw2_valid;
    wire [31:0] dw2_data;
    wire pw2_valid;
    wire [63:0] pw2_data;
    wire pool1_valid;
    wire [63:0] pool1_data;
    wire dw3_valid;
    wire [63:0] dw3_data;
    wire pw3_valid;
    wire [127:0] pw3_data;
    wire pool2_valid;
    wire [127:0] pool2_data;
    wire dw4_valid;
    wire [127:0] dw4_data;
    wire pw4_valid;
    wire [255:0] pw4_data;
    wire gap_valid;
    wire [2047:0] gap_data;
    wire signed [31:0] score0;
    wire signed [31:0] score1;
    wire signed [31:0] score2;

    conv1_layer conv1_inst (
        .clk(clk),
        .rst(rst),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .valid_out(conv1_valid),
        .data_out(conv1_data)
    );

    bconv_dw_layer #(
        .CHANNELS(32),
        .IMG_W(32),
        .IMG_H(32),
        .WEIGHT_FILE("../../weights/bconv2_dw_weights_packed.hex"),
        .THRESH_FILE("../../weights/bconv2_bn_threshold.hex"),
        .WORD_COUNT(9)
    ) bconv2_dw_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(conv1_valid),
        .data_in(conv1_data),
        .valid_out(dw2_valid),
        .data_out(dw2_data)
    );

    bconv_pw_layer #(
        .IN_CHANNELS(32),
        .OUT_CHANNELS(64),
        .IMG_W(32),
        .IMG_H(32),
        .WORDS_PER_OUT(1),
        .WEIGHT_FILE("../../weights/bconv2_pw_weights_packed.hex"),
        .THRESH_FILE("../../weights/bconv2_bn_threshold.hex")
    ) bconv2_pw_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(dw2_valid),
        .data_in(dw2_data),
        .valid_out(pw2_valid),
        .data_out(pw2_data)
    );

    maxpool_layer #(.CHANNELS(64), .IMG_W(32), .IMG_H(32)) pool1_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(pw2_valid),
        .data_in(pw2_data),
        .valid_out(pool1_valid),
        .data_out(pool1_data)
    );

    bconv_dw_layer #(
        .CHANNELS(64),
        .IMG_W(16),
        .IMG_H(16),
        .WEIGHT_FILE("../../weights/bconv3_dw_weights_packed.hex"),
        .THRESH_FILE("../../weights/bconv3_bn_threshold.hex"),
        .WORD_COUNT(18)
    ) bconv3_dw_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(pool1_valid),
        .data_in(pool1_data),
        .valid_out(dw3_valid),
        .data_out(dw3_data)
    );

    bconv_pw_layer #(
        .IN_CHANNELS(64),
        .OUT_CHANNELS(128),
        .IMG_W(16),
        .IMG_H(16),
        .WORDS_PER_OUT(2),
        .WEIGHT_FILE("../../weights/bconv3_pw_weights_packed.hex"),
        .THRESH_FILE("../../weights/bconv3_bn_threshold.hex")
    ) bconv3_pw_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(dw3_valid),
        .data_in(dw3_data),
        .valid_out(pw3_valid),
        .data_out(pw3_data)
    );

    maxpool_layer #(.CHANNELS(128), .IMG_W(16), .IMG_H(16)) pool2_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(pw3_valid),
        .data_in(pw3_data),
        .valid_out(pool2_valid),
        .data_out(pool2_data)
    );

    bconv_dw_layer #(
        .CHANNELS(128),
        .IMG_W(8),
        .IMG_H(8),
        .WEIGHT_FILE("../../weights/bconv4_dw_weights_packed.hex"),
        .THRESH_FILE("../../weights/bconv4_bn_threshold.hex"),
        .WORD_COUNT(36)
    ) bconv4_dw_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(pool2_valid),
        .data_in(pool2_data),
        .valid_out(dw4_valid),
        .data_out(dw4_data)
    );

    bconv_pw_layer #(
        .IN_CHANNELS(128),
        .OUT_CHANNELS(256),
        .IMG_W(8),
        .IMG_H(8),
        .WORDS_PER_OUT(4),
        .WEIGHT_FILE("../../weights/bconv4_pw_weights_packed.hex"),
        .THRESH_FILE("../../weights/bconv4_bn_threshold.hex")
    ) bconv4_pw_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(dw4_valid),
        .data_in(dw4_data),
        .valid_out(pw4_valid),
        .data_out(pw4_data)
    );

    gap_layer #(.CHANNELS(256), .IMG_W(8), .IMG_H(8), .COUNT_WIDTH(8)) gap_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(pw4_valid),
        .data_in(pw4_data),
        .valid_out(gap_valid),
        .data_out(gap_data)
    );

    fc_layer fc_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(gap_valid),
        .data_in(gap_data),
        .valid_out(result_valid),
        .class_out(class_out),
        .score0(score0),
        .score1(score1),
        .score2(score2)
    );
endmodule
