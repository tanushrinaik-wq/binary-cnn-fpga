`timescale 1ns / 1ps

module bconv_dw_layer #(
    parameter CHANNELS = 32,
    parameter IMG_W = 32,
    parameter IMG_H = 32,
    parameter WEIGHT_FILE = "../../weights/bconv2_dw_weights_packed.hex",
    parameter THRESH_FILE = "../../weights/bconv2_bn_threshold.hex",
    parameter WORD_COUNT = 9
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,
    output reg  valid_out,
    output reg  [CHANNELS-1:0] data_out
);
    reg [CHANNELS-1:0] fmap [0:(IMG_W*IMG_H)-1];
    reg [31:0] weight_words [0:WORD_COUNT-1];
    reg signed [15:0] threshold_mem [0:CHANNELS-1];
    reg [15:0] load_count;
    reg [15:0] out_x;
    reg [15:0] out_y;
    reg computing;
    integer c;
    integer ky;
    integer kx;
    integer bit_idx;
    integer map_index;
    integer gx;
    integer gy;
    integer global_bit;
    reg [8:0] patch;
    reg [3:0] match_count;
    reg weight_bit;

    initial begin
        $readmemh(WEIGHT_FILE, weight_words);
        $readmemh(THRESH_FILE, threshold_mem);
    end

    always @(posedge clk) begin
        if (rst) begin
            load_count <= 0;
            out_x <= 0;
            out_y <= 0;
            computing <= 0;
            valid_out <= 0;
            data_out <= 0;
        end else begin
            valid_out <= 0;
            if (!computing) begin
                if (valid_in) begin
                    fmap[load_count] <= data_in;
                    if (load_count == (IMG_W*IMG_H)-1) begin
                        computing <= 1'b1;
                        load_count <= 0;
                        out_x <= 0;
                        out_y <= 0;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end else begin
                for (c = 0; c < CHANNELS; c = c + 1) begin
                    bit_idx = 0;
                    patch = 0;
                    for (ky = 0; ky < 3; ky = ky + 1) begin
                        for (kx = 0; kx < 3; kx = kx + 1) begin
                            gx = out_x + kx - 1;
                            gy = out_y + ky - 1;
                            if (gx < 0 || gx >= IMG_W || gy < 0 || gy >= IMG_H)
                                patch[bit_idx] = 1'b0;
                            else begin
                                map_index = (gy * IMG_W) + gx;
                                patch[bit_idx] = fmap[map_index][c];
                            end
                            bit_idx = bit_idx + 1;
                        end
                    end
                    match_count = 0;
                    for (bit_idx = 0; bit_idx < 9; bit_idx = bit_idx + 1) begin
                        global_bit = c * 9 + bit_idx;
                        weight_bit = weight_words[global_bit / 32][31 - (global_bit % 32)];
                        match_count = match_count + ~(patch[bit_idx] ^ weight_bit);
                    end
                    data_out[c] <= (($signed({12'd0, match_count}) <<< 8) > threshold_mem[c]);
                end
                valid_out <= 1'b1;
                if (out_x == IMG_W - 1) begin
                    out_x <= 0;
                    if (out_y == IMG_H - 1) begin
                        out_y <= 0;
                        computing <= 1'b0;
                    end else begin
                        out_y <= out_y + 1'b1;
                    end
                end else begin
                    out_x <= out_x + 1'b1;
                end
            end
        end
    end
endmodule
