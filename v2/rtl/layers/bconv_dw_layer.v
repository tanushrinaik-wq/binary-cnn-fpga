`timescale 1ns / 1ps

module bconv_dw_layer #(
    parameter CHANNELS = 32,
    parameter IMG_W = 32,
    parameter IMG_H = 32,
    parameter WEIGHT_FILE = "../../weights/bconv2_dw_weights_packed.hex",
    parameter WORD_COUNT = 9
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,
    output reg  valid_out,
    output reg  [CHANNELS-1:0] data_out
);
    localparam TOTAL_POSITIONS = IMG_W * IMG_H;
    localparam MATCH_THRESHOLD = 5;

    reg [CHANNELS-1:0] fmap [0:TOTAL_POSITIONS-1];
    reg [31:0] weight_words [0:WORD_COUNT-1];

    reg [15:0] load_count;
    reg compute_en;
    reg [15:0] out_x;
    reg [15:0] out_y;
    reg [8:0] out_ch;
    reg [3:0] tap_idx;
    reg [3:0] match_acc;
    reg [CHANNELS-1:0] out_bits_accum;

    integer gx;
    integer gy;
    integer map_index;
    integer global_bit;
    reg pixel_bit;
    reg weight_bit;
    reg [3:0] match_work;
    reg [CHANNELS-1:0] next_bits;

    initial begin
        $readmemh(WEIGHT_FILE, weight_words);
    end

    always @(posedge clk) begin
        if (rst) begin
            load_count <= 0;
            compute_en <= 1'b0;
            out_x <= 0;
            out_y <= 0;
            out_ch <= 0;
            tap_idx <= 0;
            match_acc <= 0;
            out_bits_accum <= 0;
            valid_out <= 0;
            data_out <= 0;
        end else begin
            valid_out <= 1'b0;

            if (!compute_en) begin
                if (valid_in) begin
                    fmap[load_count] <= data_in;
                    if (load_count == TOTAL_POSITIONS - 1) begin
                        load_count <= 0;
                        compute_en <= 1'b1;
                        out_x <= 0;
                        out_y <= 0;
                        out_ch <= 0;
                        tap_idx <= 0;
                        match_acc <= 0;
                        out_bits_accum <= 0;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end else begin
                gx = out_x + (tap_idx % 3) - 1;
                gy = out_y + (tap_idx / 3) - 1;

                if (gx < 0 || gx >= IMG_W || gy < 0 || gy >= IMG_H)
                    pixel_bit = 1'b0;
                else begin
                    map_index = (gy * IMG_W) + gx;
                    pixel_bit = fmap[map_index][out_ch];
                end

                global_bit = (out_ch * 9) + tap_idx;
                weight_bit = weight_words[global_bit / 32][31 - (global_bit % 32)];
                match_work = match_acc + ~(pixel_bit ^ weight_bit);

                if (tap_idx == 8) begin
                    next_bits = out_bits_accum;
                    next_bits[out_ch] = (match_work >= MATCH_THRESHOLD);

                    if (out_ch == CHANNELS - 1) begin
                        data_out <= next_bits;
                        valid_out <= 1'b1;
                        out_bits_accum <= 0;
                        out_ch <= 0;

                        if (out_x == IMG_W - 1) begin
                            out_x <= 0;
                            if (out_y == IMG_H - 1) begin
                                out_y <= 0;
                                compute_en <= 1'b0;
                            end else begin
                                out_y <= out_y + 1'b1;
                            end
                        end else begin
                            out_x <= out_x + 1'b1;
                        end
                    end else begin
                        out_bits_accum <= next_bits;
                        out_ch <= out_ch + 1'b1;
                    end

                    tap_idx <= 0;
                    match_acc <= 0;
                end else begin
                    tap_idx <= tap_idx + 1'b1;
                    match_acc <= match_work;
                end
            end
        end
    end
endmodule
