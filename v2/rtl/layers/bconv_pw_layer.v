`timescale 1ns / 1ps

module bconv_pw_layer #(
    parameter IN_CHANNELS = 32,
    parameter OUT_CHANNELS = 64,
    parameter IMG_W = 32,
    parameter IMG_H = 32,
    parameter WORDS_PER_OUT = 1,
    parameter WEIGHT_FILE = "../../weights/bconv2_pw_weights_packed.hex",
    parameter THRESH_FILE = "../../weights/bconv2_bn_threshold.hex"
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [IN_CHANNELS-1:0] data_in,
    output reg  valid_out,
    output reg  [OUT_CHANNELS-1:0] data_out
);
    localparam TOTAL_POSITIONS = IMG_W * IMG_H;

    reg [IN_CHANNELS-1:0] fmap [0:TOTAL_POSITIONS-1];
    reg [31:0] weight_words [0:(OUT_CHANNELS*WORDS_PER_OUT)-1];
    reg signed [15:0] threshold_mem [0:OUT_CHANNELS-1];

    reg [15:0] load_count;
    reg compute_en;
    reg [15:0] out_index;
    reg [8:0] out_ch;
    reg [3:0] word_idx;
    reg [15:0] match_acc;
    reg [OUT_CHANNELS-1:0] out_bits_accum;

    wire [31:0] input_word;
    wire [31:0] weight_word;
    wire [5:0] popcount_out;

    reg [15:0] match_work;
    reg signed [15:0] signed_score_q88;
    reg [OUT_CHANNELS-1:0] next_bits;

    assign input_word = fmap[out_index][word_idx*32 +: 32];
    assign weight_word = weight_words[(out_ch * WORDS_PER_OUT) + word_idx];

    xnor_popcount #(.N(32), .COUNT_WIDTH(6)) popcount_inst (
        .a(input_word),
        .b(weight_word),
        .out(popcount_out)
    );

    initial begin
        $readmemh(WEIGHT_FILE, weight_words);
        $readmemh(THRESH_FILE, threshold_mem);
    end

    always @(posedge clk) begin
        if (rst) begin
            load_count <= 0;
            compute_en <= 1'b0;
            out_index <= 0;
            out_ch <= 0;
            word_idx <= 0;
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
                        out_index <= 0;
                        out_ch <= 0;
                        word_idx <= 0;
                        match_acc <= 0;
                        out_bits_accum <= 0;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end else begin
                match_work = match_acc + popcount_out;

                if (word_idx == WORDS_PER_OUT - 1) begin
                    signed_score_q88 = (($signed({1'b0, match_work}) <<< 9) - (IN_CHANNELS <<< 8));
                    next_bits = out_bits_accum;
                    next_bits[out_ch] = (signed_score_q88 > threshold_mem[out_ch]);

                    if (out_ch == OUT_CHANNELS - 1) begin
                        data_out <= next_bits;
                        valid_out <= 1'b1;
                        out_bits_accum <= 0;
                        out_ch <= 0;

                        if (out_index == TOTAL_POSITIONS - 1) begin
                            out_index <= 0;
                            compute_en <= 1'b0;
                        end else begin
                            out_index <= out_index + 1'b1;
                        end
                    end else begin
                        out_bits_accum <= next_bits;
                        out_ch <= out_ch + 1'b1;
                    end

                    word_idx <= 0;
                    match_acc <= 0;
                end else begin
                    word_idx <= word_idx + 1'b1;
                    match_acc <= match_work;
                end
            end
        end
    end
endmodule
