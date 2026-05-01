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
    reg [IN_CHANNELS-1:0] fmap [0:(IMG_W*IMG_H)-1];
    reg [31:0] weight_words [0:(OUT_CHANNELS*WORDS_PER_OUT)-1];
    reg signed [15:0] threshold_mem [0:OUT_CHANNELS-1];
    reg [15:0] load_count;
    reg [15:0] out_index;
    reg computing;
    integer oc;
    integer ic;
    integer word_index;
    integer bit_index;
    reg match_bit;
    reg [15:0] match_count;

    initial begin
        $readmemh(WEIGHT_FILE, weight_words);
        $readmemh(THRESH_FILE, threshold_mem);
    end

    always @(posedge clk) begin
        if (rst) begin
            load_count <= 0;
            out_index <= 0;
            computing <= 0;
            valid_out <= 0;
            data_out <= 0;
        end else begin
            valid_out <= 0;
            if (!computing) begin
                if (valid_in) begin
                    fmap[load_count] <= data_in;
                    if (load_count == (IMG_W*IMG_H)-1) begin
                        load_count <= 0;
                        computing <= 1'b1;
                        out_index <= 0;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end else begin
                for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin
                    match_count = 0;
                    for (ic = 0; ic < IN_CHANNELS; ic = ic + 1) begin
                        word_index = oc * WORDS_PER_OUT + (ic / 32);
                        bit_index = 31 - (ic % 32);
                        match_bit = ~(fmap[out_index][ic] ^ weight_words[word_index][bit_index]);
                        match_count = match_count + match_bit;
                    end
                    data_out[oc] <= (($signed(match_count) <<< 8) > threshold_mem[oc]);
                end
                valid_out <= 1'b1;
                if (out_index == (IMG_W*IMG_H)-1) begin
                    out_index <= 0;
                    computing <= 1'b0;
                end else begin
                    out_index <= out_index + 1'b1;
                end
            end
        end
    end
endmodule
