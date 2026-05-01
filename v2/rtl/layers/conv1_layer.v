`timescale 1ns / 1ps

module conv1_layer #(
    parameter IMG_W = 64,
    parameter IMG_H = 64,
    parameter IN_CHANNELS = 3,
    parameter OUT_CHANNELS = 32,
    parameter STRIDE = 2,
    parameter OUT_W = 32,
    parameter OUT_H = 32
)(
    input  wire clk,
    input  wire rst,
    input  wire [7:0] pixel_in,
    input  wire pixel_valid,
    output reg  valid_out,
    output reg  [OUT_CHANNELS-1:0] data_out
);
    localparam TOTAL_PIXELS = IMG_W * IMG_H * IN_CHANNELS;
    localparam TOTAL_TAPS = IN_CHANNELS * 3 * 3;

    reg [7:0] image_mem [0:TOTAL_PIXELS-1];
    reg signed [7:0] weight_mem [0:(OUT_CHANNELS*TOTAL_TAPS)-1];
    reg signed [15:0] threshold_mem [0:OUT_CHANNELS-1];

    reg [15:0] load_count;
    reg compute_en;
    reg [15:0] out_x;
    reg [15:0] out_y;
    reg [7:0] out_ch;
    reg [5:0] tap_idx;
    reg [OUT_CHANNELS-1:0] out_bits_accum;
    reg signed [31:0] acc;

    integer ic_i;
    integer ky_i;
    integer kx_i;
    integer img_index;
    integer w_index;
    reg signed [31:0] acc_work;
    reg [OUT_CHANNELS-1:0] next_bits;

    initial begin
        $readmemh("../../weights/conv1_weights.hex", weight_mem);
        $readmemh("../../weights/bn1_threshold.hex", threshold_mem);
    end

    always @(posedge clk) begin
        if (rst) begin
            load_count <= 0;
            compute_en <= 1'b0;
            out_x <= 0;
            out_y <= 0;
            out_ch <= 0;
            tap_idx <= 0;
            out_bits_accum <= 0;
            acc <= 0;
            valid_out <= 0;
            data_out <= 0;
        end else begin
            valid_out <= 1'b0;

            if (!compute_en) begin
                if (pixel_valid) begin
                    image_mem[load_count] <= pixel_in;
                    if (load_count == TOTAL_PIXELS - 1) begin
                        load_count <= 0;
                        compute_en <= 1'b1;
                        out_x <= 0;
                        out_y <= 0;
                        out_ch <= 0;
                        tap_idx <= 0;
                        out_bits_accum <= 0;
                        acc <= 0;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end else begin
                ic_i = tap_idx % IN_CHANNELS;
                ky_i = (tap_idx / IN_CHANNELS) / 3;
                kx_i = (tap_idx / IN_CHANNELS) % 3;
                img_index = (((out_y * STRIDE) + ky_i) * IMG_W + ((out_x * STRIDE) + kx_i)) * IN_CHANNELS + ic_i;
                w_index = (out_ch * TOTAL_TAPS) + tap_idx;

                acc_work = acc + ($signed({1'b0, image_mem[img_index]}) * $signed(weight_mem[w_index]));

                if (tap_idx == TOTAL_TAPS - 1) begin
                    next_bits = out_bits_accum;
                    next_bits[out_ch] = (acc_work > threshold_mem[out_ch]);

                    if (out_ch == OUT_CHANNELS - 1) begin
                        data_out <= next_bits;
                        valid_out <= 1'b1;
                        out_bits_accum <= 0;
                        out_ch <= 0;

                        if (out_x == OUT_W - 1) begin
                            out_x <= 0;
                            if (out_y == OUT_H - 1) begin
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
                    acc <= 0;
                end else begin
                    tap_idx <= tap_idx + 1'b1;
                    acc <= acc_work;
                end
            end
        end
    end
endmodule
