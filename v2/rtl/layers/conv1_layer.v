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
    reg [7:0] image_mem [0:(IMG_W*IMG_H*IN_CHANNELS)-1];
    reg signed [7:0] weight_mem [0:(OUT_CHANNELS*IN_CHANNELS*3*3)-1];
    reg signed [15:0] threshold_mem [0:OUT_CHANNELS-1];
    reg [15:0] load_count;
    reg [15:0] out_x;
    reg [15:0] out_y;
    reg computing;
    integer oc;
    integer ic;
    integer ky;
    integer kx;
    integer img_index;
    integer w_index;
    reg signed [31:0] acc;

    initial begin
        $readmemh("../../weights/conv1_weights.hex", weight_mem);
        $readmemh("../../weights/bn1_threshold.hex", threshold_mem);
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
                if (pixel_valid) begin
                    image_mem[load_count] <= pixel_in;
                    if (load_count == (IMG_W*IMG_H*IN_CHANNELS)-1) begin
                        load_count <= 0;
                        computing <= 1'b1;
                        out_x <= 0;
                        out_y <= 0;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end else begin
                for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin
                    acc = 0;
                    for (ic = 0; ic < IN_CHANNELS; ic = ic + 1) begin
                        for (ky = 0; ky < 3; ky = ky + 1) begin
                            for (kx = 0; kx < 3; kx = kx + 1) begin
                                img_index = (((out_y * STRIDE) + ky) * IMG_W + ((out_x * STRIDE) + kx)) * IN_CHANNELS + ic;
                                w_index = (((oc * IN_CHANNELS) + ic) * 3 + ky) * 3 + kx;
                                acc = acc + $signed({1'b0, image_mem[img_index]}) * weight_mem[w_index];
                            end
                        end
                    end
                    data_out[oc] <= (acc > threshold_mem[oc]);
                end
                valid_out <= 1'b1;
                if (out_x == OUT_W - 1) begin
                    out_x <= 0;
                    if (out_y == OUT_H - 1) begin
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

