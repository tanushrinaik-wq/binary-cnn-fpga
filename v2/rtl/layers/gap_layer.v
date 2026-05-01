`timescale 1ns / 1ps

module gap_layer #(
    parameter CHANNELS = 256,
    parameter IMG_W = 8,
    parameter IMG_H = 8,
    parameter COUNT_WIDTH = 8
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,
    output reg  valid_out,
    output reg  [CHANNELS*COUNT_WIDTH-1:0] data_out
);
    reg [COUNT_WIDTH-1:0] acc [0:CHANNELS-1];
    reg [15:0] count;
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 0;
            data_out <= 0;
            count <= 0;
            for (i = 0; i < CHANNELS; i = i + 1)
                acc[i] <= 0;
        end else begin
            valid_out <= 0;
            if (valid_in) begin
                for (i = 0; i < CHANNELS; i = i + 1)
                    acc[i] <= acc[i] + data_in[i];
                if (count == (IMG_W * IMG_H) - 1) begin
                    count <= 0;
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        data_out[i*COUNT_WIDTH +: COUNT_WIDTH] <= acc[i] + data_in[i];
                        acc[i] <= 0;
                    end
                    valid_out <= 1'b1;
                end else begin
                    count <= count + 1'b1;
                end
            end
        end
    end
endmodule

