`timescale 1ns / 1ps

module maxpool_layer #(
    parameter CHANNELS = 64,
    parameter IMG_W = 32,
    parameter IMG_H = 32
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,
    output reg  valid_out,
    output reg  [CHANNELS-1:0] data_out
);
    reg [CHANNELS-1:0] row_buf [0:IMG_W-1];
    reg [15:0] col;
    reg [15:0] row;
    reg [CHANNELS-1:0] top_left;
    reg [CHANNELS-1:0] top_right;
    reg [CHANNELS-1:0] bottom_left;
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            col <= 0;
            row <= 0;
            valid_out <= 0;
            data_out <= 0;
            top_left <= 0;
            top_right <= 0;
            bottom_left <= 0;
        end else begin
            valid_out <= 0;
            if (valid_in) begin
                if (row[0] == 1'b0 && col[0] == 1'b0) begin
                    top_left <= data_in;
                end else if (row[0] == 1'b0 && col[0] == 1'b1) begin
                    top_right <= data_in;
                end else if (row[0] == 1'b1 && col[0] == 1'b0) begin
                    bottom_left <= data_in;
                end else begin
                    data_out <= top_left | top_right | bottom_left | data_in;
                    valid_out <= 1'b1;
                end

                if (col == IMG_W - 1) begin
                    col <= 0;
                    if (row == IMG_H - 1)
                        row <= 0;
                    else
                        row <= row + 1'b1;
                end else begin
                    col <= col + 1'b1;
                end
            end
        end
    end
endmodule

