// ============================================================================
// Module : line_buffer.v (CLEAN - SIM SAFE)
// ============================================================================

`timescale 1ns / 1ps

module line_buffer #(
    parameter IMG_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    input  wire valid_in,
    input  wire pixel_in,

    output reg  valid_out,
    output reg  [8:0] window_out
);

    // COLUMN
    reg [$clog2(IMG_WIDTH)-1:0] col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col <= 0;
        else if (valid_in)
            col <= (col == IMG_WIDTH-1) ? 0 : col + 1;
    end

    // INPUT DELAY
    reg pixel_in_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pixel_in_d <= 0;
        else if (valid_in)
            pixel_in_d <= pixel_in;
    end

    // ROW BUFFERS (removed ramstyle)
    reg row_buf1 [0:IMG_WIDTH-1];
    reg row_buf2 [0:IMG_WIDTH-1];

    reg row1_data, row2_data;

    always @(posedge clk) begin
        if (valid_in) begin
            row1_data <= row_buf1[col];
            row2_data <= row_buf2[col];

            row_buf2[col] <= row_buf1[col];
            row_buf1[col] <= pixel_in;
        end
    end

    // SHIFT REG
    reg [2:0] shift_row0, shift_row1, shift_row2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_row0 <= 0;
            shift_row1 <= 0;
            shift_row2 <= 0;
        end else if (valid_in) begin
            shift_row0 <= {shift_row0[1:0], pixel_in_d};
            shift_row1 <= {shift_row1[1:0], row1_data};
            shift_row2 <= {shift_row2[1:0], row2_data};
        end
    end

    // ROW COUNT
    reg [$clog2(IMG_WIDTH):0] row_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            row_count <= 0;
        else if (valid_in && col == IMG_WIDTH-1)
            row_count <= row_count + 1;
    end

    wire window_ready = (row_count >= 2) && (col >= 2);

    reg valid_in_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_in_d <= 0;
        else
            valid_in_d <= valid_in;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 0;
            window_out <= 0;
        end else begin
            valid_out <= window_ready && valid_in_d;

            if (window_ready && valid_in_d) begin
                window_out <= {
                    shift_row2[2], shift_row2[1], shift_row2[0],
                    shift_row1[2], shift_row1[1], shift_row1[0],
                    shift_row0[2], shift_row0[1], shift_row0[0]
                };
            end
        end
    end

// synthesis translate_off
    integer i;
    initial begin
        for (i = 0; i < IMG_WIDTH; i = i + 1) begin
            row_buf1[i] = 0;
            row_buf2[i] = 0;
        end
    end
// synthesis translate_on

endmodule