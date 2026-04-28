// ============================================================================
// Module : line_buffer.v (FINAL SPEC-COMPLIANT)
// Description:
//   True 3x3 sliding window generator
//   BRAM-friendly row buffers + shift registers
// ============================================================================

`timescale 1ns / 1ps

module line_buffer #(
    parameter IMG_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Input stream
    input  wire valid_in,
    input  wire pixel_in,   // 1-bit

    // Output
    output reg  valid_out,
    output reg  [8:0] window_out
);

    // =========================================================================
    // 1. Column counter
    // =========================================================================
    reg [$clog2(IMG_WIDTH)-1:0] col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col <= 0;
        else if (valid_in) begin
            if (col == IMG_WIDTH - 1)
                col <= 0;
            else
                col <= col + 1;
        end
    end

    // =========================================================================
    // 2. Row buffers (BRAM-friendly)
    // =========================================================================
    (* ramstyle = "M4K" *) reg row_buf1 [0:IMG_WIDTH-1]; // N-1
    (* ramstyle = "M4K" *) reg row_buf2 [0:IMG_WIDTH-1]; // N-2

    reg row1_data, row2_data;

    always @(posedge clk) begin
        if (valid_in) begin
            row1_data <= row_buf1[col];
            row2_data <= row_buf2[col];

            row_buf2[col] <= row_buf1[col];
            row_buf1[col] <= pixel_in;
        end
    end

    // =========================================================================
    // 3. Horizontal shift registers (window columns)
    // =========================================================================
    reg [2:0] shift_row0;
    reg [2:0] shift_row1;
    reg [2:0] shift_row2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_row0 <= 0;
            shift_row1 <= 0;
            shift_row2 <= 0;
        end else if (valid_in) begin
            shift_row0 <= {shift_row0[1:0], pixel_in};
            shift_row1 <= {shift_row1[1:0], row1_data};
            shift_row2 <= {shift_row2[1:0], row2_data};
        end
    end

    // =========================================================================
    // 4. Row counter (to know when enough rows are filled)
    // =========================================================================
    reg [$clog2(IMG_WIDTH):0] row_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            row_count <= 0;
        else if (valid_in && col == IMG_WIDTH - 1)
            row_count <= row_count + 1;
    end

    // =========================================================================
    // 5. Valid generation
    // =========================================================================
    wire window_ready = (row_count >= 2) && (col >= 2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 0;
            window_out <= 0;
        end else begin
            valid_out <= window_ready && valid_in;

            if (window_ready && valid_in) begin
                window_out <= {
                    shift_row2[2], shift_row2[1], shift_row2[0],
                    shift_row1[2], shift_row1[1], shift_row1[0],
                    shift_row0[2], shift_row0[1], shift_row0[0]
                };
            end
        end
    end

endmodule