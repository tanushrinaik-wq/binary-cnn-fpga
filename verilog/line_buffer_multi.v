// ============================================================================
// Module : line_buffer_multi.v
// Description:
//   Multi-channel 3x3 sliding window generator
//   Used for L2
// ============================================================================

`timescale 1ns / 1ps

module line_buffer_multi #(
    parameter IMG_WIDTH = 30,   // reduced after L1
    parameter CHANNELS = 8
)(
    input  wire clk,
    input  wire rst_n,

    input  wire valid_in,
    input  wire [CHANNELS-1:0] pixel_in,

    output reg  valid_out,
    output reg  [CHANNELS*9-1:0] window_out
);

    // Column index
    reg [$clog2(IMG_WIDTH)-1:0] col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col <= 0;
        else if (valid_in)
            col <= (col == IMG_WIDTH-1) ? 0 : col + 1;
    end

    // Row buffers
    (* ramstyle = "M4K" *)
    reg [CHANNELS-1:0] row_buf1 [0:IMG_WIDTH-1];

    (* ramstyle = "M4K" *)
    reg [CHANNELS-1:0] row_buf2 [0:IMG_WIDTH-1];

    reg [CHANNELS-1:0] row1_data, row2_data;

    always @(posedge clk) begin
        if (valid_in) begin
            row1_data <= row_buf1[col];
            row2_data <= row_buf2[col];

            row_buf2[col] <= row_buf1[col];
            row_buf1[col] <= pixel_in;
        end
    end

    // Horizontal shift registers per channel
    reg [2:0] shift0 [0:CHANNELS-1];
    reg [2:0] shift1 [0:CHANNELS-1];
    reg [2:0] shift2 [0:CHANNELS-1];

    integer c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (c = 0; c < CHANNELS; c = c + 1) begin
                shift0[c] <= 0;
                shift1[c] <= 0;
                shift2[c] <= 0;
            end
        end else if (valid_in) begin
            for (c = 0; c < CHANNELS; c = c + 1) begin
                shift0[c] <= {shift0[c][1:0], pixel_in[c]};
                shift1[c] <= {shift1[c][1:0], row1_data[c]};
                shift2[c] <= {shift2[c][1:0], row2_data[c]};
            end
        end
    end

    // Row tracking
    reg [$clog2(IMG_WIDTH):0] row_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            row_count <= 0;
        else if (valid_in && col == IMG_WIDTH-1)
            row_count <= row_count + 1;
    end

    wire ready = (row_count >= 2) && (col >= 2);

    // Output assembly
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 0;
            window_out <= 0;
        end else begin
            valid_out <= ready && valid_in;

            if (ready && valid_in) begin
                for (i = 0; i < CHANNELS; i = i + 1) begin
                    window_out[i*9 +: 9] <= {
                        shift2[i][2], shift2[i][1], shift2[i][0],
                        shift1[i][2], shift1[i][1], shift1[i][0],
                        shift0[i][2], shift0[i][1], shift0[i][0]
                    };
                end
            end
        end
    end

endmodule