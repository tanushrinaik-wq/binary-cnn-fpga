// ============================================================================
// Module : line_buffer_multi.v (CLEAN - SIM SAFE)
// ============================================================================

`timescale 1ns / 1ps

module line_buffer_multi #(
    parameter IMG_WIDTH = 30,
    parameter CHANNELS = 8
)(
    input  wire clk,
    input  wire rst_n,

    input  wire valid_in,
    input  wire [CHANNELS-1:0] pixel_in,

    output reg  valid_out,
    output reg  [CHANNELS*9-1:0] window_out
);

    reg [$clog2(IMG_WIDTH)-1:0] col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            col <= 0;
        else if (valid_in)
            col <= (col == IMG_WIDTH-1) ? 0 : col + 1;
    end

    reg [CHANNELS-1:0] pixel_in_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pixel_in_d <= 0;
        else if (valid_in)
            pixel_in_d <= pixel_in;
    end

    // ROW BUFFERS (removed ramstyle)
    reg [CHANNELS-1:0] row_buf1 [0:IMG_WIDTH-1];
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
                shift0[c] <= {shift0[c][1:0], pixel_in_d[c]};
                shift1[c] <= {shift1[c][1:0], row1_data[c]};
                shift2[c] <= {shift2[c][1:0], row2_data[c]};
            end
        end
    end

    reg [$clog2(IMG_WIDTH):0] row_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            row_count <= 0;
        else if (valid_in && col == IMG_WIDTH-1)
            row_count <= row_count + 1;
    end

    wire ready = (row_count >= 2) && (col >= 2);

    reg valid_in_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_in_d <= 0;
        else
            valid_in_d <= valid_in;
    end

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 0;
            window_out <= 0;
        end else begin
            valid_out <= ready && valid_in_d;

            if (ready && valid_in_d) begin
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

// synthesis translate_off
    integer j;
    initial begin
        for (j = 0; j < IMG_WIDTH; j = j + 1) begin
            row_buf1[j] = 0;
            row_buf2[j] = 0;
        end
    end
// synthesis translate_on

endmodules