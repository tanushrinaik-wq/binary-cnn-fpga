// ============================================================================
// Module : pool_classifier.v (SPEC-COMPLIANT)
// Description:
//   Global Average Pooling + Argmax Classifier
//   Works on streaming BCNN outputs
// ============================================================================

`timescale 1ns / 1ps

module pool_classifier #(
    parameter CHANNELS = 8,
    parameter IMG_WIDTH = 32,
    parameter IMG_HEIGHT = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Control
    input  wire start,      // start of frame
    input  wire end_frame,  // end of frame

    // Streaming input from BCNN
    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,

    // Output
    output reg  valid_out,
    output reg  [$clog2(CHANNELS)-1:0] class_out
);

    // =========================================================================
    // 1. Accumulators (per channel)
    // =========================================================================
    localparam ACC_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT + 1);

    reg [ACC_WIDTH-1:0] acc [0:CHANNELS-1];

    integer i;

    // =========================================================================
    // 2. Accumulation logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < CHANNELS; i = i + 1)
                acc[i] <= 0;
        end else begin

            // Reset at start of new frame
            if (start) begin
                for (i = 0; i < CHANNELS; i = i + 1)
                    acc[i] <= 0;
            end

            // Accumulate
            else if (valid_in) begin
                for (i = 0; i < CHANNELS; i = i + 1) begin
                    acc[i] <= acc[i] + data_in[i];
                end
            end
        end
    end

    // =========================================================================
    // 3. Argmax classifier
    // =========================================================================
    reg [ACC_WIDTH-1:0] max_val;
    reg [$clog2(CHANNELS)-1:0] max_idx;

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_val  <= 0;
            max_idx  <= 0;
            class_out <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= 0;

            if (end_frame) begin
                max_val = acc[0];
                max_idx = 0;

                for (j = 1; j < CHANNELS; j = j + 1) begin
                    if (acc[j] > max_val) begin
                        max_val = acc[j];
                        max_idx = j;
                    end
                end

                class_out <= max_idx;
                valid_out <= 1'b1;
            end
        end
    end

endmodule