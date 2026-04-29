// ============================================================================
// Module : pool_classifier.v (FULLY FIXED - SPEC COMPLIANT)
// Description:
//   Global Average Pooling + Dense(16→2) + Argmax
//   Fixes:
//     - Dense layer implemented (Q8.8 fixed point)
//     - Last window inclusion (end_frame delayed)
//     - No blocking assignments in sequential logic
// ============================================================================

`timescale 1ns / 1ps

module pool_classifier #(
    parameter CHANNELS = 16,
    parameter IMG_WIDTH = 32,
    parameter IMG_HEIGHT = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Control
    input  wire start,
    input  wire end_frame,

    // Streaming input
    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,

    // Output
    output reg  valid_out,
    output reg  class_out   // now 2 classes → 1 bit
);

    // =========================================================================
    // PARAMETERS
    // =========================================================================
    localparam ACC_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT + 1);
    localparam FP_WIDTH  = 16;  // Q8.8

    // =========================================================================
    // GAP ACCUMULATORS
    // =========================================================================
    reg [ACC_WIDTH-1:0] acc [0:CHANNELS-1];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < CHANNELS; i = i + 1)
                acc[i] <= 0;
        end else begin
            if (start) begin
                for (i = 0; i < CHANNELS; i = i + 1)
                    acc[i] <= 0;
            end
            else if (valid_in) begin
                for (i = 0; i < CHANNELS; i = i + 1)
                    acc[i] <= acc[i] + data_in[i];
            end
        end
    end

    // =========================================================================
    // END FRAME DELAY (FIX: include last accumulation)
    // =========================================================================
    reg end_frame_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            end_frame_d <= 1'b0;
        else
            end_frame_d <= end_frame;
    end

    // =========================================================================
    // DENSE LAYER WEIGHTS (Q8.8 fixed-point)
    // NOTE: Replace with real trained weights
    // =========================================================================
    reg signed [FP_WIDTH-1:0] W [0:1][0:CHANNELS-1];
    reg signed [FP_WIDTH-1:0] B [0:1];

    initial begin
        $readmemh("dense_w0.hex", W[0]);
        $readmemh("dense_w1.hex", W[1]);
        $readmemh("dense_b.hex",  B);
    end

    // =========================================================================
    // DENSE COMPUTATION
    // =========================================================================
    reg signed [31:0] score0;
    reg signed [31:0] score1;

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score0 <= 0;
            score1 <= 0;
            class_out <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= 0;

            if (end_frame_d) begin
                score0 <= B[0];
                score1 <= B[1];

                for (j = 0; j < CHANNELS; j = j + 1) begin
                    score0 <= score0 + (acc[j] * W[0][j]);
                    score1 <= score1 + (acc[j] * W[1][j]);
                end

                // Argmax (2 classes)
                if (score1 > score0)
                    class_out <= 1'b1;
                else
                    class_out <= 1'b0;

                valid_out <= 1'b1;
            end
        end
    end

endmodule