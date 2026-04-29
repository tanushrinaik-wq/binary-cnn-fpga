// ============================================================================
// Module : pool_classifier.v (FINAL - CORRECT + NO MULTIPLIERS)
// ============================================================================

`timescale 1ns / 1ps

module pool_classifier #(
    parameter CHANNELS = 16,
    parameter IMG_WIDTH = 32,
    parameter IMG_HEIGHT = 32
)(
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire end_frame,

    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,

    output reg  valid_out,
    output reg  class_out   // 1-bit (2 classes)
);

    // =========================================================================
    // PARAMETERS
    // =========================================================================
    localparam ACC_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT + 1);
    localparam FP_WIDTH  = 16;

    // =========================================================================
    // GAP ACCUMULATION
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
    // END FRAME DELAY
    // =========================================================================
    reg end_frame_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            end_frame_d <= 0;
        else
            end_frame_d <= end_frame;
    end

    // =========================================================================
    // WEIGHTS (Q8.8 FIXED POINT)
    // =========================================================================
    reg signed [FP_WIDTH-1:0] W [0:1][0:CHANNELS-1];
    reg signed [FP_WIDTH-1:0] B [0:1];

    initial begin
        $readmemh("dense_w0.hex", W[0]);
        $readmemh("dense_w1.hex", W[1]);
        $readmemh("dense_b.hex",  B);
    end

    // =========================================================================
    // SHIFT-ADD MULTIPLIER (NO '*')
    // =========================================================================
    function signed [31:0] mult_shift_add;
        input [ACC_WIDTH-1:0] a;
        input signed [FP_WIDTH-1:0] w;
        integer k;
        begin
            mult_shift_add = 0;
            for (k = 0; k < FP_WIDTH; k = k + 1) begin
                if (w[k])
                    mult_shift_add = mult_shift_add + (a <<< k);
            end
            if (w[FP_WIDTH-1]) // sign bit correction
                mult_shift_add = -mult_shift_add;
        end
    endfunction

    // =========================================================================
    // COMBINATIONAL DENSE (FIXED NBA BUG)
    // =========================================================================
    reg signed [31:0] comb_score0;
    reg signed [31:0] comb_score1;

    integer j;

    always @(*) begin
        comb_score0 = {{(32-FP_WIDTH){B[0][FP_WIDTH-1]}}, B[0]};
        comb_score1 = {{(32-FP_WIDTH){B[1][FP_WIDTH-1]}}, B[1]};

        for (j = 0; j < CHANNELS; j = j + 1) begin
            comb_score0 = comb_score0 + mult_shift_add(acc[j], W[0][j]);
            comb_score1 = comb_score1 + mult_shift_add(acc[j], W[1][j]);
        end
    end

    // =========================================================================
    // OUTPUT REGISTER
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            class_out <= 0;
        end else begin
            valid_out <= end_frame_d;

            if (end_frame_d)
                class_out <= (comb_score1 > comb_score0);
        end
    end

endmodule