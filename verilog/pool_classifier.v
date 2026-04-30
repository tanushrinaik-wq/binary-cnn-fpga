// ============================================================================
// Module : pool_classifier.v (FIXED - PIPELINED DSP)
//
// Changes from original:
//   1. Removed mult_shift_add function entirely.
//   2. Stage 1 (clocked): multiplies acc[i] * W using * operator so Quartus
//      infers DSP blocks instead of a shift-add LUT chain.
//   3. Stage 2 (clocked): accumulates the registered products + bias.
//   4. valid_out is now delayed 2 extra cycles relative to end_frame_d.
//      This is harmless — the FSM waits on valid_out anyway.
// ============================================================================

`timescale 1ns / 1ps

module pool_classifier #(
    parameter CHANNELS   = 16,
    parameter IMG_WIDTH  = 28,
    parameter IMG_HEIGHT = 28
)(
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire end_frame,

    input  wire valid_in,
    input  wire [CHANNELS-1:0] data_in,

    output reg  valid_out,
    output reg  class_out
);

    // =========================================================================
    // PARAMETERS
    // =========================================================================
    localparam ACC_WIDTH  = $clog2(IMG_WIDTH * IMG_HEIGHT + 1);
    localparam FP_WIDTH   = 16;
    // Product of ACC_WIDTH-bit unsigned * FP_WIDTH-bit signed
    localparam PROD_WIDTH = ACC_WIDTH + FP_WIDTH;
    // Sum of CHANNELS products — add log2(CHANNELS) bits to avoid overflow
    localparam SUM_WIDTH  = PROD_WIDTH + $clog2(CHANNELS) + 1;

    // =========================================================================
    // GAP ACCUMULATION (unchanged)
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
            else if (valid_in && !end_frame) begin
                for (i = 0; i < CHANNELS; i = i + 1)
                    acc[i] <= acc[i] + data_in[i];
            end
        end
    end

    // =========================================================================
    // END FRAME DELAY (unchanged)
    // =========================================================================
    reg end_frame_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            end_frame_d <= 0;
        else
            end_frame_d <= end_frame;
    end

    // =========================================================================
    // WEIGHTS (unchanged)
    // =========================================================================
    reg signed [FP_WIDTH-1:0] W0   [0:CHANNELS-1];
    reg signed [FP_WIDTH-1:0] W1   [0:CHANNELS-1];
    reg signed [FP_WIDTH-1:0] B0, B1;
    reg signed [FP_WIDTH-1:0] Btmp [0:1];

    initial begin
        $readmemh("dense_w0.hex", W0);
        $readmemh("dense_w1.hex", W1);
        $readmemh("dense_b.hex",  Btmp);
        B0 = Btmp[0];
        B1 = Btmp[1];
    end

    // =========================================================================
    // PIPELINE STAGE 1 — MULTIPLY
    // The * operator on a signed * unsigned expression is inferred as a DSP
    // block on Cyclone IV. Each multiply completes in one clock cycle inside
    // the DSP hard block, replacing the 16-iteration LUT shift-add chain.
    // multstyle attribute hints Quartus to prefer DSP over logic cells.
    // =========================================================================
    (* multstyle = "dsp" *) reg signed [PROD_WIDTH-1:0] prod0 [0:CHANNELS-1];
    (* multstyle = "dsp" *) reg signed [PROD_WIDTH-1:0] prod1 [0:CHANNELS-1];
    reg pipe1_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe1_valid <= 0;
            for (i = 0; i < CHANNELS; i = i + 1) begin
                prod0[i] <= 0;
                prod1[i] <= 0;
            end
        end else begin
            pipe1_valid <= end_frame_d;
            if (end_frame_d) begin
                for (i = 0; i < CHANNELS; i = i + 1) begin
                    // Zero-extend acc to signed, then multiply signed weight.
                    // Quartus maps this directly to a DSP18 block.
                    prod0[i] <= $signed({1'b0, acc[i]}) * $signed(W0[i]);
                    prod1[i] <= $signed({1'b0, acc[i]}) * $signed(W1[i]);
                end
            end
        end
    end

    // =========================================================================
    // PIPELINE STAGE 2 — ACCUMULATE
    // Products are already registered. This stage is just 16 additions of
    // known-width values — a short adder tree, well within 20 ns.
    // =========================================================================

    // Combinational sum of registered products
    reg signed [SUM_WIDTH-1:0] sum0_comb;
    reg signed [SUM_WIDTH-1:0] sum1_comb;

    integer j;

    always @(*) begin
        sum0_comb = {{(SUM_WIDTH-FP_WIDTH){B0[FP_WIDTH-1]}}, B0};
        sum1_comb = {{(SUM_WIDTH-FP_WIDTH){B1[FP_WIDTH-1]}}, B1};
        for (j = 0; j < CHANNELS; j = j + 1) begin
            sum0_comb = sum0_comb +
                {{(SUM_WIDTH-PROD_WIDTH){prod0[j][PROD_WIDTH-1]}}, prod0[j]};
            sum1_comb = sum1_comb +
                {{(SUM_WIDTH-PROD_WIDTH){prod1[j][PROD_WIDTH-1]}}, prod1[j]};
        end
    end

    // Register stage 2 result
    reg signed [SUM_WIDTH-1:0] score0, score1;
    reg pipe2_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe2_valid <= 0;
            score0      <= 0;
            score1      <= 0;
        end else begin
            pipe2_valid <= pipe1_valid;
            if (pipe1_valid) begin
                score0 <= sum0_comb;
                score1 <= sum1_comb;
            end
        end
    end

    // =========================================================================
    // OUTPUT — now driven by pipe2_valid instead of end_frame_d
    // Latency vs original: +2 clock cycles. Irrelevant to system correctness
    // because fsm_controller waits on classifier_valid (= valid_out).
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            class_out <= 0;
        end else begin
            valid_out <= pipe2_valid;
            if (pipe2_valid)
                class_out <= (score1 > score0);
        end
    end

endmodule