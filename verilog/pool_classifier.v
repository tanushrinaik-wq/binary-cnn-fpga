// ============================================================================
// Module : pool_classifier.v (FINAL - VERILOG2001 + CORRECT MATH)
// ============================================================================

`timescale 1ns / 1ps

module pool_classifier #(
    parameter CHANNELS  = 16,
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
    output reg  class_out
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
            else if (valid_in && !end_frame) begin
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
    // WEIGHTS (FLATTENED - VERILOG 2001 SAFE)
    // =========================================================================
    reg signed [FP_WIDTH-1:0] W0 [0:CHANNELS-1];
    reg signed [FP_WIDTH-1:0] W1 [0:CHANNELS-1];
    reg signed [FP_WIDTH-1:0] B0, B1;
    reg signed [FP_WIDTH-1:0] Btmp [0:1];

    initial begin
        $readmemh("dense_w0.hex", W0);
        $readmemh("dense_w1.hex", W1);

        // Load biases via temp array
       
        $readmemh("dense_b.hex", Btmp);
        B0 = Btmp[0];
        B1 = Btmp[1];
    end

    // =========================================================================
    // CORRECT TWO'S COMPLEMENT SHIFT-ADD MULTIPLIER
    // =========================================================================
    function signed [31:0] mult_shift_add;
        input [ACC_WIDTH-1:0] a;
        input signed [FP_WIDTH-1:0] w;

        integer k;
        reg signed [31:0] result;
        reg [31:0] a_ext;

        begin
            result = 0;

            // zero-extend 'a'
            a_ext = {{(32-ACC_WIDTH){1'b0}}, a};

            // positive bits [0 .. FP_WIDTH-2]
            for (k = 0; k <= FP_WIDTH-2; k = k + 1) begin
                if (w[k])
                    result = result + (a_ext << k);
            end

            // MSB = negative weight
            if (w[FP_WIDTH-1])
                result = result - (a_ext << (FP_WIDTH-1));

            mult_shift_add = result;
        end
    endfunction

    // =========================================================================
    // COMBINATIONAL DENSE
    // =========================================================================
    reg signed [31:0] comb_score0;
    reg signed [31:0] comb_score1;

    integer j;

    always @(*) begin
        // sign-extend biases
        comb_score0 = {{(32-FP_WIDTH){B0[FP_WIDTH-1]}}, B0};
        comb_score1 = {{(32-FP_WIDTH){B1[FP_WIDTH-1]}}, B1};

        for (j = 0; j < CHANNELS; j = j + 1) begin
            comb_score0 = comb_score0 + mult_shift_add(acc[j], W0[j]);
            comb_score1 = comb_score1 + mult_shift_add(acc[j], W1[j]);
        end
    end

    // =========================================================================
    // OUTPUT
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