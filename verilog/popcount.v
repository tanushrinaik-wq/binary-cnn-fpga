// ============================================================================
// Module : popcount.v (SPEC-COMPLIANT)
// Description:
//   XNOR + balanced adder tree popcount
//   Designed for high Fmax on FPGA
// ============================================================================

`timescale 1ns / 1ps

module popcount #(
    parameter N = 9  // must be power-friendly (e.g., 9, 16, 32)
)(
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    output wire [$clog2(N+1)-1:0] count
);

    // =========================================================================
    // 1. XNOR stage
    // =========================================================================
    wire [N-1:0] x;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : XNOR_STAGE
            assign x[i] = ~(a[i] ^ b[i]);
        end
    endgenerate

    // =========================================================================
    // 2. Balanced reduction tree
    // =========================================================================
    // Level 1: pairwise add
    localparam L1 = (N+1)/2;
    wire [1:0] level1 [0:L1-1];

    generate
        for (i = 0; i < L1; i = i + 1) begin : LEVEL1
            if (2*i+1 < N)
                assign level1[i] = x[2*i] + x[2*i+1];
            else
                assign level1[i] = x[2*i];
        end
    endgenerate

    // Level 2
    localparam L2 = (L1+1)/2;
    wire [2:0] level2 [0:L2-1];

    generate
        for (i = 0; i < L2; i = i + 1) begin : LEVEL2
            if (2*i+1 < L1)
                assign level2[i] = level1[2*i] + level1[2*i+1];
            else
                assign level2[i] = level1[2*i];
        end
    endgenerate

    // Level 3
    localparam L3 = (L2+1)/2;
    wire [3:0] level3 [0:L3-1];

    generate
        for (i = 0; i < L3; i = i + 1) begin : LEVEL3
            if (2*i+1 < L2)
                assign level3[i] = level2[2*i] + level2[2*i+1];
            else
                assign level3[i] = level2[2*i];
        end
    endgenerate

    // Final reduction (small enough now)
    reg [$clog2(N+1)-1:0] final_sum;
    integer j;

    always @(*) begin
        final_sum = 0;
        for (j = 0; j < L3; j = j + 1)
            final_sum = final_sum + level3[j];
    end

    assign count = final_sum;

endmodule