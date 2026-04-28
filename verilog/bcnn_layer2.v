// ============================================================================
// Module : bcnn_layer2.v (SPEC-COMPLIANT)
// Description:
//   Multi-channel BCNN layer (Layer 2)
//   XNOR + popcount per channel + accumulation + threshold
// ============================================================================

`timescale 1ns / 1ps

module bcnn_layer2 #(
    parameter K = 3,
    parameter IN_CH = 8,
    parameter OUT_CH = 16
)(
    input  wire clk,
    input  wire rst_n,

    // Input: flattened window per channel
    input  wire valid_in,
    input  wire [IN_CH*K*K-1:0] window_in,

    // Output
    output reg  valid_out,
    output reg  [OUT_CH-1:0] out_bits
);

    localparam N = K*K;
    localparam PC_WIDTH = $clog2(N+1);
    localparam SUM_WIDTH = $clog2(IN_CH * N + 1);

    // =========================================================================
    // 1. Weight storage
    // =========================================================================
    (* ramstyle = "M4K" *)
    reg [IN_CH*K*K-1:0] weights [0:OUT_CH-1];

    (* ramstyle = "M4K" *)
    reg [SUM_WIDTH-1:0] thresholds [0:OUT_CH-1];

    // =========================================================================
    // 2. Popcount per channel
    // =========================================================================
    wire [PC_WIDTH-1:0] pc [0:OUT_CH-1][0:IN_CH-1];

    genvar f, c;

    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : FILTERS
            for (c = 0; c < IN_CH; c = c + 1) begin : CHANNELS

                popcount #(.N(N)) pc_inst (
                    .a(window_in[c*N +: N]),
                    .b(weights[f][c*N +: N]),
                    .count(pc[f][c])
                );

            end
        end
    endgenerate

    // =========================================================================
    // 3. Accumulation across channels
    // =========================================================================
    reg [SUM_WIDTH-1:0] sum [0:OUT_CH-1];

    integer i, j;

    always @(*) begin
        for (i = 0; i < OUT_CH; i = i + 1) begin
            sum[i] = 0;
            for (j = 0; j < IN_CH; j = j + 1) begin
                sum[i] = sum[i] + pc[i][j];
            end
        end
    end

    // =========================================================================
    // 4. Threshold + register
    // =========================================================================
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_bits  <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_in;

            if (valid_in) begin
                for (k = 0; k < OUT_CH; k = k + 1) begin
                    out_bits[k] <= (sum[k] >= thresholds[k]);
                end
            end
        end
    end

    // =========================================================================
    // 5. Memory init
    // =========================================================================
    initial begin
        $readmemh("weights_l2.mif", weights);
        $readmemh("thresholds_l2.mif", thresholds);
    end

endmodule