// ============================================================================
// Module : bcnn_layer.v (SPEC-COMPLIANT)
// Description:
//   Streaming BCNN layer (XNOR + popcount + threshold)
// ============================================================================

`timescale 1ns / 1ps

module bcnn_layer #(
    parameter K = 3,
    parameter IN_CH = 1,
    parameter OUT_CH = 8
)(
    input  wire clk,
    input  wire rst_n,

    // Streaming input window (from line buffer)
    input  wire valid_in,
    input  wire [K*K*IN_CH-1:0] window_in,

    // Output
    output reg  valid_out,
    output reg  [OUT_CH-1:0] out_bits
);

    localparam N = K*K*IN_CH;
    localparam COUNT_W = $clog2(N+1);

    // =========================================================================
    // 1. Weight storage (ROM-friendly)
    // =========================================================================
    (* ramstyle = "M4K" *) reg [N-1:0] weights [0:OUT_CH-1];

    // =========================================================================
    // 2. Threshold storage
    // =========================================================================
    (* ramstyle = "M4K" *) reg [COUNT_W-1:0] thresholds [0:OUT_CH-1];

    // =========================================================================
    // 3. Popcount results
    // =========================================================================
    wire [COUNT_W-1:0] pc [0:OUT_CH-1];

    genvar i;
    generate
        for (i = 0; i < OUT_CH; i = i + 1) begin : FILTERS
            popcount #(.N(N)) pc_inst (
                .a(window_in),
                .b(weights[i]),
                .count(pc[i])
            );
        end
    endgenerate

    // =========================================================================
    // 4. Registered output (pipeline stage)
    // =========================================================================
    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_bits  <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_in;

            if (valid_in) begin
                for (j = 0; j < OUT_CH; j = j + 1) begin
                    out_bits[j] <= (pc[j] >= thresholds[j]);
                end
            end
        end
    end

    // =========================================================================
    // 5. Memory initialization
    // =========================================================================
    initial begin
        $readmemh("weights_layer.mif", weights);
        $readmemh("thresholds_layer.mif", thresholds);
    end

endmodule