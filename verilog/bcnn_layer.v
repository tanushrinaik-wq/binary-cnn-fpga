// ============================================================================
// Module : bcnn_layer.v (FIXED - SPEC COMPLIANT)
// ============================================================================

`timescale 1ns / 1ps

module bcnn_layer #(
    parameter K = 3,
    parameter IN_CH = 1,
    parameter OUT_CH = 8
)(
    input  wire clk,
    input  wire rst_n,

    input  wire valid_in,
    input  wire [K*K*IN_CH-1:0] window_in,

    output reg  valid_out,
    output reg  [OUT_CH-1:0] out_bits
);

    localparam N = K*K*IN_CH;
    localparam COUNT_W = $clog2(N+1);

    // =========================================================================
    // WEIGHTS + THRESHOLDS + FLIP FLAGS
    // =========================================================================
    (* ramstyle = "M4K" *) reg [N-1:0] weights [0:OUT_CH-1];
    (* ramstyle = "M4K" *) reg [COUNT_W-1:0] thresholds [0:OUT_CH-1];
    reg flip [0:OUT_CH-1];

    // =========================================================================
    // POPCOUNT
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
    // OUTPUT LOGIC (WITH FLIP)
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
                    out_bits[j] <= flip[j] ?
                        (pc[j] < thresholds[j]) :
                        (pc[j] >= thresholds[j]);
                end
            end
        end
    end

    // =========================================================================
    // MEMORY INIT (FIXED FILE NAMES + HEX FORMAT)
    // =========================================================================
    initial begin
        $readmemh("conv1_weights.hex", weights);
        $readmemh("conv1_thresh.hex",  thresholds);
        $readmemb("conv1_flip.mif",    flip);   // binary file OK
    end

endmodule