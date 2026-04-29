// ============================================================================
// Module : bcnn_layer2.v (FINAL FIX - VERILOG-2001 SAFE)
// ============================================================================

`timescale 1ns / 1ps

module bcnn_layer2 #(
    parameter K = 3,
    parameter IN_CH = 8,
    parameter OUT_CH = 16
)(
    input  wire clk,
    input  wire rst_n,

    input  wire valid_in,
    input  wire [IN_CH*K*K-1:0] window_in,

    output reg  valid_out,
    output reg  [OUT_CH-1:0] out_bits
);

    localparam N = K*K;
    localparam PC_WIDTH  = $clog2(N+1);
    localparam SUM_WIDTH = $clog2(IN_CH * N + 1);

    // =========================================================================
    // WEIGHTS / THRESHOLDS / FLIP
    // =========================================================================
    (* ramstyle = "M4K" *)
    reg [IN_CH*K*K-1:0] weights [0:OUT_CH-1];

    (* ramstyle = "M4K" *)
    reg [SUM_WIDTH-1:0] thresholds [0:OUT_CH-1];

    reg flip [0:OUT_CH-1];

    // =========================================================================
    // POPCOUNT (FLATTENED ARRAY FIX)
    // =========================================================================
    wire [PC_WIDTH-1:0] pc [0:(OUT_CH*IN_CH)-1];

    genvar f, c;
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : FILTERS
            for (c = 0; c < IN_CH; c = c + 1) begin : CHANNELS
                popcount #(.N(N)) pc_inst (
                    .a(window_in[c*N +: N]),
                    .b(weights[f][c*N +: N]),
                    .count(pc[f*IN_CH + c])   // ✅ flattened indexing
                );
            end
        end
    endgenerate

    // =========================================================================
    // ACCUMULATION (UPDATED INDEXING)
    // =========================================================================
    reg [SUM_WIDTH-1:0] sum [0:OUT_CH-1];

    integer i, j;

    always @(*) begin
        for (i = 0; i < OUT_CH; i = i + 1) begin
            sum[i] = 0;
            for (j = 0; j < IN_CH; j = j + 1) begin
                sum[i] = sum[i] + pc[i*IN_CH + j];  // ✅ fixed
            end
        end
    end

    // =========================================================================
    // OUTPUT (WITH FLIP FLAGS)
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
                    out_bits[k] <= flip[k] ?
                        (sum[k] < thresholds[k]) :
                        (sum[k] >= thresholds[k]);
                end
            end
        end
    end

    // =========================================================================
    // MEMORY INIT
    // =========================================================================
    initial begin
        $readmemh("conv2_weights.hex", weights);
        $readmemh("conv2_thresh.hex",  thresholds);
        $readmemb("conv2_flip.mif",    flip);
    end

endmodule