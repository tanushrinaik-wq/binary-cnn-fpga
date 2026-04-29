// ============================================================================
// Module : fsm_controller.v (FINAL - L2 AWARE)
// Description:
//   Controls frame lifecycle for 2-layer BCNN pipeline
//   Counts L2 valid outputs
// ============================================================================

`timescale 1ns / 1ps

module fsm_controller #(
    parameter IMG_W = 32,
    parameter IMG_H = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Frame control (from SPI)
    input  wire frame_active,

    // Pipeline signal (L2 valid outputs)
    input  wire l2_valid,

    // Classifier done signal
    input  wire classifier_valid,

    // Outputs
    output reg  start_frame,   // 1-cycle pulse
    output reg  end_frame      // 1-cycle pulse
);

    // =========================================================================
    // FSM States
    // =========================================================================
    localparam IDLE       = 2'd0;
    localparam RUNNING    = 2'd1;
    localparam WAIT_CLASS = 2'd2;

    reg [1:0] state, next_state;

    // =========================================================================
    // Total valid convolution windows
    // 2 layers of 3x3 valid conv:
    // 32 → 30 → 28 ⇒ 28x28 = 784
    // =========================================================================
    localparam TOTAL_WINDOWS = (IMG_W - 4) * (IMG_H - 4);

    reg [$clog2(TOTAL_WINDOWS+1)-1:0] window_count;

    // =========================================================================
    // State register
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always @(*) begin
        case (state)
            IDLE:
                next_state = frame_active ? RUNNING : IDLE;

            RUNNING:
                // Transition AFTER last window is counted
                next_state = (window_count == TOTAL_WINDOWS) ? WAIT_CLASS : RUNNING;

            WAIT_CLASS:
                next_state = classifier_valid ? IDLE : WAIT_CLASS;

            default:
                next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Output + counter logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_count <= {($clog2(TOTAL_WINDOWS+1)){1'b0}};
            start_frame  <= 1'b0;
            end_frame    <= 1'b0;
        end else begin
            // Default outputs
            start_frame <= 1'b0;
            end_frame   <= 1'b0;

            case (state)

                IDLE: begin
                    window_count <= {($clog2(TOTAL_WINDOWS+1)){1'b0}};
                    if (frame_active)
                        start_frame <= 1'b1;
                end

                RUNNING: begin
                    if (l2_valid) begin
                        window_count <= window_count + 1'b1;

                        // Assert on last valid window
                        // NOTE: classifier must delay internally by 1 cycle
                        if (window_count == TOTAL_WINDOWS - 1)
                            end_frame <= 1'b1;
                    end
                end

                WAIT_CLASS: begin
                    // wait for classifier_valid
                end

            endcase
        end
    end

endmodule