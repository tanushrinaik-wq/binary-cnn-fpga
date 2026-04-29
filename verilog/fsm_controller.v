// ============================================================================
// Module : fsm_controller.v (FINAL - DEADLOCK FREE, SPEC CORRECT)
// ============================================================================

`timescale 1ns / 1ps

module fsm_controller #(
    parameter IMG_W = 32,
    parameter IMG_H = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Frame control
    input  wire frame_active,
    input  wire frame_done,        // (kept for interface completeness)

    // Pipeline
    input  wire l2_valid,

    // Classifier
    input  wire classifier_valid,

    // Outputs
    output reg  start_frame,
    output reg  end_frame
);

    // =========================================================================
    // STATES
    // =========================================================================
    localparam IDLE       = 2'd0;
    localparam RUNNING    = 2'd1;
    localparam WAIT_CLASS = 2'd2;

    reg [1:0] state, next_state;

    // =========================================================================
    // WINDOWS
    // =========================================================================
    localparam TOTAL_WINDOWS = (IMG_W - 4) * (IMG_H - 4);

    reg [$clog2(TOTAL_WINDOWS+1)-1:0] window_count;

    // =========================================================================
    // STATE REGISTER
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // =========================================================================
    // NEXT STATE (FIXED)
    // =========================================================================
    always @(*) begin
        case (state)

            IDLE:
                next_state = frame_active ? RUNNING : IDLE;

            RUNNING:
                next_state = (window_count == TOTAL_WINDOWS) ? WAIT_CLASS : RUNNING;

            WAIT_CLASS:
                next_state = classifier_valid ? IDLE : WAIT_CLASS;

            default:
                next_state = IDLE;
        endcase
    end

    // =========================================================================
    // OUTPUT + COUNTER
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_count <= 0;
            start_frame  <= 0;
            end_frame    <= 0;
        end else begin
            start_frame <= 0;
            end_frame   <= 0;

            case (state)

                IDLE: begin
                    window_count <= 0;

                    if (frame_active)
                        start_frame <= 1'b1;
                end

                RUNNING: begin
                    if (l2_valid) begin
                        window_count <= window_count + 1'b1;

                        if (window_count == TOTAL_WINDOWS - 1)
                            end_frame <= 1'b1;
                    end
                end

                WAIT_CLASS: begin
                    // hold state until classifier_valid
                end

            endcase
        end
    end

endmodule