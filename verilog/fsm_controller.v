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

    // From SPI
    input  wire frame_active,
    input  wire frame_done,

    // From L2 (IMPORTANT CHANGE)
    input  wire l2_valid,

    // From classifier
    input  wire classifier_valid,

    // Outputs
    output reg start_frame,
    output reg end_frame
);

    // =========================================================================
    // 1. Parameters
    // =========================================================================
    localparam TOTAL_WINDOWS = (IMG_W - 4) * (IMG_H - 4);

    // =========================================================================
    // 2. State machine
    // =========================================================================
    typedef enum reg [1:0] {
        IDLE,
        RUNNING,
        WAIT_CLASS
    } state_t;

    state_t state, next_state;

    // =========================================================================
    // 3. Window counter (counts L2 outputs)
    // =========================================================================
    reg [$clog2(TOTAL_WINDOWS+1)-1:0] window_count;

    // =========================================================================
    // 4. State register
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // =========================================================================
    // 5. Next-state logic
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
    // 6. Control + counter logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_count <= 0;
            start_frame  <= 0;
            end_frame    <= 0;
        end else begin
            // Default outputs
            start_frame <= 0;
            end_frame   <= 0;

            case (state)

                // -------------------------------------------------------------
                IDLE
                // -------------------------------------------------------------
                IDLE: begin
                    window_count <= 0;

                    if (frame_active)
                        start_frame <= 1;
                end

                // -------------------------------------------------------------
                RUNNING
                // -------------------------------------------------------------
                RUNNING: begin
                    if (l2_valid) begin
                        window_count <= window_count + 1;

                        // Assert end_frame on LAST valid window
                        if (window_count == TOTAL_WINDOWS - 1)
                            end_frame <= 1;
                    end
                end

                // -------------------------------------------------------------
                WAIT_CLASS
                // -------------------------------------------------------------
                WAIT_CLASS: begin
                    // wait for classifier_valid
                end

            endcase
        end
    end

endmodule