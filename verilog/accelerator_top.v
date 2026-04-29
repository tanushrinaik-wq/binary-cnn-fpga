// ============================================================================
// Module : accelerator_top.v (FIXED - SPEC COMPLIANT)
// Description:
//   Pure datapath integration for BCNN accelerator
//   FSM is external (fsm_controller.v)
// ============================================================================

`timescale 1ns / 1ps

module accelerator_top #(
    parameter IMG_W = 32,
    parameter IMG_H = 32,
    parameter CHANNELS = 8
)(
    input  wire clk,
    input  wire rst_n,

    // SPI
    input  wire spi_sck,
    input  wire spi_mosi,
    input  wire spi_cs_n,
    output wire spi_miso,

    // Control from FSM
    input  wire start_frame,
    input  wire end_frame,

    // Status to FSM
    output wire frame_active,
    output wire frame_done,
    output wire bcnn_valid,

    // Final output
    output wire valid_out,
    output wire [$clog2(CHANNELS)-1:0] class_out
);

    // =========================================================================
    // LOCALPARAMS (FIX: missing declarations)
    // =========================================================================
    localparam CH1 = 8;
    localparam CH2 = 16;

    // =========================================================================
    // 1. SPI SLAVE
    // =========================================================================
    wire [7:0] spi_data;
    wire spi_valid;

    spi_slave spi_inst (
        .sys_clk(clk),
        .sys_rst_n(rst_n),
        .spi_sck(spi_sck),
        .spi_mosi(spi_mosi),
        .spi_cs_n(spi_cs_n),
        .spi_miso(spi_miso),
        .byte_data(spi_data),
        .byte_valid(spi_valid),
        .frame_active(frame_active),
        .frame_done(frame_done)
    );

    // =========================================================================
    // 2. FIFO (streaming)
    // =========================================================================
    wire [7:0] fifo_out;
    wire fifo_empty;

    wire fifo_rd = !fifo_empty;

    spi_fifo fifo_inst (
        .sys_clk(clk),
        .sys_rst_n(rst_n),
        .write_en(spi_valid),
        .write_data(spi_data),
        .read_en(fifo_rd),
        .read_data(fifo_out),
        .full(),
        .empty(fifo_empty),
        .overflow(),
        .underflow(),
        .count()
    );

    // =========================================================================
    // 2.1 FIFO READ ALIGNMENT (FIX: 1-cycle latency compensation)
    // =========================================================================
    reg fifo_rd_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fifo_rd_d <= 1'b0;
        else
            fifo_rd_d <= fifo_rd;
    end

    // =========================================================================
    // 3. BINARIZER
    // =========================================================================
    wire pixel_bin = (fifo_out > 8'd127);

    // =========================================================================
    // 4. LINE BUFFER
    // =========================================================================
    wire lb_valid;
    wire [8:0] window;

    line_buffer #(.IMG_WIDTH(IMG_W)) lb_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fifo_rd_d),   // FIXED alignment
        .pixel_in(pixel_bin),
        .valid_out(lb_valid),
        .window_out(window)
    );

    // ======================= L1 =======================
    wire l1_valid;
    wire [CH1-1:0] l1_out;

    bcnn_layer #(
        .K(3),
        .IN_CH(1),
        .OUT_CH(CH1)
    ) l1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(lb_valid),
        .window_in(window),
        .valid_out(l1_valid),
        .out_bits(l1_out)
    );

    // ======================= LB2 =======================
    wire lb2_valid;
    wire [CH1*9-1:0] window_l2;

    line_buffer_multi #(
        .IMG_WIDTH(IMG_W-2),
        .CHANNELS(CH1)
    ) lb2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(l1_valid),
        .pixel_in(l1_out),
        .valid_out(lb2_valid),
        .window_out(window_l2)
    );

    // ======================= L2 =======================
    wire l2_valid;
    wire [CH2-1:0] l2_out;

    bcnn_layer2 #(
        .K(3),
        .IN_CH(CH1),
        .OUT_CH(CH2)
    ) l2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(lb2_valid),
        .window_in(window_l2),
        .valid_out(l2_valid),
        .out_bits(l2_out)
    );

    // =========================================================================
    // STATUS OUTPUT (FIX: previously undriven)
    // =========================================================================
    assign bcnn_valid = l2_valid;

    // ======================= POOL =======================
    pool_classifier #(
        .CHANNELS(CH2),
        .IMG_WIDTH(IMG_W-4),
        .IMG_HEIGHT(IMG_H-4)
    ) pc (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_frame),
        .end_frame(end_frame),
        .valid_in(l2_valid),
        .data_in(l2_out),
        .valid_out(valid_out),
        .class_out(class_out)
    );

endmodule