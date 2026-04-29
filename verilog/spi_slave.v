// ============================================================================
// Module  : spi_slave.v (FINAL - CDC SAFE, HARDWARE CORRECT)
// ============================================================================

`timescale 1ns / 1ps

module spi_slave #(
    parameter DATA_BITS = 8
)(
    input  wire                  sys_clk,
    input  wire                  sys_rst_n,

    input  wire                  spi_sck,
    input  wire                  spi_mosi,
    input  wire                  spi_cs_n,
    output wire                  spi_miso,

    output reg  [DATA_BITS-1:0]  byte_data,
    output reg                   byte_valid,

    output wire                  frame_active,
    output reg                   frame_done
);

    // =========================================================================
    // 2-FF SYNCHRONIZERS
    // =========================================================================
    reg [1:0] sck_sync;
    reg [1:0] mosi_sync;
    reg [1:0] cs_sync;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            sck_sync  <= 2'b00;
            mosi_sync <= 2'b00;
            cs_sync   <= 2'b11;
        end else begin
            sck_sync  <= {sck_sync[0], spi_sck};
            mosi_sync <= {mosi_sync[0], spi_mosi};
            cs_sync   <= {cs_sync[0],  spi_cs_n};
        end
    end

    // =========================================================================
    // STABLE SIGNALS (ONLY USE [1])
    // =========================================================================
    wire sck_s  = sck_sync[1];
    wire cs_s   = cs_sync[1];
    wire mosi_s = mosi_sync[1];

    assign frame_active = ~cs_s;

    // =========================================================================
    // EDGE DETECTION (SAFE)
    // =========================================================================
    reg sck_prev;
    reg cs_prev;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            sck_prev <= 1'b0;
            cs_prev  <= 1'b1;
        end else begin
            sck_prev <= sck_s;
            cs_prev  <= cs_s;
        end
    end

    wire sck_rising = sck_s & ~sck_prev;
    wire cs_rising  = cs_s  & ~cs_prev;
    wire cs_falling = ~cs_s &  cs_prev;

    // =========================================================================
    // SHIFT REGISTER
    // =========================================================================
    reg [DATA_BITS-1:0] shift_reg;
    reg [$clog2(DATA_BITS):0] bit_cnt;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            shift_reg  <= 0;
            bit_cnt    <= 0;
            byte_data  <= 0;
            byte_valid <= 0;
            frame_done <= 0;
        end else begin
            byte_valid <= 1'b0;
            frame_done <= 1'b0;

            // Frame start
            if (cs_falling) begin
                shift_reg <= 0;
                bit_cnt   <= 0;
            end

            // Shift only when CS active (stable signal)
            if (!cs_s && sck_rising) begin
                shift_reg <= {shift_reg[DATA_BITS-2:0], mosi_s};
                bit_cnt   <= bit_cnt + 1'b1;

                if (bit_cnt == DATA_BITS - 1) begin
                    byte_data  <= {shift_reg[DATA_BITS-2:0], mosi_s};
                    byte_valid <= 1'b1;
                    bit_cnt    <= 0;
                end
            end

            // Frame end pulse
            if (cs_rising) begin
                frame_done <= 1'b1;
            end
        end
    end

    // =========================================================================
    // MISO (unused)
    // =========================================================================
    assign spi_miso = 1'b0;

endmodule