// ============================================================================
// Module  : spi_slave.v (FIXED - EDGE + FRAME CORRECT)
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
    // SYNCHRONIZERS
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

    wire sck_old = sck_sync[1];
    wire sck_new = sck_sync[0];

    wire cs_old  = cs_sync[1];
    wire cs_new  = cs_sync[0];

    wire mosi_s  = mosi_sync[1];

    // =========================================================================
    // FIXED EDGE DETECTION
    // =========================================================================
    wire sck_rising = sck_new & (~sck_old);   // rising edge
    wire cs_rising  = cs_new  & (~cs_old);    // CS: 0→1 (frame end)
    wire cs_falling = (~cs_new) & cs_old;     // CS: 1→0 (frame start)

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
        end else begin
            byte_valid <= 1'b0;

            if (!cs_new) begin
                if (sck_rising) begin
                    shift_reg <= {shift_reg[DATA_BITS-2:0], mosi_s};
                    bit_cnt   <= bit_cnt + 1'b1;

                    if (bit_cnt == DATA_BITS - 1) begin
                        byte_data  <= {shift_reg[DATA_BITS-2:0], mosi_s};
                        byte_valid <= 1'b1;
                        bit_cnt    <= 0;
                    end
                end
            end else begin
                shift_reg <= 0;
                bit_cnt   <= 0;
            end
        end
    end

    // =========================================================================
    // FRAME SIGNALS (FIXED)
    // =========================================================================
    assign frame_active = ~cs_new;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            frame_done <= 0;
        else
            frame_done <= cs_rising;   // ✅ fires on CS_N rising (frame END)
    end

    // =========================================================================
    // MISO
    // =========================================================================
    assign spi_miso = 1'b0;

endmodule