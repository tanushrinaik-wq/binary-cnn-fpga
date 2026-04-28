// ============================================================================
// Module  : spi_slave.v
// Project : BCNN Edge Vision Accelerator
// Target  : Intel Cyclone IV EP4CE115F29C7  (Quartus II 13.0)
//
// Description
// -----------
//   SPI Mode-0 (CPOL=0, CPHA=0) slave receiver.
//   Accepts a stream of 8-bit pixel bytes from the Arduino Nicla Vision at
//   up to 20 MHz SPI clock while the FPGA runs on a 50 MHz system clock.
//
//   Clock-domain crossing strategy
//   --------------------------------
//   SCK, MOSI and CS_N arrive in an asynchronous clock domain.  They are
//   brought into the sys_clk domain through a 2-stage synchroniser before
//   any logic acts on them.  SCK edges are detected by comparing the
//   current and previous synchronised values (one-cycle delay).
//
//   Output interface
//   ----------------
//   When a complete 8-bit word has been shifted in, byte_data[7:0] is
//   presented and byte_valid is asserted for exactly ONE sys_clk cycle.
//   The downstream async FIFO (spi_fifo.v) latches on this pulse.
//
//   Frame protocol
//   --------------
//   CS_N low  = frame active.  The master drives CS_N low, then clocks
//   exactly (IMG_H * IMG_W) = 1024 bytes.  CS_N high resets the
//   shift register and byte counter so a new frame can begin cleanly.
//
// Parameters
// ----------
//   DATA_BITS   : bits per SPI word (8 — matches hex file format)
//
// Ports
// -----
//   sys_clk     : FPGA system clock (50 MHz)
//   sys_rst_n   : active-low synchronous reset
//   spi_sck     : SPI clock from master  (async, up to 20 MHz)
//   spi_mosi    : SPI MOSI from master   (async)
//   spi_cs_n    : SPI chip-select, active-low (async)
//   spi_miso    : SPI MISO to master (tied low — receive-only slave)
//   byte_data   : 8-bit received byte
//   byte_valid  : pulses HIGH for 1 sys_clk when byte_data is valid
//   frame_active: HIGH while CS_N is asserted (useful for FSM)
//   frame_done  : pulses HIGH for 1 sys_clk on CS_N rising edge
//
// ============================================================================

`timescale 1ns / 1ps

module spi_slave #(
    parameter DATA_BITS = 8         // SPI word width — do not change without
                                    // updating downstream FIFO width
) (
    // System domain
    input  wire                  sys_clk,
    input  wire                  sys_rst_n,

    // SPI pins (async input domain)
    input  wire                  spi_sck,
    input  wire                  spi_mosi,
    input  wire                  spi_cs_n,
    output wire                  spi_miso,   // receive-only; drive low

    // Output to async FIFO (sys_clk domain)
    output reg  [DATA_BITS-1:0]  byte_data,
    output reg                   byte_valid,

    // Frame status (sys_clk domain)
    output wire                  frame_active,
    output reg                   frame_done
);

    // =========================================================================
    // 1.  2-Stage Synchronisers  (metastability hardening)
    // =========================================================================
    // Each external signal gets two back-to-back flip-flops clocked by
    // sys_clk.  This keeps the mean-time-between-failure (MTBF) well above
    // the operational lifetime of the device.

    // -- SCK synchroniser --
    reg [1:0] sck_sync;
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            sck_sync <= 2'b00;
        else
            sck_sync <= {sck_sync[0], spi_sck};
    end
    wire sck_s  = sck_sync[1];         // synchronised SCK (current)
    wire sck_s0 = sck_sync[0];         // one pipeline stage earlier (for edge detect)

    // -- MOSI synchroniser --
    reg [1:0] mosi_sync;
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            mosi_sync <= 2'b00;
        else
            mosi_sync <= {mosi_sync[0], spi_mosi};
    end
    wire mosi_s = mosi_sync[1];        // synchronised MOSI

    // -- CS_N synchroniser --
    reg [1:0] cs_sync;
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            cs_sync <= 2'b11;          // default: de-asserted
        else
            cs_sync <= {cs_sync[0], spi_cs_n};
    end
    wire cs_n_s = cs_sync[1];          // synchronised CS_N (current)
    wire cs_n_s0 = cs_sync[0];         // one stage earlier

    // =========================================================================
    // 2.  Edge Detection
    // =========================================================================
    // SPI Mode-0: sample MOSI on the RISING edge of SCK.
    // We detect this as:  previous SCK = 0, current SCK = 1.
    //
    // CS_N rising edge marks end-of-frame.

    wire sck_rising  = (~sck_s0) &  sck_s;   // 0→1 transition detected
    // (sck_falling not needed for Mode-0 receive-only)

    wire cs_rising   = (~cs_n_s0) &  cs_n_s; // CS_N 0→1 = frame end
    wire cs_falling  =  cs_n_s0  & (~cs_n_s);// CS_N 1→0 = frame start (optional)

    // =========================================================================
    // 3.  Shift Register & Bit Counter
    // =========================================================================
    reg [DATA_BITS-1:0] shift_reg;
    reg [$clog2(DATA_BITS):0] bit_cnt;      // counts 0..DATA_BITS (0..8)

    // --  Serial-in, MSB first  --
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            shift_reg  <= {DATA_BITS{1'b0}};
            bit_cnt    <= {($clog2(DATA_BITS)+1){1'b0}};
            byte_data  <= {DATA_BITS{1'b0}};
            byte_valid <= 1'b0;
        end else begin
            // Default: clear strobe every cycle
            byte_valid <= 1'b0;

            // CS_N asserted (active-low) = frame is running
            if (!cs_n_s) begin
                if (sck_rising) begin
                    // Shift in MOSI MSB-first
                    shift_reg <= {shift_reg[DATA_BITS-2:0], mosi_s};
                    bit_cnt   <= bit_cnt + 1'b1;

                    // On the 8th rising edge the word is complete
                    if (bit_cnt == DATA_BITS - 1) begin
                        byte_data  <= {shift_reg[DATA_BITS-2:0], mosi_s};
                        byte_valid <= 1'b1;
                        bit_cnt    <= {($clog2(DATA_BITS)+1){1'b0}};
                    end
                end
            end else begin
                // CS_N de-asserted: reset for next frame
                shift_reg <= {DATA_BITS{1'b0}};
                bit_cnt   <= {($clog2(DATA_BITS)+1){1'b0}};
            end
        end
    end

    // =========================================================================
    // 4.  Frame Status Signals
    // =========================================================================
    assign frame_active = ~cs_n_s;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n)
            frame_done <= 1'b0;
        else
            frame_done <= cs_rising;    // single-cycle pulse on CS_N release
    end

    // =========================================================================
    // 5.  MISO
    // =========================================================================
    // Receive-only slave — drive MISO low to avoid bus contention.
    assign spi_miso = 1'b0;


// ============================================================================
// Simulation / Synthesis Guards
// ============================================================================
// synthesis translate_off
    // Parameter sanity checks (ModelSim will print these at elaboration)
    initial begin
        if (DATA_BITS != 8)
            $warning("spi_slave: DATA_BITS=%0d — downstream modules assume 8",
                     DATA_BITS);
    end
// synthesis translate_on

endmodule
// ============================================================================
// End of spi_slave.v
// ============================================================================