// ============================================================================
// Module  : spi_fifo.v
// Project : BCNN Edge Vision Accelerator
// Target  : Intel Cyclone IV EP4CE115F29C7  (Quartus II 13.0)
//
// Description
// -----------
//   Synchronous FIFO that buffers pixel bytes between the SPI slave output
//   and the BCNN compute pipeline.
//
//   Clock-domain note
//   -----------------
//   spi_slave.v already synchronises SCK, MOSI and CS_N into sys_clk and
//   only asserts byte_valid as a sys_clk-domain pulse.  Both write and read
//   ports of this FIFO therefore run on the same clock (sys_clk).  A
//   Gray-code async FIFO would be required only if write_en were driven
//   directly by the raw SPI clock — that crossing is handled upstream.
//
//   Capacity
//   --------
//   DEPTH = 64 entries (2 full image rows of 32 bytes).  This absorbs any
//   cycle-accurate stall in the compute pipeline without losing bytes.
//   Depth must remain a power-of-2 for the pointer wrap logic.
//
//   Overflow / Underflow protection
//   --------------------------------
//   write_en while full  : byte is silently dropped; overflow flag asserted.
//   read_en  while empty : read_data holds last valid value; underflow flag.
//   Both flags are sticky — cleared only by reset.  The testbench monitors
//   them to catch timing bugs during simulation.
//
// Parameters
// ----------
//   DATA_WIDTH : word width in bits (8 — one pixel byte)
//   DEPTH      : number of entries  (must be power of 2)
//
// Ports
// -----
//   sys_clk    : system clock (50 MHz)
//   sys_rst_n  : active-low synchronous reset
//   write_en   : write strobe — connect to spi_slave byte_valid
//   write_data : byte to enqueue — connect to spi_slave byte_data
//   read_en    : read strobe — asserted by FSM/binarizer when consuming a byte
//   read_data  : dequeued byte (registered output — 1-cycle latency)
//   full       : FIFO cannot accept more data
//   empty      : FIFO has no data to read
//   overflow   : sticky flag — write attempted while full
//   underflow  : sticky flag — read attempted while empty
//   count      : number of entries currently stored
//
// ============================================================================

`timescale 1ns / 1ps

module spi_fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 64           // must be a power of 2
) (
    input  wire                   sys_clk,
    input  wire                   sys_rst_n,

    // Write port  (driven by spi_slave)
    input  wire                   write_en,
    input  wire [DATA_WIDTH-1:0]  write_data,

    // Read port  (driven by FSM / binarizer)
    input  wire                   read_en,
    output reg  [DATA_WIDTH-1:0]  read_data,

    // Status
    output wire                   full,
    output wire                   empty,
    output reg                    overflow,
    output reg                    underflow,
    output wire [$clog2(DEPTH):0] count       // 0..DEPTH
);

    // =========================================================================
    // 1.  Storage Array
    // =========================================================================
    // Quartus infers M4K Block RAM for arrays this size.
    // 64 × 8 = 512 bits — well within a single M4K block (4096 bits).

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // =========================================================================
    // 2.  Read / Write Pointers
    // =========================================================================
    // Extra MSB is the "wrap" bit.  When both pointers share the same lower
    // bits but differ in the MSB the FIFO is full; when they are identical
    // (including MSB) it is empty.  This avoids a separate counter for
    // full/empty decode while keeping count straightforward.

    localparam PTR_W = $clog2(DEPTH) + 1;  // e.g. 7 bits for DEPTH=64

    reg [PTR_W-1:0] wr_ptr;   // write pointer (binary)
    reg [PTR_W-1:0] rd_ptr;   // read  pointer (binary)

    // Lower bits are the memory index; MSB is the generation/wrap bit.
    wire [$clog2(DEPTH)-1:0] wr_addr = wr_ptr[$clog2(DEPTH)-1:0];
    wire [$clog2(DEPTH)-1:0] rd_addr = rd_ptr[$clog2(DEPTH)-1:0];

    // =========================================================================
    // 3.  Full / Empty / Count
    // =========================================================================
    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[$clog2(DEPTH)]   != rd_ptr[$clog2(DEPTH)]) &&
                   (wr_ptr[$clog2(DEPTH)-1:0] == rd_ptr[$clog2(DEPTH)-1:0]);

    // Count = difference of the two binary pointers (mod 2*DEPTH arithmetic
    // is handled naturally by the extra wrap bit).
    assign count = wr_ptr - rd_ptr;   // unsigned subtraction, PTR_W bits wide

    // =========================================================================
    // 4.  Write Logic
    // =========================================================================
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            wr_ptr   <= {PTR_W{1'b0}};
            overflow <= 1'b0;
        end else begin
            if (write_en) begin
                if (!full) begin
                    mem[wr_addr] <= write_data;
                    wr_ptr       <= wr_ptr + 1'b1;
                end else begin
                    overflow <= 1'b1;   // sticky — cleared only by reset
                end
            end
        end
    end

    // =========================================================================
    // 5.  Read Logic
    // =========================================================================
    // Registered (synchronous) read output — 1-cycle read latency.
    // Quartus uses this style to infer true Block RAM read ports, which
    // maximises Fmax compared to asynchronous (combinatorial) read.

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            rd_ptr    <= {PTR_W{1'b0}};
            read_data <= {DATA_WIDTH{1'b0}};
            underflow <= 1'b0;
        end else begin
            if (read_en) begin
                if (!empty) begin
                    read_data <= mem[rd_addr];
                    rd_ptr    <= rd_ptr + 1'b1;
                end else begin
                    underflow <= 1'b1;  // sticky — cleared only by reset
                end
            end
        end
    end

    // =========================================================================
    // 6.  Simulation Assertions
    // =========================================================================
// synthesis translate_off
    // Depth power-of-2 check
    initial begin
        if ((DEPTH & (DEPTH - 1)) != 0) begin
            $display("ERROR: spi_fifo DEPTH=%0d is not a power of 2.", DEPTH);
            $finish;
        end
    end

    // Runtime overflow/underflow monitors
    always @(posedge sys_clk) begin
        if (write_en && full)
            $warning("spi_fifo: OVERFLOW  at time %0t — byte dropped.", $time);
        if (read_en && empty)
            $warning("spi_fifo: UNDERFLOW at time %0t — stale data read.", $time);
    end
// synthesis translate_on

endmodule
// ============================================================================
// End of spi_fifo.v
// ============================================================================