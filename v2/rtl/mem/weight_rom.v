`timescale 1ns / 1ps

module weight_rom #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 1024,
    parameter ADDR_WIDTH = 10,
    parameter INIT_FILE_HEX = "",
    parameter INIT_FILE_BIN = ""
)(
    input  wire clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg  [DATA_WIDTH-1:0] q
);
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

`ifdef SIMULATION
    initial begin
        if (INIT_FILE_HEX != "")
            $readmemh(INIT_FILE_HEX, mem);
    end
`endif

    always @(posedge clk)
        q <= mem[addr];
endmodule

