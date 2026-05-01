`timescale 1ns / 1ps

module tb_xnor_popcount;
    reg [31:0] a;
    reg [31:0] b;
    wire [5:0] out;

    xnor_popcount #(.N(32), .COUNT_WIDTH(6)) dut (
        .a(a),
        .b(b),
        .out(out)
    );

    initial begin
        a = 32'hFFFF0000;
        b = 32'hFF00FF00;
        #10;
        if (out !== 6'd16) begin
            $display("FAIL: expected 16 got %0d", out);
            $finish;
        end
        a = 32'hAAAAAAAA;
        b = 32'hAAAAAAAA;
        #10;
        if (out !== 6'd32) begin
            $display("FAIL: expected 32 got %0d", out);
            $finish;
        end
        $display("PASS: xnor_popcount");
        $finish;
    end
endmodule

