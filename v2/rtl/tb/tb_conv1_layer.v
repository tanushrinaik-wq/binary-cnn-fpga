`timescale 1ns / 1ps

module tb_conv1_layer;
    reg clk;
    reg rst;
    reg [7:0] pixel_in;
    reg pixel_valid;
    wire valid_out;
    wire [31:0] data_out;
    integer i;

    conv1_layer dut (
        .clk(clk),
        .rst(rst),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .valid_out(valid_out),
        .data_out(data_out)
    );

    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    initial begin
        rst = 1;
        pixel_in = 0;
        pixel_valid = 0;
        #50;
        rst = 0;
        pixel_valid = 1;
        for (i = 0; i < 64*64*3; i = i + 1) begin
            pixel_in = i[7:0];
            @(posedge clk);
        end
        pixel_valid = 0;
        wait(valid_out);
        $display("conv1 first output = %h", data_out);
        #20;
        $finish;
    end
endmodule

