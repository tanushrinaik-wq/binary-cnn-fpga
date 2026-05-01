`timescale 1ns / 1ps

module tb_bcnn_top;
    reg clk;
    reg rst;
    reg [7:0] pixel_in;
    reg pixel_valid;
    wire [1:0] class_out;
    wire result_valid;
    reg [7:0] image_mem [0:(10*64*64*3)-1];
    reg [7:0] label_mem [0:9];
    integer image_idx;
    integer pixel_idx;
    integer correct;
    integer total;
    integer watchdog;
    integer watchdog_limit;

    bcnn_top dut (
        .clk(clk),
        .rst(rst),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .class_out(class_out),
        .result_valid(result_valid)
    );

    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    initial begin
        $dumpfile("bcnn_tb.vcd");
        $dumpvars(0, tb_bcnn_top);
        $readmemh("../../weights/test_images.hex", image_mem);
        $readmemh("../../weights/test_labels.hex", label_mem);

        rst = 1;
        pixel_in = 0;
        pixel_valid = 0;
        correct = 0;
        total = 0;
        watchdog_limit = 3000000;
        #100;
        rst = 0;

        for (image_idx = 0; image_idx < 10; image_idx = image_idx + 1) begin
            pixel_valid = 1;
            for (pixel_idx = 0; pixel_idx < 64*64*3; pixel_idx = pixel_idx + 1) begin
                pixel_in = image_mem[image_idx*64*64*3 + pixel_idx];
                @(posedge clk);
            end
            pixel_valid = 0;
            watchdog = 0;
            while (!result_valid && watchdog < watchdog_limit) begin
                @(posedge clk);
                watchdog = watchdog + 1;
            end
            if (watchdog >= watchdog_limit) begin
                $display("TIMEOUT on image %0d", image_idx);
                $finish;
            end
            total = total + 1;
            if (class_out == label_mem[image_idx][1:0])
                correct = correct + 1;
            $display("image=%0d pred=%0d label=%0d", image_idx, class_out, label_mem[image_idx][1:0]);
            repeat (10) @(posedge clk);
        end

        $display("total=%0d correct=%0d accuracy=%0f", total, correct, (correct * 100.0) / total);
        $finish;
    end
endmodule
