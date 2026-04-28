// ============================================================================
// Testbench : tb_accelerator.v (FULL SPEC-COMPLIANT)
// Description:
//   - SPI stimulus
//   - Image loading via $readmemh
//   - Latency measurement
// ============================================================================

`timescale 1ns / 1ps

module tb_accelerator;

    // =========================================================================
    // 1. Parameters
    // =========================================================================
    localparam IMG_W = 32;
    localparam IMG_H = 32;
    localparam TOTAL_PIXELS = IMG_W * IMG_H;

    // =========================================================================
    // 2. Clock / Reset
    // =========================================================================
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #10 clk = ~clk;   // 50 MHz
    end

    initial begin
        rst_n = 0;
        #100;
        rst_n = 1;
    end

    // =========================================================================
    // 3. SPI signals
    // =========================================================================
    reg spi_sck;
    reg spi_mosi;
    reg spi_cs_n;
    wire spi_miso;

    // SPI clock ~20 MHz
    initial begin
        spi_sck = 0;
        forever #25 spi_sck = ~spi_sck;
    end

    // =========================================================================
    // 4. DUT wiring
    // =========================================================================
    wire frame_active, frame_done;
    wire l2_valid;
    wire classifier_valid;
    wire start_frame, end_frame;

    wire valid_out;
    wire [3:0] class_out;

    accelerator_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .spi_sck(spi_sck),
        .spi_mosi(spi_mosi),
        .spi_cs_n(spi_cs_n),
        .spi_miso(spi_miso),

        .start_frame(start_frame),
        .end_frame(end_frame),

        .frame_active(frame_active),
        .frame_done(frame_done),
        .bcnn_valid(l2_valid),

        .valid_out(valid_out),
        .class_out(class_out)
    );

    fsm_controller fsm (
        .clk(clk),
        .rst_n(rst_n),
        .frame_active(frame_active),
        .frame_done(frame_done),
        .l2_valid(l2_valid),
        .classifier_valid(valid_out),
        .start_frame(start_frame),
        .end_frame(end_frame)
    );

    // =========================================================================
    // 5. Image memory
    // =========================================================================
    reg [7:0] image_mem [0:TOTAL_PIXELS-1];

    initial begin
        $readmemh("image.hex", image_mem);
    end

    // =========================================================================
    // 6. SPI transmit task
    // =========================================================================
    task send_byte;
        input [7:0] data;
        integer i;
        begin
            for (i = 7; i >= 0; i = i - 1) begin
                @(negedge spi_sck);
                spi_mosi = data[i];
            end
        end
    endtask

    // =========================================================================
    // 7. Send full image
    // =========================================================================
    integer idx;

    initial begin
        spi_cs_n = 1;
        spi_mosi = 0;

        wait(rst_n);

        #100;

        spi_cs_n = 0;  // start frame

        for (idx = 0; idx < TOTAL_PIXELS; idx = idx + 1) begin
            send_byte(image_mem[idx]);
        end

        spi_cs_n = 1;  // end frame
    end

    // =========================================================================
    // 8. Latency measurement
    // =========================================================================
    reg [31:0] cycle_counter;
    reg [31:0] start_cycle;
    reg [31:0] end_cycle;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_counter <= 0;
        else
            cycle_counter <= cycle_counter + 1;
    end

    // Detect first SPI activity
    reg started;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started <= 0;
            start_cycle <= 0;
        end else if (!started && !spi_cs_n) begin
            started <= 1;
            start_cycle <= cycle_counter;
        end
    end

    // Detect classification output
    always @(posedge clk) begin
        if (valid_out) begin
            end_cycle <= cycle_counter;

            $display("====================================");
            $display("Classification Result: %d", class_out);
            $display("Latency (cycles): %d", end_cycle - start_cycle);
            $display("====================================");

            $finish;
        end
    end

endmodule