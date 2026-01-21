// Write a directed test for the weight double buffer module. Make sure you test 
// all its ports and its behaviour when your switch banks.

// Your code starts here
module weight_double_buffer_tb_v;
  // Layer configuration parameters
  parameter IC0 = 4;
  parameter IC1 = 2;
  parameter IX0 = 5;
  parameter IY0 = 5;
  parameter OC0 = 4;
  parameter OX1 = 4;
  parameter OY1 = 4;
  parameter FX = 3;
  parameter FY = 3;
  parameter OC1 = 4;
  
  // Buffer parameters - sized for input feature map double buffer
  parameter BITWIDTH = 16;
  parameter DATA_WIDTH = BITWIDTH * IC0*OC0;      
  parameter BANK_ADDR_WIDTH = $clog2(IC0*OC0*FX*FY*IC1);
  parameter BANK_DEPTH = IC0*OC0*FX*FY*IC1;    

  //input reg
  reg clk;
  reg rst_n;
  reg switch_banks;
  reg ren;
  reg [BANK_ADDR_WIDTH-1:0] radr;
  reg wen;
  reg [BANK_ADDR_WIDTH-1:0] wadr;
  reg [DATA_WIDTH-1:0] wdata;
  
  wire [DATA_WIDTH-1:0] rdata;

  always #10 clk = ~clk;

  double_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
    .BANK_DEPTH(BANK_DEPTH)
  ) weight_double_buffer_inst (
    .clk(clk),
    .rst_n(rst_n),
    .switch_banks(switch_banks),
    .ren(ren),
    .radr(radr),
    .rdata(rdata),
    .wen(wen),
    .wadr(wadr),
    .wdata(wdata)
  );

  initial begin
    integer addr;
    addr = 0;
    clk <= 0;
    rst_n <= 0;  // Start in reset
    switch_banks <= 0;
    ren <= 0;
    radr <= 0;
    wen <= 0;
    wadr <= 0;
    wdata <= 0;

    // Apply reset pulse
    #20 rst_n <= 1;  // Release reset after 20ns
    #20;  // Wait for reset to take effect

    $display("Starting Test Case 1: Write to write-bank and switch banks and verify it's readable");
    #20 wen <= 1; wadr <= 10; wdata <= 1;
    #20 wen <= 0;
    #20 switch_banks <= 1;
    #20 wen <= 1; wadr <= 10; wdata <= 2;
    #20 wen <= 0;
    #20;    
    #20 ren <= 1; radr <= 10;
    #40; // Wait for read latency
    $display("Test 1: rdata = %h, expected = 1", rdata);
    assert(rdata == 1) else $error("Test 1 failed!");
    #20 ren <=0;

    $display("Starting Test Case 2: Simultaneous write to write-bank and read from read-bank");
    #20 wen <= 1; wadr <= 11; wdata <= 3;
        ren <= 1; radr <= 10;
    #40; // Wait for read latency
    $display("Test 2a: rdata = %h, expected = 1", rdata);
    assert(rdata == 1) else $error("Test 2a failed!");
    #20 switch_banks <= 0;
    #20;
    #20 ren <= 1; radr <= 11;
    #40;
    $display("Test 2b: rdata = %h, expected = 3", rdata);
    assert(rdata == 3) else $error("Test 2b failed!");
    #20 wen <= 0; ren <= 0;

    $display("Starting Test Case 3: Sequentially write to all weight addresses in bank (simulate weight tiling) and read from it after switching");
    #20;
    // Loop order: IC1, FX, FY, OC0, IC0 (matches weight buffer layout from Figure 4)
    for (int i = 0; i < IC1; i++) begin
      for (int j = 0; j < FX; j++) begin
        for (int k = 0; k < FY; k++) begin
          for (int l = 0; l < OC0; l++) begin
            for (int m = 0; m < IC0; m++) begin
              #40; 
              #20 addr = m + l*IC0 + k*IC0*OC0 + j*IC0*OC0*FY + i*IC0*OC0*FY*FX;
              #20;
              #20 wen <= 1; wadr <= addr; wdata <= addr;
              #40;
              #20 wen <=0;
            end
          end
        end
      end
    end
    #20 wen <=0;
    #20 switch_banks <= 1;
    #20;
    for (int i = 0; i < IC1; i++) begin
      for (int j = 0; j < FX; j++) begin
        for (int k = 0; k < FY; k++) begin
          for (int l = 0; l < OC0; l++) begin
            for (int m = 0; m < IC0; m++) begin
              #20 addr = m + l*IC0 + k*IC0*OC0 + j*IC0*OC0*FY + i*IC0*OC0*FY*FX;
              #20 ren <=1; radr <= addr;
              #40;
              $display("Test 3: addr=%0d rdata = %h, expect = %h", addr, rdata, addr);
              assert(rdata == addr) else $error("Test 3 failed at addr %0d!", addr);
              #20 ren <= 0;
            end
          end
        end
      end
    end
    #20;

    $display("Starting Test Case 4: Check both ren and wen disabled don't change outputs");
    #20 wen <= 1; wadr <= 22; wdata <= 1;
    #20;
    #20 wen <= 0; wadr <= 22; wdata <= 2;
    #20 switch_banks <= 0;
    #20;
    #20 ren <= 1; radr <= 22;
    #40 
    $display("Test 4a: rdata = %h, expected = 1", rdata);
    assert(rdata == 1) else $error("Test 4a failed! it wrote data when wen=0!");
    #20 switch_banks <= 1;
    #20;
    #20 wen <= 1; wadr <= 23; wdata <= 3;
    #20
    #20 switch_banks <= 0;
    #20;
    #20 ren <= 0; radr <= 23;
    #40
    $display("Test 4b: rdata = %h, expected = 1", rdata);
    assert(rdata == 1) else $error("Test 4b failed! it read data when ren=0!");

end

  initial begin
    $vcdplusfile("weight_double_buffer_dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, weight_double_buffer_tb_v);
    #20000000;
    $finish(2);
  end

endmodule
// Your code ends here
