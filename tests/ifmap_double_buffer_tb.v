// Write a directed test for the ifmap double buffer module. Make sure you test 
// all its ports and its behaviour when your switch banks.

// Your code starts here
module ifmap_double_buffer_tb_v;
  // Layer configuration parameters
  parameter IC0 = 4;
  parameter IC1 = 2;
  parameter IX0 = 5;
  parameter IY0 = 5;
  parameter FX = 3;
  parameter FY = 3;
  parameter OC1 = 4;
  
  // Buffer parameters - sized for input feature map double buffer
  parameter BITWIDTH = 16;
  parameter DATA_WIDTH = BITWIDTH * IC0;      
  parameter BANK_ADDR_WIDTH = $clog2(IC0*IX0*IY0*IC1);
  parameter BANK_DEPTH = IC0*IX0*IY0*IC1;    

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
  ) ifmap_double_buffer_inst (
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
    #20 wen <= 1; wadr <= 7'd10; wdata <= 64'h1;
    #20 wen <= 0;
    #20 switch_banks <= 1;
    #20;
    #20 wen <= 1; wadr <= 7'd10; wdata <= 64'h2;
    #20 wen <= 0;
    #20;    
    #20 ren <= 1; radr <= 7'd10;
    #40; // Wait for read latency
    $display("Test 1: rdata = %h, expected = 1", rdata);
    assert(rdata == 64'h1) else $error("Test 1 failed!");
    #20 ren <=0;

    $display("Starting Test Case 2: Simultaneous write to write-bank and read from read-bank");
    #20 wen <= 1; wadr <= 7'd11; wdata <= 64'h3;
        ren <= 1; radr <= 7'd10;
    #40; // Wait for read latency
    $display("Test 2a: rdata = %h, expected = 1", rdata);
    assert(rdata == 64'h1) else $error("Test 2a failed!");
    #20 switch_banks <=0;
    #20;
    #20 ren <= 1; radr <= 7'd11;
    #40;
    $display("Test 2b: rdata = %h, expected = 3", rdata);
    assert(rdata == 64'h3) else $error("Test 2b failed!");
    #20 wen <= 0; ren <= 0;

    $display("Starting Test Case 3: Sequentially write to a bank and read from it after switching");
    #20;
    for (int i = 0; i < IC0; i++) begin
      for (int j = 0; j < IC1; j++) begin
        for (int k = 0; k < IY0; k++) begin
          for (int l = 0; l < IX0; l++) begin
            #20 
            #20 addr = l + k*IX0 + j*IX0*IY0 + i*IX0*IY0*IC1;
            #20 wen <= 1; wadr <= addr; wdata <= 64'h100;

          end
        end
      end
    end
    #20 wen <=0;
    #20 switch_banks <= 1;
    #20;
    for (int i = 0; i < IC0; i++) begin
      for (int j = 0; j < IC1; j++) begin
        for (int k = 0; k < IY0; k++) begin
          for (int l = 0; l < IX0; l++) begin
            #20 
            #20 addr = l + k*IX0 + j*IX0*IY0 + i*IX0*IY0*IC1;
            #20 ren <=1; radr <= addr;
            #40
            assert(rdata == 64'h100) else $error("Test 3 failed!");
          end
        end
      end
    end
    #20 wen <=0; ren <=0;

    $display("Starting Test Case 4: Check both ren and wen disabled don't change outputs");
    #20 wen <= 1; wadr <= 7'd22; wdata <= 64'd1;
    #20;
    #20 wen <= 0; wadr <= 7'd22; wdata <= 64'd2;
    #20 switch_banks <= 0;
    #20;
    #20 ren <= 1; radr <= 7'd22;
    #40 
    $display("Test 4a: rdata = %h, expected = 1", rdata);
    assert(rdata ==64'd1) else $error("Test 4a failed! it wrote data when wen=0!");
    #20 switch_banks <= 1;
    #20;
    #20 wen <= 1; wadr <= 7'd22; wdata <= 64'd3;
    #20
    #20 switch_banks <= 0;
    #20;
    #20 ren <= 0; radr <= 7'd22;
    #40
    $display("Test 4b: rdata = %h, expected = 1", rdata);
    assert(rdata ==64'd1) else $error("Test 4b failed! it read data when ren=0!");

end

  initial begin
    $vcdplusfile("ifmap_double_buffer_dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, ifmap_double_buffer_tb_v);
    #20000000;
    $finish(2);
  end

endmodule
// Your code ends here
