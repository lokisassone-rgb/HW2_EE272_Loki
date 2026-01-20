// Write a directed test for the accumulation buffer module. Make sure you test 
// all its ports and its behaviour when your switch banks.

// Your code starts here
module accumulation_buffer_tb;
  // Parameters
  parameter DATA_WIDTH = 64;
  parameter BANK_ADDR_WIDTH = 7;
  parameter BANK_DEPTH = 128; 

  //input reg
  reg clk;
  reg rst_n;
  reg switch_banks;
  reg ren;
  reg [BANK_ADDR_WIDTH-1:0] radr;
  reg wen;
  reg [BANK_ADDR_WIDTH-1:0] wadr;
  reg [DATA_WIDTH-1:0] wdata;
  reg ren_wb;
  reg [BANK_ADDR_WIDTH-1:0] radr_wb;
  
  wire [DATA_WIDTH-1:0] rdata;
  wire [DATA_WIDTH-1:0] rdata_wb;

  always #10 clk =~clk;

  accumulation_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
    .BANK_DEPTH(BANK_DEPTH)
  ) accumulation_buffer_inst (
    .clk(clk),
    .rst_n(rst_n),
    .switch_banks(switch_banks),
    .ren(ren),
    .radr(radr),
    .rdata(rdata),
    .wen(wen),
    .wadr(wadr),
    .wdata(wdata),
    .ren_wb(ren_wb),
    .radr_wb(radr_wb),
    .rdata_wb(rdata_wb)
  );

  initial begin
    clk <= 0;
    rst_n <= 1;
    switch_banks <= 0;
    ren <= 0;
    radr <= 0;
    wen <= 0;
    wadr <= 0;
    wdata <= 0;
    ren_wb <= 0;
    radr_wb <= 0;

    $display("Starting Test Case 1: Write to Systolic Bank and read back");
    #20 wen <=1; wadr <= 7'b0000111; wdata <= 64'hABCD;
    #20 wen <=0;
    #20 ren <=1; radr <= 7'b0000111;
    #40 //wait for read one cycle latency
    $display("Test 1: rdata = %h, expected = ABCD", rdata);
    assert(rdata == 64'hABCD) else $error("Test 1 failed!");
    #20 ren <=0;

    $display("Starting Test Case 2: Check correct bank is being written/read and switching works");
    #20 switch_banks <= 0;
    #20 wen <= 1; wadr <= 7'd20; wdata <= 64'hBBBB;
    #20 switch_banks <= 1;
    #20 wen <= 1; wadr <= 7'd20; wdata <= 64'hAAAA;
    #20 ren_wb <= 1; radr_wb <= 7'd20;
    #40 // Wait for read latency
    $display("Test 2a: rdata_wb = %h, expected = BBBB", rdata_wb);
    assert(rdata_wb == 64'hBBBB) else $error("Test 2a failed! writeback bank was modified!");
    #20 wen <=0; ren_wb<=0;
    #20 ren <= 1; radr <= 7'd20;
    #40 //wait for read
    $display("Test 2b: rdata = %h, expected = AAAA", rdata);
    assert(rdata == 64'hAAAA) else $error("Test 2b failed! systolic bank was modified!");
    #20 ren <=0;

    $display("Starting Test Case 3: Check both ren and wen disabled don't change outputs");
    #20 wen <= 1; wadr <= 7'd22; wdata <= 64'd1;
    #20 wen <= 0; wadr <= 7'd22; wdata <= 64'd2;
    #20 ren <= 1; radr <= 7'd22;
    #40 
    $display("Test 3a: rdata = %h, expected = 1", rdata);
    assert(rdata ==64'd1) else $error("Test 3a failed! it wrote data when wen=0!");
    #20 wen <= 1; wadr <= 7'd23; wdata <= 64'd3;
    #20 ren <= 0; radr <= 7'd23;
    #40
    $display("Test 3b: rdata = %h, expected = 1", rdata);
    assert(rdata ==64'd1) else $error("Test 3b failed! it read data when ren=0!");
    
    $display("All tests passed!");
  end

  initial begin
    $vcdplusfile("accumulation_buffer_dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, accumulation_buffer_tb);
    #20000000;
    $finish(2);
  end

endmodule
// Your code ends here
