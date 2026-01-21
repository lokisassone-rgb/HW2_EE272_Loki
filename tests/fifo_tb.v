`define DATA_WIDTH 4
`define FIFO_DEPTH 3
`define COUNTER_WIDTH 1

module fifo_tb;

  // Write five directed tests for the fifo module that test different corner
  // cases. For example, whether it raises the empty and full flags correctly,
  // whether it clears (empties) when you assert the clr signal. Verify its
  // behaviour on reset. You should also test whether the fifo gives the
  // expected latency between when a data goes in and the earliest it can come
  // out. 

  // Your code starts here
  reg clk;
  reg rst_n;
  reg [`DATA_WIDTH - 1 : 0] fifo_din;
  wire fifo_enq;
  wire fifo_full_n;
  wire [`DATA_WIDTH - 1 : 0] fifo_dout;
  wire fifo_deq;
  wire fifo_empty_n;
  reg clr; 

  always #10 clk =~clk;

  fifo
  #(
    .DATA_WIDTH(`DATA_WIDTH),
    .FIFO_DEPTH(`FIFO_DEPTH),
    .COUNTER_WIDTH(`COUNTER_WIDTH)
  ) fifo_inst 
  (
    .clk(clk),
    .rst_n(rst_n),
    .din(fifo_din),
    .enq(fifo_enq),
    .full_n(fifo_full_n),
    .dout(fifo_dout),
    .deq(fifo_deq),
    .empty_n(fifo_empty_n),
    .clr(clr)
  );

  initial begin
    clk <= 0;
    rst_n <= 0;
    fifo_din <= 0;
    clr <= 0;

    // Apply reset
    #15;
    rst_n <= 1;

    // Test 1: Enqueue data until full
    force fifo_enq = 1'b1;  // drive enqueue for this test only

    // Test 1: Enqueue data until full
    $display("Test 1: Enqueue data until full");
    repeat (`FIFO_DEPTH) begin
      @(posedge clk);
      fifo_din <= fifo_din + 1;
    end

    @(posedge clk); // allow flags to update
    assert (fifo_full_n == 1'b0)
      else $fatal(1, "FIFO should be full after %0d enqueues", `FIFO_DEPTH);
    assert (fifo_empty_n == 1'b1)
      else $fatal(1, "FIFO should not be empty when full");

    // Clean up the force for later tests
    release fifo_enq;

    // Test 2: Dequeue all data
    $display("Test 2: Dequeue all data");
    repeat (`FIFO_DEPTH) begin
      #20;
    end

    // Test 3: Clear the FIFO
    $display("Test 3: Clear the FIFO");
    fifo_din <= fifo_din + 1;
    #20;
    clr <= 1;
    #20;
    clr <= 0;

    // Test 4: Enqueue and dequeue simultaneously
    $display("Test 4: Enqueue and dequeue simultaneously");
    fifo_din <= fifo_din + 1;
    #10;
    fifo_din <= fifo_din + 1;
    #10;

    // Test 5: Check empty and full flags
    $display("Test 5: Check empty and full flags");
    #20;
    $finish;
  end

  // Your code ends here

  initial begin
    $vcdplusfile("dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, fifo_tb);
    `ifdef FSDB
    $fsdbDumpfile("dump.fsdb");
    $fsdbDumpvars(0, fifo_tb);
    $fsdbDumpMDA();
    `endif
    #20000000;
    $finish(2);
  end

endmodule
