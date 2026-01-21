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
  reg fifo_enq;
  wire fifo_full_n;
  wire [`DATA_WIDTH - 1 : 0] fifo_dout;
  reg fifo_deq;
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
    integer i, errors;
    // Init
    clk      = 0;
    rst_n    = 0;
    fifo_din = 0;
    fifo_enq = 0;
    fifo_deq = 0;
    clr      = 0;
    errors   = 0;

    // Apply reset
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Verify reset behaviour
    $display("Reset check");
    if (fifo_empty_n !== 1'b0) begin $error("EMPTY_N should be 0 after reset"); errors = errors + 1; end
    if (fifo_full_n  !== 1'b1) begin $error("FULL_N should be 1 after reset"); errors = errors + 1; end

    // Test 1: Enqueue data until full
    $display("Test 1: Enqueue data until full");
    for (i = 0; i < `FIFO_DEPTH; i = i + 1) begin
      // Only enqueue if not full
      if (!fifo_full_n) begin $error("FIFO reported full before reaching capacity at enq %0d", i); errors = errors + 1; end
      fifo_din = i + 1;
      fifo_enq = 1;
      @(posedge clk);
      fifo_enq = 0;
    end
    @(posedge clk);
    if (fifo_full_n !== 1'b0) begin $error("FULL_N did not deassert when FIFO reached capacity"); errors = errors + 1; end
    $display("Test 1 passed: FIFO becomes full after %0d enqueues", `FIFO_DEPTH);

    // Test 2: Dequeue all data
    $display("Test 2: Dequeue all data");
    for (i = 0; i < `FIFO_DEPTH; i = i + 1) begin
      if (!fifo_empty_n) begin $error("EMPTY_N unexpectedly low before dequeue %0d", i); errors = errors + 1; end
      // Check current D_OUT before dequeuing; then advance to next on posedge
      if (fifo_dout !== (i + 1)) begin $error("Data mismatch: expected %0d, got %0d", (i + 1), fifo_dout); errors = errors + 1; end
      fifo_deq = 1;
      @(posedge clk);
      fifo_deq = 0;
    end
    @(posedge clk);
    if (fifo_empty_n !== 1'b0) begin $error("EMPTY_N did not assert (go low) after draining FIFO"); errors = errors + 1; end
    if (fifo_full_n  !== 1'b1) begin $error("FULL_N not high after draining FIFO"); errors = errors + 1; end
    $display("Test 2 passed: drained FIFO in order and flags updated");

    // Test 3: Clear the FIFO
    $display("Test 3: Clear the FIFO");
    // Fill partially
    for (i = 0; i < 2; i = i + 1) begin
      if (!fifo_full_n) begin $error("FIFO unexpectedly full during pre-clear enqueue %0d", i); errors = errors + 1; end
      fifo_din = i + 5;
      fifo_enq = 1;
      @(posedge clk);
      fifo_enq = 0;
    end
    @(posedge clk);
    // Now clear
    clr = 1;
    @(posedge clk);
    clr = 0;
    @(posedge clk);
    if (fifo_empty_n !== 1'b0) begin $error("EMPTY_N should be 0 immediately after CLR"); errors = errors + 1; end
    if (fifo_full_n  !== 1'b1) begin $error("FULL_N should be 1 immediately after CLR"); errors = errors + 1; end
    $display("Test 3 passed: CLR empties FIFO and resets flags");

    // Test 4: Enqueue and dequeue simultaneously (avoid warning, non-empty case)
    $display("Test 4: Enqueue and dequeue simultaneously");
    // Ensure we have at least one element so DEQ is legal
    if (fifo_empty_n === 1'b0) begin
      fifo_din = 21;
      fifo_enq = 1;
      @(posedge clk);
      fifo_enq = 0;
      @(posedge clk);
    end
    // Capture current head value
    integer head_before;
    head_before = fifo_dout;
    // Perform simultaneous ENQ+DEQ; on non-empty, D_OUT should be old head value
    fifo_din = 31;
    fifo_enq = 1;
    fifo_deq = 1;
    @(posedge clk);
    if (fifo_dout !== head_before) begin $error("Simultaneous enq+deq (non-empty) should output old head: expected %0d, got %0d", head_before, fifo_dout); errors = errors + 1; end
    fifo_enq = 0;
    fifo_deq = 0;
    @(posedge clk);
    // FIFO occupancy should remain non-empty (one in, one out)
    if (fifo_empty_n !== 1'b1) begin $error("FIFO should be non-empty after simultaneous enq+deq with preload"); errors = errors + 1; end
    $display("Test 4 passed: simultaneous enq+deq on non-empty avoids warnings");

    // Test 5: Check empty and full flags transitions around boundaries
    $display("Test 5: Check empty and full flags");
    // Enqueue up to depth-1: should not be full
    for (i = 0; i < (`FIFO_DEPTH - 1); i = i + 1) begin
      if (!fifo_full_n) begin $error("FULL_N should be high before reaching capacity at count %0d", i); errors = errors + 1; end
      fifo_din = i + 20;
      fifo_enq = 1;
      @(posedge clk);
      fifo_enq = 0;
    end
    @(posedge clk);
    if (fifo_full_n !== 1'b1) begin $error("FULL_N should still be high at depth-1"); errors = errors + 1; end
    // Enqueue last element to reach full
    fifo_din = 99;
    fifo_enq = 1;
    @(posedge clk);
    fifo_enq = 0;
    @(posedge clk);
    if (fifo_full_n !== 1'b0) begin $error("FULL_N should be low when FIFO is full"); errors = errors + 1; end
    // Dequeue one: should clear full condition
    if (!fifo_empty_n) begin $error("EMPTY_N unexpectedly low before boundary dequeue"); errors = errors + 1; end
    fifo_deq = 1;
    @(posedge clk);
    fifo_deq = 0;
    @(posedge clk);
    if (fifo_full_n !== 1'b1) begin $error("FULL_N should return high after one dequeue from full"); errors = errors + 1; end
    // Drain all to empty
    while (fifo_empty_n) begin
      fifo_deq = 1;
      @(posedge clk);
      fifo_deq = 0;
      @(posedge clk);
    end
    if (fifo_empty_n !== 1'b0) begin $error("EMPTY_N should be low when FIFO is empty after draining"); errors = errors + 1; end
    $display("Test 5 passed: flags behave correctly at empty/full boundaries");

    if (errors == 0) begin
      $display("All FIFO tests passed.");
    end else begin
      $display("FIFO tests completed with %0d errors.", errors);
    end
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
