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

  integer pass_count;
  integer fail_count;

  task automatic check;
    input cond;
    input [127:0] msg;
    begin
      if (!cond) begin
        fail_count = fail_count + 1;
        $display("[FAIL] %0s", msg);
      end else begin
        pass_count = pass_count + 1;
        $display("[PASS] %0s", msg);
      end
    end
  endtask

  task automatic push;
    input [`DATA_WIDTH - 1 : 0] val;
    begin
      // Wait until FIFO can accept data
      @(negedge clk);
      fifo_din <= val;
      fifo_enq <= 1'b1;
      @(posedge clk);
      fifo_enq <= 1'b0;
      // allow design NBAs to settle this cycle
      #1;
    end
  endtask

  task automatic pop;
    output [`DATA_WIDTH - 1 : 0] val;
    begin
      @(negedge clk);
      fifo_deq <= 1'b1;
      @(posedge clk);
      // allow design NBAs to update D_OUT then sample
      #1;
      val = fifo_dout;
      fifo_deq <= 1'b0;
    end
  endtask

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
    clk      <= 0;
    rst_n    <= 0;
    fifo_din <= 0;
    fifo_enq <= 0;
    fifo_deq <= 0;
    clr      <= 0;
    pass_count = 0;
    fail_count = 0;

    // Apply reset and check initial flags
    @(posedge clk);
    @(posedge clk);
    rst_n <= 1;
    @(posedge clk);
    check(fifo_empty_n == 1'b0, "After reset: EMPTY_N==0");
    check(fifo_full_n  == 1'b1, "After reset: FULL_N==1");

    // Test 1: Fill to full
    push(4'hA);
    check(fifo_empty_n == 1'b1, "After 1 enq: EMPTY_N==1");
    push(4'hB);
    check(fifo_full_n == 1'b1, "After 2 enq: FULL_N==1 (not full)");
    push(4'hC);
    check(fifo_full_n == 1'b0, "After 3 enq: FULL_N==0 (full)");

    // Test 2: Drain to empty and order preservation
    begin
      reg [`DATA_WIDTH-1:0] val;
      pop(val); check(val == 4'hA, "Deq #1 == 0xA");
      pop(val); check(val == 4'hB, "Deq #2 == 0xB");
      pop(val); check(val == 4'hC, "Deq #3 == 0xC");
      // allow flags to settle after final pop
      #1; check(fifo_empty_n == 1'b0, "After drain: EMPTY_N==0");
      check(fifo_full_n  == 1'b1, "After drain: FULL_N==1");
    end

    // Test 3: Clear behavior
    push(4'h1);
    push(4'h2);
    @(negedge clk);
    clr <= 1'b1;
    @(posedge clk);
    clr <= 1'b0;
    // sample flags a delta after clr edge
    #1; check(fifo_empty_n == 1'b0, "After CLR: EMPTY_N==0");
    #1; check(fifo_full_n  == 1'b1, "After CLR: FULL_N==1");
    begin
      reg [`DATA_WIDTH-1:0] val2;
      // Deq after clear should not return old data; fifo is empty
      @(negedge clk); fifo_deq <= 1'b1; @(posedge clk); val2 = fifo_dout; fifo_deq <= 1'b0;
      check(fifo_empty_n == 1'b0, "Post-clear deq keeps EMPTY_N==0");
    end
    push(4'h3);
    begin
      reg [`DATA_WIDTH-1:0] val3; pop(val3); check(val3 == 4'h3, "Post-clear enq/deq returns new data only");
    end

    // Test 4: Simultaneous ENQ/DEQ (bypass and steady-state)
    // Case A: empty bypass
    @(negedge clk); fifo_din <= 4'h7; fifo_enq <= 1'b1; fifo_deq <= 1'b1; @(posedge clk);
    #1; check(fifo_dout == 4'h7, "Bypass when empty: D_OUT==DIN");
    fifo_enq <= 1'b0; fifo_deq <= 1'b0;
    check(fifo_empty_n == 1'b0, "Bypass leaves FIFO empty");
    // Case B: occupied steady-state
    push(4'h9);
    @(negedge clk); fifo_din <= 4'hE; fifo_enq <= 1'b1; fifo_deq <= 1'b1; @(posedge clk);
    #1; check(fifo_dout == 4'h9, "Simultaneous on occupied: D_OUT old head");
    fifo_enq <= 1'b0; fifo_deq <= 1'b0;
    begin
      reg [`DATA_WIDTH-1:0] val4; pop(val4); check(val4 == 4'hE, "Next deq returns newly enqueued"); end

    // Test 5: Wrap-around pointers with depth=3
    push(4'h1);
    push(4'h2);
    begin reg [`DATA_WIDTH-1:0] v; pop(v); check(v==4'h1, "Wrap seq deq #1"); end
    push(4'h3);
    #1; check(fifo_full_n == 1'b1, "After wrap fill step: not full yet");
    push(4'h4);
    #1; check(fifo_full_n == 1'b0, "After wrap fill: full");
    begin
      reg [`DATA_WIDTH-1:0] v1;
      reg [`DATA_WIDTH-1:0] v2;
      reg [`DATA_WIDTH-1:0] v3;
      pop(v1); check(v1==4'h2, "Wrap deq #2");
      pop(v2); check(v2==4'h3, "Wrap deq #3");
      pop(v3); check(v3==4'h4, "Wrap deq #4");
    end
    #1; check(fifo_empty_n == 1'b0, "Wrap: empty at end");

    $display("\nTest Summary: %0d PASS, %0d FAIL\n", pass_count, fail_count);
    if (fail_count == 0) $finish; else $finish(2);
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
