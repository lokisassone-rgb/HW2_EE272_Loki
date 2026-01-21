// Write a UVM test for the MAC unit. Especially make sure that the MAC unit
// works correctly with random stalls (meaning if en goes low intermittently)
// and that it resets properly.

// Your code starts here
`define IFMAP_WIDTH (16)
`define WEIGHT_WIDTH (16)
`define OFMAP_WIDTH (32)
`define TEST_LENGTH (100)

interface mac_if (input bit clk);
  logic rst_n;
  logic en;
  logic weight_wen;
  logic signed [`IFMAP_WIDTH - 1 : 0] ifmap_in;
  logic signed [`WEIGHT_WIDTH - 1 : 0] weight_in;
  logic signed [`OFMAP_WIDTH - 1 : 0] ofmap_in;
  logic signed [`IFMAP_WIDTH - 1 : 0] ifmap_out;
  logic signed [`OFMAP_WIDTH - 1 : 0] ofmap_out;
endinterface

class mac_item;
  rand bit en;
  rand bit weight_wen;
  rand bit signed [`IFMAP_WIDTH - 1 : 0] ifmap_in;
  rand bit signed [`WEIGHT_WIDTH - 1 : 0] weight_in;
  rand bit signed [`OFMAP_WIDTH - 1 : 0] ofmap_in;
  bit signed [`IFMAP_WIDTH - 1 : 0] ifmap_out;
  bit signed [`OFMAP_WIDTH - 1 : 0] ofmap_out; 

  function void print(int id = "");
    $display("T=%0t [transaction_id=%0d] en=%b weight_wen=%b ifmap_in=%h weight_in=%h ofmap_in=%h ifmap_out=%h ofmap_out=%h", $time, id, en, weight_wen, ifmap_in, weight_in, ofmap_in, ifmap_out, ofmap_out);
  endfunction
endclass;

class driver;
  virtual mac_if vif;
  mailbox drv_mbx;
  
  task run();
    $display ("T=%0t [Write driver] Starting ...", $time);
    @ (negedge vif.clk);
    forever begin
      mac_item transaction;
      drv_mbx.get(transaction);

      vif.en = transaction.en;
      vif.weight_wen = transaction.weight_wen;
      vif.ifmap_in = transaction.ifmap_in;
      vif.weight_in = transaction.weight_in;
      vif.ofmap_in = transaction.ofmap_in;

      @ (negedge vif.clk);
    end
  endtask
endclass

class monitor; 
  virtual mac_if vif;
  mailbox scb_mbx; // mailbox connected to scoreboard

  task run();
    $display("T=%0t [Read monitor] Starting ...", $time);
    forever begin
      mac_item transaction = new;

      @ (posedge vif.clk);
      transaction.en = vif.en;
      transaction.weight_wen = vif.weight_wen;
      transaction.ifmap_in = vif.ifmap_in;
      transaction.weight_in = vif.weight_in;
      transaction.ofmap_in = vif.ofmap_in;

      transaction.ifmap_out = vif.ifmap_out;
      transaction.ofmap_out = vif.ofmap_out;

      // send transaction to scoreboard
      scb_mbx.put(transaction);
    end
  endtask
endclass

class scoreboard;
  mailbox scb_mbx;
  int resp_id;
  //golden internal registers below
  reg signed [`WEIGHT_WIDTH - 1 : 0] weight_r_golden;
  reg signed [`IFMAP_WIDTH - 1 : 0] ifmap_r_golden;
  reg signed [`OFMAP_WIDTH - 1 : 0] ofmap_r_golden;

  task run();
    forever begin
      mac_item transaction;
      resp_id = 0;
    // the initial data is garbage since no stimulus applied yet, skip it
      scb_mbx.get(transaction);  // at 30ns
      // set the initial expected_data to the initial dout to pass the first comparison
      weight_r_golden = 0;
      ifmap_r_golden = 0;
      ofmap_r_golden = 0;

      while(resp_id < `TEST_LENGTH) begin
        scb_mbx.get(transaction);  // first mail arrives at 50ns
        transaction.print(resp_id);

        // Comparison of the 2 outputs
        if (transaction.ofmap_out !== ofmap_r_golden) begin
          $display("T=%0t [Scoreboard] Ofmap_out Error! Received = %h, expected = %h", $time, transaction.ofmap_out, ofmap_r_golden);
        end else begin
          $display("T=%0t [Scoreboard] Ofmap_out Pass! Received = %h, expected = %h", $time, transaction.ofmap_out, ofmap_r_golden);
        end
        if (transaction.ifmap_out !== ifmap_r_golden) begin
          $display("T=%0t [Scoreboard] Ifmap_out Error! Received = %h, expected = %h", $time, transaction.ifmap_out, ifmap_r_golden);
        end else begin
          $display("T=%0t [Scoreboard] Ifmap_out Pass! Received = %h, expected = %h", $time, transaction.ifmap_out, ifmap_r_golden);
        end
        
        if (transaction.weight_wen) begin
          weight_r_golden = transaction.weight_in;
        end

        if (transaction.en) begin
          ifmap_r_golden = transaction.ifmap_in;
          ofmap_r_golden = weight_r_golden * transaction.ifmap_in + transaction.ofmap_in;
        end

        resp_id = resp_id + 1;
      end
      $finish;
    end
  endtask
endclass

class env;
  driver d0;
  monitor m0;
  scoreboard s0;
  mailbox scb_mbx;
  virtual mac_if vif;

  function new();
    d0 = new; 
    m0 = new;
    s0 = new;
    scb_mbx = new();
  endfunction

  virtual task run();
    d0.vif = vif;
    m0.vif = vif;
    m0.scb_mbx = scb_mbx;
    s0.scb_mbx = scb_mbx;

    fork
      s0.run();
      d0.run();
      m0.run();    
    join_any 
    endtask   
endclass    

class test;
  env e0;
  mailbox drv_mbx;
  int stim_id;

  function new();
    drv_mbx = new();
    e0 = new();
  endfunction

  virtual task run();
    e0.d0.drv_mbx = drv_mbx;
    fork 
      e0.run();
    join_none

    apply_stim();
  endtask

  virtual task apply_stim();
    mac_item transaction;
    $display ("T=%0t [Test] Starting write stimulus ...", $time);

    stim_id = 0;
    while(stim_id < `TEST_LENGTH) begin
      transaction = new;
      transaction.randomize();
      stim_id = stim_id + 1;
      drv_mbx.put(transaction);
    end

  endtask
endclass

module mac_tb;

    reg clk;
    reg rst_n;

    always #10 clk =~clk;
    
    mac_if _if (clk);

    mac #(
      .IFMAP_WIDTH(`IFMAP_WIDTH), 
      .WEIGHT_WIDTH(`WEIGHT_WIDTH), 
      .OFMAP_WIDTH(`OFMAP_WIDTH)
    ) dut (
      .clk(_if.clk),
      .rst_n(_if.rst_n),
      .en(_if.en),
      .weight_wen(_if.weight_wen),
      .ifmap_in(_if.ifmap_in),
      .weight_in(_if.weight_in),
      .ofmap_in(_if.ofmap_in),
      .ifmap_out(_if.ifmap_out),
      .ofmap_out(_if.ofmap_out)
    );

    assign _if.rst_n = rst_n;

    initial begin
        test t0;

        clk <= 0;
        rst_n <= 0;
        #40 rst_n <= 1;
        t0 = new(); 
        t0.e0.vif = _if;
        t0.run();
    end

    initial begin
        $vcdplusfile("dump.vcd");
        $vcdplusmemon();
        $vcdpluson(0, mac_tb);
        #20000000;
        $finish(2);
    end

endmodule

// Your code ends here
