// Write a UVM test for the sequential address generator (adr_gen_sequential).
// Especially make sure that the address generator works correctly with random
// stalls (meaning if adr_en goes low intermittently), that it resets properly, 
// and that it goes back to zero after reaching the maximum configured value.

// Your code starts here
`define BANK_ADDR_WIDTH (32)
`define TEST_LENGTH (100)

interface adr_gen_sequential_if (input bit clk);
  logic rst_n;
  logic adr_en;
  logic [`BANK_ADDR_WIDTH - 1 : 0] adr;
  logic config_en;
  logic [`BANK_ADDR_WIDTH - 1 : 0] config_data;
endinterface

// Transaction Object
class adr_gen_sequential_item;
    rand bit adr_en;
    rand bit [`BANK_ADDR_WIDTH - 1 : 0] config_data;
    rand bit config_en;
    bit [`BANK_ADDR_WIDTH - 1 : 0] adr;
    
    function void print(int id = "");
        $display("T=%0t [transaction_id=%0d] adr_en=%b config_data=%h adr=%h", $time, id, adr_en, config_data, adr);
    endfunction
endclass;

// Driver applies the generated stimulus to DUT
class driver; 
    virtual adr_gen_sequential_if vif;
    mailbox drv_mbx;
    
    task run();
        $display ("T=%0t [adr_gen_sequential driver] Starting ...", $time);
        @ (negedge vif.clk);
        forever begin
            adr_gen_sequential_item transaction;
            drv_mbx.get(transaction);

            vif.adr_en = transaction.adr_en;
            vif.config_en = transaction.config_en;
            vif.config_data = transaction.config_data;

            @ (negedge vif.clk);
        end
    endtask
endclass

// Monitor observes DUT outputs and sends transactions to scoreboard
class monitor; 
    virtual adr_gen_sequential_if vif; 
    mailbox scb_mbx; // mailbox connected to scoreboard

    task run();
        $display ("T=%0t [adr_gen_sequential monitor] Starting ...", $time);
        forever begin
            adr_gen_sequential_item transaction = new;
            
            @ (posedge vif.clk);
            transaction.adr_en = vif.adr_en; 
            transaction.config_en = vif.config_en;
            transaction.config_data = vif.config_data;
            
            transaction.adr = vif.adr;

            scb_mbx.put(transaction);
        end
    endtask
endclass

class scoreboard; 
    mailbox scb_mbx; // mailbox connected to monitor
    int resp_id; 
    
    reg signed [`BANK_ADDR_WIDTH - 1 : 0] config_block_max;

    task run();
        forever begin
            adr_gen_sequential_item transaction;
            resp_id = 0;

            scb_mbx.get(transaction);

            // Check for config_en
            if (transaction.config_en) begin
                config_block_max = transaction.config_data;
                expected_adr = 0; // Reset expected address on new config
            end

            // Check for adr_en
            if (transaction.adr_en) begin
                if (transaction.adr !== expected_adr) begin
                    $error("Mismatch at transaction_id=%0d: expected adr=%h, got adr=%h", resp_id, expected_adr, transaction.adr);
                end

                // Update expected address
                if (expected_adr == config_block_max) begin
                    expected_adr = 0;
                end else begin
                    expected_adr++;
                end
            end

            resp_id++;
        end
        $finish;
    endtask
endclass

// Environment class to instantiate driver, monitor, and scoreboard
class env; 
    driver d0;
    monitor m0;
    scoreboard s0;
    mailbox scb_mbx; // mailbox connecting monitor and scoreboard
    virtual adr_gen_sequential_if vif;

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
        $display ("T=%0t [adr_gen_sequential test] Starting ...", $time);

        stim_id = 0;
        // Initial reset
        e0.vif.rst_n = 0;
        e0.vif.adr_en = 0;
        e0.vif.config_en = 0;
        e0.vif.config_data = 0;
        @ (negedge e0.vif.clk);
        @ (negedge e0.vif.clk);
        e0.vif.rst_n = 1;

        // Apply configuration
        adr_gen_sequential_item config_transaction = new;
        config_transaction.config_en = 1;
        config_transaction.config_data = 10; // Example max address
        config_transaction.adr_en = 0;
        drv_mbx.put(config_transaction);
        @ (negedge e0.vif.clk);

        // Disable config after one cycle
        adr_gen_sequential_item disable_config_transaction = new;
        disable_config_transaction.config_en = 0;
        disable_config_transaction.adr_en = 0;
        drv_mbx.put(disable_config_transaction);
        @ (negedge e0.vif.clk);

        // Generate address sequence with random stalls
        for (int i = 0; i < `TEST_LENGTH; i++) begin
            adr_gen_sequential_item transaction = new;
            transaction.config_en = 0;

            // Randomly decide to enable or disable adr_en
            if ($urandom_range(0, 1)) begin
                transaction.adr_en = 1;
            end else begin
                transaction.adr_en = 0;
            end

            drv_mbx.put(transaction);
            @ (negedge e0.vif.clk);
            stim_id++;
        end

        // Finish simulation
        #20;
        $finish;
    endtask
endclass

module adr_gen_sequential_tb;

    reg clk;

    always #10 clk =~clk;

    adr_gen_sequential_if vif (clk);

    adr_gen_sequential #(
        .BANK_ADDR_WIDTH(`BANK_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(vif.rst_n),
        .adr_en(vif.adr_en),
        .adr(vif.adr),
        .config_en(vif.config_en),
        .config_data(vif.config_data)
    );

    assign vif.rst_n = rst_n;

    initial begin
        test t0;

        clk <= 0;
        rst_n <= 0;
        #40 rst_n <= 1;
        t0 = new(); 
        t0.e0.vif = vif;
        t0.run();

    end

    initial begin
        $vcdplusfile("dump.vcd");
        $vcdplusmemon();
        $vcdpluson(0, adr_gen_sequential_tb);
        #200000;
        $finish(2);
    end

endmodule





// Your code ends here
