module double_buffer
#( 
  parameter DATA_WIDTH = 64,
  parameter BANK_ADDR_WIDTH = 7,
  parameter [BANK_ADDR_WIDTH : 0] BANK_DEPTH = 128
)(
  input clk,
  input rst_n,
  input switch_banks,
  input ren,
  input [BANK_ADDR_WIDTH - 1 : 0] radr,
  output [DATA_WIDTH - 1 : 0] rdata,
  input wen,
  input [BANK_ADDR_WIDTH - 1 : 0] wadr,
  input [DATA_WIDTH - 1 : 0] wdata
);
  // Implement a double buffer with the dual-port SRAM (ram_sync_1r1w)
  // provided. This SRAM allows one read and one write every cycle. To read
  // from it you need to supply the address on radr and turn ren (read enable)
  // high. The read data will appear on rdata port after 1 cycle (1 cycle
  // latency). To write into the SRAM, provide write address and data on wadr
  // and wdata respectively and turn write enable (wen) high. 
  
  // You can implement both double buffer banks with one dual-port SRAM.
  // Think of one bank consisting of the first half of the addresses of the
  // SRAM, and the second bank consisting of the second half of the addresses.
  // If switch_banks is high, you need to switch the bank you are reading with
  // the bank you are writing on the clock edge.

  // Your code starts here

  reg read_bank; //which bank is on read mode so 0 means read from bank 0 and write from bank 1
  
  wire [BANK_ADDR_WIDTH:0] radr_with_bank;
  wire [BANK_ADDR_WIDTH:0] wadr_with_bank;

  //as document says everytime on clock edge we check if rst_n is set to 0 to reset the read bank to 0 or
  //if we want to switch which bank is read

  always_ff @(posedge clk) begin 
    if (!rst_n) begin
      read_bank <= 0;
    end else if (switch_banks) begin
      read_bank <= ~read_bank;
    end
  end

  //need to add which bank to read and write from based on read_bank as MSB partitions SRAM into 2 buffers

  assign radr_with_bank = {read_bank, radr};
  assign wadr_with_bank = {~read_bank, wadr};

  //now need to instantiate SRAM 1r1w based on above variables, need to increase addr_width by 1 because
  //we added 1 bit for splitting the SRAM into 2 for double buffering. Also need to multiply depth by 2
  //because depth is correlated with bank addr width

  ram_sync_1r1w
    #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BANK_ADDR_WIDTH + 1),
      .DEPTH(BANK_DEPTH*2)
    ) ram_inst_read_0 (
      .clk(clk),
      .ren(ren),
      .wen(wen),
      .wadr(wadr_with_bank),
      .wdata(wdata),
      .radr(radr_with_bank),
      .rdata(rdata)
    );
  

  // Your code ends here
endmodule
