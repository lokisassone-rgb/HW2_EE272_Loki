module accumulation_buffer
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
  input [DATA_WIDTH - 1 : 0] wdata,

  input ren_wb,
  input [BANK_ADDR_WIDTH - 1 : 0] radr_wb,
  output [DATA_WIDTH - 1 : 0] rdata_wb
);

  // Implement an accumulation buffer with the dual-port SRAM (ram_sync_1r1w)
  // provided. This SRAM allows one read and one write every cycle. To read
  // from it you need to supply the address on radr and turn ren (read enable)
  // high. The read data will appear on rdata port after 1 cycle (1 cycle
  // latency). To write into the SRAM, provide write address and data on wadr
  // and wdata respectively and turn write enable (wen) high. 
  
  // Accumulation buffer is similar to a double buffer, but one of its banks
  // has both a read port (ren, radr, rdata) and a write port (wen, wadr,
  // wdata). This bank is used by the systolic array to store partial sums and
  // then read them back out. The other bank has a read port only (ren_wb,
  // radr_wb, rdata_wb). This bank is used to read out the final output (after
  // accumulation is complete) and send it out of the chip. The reason for
  // adopting two banks is so that we can overlap systolic array processing,
  // and data transfer out of the accelerator (otherwise one of them will
  // stall while the other is taking place). Note: both srams will be 1r1w, 
  // but the logical operation will be as described above.

  // If switch_banks is high, you need to switch the functionality of the two
  // banks at the positive edge of the clock. That means, you will use the bank
  // you were previously using for data transfer out of the chip for systolic
  // array and vice versa.

  // Your code starts here
  reg read_only_bank; //which bank is on read only mode so 0 means read only from bank 0 and write/read from bank 1

  //as document says everytime on clock edge we check if rst_n is set to 0 to reset the read bank to 0 or
  //if we want to switch which bank is read

  always_ff @(posedge clk) begin 
    if (!rst_n) begin
      read_only_bank <= 0;
    end else if (switch_banks) begin
      read_only_bank <= ~read_only_bank;
    end
  end

  wire wen_bank0; //to enable bank0 wen or not
  wire wen_bank1; //to enable bank1 wen or not
  wire [BANK_ADDR_WIDTH - 1 : 0] wadr_bank0; //write address for bank0
  wire [BANK_ADDR_WIDTH - 1 : 0] wadr_bank1; //write address for bank1
  wire [DATA_WIDTH - 1 : 0] wdata_bank0; //write data for bank0
  wire [DATA_WIDTH - 1 : 0] wdata_bank1; //write data for bank1
  wire [BANK_ADDR_WIDTH - 1 : 0] radr_bank0; //read address for bank0
  wire [BANK_ADDR_WIDTH - 1 : 0] radr_bank1; //read address for bank1
  wire ren_bank0; //bank 0 ren
  wire ren_bank1; //bank 1 ren
  wire [DATA_WIDTH - 1 : 0] rdata_bank0; //bank 0 rdata
  wire [DATA_WIDTH - 1 : 0] rdata_bank1; //bank 1 rdata

  assign wen_bank0 = (read_only_bank==0) ? 0 : wen;
  assign wen_bank1 = (read_only_bank==1) ? 0 : wen;
  assign wadr_bank0 = (read_only_bank==0) ? 0 : wadr;
  assign wadr_bank1 = (read_only_bank==1) ? 0 : wadr;
  assign wdata_bank0 = (read_only_bank==0) ? 0 : wdata;
  assign wdata_bank1 = (read_only_bank==1) ? 0 : wdata;
  assign radr_bank0 = (read_only_bank==0) ? radr_wb : radr;
  assign radr_bank1 = (read_only_bank==1) ? radr_wb : radr;
  assign ren_bank0 = (read_only_bank==0) ? ren_wb : ren;
  assign ren_bank1 = (read_only_bank==1) ? ren_wb : ren;

  ram_sync_1r1w 
    #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BANK_ADDR_WIDTH),
      .DEPTH(BANK_DEPTH)
    ) sram_bank0 (
      .clk(clk),
      .ren(ren_bank0),
      .wen(wen_bank0),
      .wadr(wadr_bank0),
      .wdata(wdata_bank0),
      .radr(radr_bank0),
      .rdata(rdata_bank0)
    );
  ram_sync_1r1w
    #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BANK_ADDR_WIDTH),
      .DEPTH(BANK_DEPTH)
    ) sram_bank1 (
      .clk(clk),
      .ren(ren_bank1),
      .wen(wen_bank1),
      .wadr(wadr_bank1),
      .wdata(wdata_bank1),
      .radr(radr_bank1),
      .rdata(rdata_bank1)
    );

  assign rdata = (read_only_bank == 0) ? rdata_bank1 : rdata_bank0;
  assign rdata_wb = (read_only_bank == 0) ? rdata_bank0 : rdata_bank1;
  
  // Your code ends here
endmodule
