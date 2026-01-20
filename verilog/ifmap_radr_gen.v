module ifmap_radr_gen
#( 
  parameter BANK_ADDR_WIDTH = 8
)(
  input clk,
  input rst_n,
  input adr_en,
  output reg [BANK_ADDR_WIDTH - 1 : 0] adr, //Changed to reg type
  input config_en,
  input [BANK_ADDR_WIDTH*8 - 1 : 0] config_data
);

  reg [BANK_ADDR_WIDTH - 1 : 0] config_OX0, config_OY0, config_FX, config_FY, 
    config_STRIDE, config_IX0, config_IY0, config_IC1;
  
  always @ (posedge clk) begin
    if (rst_n) begin
      if (config_en) begin
        {config_OX0, config_OY0, config_FX, config_FY, config_STRIDE, 
         config_IX0, config_IY0, config_IC1} <= config_data; 
      end
    end else begin
      {config_OX0, config_OY0, config_FX, config_FY, config_STRIDE, 
       config_IX0, config_IY0, config_IC1} <= 0;
    end
  end
  
  // This is the read address generator for the input double buffer. It is
  // more complex than the sequential address generator because there are
  // overlaps between the input tiles that are read out.  We have already
  // instantiated for you all the configuration registers that will hold the
  // various tiling parameters (OX0, OY0, FX, FY, STRIDE, IX0, IY0, IC1).
  // You need to generate address (adr) for the input buffer in the same
  // sequence as the C++ tiled convolution that you implemented. Make sure you
  // increment/step the address generator only when adr_en is high. Also reset
  // all registers when rst_n is low.  
  
  // Your code starts here
  //Adding Five counters for OX, OY, FX, FY, IC
  reg [BANK_ADDR_WIDTH - 1 : 0] ox_cnt, oy_cnt, fx_cnt, fy_cnt, ic_cnt; 
  
  always @(posedge clk) begin 
    if (!rst_n) begin
      ox_cnt <= 0;
      oy_cnt <= 0;
      fx_cnt <= 0;
      fy_cnt <= 0;
      ic_cnt <= 0;
      adr <= 0;
    end else begin
      if (adr_en) begin
        // Compute address from current counters
        adr <= (ic_cnt * (config_IX0 * config_IY0))
               + ((fy_cnt + (config_STRIDE * oy_cnt)) * config_IX0)
               + (fx_cnt + (config_STRIDE * ox_cnt));

        // Increment counters with carry: ox -> oy -> fx -> fy -> ic
        if (ox_cnt < (config_OX0 - 1)) begin
          ox_cnt <= ox_cnt + 1;
        end else begin
          ox_cnt <= 0;
          if (oy_cnt < (config_OY0 - 1)) begin
            oy_cnt <= oy_cnt + 1;
          end else begin
            oy_cnt <= 0;
            if (fx_cnt < (config_FX - 1)) begin
              fx_cnt <= fx_cnt + 1;
            end else begin
              fx_cnt <= 0;
              if (fy_cnt < (config_FY - 1)) begin
                fy_cnt <= fy_cnt + 1;
              end else begin
                fy_cnt <= 0;
                if (ic_cnt < (config_IC1 - 1)) begin
                  ic_cnt <= ic_cnt + 1;
                end else begin
                  ic_cnt <= 0;
                end
              end
            end
          end
        end
      end
    end
  end

  // Your code ends here
endmodule
