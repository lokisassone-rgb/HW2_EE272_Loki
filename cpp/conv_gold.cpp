#include <stdio.h>
#include <cassert>
#include <string.h>
#include <cstdint>

template <int OY, int OX, int OC, int IC, int FY, int FX, int STRIDE>
void conv_gold(int16_t ifmap[(OY-1)*STRIDE+FY][(OX-1)*STRIDE+FX][IC],
               int16_t weight[FY][FX][IC][OC],
               int32_t ofmap[OY][OX][OC]){

  // Implement the functionality of a convolutional layer, which convolves
  // ifmap with weight to produce ofmap. Your code should assign values to the
  // ofmap array. Make sure you take STRIDE into account.
 
  // Your code starts here
  memset(ofmap, 0, sizeof(int32_t)*OY*OX*OC); //initialize ofmap to zero

  for (int oc =0; oc< OC; oc++) { //output channels
    for (int ic =0; ic< IC; ic++){ //intput channels
      for (int oy=0; oy< OY; oy++){ //output height
        for (int ox=0; ox<OX; ox++){ //output width
          for (int fy=0; fy< FY; fy++){ //kernel height
            for (int fx=0; fx<FX; fx++){ //kernel width
              ofmap[oy][ox][oc] += ifmap[fy + oy*STRIDE][fx + ox*STRIDE][ic] *weight[fy][fx][ic][oc]; //element-wise MAC with stride
            }
          }
        }
      }
    }
  }
}
