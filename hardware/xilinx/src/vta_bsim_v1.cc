
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vta_bsim_v1.h"

void fetch(
  ap_uint<32> insn_count,
  ap_uint<128>* insns,
  hls::stream<ap_uint<128>> &load_queue,
  hls::stream<ap_uint<128>> &gemm_queue,
  hls::stream<ap_uint<128>> &store_queue)
{
                                   //store_queue[0] = (ap_uint<128>)0;
  //load_queue[0] = (ap_uint<128>)0;
  //gemm_queue[0] = (ap_uint<128>)0;
  ap_int<32> _top;
  ap_int<32> fetch;
  for (ap_int<32> i = 0; i < ((ap_int<32>)insn_count); ++i) {
    ap_uint<3> opcode;
    opcode = ((ap_uint<3>)insns[i](2, 0));
    ap_uint<2> memory_type;
    memory_type = ((ap_uint<2>)insns[i](8, 7));
    if (((ap_int<32>)opcode) == 1) {
      //store_queue[i] = insns[i];
      store_queue.write(insns[i]);
    } else {
      if (((ap_int<32>)opcode) == 0) {
        if (((ap_int<32>)memory_type) == 2) {
          //load_queue[i] = insns[i];
          load_queue.write(insns[i]);
        } else {
          if (((ap_int<32>)memory_type) == 1) {
            //load_queue[i] = insns[i];
            load_queue.write(insns[i]);
          } else {
            //gemm_queue[i] = insns[i];
            gemm_queue.write(insns[i]);
          }
        }
      } else {
        //gemm_queue[i] = insns[i];
            gemm_queue.write(insns[i]);
      }
    }
  }
}


void load(ap_uint<8>* inputs, ap_uint<8>* weights, hls::stream<ap_uint<128>> &load_queue, hls::stream<bool> & g2l_dep_queue, hls::stream<bool> & l2g_dep_queue, 
ap_uint<8>* inp_mem, ap_uint<8>* wgt_mem) {
  //l2g_dep_queue[0] = (ap_uint<1>)0;
  //load_queue[0] = (ap_uint<128>)0;
  //g2l_dep_queue[0] = (ap_uint<1>)0;
  ap_int<32> _top;
  ap_uint<128> raw_insn;
  //raw_insn = load_queue[0];
  raw_insn = load_queue.read();
  ap_int<32> memory_type;
  memory_type = ((ap_int<32>)raw_insn(8, 7));
  ap_int<32> sram_base;
  sram_base = ((ap_int<32>)raw_insn(24, 9));
  ap_int<32> dram_base;
  dram_base = ((ap_int<32>)raw_insn(56, 25));
  ap_int<32> is_pad_min_value;
  is_pad_min_value = ((ap_int<32>)raw_insn(57, 57));
  ap_int<32> y_size;
  y_size = ((ap_int<32>)raw_insn(79, 64));
  ap_int<32> x_size;
  x_size = ((ap_int<32>)raw_insn(95, 80));
  ap_int<32> x_stride;
  x_stride = ((ap_int<32>)raw_insn(111, 96));
  ap_int<32> y_pad_0;
  y_pad_0 = ((ap_int<32>)raw_insn(115, 112));
  ap_int<32> y_pad_1;
  y_pad_1 = ((ap_int<32>)raw_insn(119, 116));
  ap_int<32> x_pad_0;
  x_pad_0 = ((ap_int<32>)raw_insn(123, 120));
  ap_int<32> x_pad_1;
  x_pad_1 = ((ap_int<32>)raw_insn(127, 124));
  if (((ap_int<128>)raw_insn(4, 4)) == (ap_int<128>)1) {
    //ap_int<32> scalar0;
    //scalar0 = ((ap_int<32>)g2l_dep_queue[0]);
    g2l_dep_queue.read();
  }
  if (memory_type == 2) {
    ap_int<32> load_2d;
    ap_int<32> pad_top;
    for (ap_int<32> y = 0; y < y_pad_0; ++y) {
      for (ap_int<32> x = 0; x < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x) {
        ap_int<32> pad_clear;
        for (ap_int<32> col = 0; col < 16; ++col) {
          inp_mem[(col + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * y)) + x) * 16))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
        }
      }
    }
    ap_int<32> pad_left;
    for (ap_int<32> y1 = 0; y1 < y_size; ++y1) {
      for (ap_int<32> x1 = 0; x1 < x_pad_0; ++x1) {
        ap_int<32> pad_clear1;
        for (ap_int<32> col1 = 0; col1 < 16; ++col1) {
          inp_mem[(col1 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y1))) + x1) * 16))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
        }
      }
    }
    ap_int<32> load_data;
    for (ap_int<32> y2 = 0; y2 < y_size; ++y2) {
      for (ap_int<32> x2 = 0; x2 < x_size; ++x2) {
        ap_uint<8> compute0[16];
        for (ap_int<32> c = 0; c < 16; ++c) {
          compute0[c] = inputs[((((dram_base + (y2 * x_stride)) + x2) * 16) + c)];
        }
        ap_uint<8> pack0[16];
        for (ap_int<32> temp = 0; temp < 16; ++temp) {
          ap_uint<8> pack0_temp;
          pack0_temp = (ap_uint<8>)0;
          pack0_temp(7, 0) = compute0[temp];
          pack0[temp] = pack0_temp;
        }
        ap_int<32> mutate0;
        for (ap_int<32> col2 = 0; col2 < 16; ++col2) {
          inp_mem[(col2 + ((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y2))) + x2) + x_pad_0) * 16))] = pack0[col2];
        }
      }
    }
    ap_int<32> pad_right;
    for (ap_int<32> y3 = 0; y3 < y_size; ++y3) {
      for (ap_int<32> x3 = 0; x3 < x_pad_1; ++x3) {
        ap_int<32> pad_clear2;
        for (ap_int<32> col3 = 0; col3 < 16; ++col3) {
          inp_mem[(col3 + (((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y3))) + x3) + x_pad_0) + x_size) * 16))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
        }
      }
    }
    ap_int<32> pad_bottom;
    for (ap_int<32> y4 = 0; y4 < y_pad_1; ++y4) {
      for (ap_int<32> x4 = 0; x4 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x4) {
        ap_int<32> pad_clear3;
        for (ap_int<32> col4 = 0; col4 < 16; ++col4) {
          inp_mem[(col4 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * ((y_pad_0 + y_size) + y4))) + x4) * 16))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
        }
      }
    }
  }
  if (memory_type == 1) {
    ap_int<32> load_2d1;
    ap_int<32> pad_top1;
    for (ap_int<32> y5 = 0; y5 < y_pad_0; ++y5) {
      for (ap_int<32> x5 = 0; x5 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x5) {
        ap_int<32> pad_clear4;
        for (ap_int<32> row = 0; row < 16; ++row) {
          for (ap_int<32> col5 = 0; col5 < 16; ++col5) {
            wgt_mem[((col5 + (row * 16)) + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * y5)) + x5) * 256))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
          }
        }
      }
    }
    ap_int<32> pad_left1;
    for (ap_int<32> y6 = 0; y6 < y_size; ++y6) {
      for (ap_int<32> x6 = 0; x6 < x_pad_0; ++x6) {
        ap_int<32> pad_clear5;
        for (ap_int<32> row1 = 0; row1 < 16; ++row1) {
          for (ap_int<32> col6 = 0; col6 < 16; ++col6) {
            wgt_mem[((col6 + (row1 * 16)) + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y6))) + x6) * 256))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
          }
        }
      }
    }
    ap_int<32> load_data1;
    for (ap_int<32> y7 = 0; y7 < y_size; ++y7) {
      for (ap_int<32> x7 = 0; x7 < x_size; ++x7) {
        ap_uint<8> compute1[256];
        for (ap_int<32> r = 0; r < 16; ++r) {
          for (ap_int<32> c1 = 0; c1 < 16; ++c1) {
            compute1[(c1 + (r * 16))] = weights[((((((dram_base + (y7 * x_stride)) + x7) * 16) + r) * 16) + c1)];
          }
        }
        ap_uint<8> pack1[256];
        for (ap_int<32> indices = 0; indices < 16; ++indices) {
          for (ap_int<32> temp1 = 0; temp1 < 16; ++temp1) {
            ap_uint<8> pack1_temp;
            pack1_temp = (ap_uint<8>)0;
            pack1_temp(7, 0) = compute1[(temp1 + (indices * 16))];
            pack1[(temp1 + (indices * 16))] = pack1_temp;
          }
        }
        ap_int<32> mutate1;
        for (ap_int<32> row2 = 0; row2 < 16; ++row2) {
          for (ap_int<32> col7 = 0; col7 < 16; ++col7) {
            wgt_mem[((col7 + (row2 * 16)) + ((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y7))) + x7) + x_pad_0) * 256))] = pack1[(col7 + (row2 * 16))];
          }
        }
      }
    }
    ap_int<32> pad_right1;
    for (ap_int<32> y8 = 0; y8 < y_size; ++y8) {
      for (ap_int<32> x8 = 0; x8 < x_pad_1; ++x8) {
        ap_int<32> pad_clear6;
        for (ap_int<32> row3 = 0; row3 < 16; ++row3) {
          for (ap_int<32> col8 = 0; col8 < 16; ++col8) {
            wgt_mem[((col8 + (row3 * 16)) + (((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y8))) + x8) + x_pad_0) + x_size) * 256))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
          }
        }
      }
    }
    ap_int<32> pad_bottom1;
    for (ap_int<32> y9 = 0; y9 < y_pad_1; ++y9) {
      for (ap_int<32> x9 = 0; x9 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x9) {
        ap_int<32> pad_clear7;
        for (ap_int<32> row4 = 0; row4 < 16; ++row4) {
          for (ap_int<32> col9 = 0; col9 < 16; ++col9) {
            wgt_mem[((col9 + (row4 * 16)) + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * ((y_pad_0 + y_size) + y9))) + x9) * 256))] = ((ap_uint<8>)((is_pad_min_value == 1) ? 128 : 0));
          }
        }
      }
    }
  }
  if (((ap_int<128>)raw_insn(6, 6)) == (ap_int<128>)1) {
   // ap_int<32> update0;
    //l2g_dep_queue[0] = (ap_uint<1>)1;
    l2g_dep_queue.write(1);
  }
}


void store(ap_uint<8>* output, hls::stream<ap_uint<128>>& store_queue, hls::stream<bool> & g2s_dep_queue, hls::stream<bool> & s2g_dep_queue, ap_uint<8>* out_mem) {
    //g2s_dep_queue[0] = (ap_uint<1>)0;
  //s2g_dep_queue[0] = (ap_uint<1>)0;
  //store_queue[0] = (ap_uint<128>)0;
  ap_int<32> _top;
  ap_uint<128> raw_insn;
  //raw_insn = store_queue[0];
  raw_insn = store_queue.read();

  ap_int<32> sram_base;
  sram_base = ((ap_int<32>)raw_insn(24, 9));
  ap_int<32> dram_base;
  dram_base = ((ap_int<32>)raw_insn(56, 25));
  ap_int<32> y_size;
  y_size = ((ap_int<32>)raw_insn(79, 64));
  ap_int<32> x_size;
  x_size = ((ap_int<32>)raw_insn(95, 80));
  ap_int<32> x_stride;
  x_stride = ((ap_int<32>)raw_insn(111, 96));
  if (((ap_int<128>)raw_insn(3, 3)) == (ap_int<128>)1) {
    //ap_int<32> scalar0;
    //scalar0 = ((ap_int<32>)g2s_dep_queue[0]);
    g2s_dep_queue.read();
  }
  if (x_size == 0) {
  } else {
    ap_int<32> store;
    ap_int<32> store_data;
    for (ap_int<32> y = 0; y < y_size; ++y) {
      for (ap_int<32> x = 0; x < x_size; ++x) {
        ap_int<32> mutate0;
        for (ap_int<32> col = 0; col < 16; ++col) {
          output[((((dram_base + (y * x_stride)) + x) * 16) + col)] = out_mem[(col + (((sram_base + (y * x_size)) + x) * 16))];
        }
      }
    }
  }
  if (((ap_int<128>)raw_insn(5, 5)) == (ap_int<128>)1) {
    //ap_int<32> update0;
    //s2g_dep_queue[0] = (ap_uint<1>)1;
    s2g_dep_queue.write(1);
  }
}

  ap_uint<32> acc_mem[2097152];
  ap_uint<32> uop_mem[32768];
void compute(ap_uint<1> &done, ap_uint<8>* uops, ap_uint<8>* biases, hls::stream<ap_uint<128>>& gemm_queue, hls::stream<bool> & l2g_dep_queue, hls::stream<bool> & s2g_dep_queue, 
hls::stream<bool> & g2l_dep_queue, hls::stream<bool> & g2s_dep_queue, ap_uint<8>* inp_mem, ap_uint<8>* wgt_mem, ap_uint<8>* out_mem) {
//                                                                         s2g_dep_queue[0] = (ap_uint<1>)0;
//   g2l_dep_queue[0] = (ap_uint<1>)0;
//   l2g_dep_queue[0] = (ap_uint<1>)0;
//   g2s_dep_queue[0] = (ap_uint<1>)0;
//   gemm_queue[0] = (ap_uint<128>)0;
  ap_int<32> _top;

/*   for (ap_int<32> x = 0; x < 32768; ++x) {
    uop_mem[x] = 0U;
  } */

//   for (ap_int<32> x1 = 0; x1 < 131072; ++x1) {
//     for (ap_int<32> args1 = 0; args1 < 16; ++args1) {
//       acc_mem[(args1 + (x1 * 16))] = 0U;
//     }
//   }
  ap_uint<128> raw_insn;
  //raw_insn = gemm_queue[0];
  raw_insn = gemm_queue.read();

  ap_int<32> stage0;
  if (((ap_int<128>)raw_insn(3, 3)) == (ap_int<128>)1) {
    // ap_int<32> scalar0;
    // scalar0 = ((ap_int<32>)l2g_dep_queue[0]);
    l2g_dep_queue.read();
  }
  ap_int<32> stage1;
  if (((ap_int<128>)raw_insn(4, 4)) == (ap_int<128>)1) {
    // ap_int<32> scalar1;
    // scalar1 = ((ap_int<32>)s2g_dep_queue[0]);
    s2g_dep_queue.read();
  }
  ap_int<32> done1;
  done = (ap_uint<1>)0;
  ap_int<32> opcode;
  opcode = ((ap_int<32>)raw_insn(2, 0));
  ap_int<32> compute2;
  if (opcode == 3) {
    ap_int<32> done2;
    done = (ap_uint<1>)1;
  } else {
    if (opcode == 0) {
      ap_int<32> memory_type;
      memory_type = ((ap_int<32>)raw_insn(8, 7));
      ap_int<32> sram_base;
      sram_base = ((ap_int<32>)raw_insn(24, 9));
      ap_int<32> dram_base;
      dram_base = ((ap_int<32>)raw_insn(56, 25));
      ap_int<32> is_pad_min_value;
      is_pad_min_value = ((ap_int<32>)raw_insn(57, 57));
      ap_int<32> y_size;
      y_size = ((ap_int<32>)raw_insn(79, 64));
      ap_int<32> x_size;
      x_size = ((ap_int<32>)raw_insn(95, 80));
      ap_int<32> x_stride;
      x_stride = ((ap_int<32>)raw_insn(111, 96));
      ap_int<32> y_pad_0;
      y_pad_0 = ((ap_int<32>)raw_insn(115, 112));
      ap_int<32> y_pad_1;
      y_pad_1 = ((ap_int<32>)raw_insn(119, 116));
      ap_int<32> x_pad_0;
      x_pad_0 = ((ap_int<32>)raw_insn(123, 120));
      ap_int<32> x_pad_1;
      x_pad_1 = ((ap_int<32>)raw_insn(127, 124));
      ap_int<32> stage2;
      if (x_size == 0) {
      } else {
        if (memory_type == 0) {
          ap_int<32> load_uop;
          for (ap_int<32> x2 = 0; x2 < x_size; ++x2) {
            ap_uint<8> burst[4];
            for (ap_int<32> i = 0; i < 4; ++i) {
              burst[i] = uops[(((dram_base + x2) * 4) + i)];
            }
            ap_uint<32> uop;
            ap_uint<32> uop_temp;
            uop_temp = 0U;
            for (ap_int<32> i1 = 0; i1 < 4; ++i1) {
              uop_temp(((i1 * 8) + 7), (i1 * 8)) = burst[i1];
            }
            uop = uop_temp;
            uop_mem[(sram_base + x2)] = uop;
          }
        } else {
          if (memory_type == 3) {
            ap_int<32> load_2d;
            ap_int<32> pad_top;
            for (ap_int<32> y = 0; y < y_pad_0; ++y) {
              for (ap_int<32> x3 = 0; x3 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x3) {
                ap_int<32> pad_clear;
                for (ap_int<32> col = 0; col < 16; ++col) {
                  acc_mem[(col + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * y)) + x3) * 16))] = ((ap_uint<32>)((is_pad_min_value == 1) ? -2147483648 : 0));
                }
              }
            }
            ap_int<32> pad_left;
            for (ap_int<32> y1 = 0; y1 < y_size; ++y1) {
              for (ap_int<32> x4 = 0; x4 < x_pad_0; ++x4) {
                ap_int<32> pad_clear1;
                for (ap_int<32> col1 = 0; col1 < 16; ++col1) {
                  acc_mem[(col1 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y1))) + x4) * 16))] = ((ap_uint<32>)((is_pad_min_value == 1) ? -2147483648 : 0));
                }
              }
            }
            ap_int<32> load_data;
            for (ap_int<32> y2 = 0; y2 < y_size; ++y2) {
              for (ap_int<32> x5 = 0; x5 < x_size; ++x5) {
                ap_uint<8> compute0[64];
                for (ap_int<32> c = 0; c < 64; ++c) {
                  compute0[c] = biases[((((dram_base + (y2 * x_stride)) + x5) * 64) + c)];
                }
                ap_uint<32> pack0[16];
                for (ap_int<32> temp = 0; temp < 16; ++temp) {
                  ap_uint<32> pack0_temp;
                  pack0_temp = 0U;
                  for (ap_int<32> i2 = 0; i2 < 4; ++i2) {
                    pack0_temp(((i2 * 8) + 7), (i2 * 8)) = compute0[((temp * 4) + i2)];
                  }
                  pack0[temp] = pack0_temp;
                }
                ap_int<32> mutate0;
                for (ap_int<32> col2 = 0; col2 < 16; ++col2) {
                  acc_mem[(col2 + ((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y2))) + x5) + x_pad_0) * 16))] = pack0[col2];
                }
              }
            }
            ap_int<32> pad_right;
            for (ap_int<32> y3 = 0; y3 < y_size; ++y3) {
              for (ap_int<32> x6 = 0; x6 < x_pad_1; ++x6) {
                ap_int<32> pad_clear2;
                for (ap_int<32> col3 = 0; col3 < 16; ++col3) {
                  acc_mem[(col3 + (((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y3))) + x6) + x_pad_0) + x_size) * 16))] = ((ap_uint<32>)((is_pad_min_value == 1) ? -2147483648 : 0));
                }
              }
            }
            ap_int<32> pad_bottom;
            for (ap_int<32> y4 = 0; y4 < y_pad_1; ++y4) {
              for (ap_int<32> x7 = 0; x7 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x7) {
                ap_int<32> pad_clear3;
                for (ap_int<32> col4 = 0; col4 < 16; ++col4) {
                  acc_mem[(col4 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * ((y_pad_0 + y_size) + y4))) + x7) * 16))] = ((ap_uint<32>)((is_pad_min_value == 1) ? -2147483648 : 0));
                }
              }
            }
          }
        }
      }
    } else {
      if (opcode == 2) {
       ap_int<32> gemm;
        ap_int<32> reset_reg;
        reset_reg = ((ap_int<32>)raw_insn(7, 7));
        ap_int<32> uop_bgn;
        uop_bgn = ((ap_int<32>)raw_insn(20, 8));
        ap_int<32> uop_end;
        uop_end = ((ap_int<32>)raw_insn(34, 21));
        ap_int<32> iter_out;
        iter_out = ((ap_int<32>)raw_insn(48, 35));
        ap_int<32> iter_in;
        iter_in = ((ap_int<32>)raw_insn(62, 49));
        ap_int<32> dst_factor_out;
        dst_factor_out = ((ap_int<32>)raw_insn(74, 64));
        ap_int<32> dst_factor_in;
        dst_factor_in = ((ap_int<32>)raw_insn(85, 75));
        ap_int<32> src_factor_out;
        src_factor_out = ((ap_int<32>)raw_insn(96, 86));
        ap_int<32> src_factor_in;
        src_factor_in = ((ap_int<32>)raw_insn(107, 97));
        ap_int<32> wgt_factor_out;
        wgt_factor_out = ((ap_int<32>)raw_insn(117, 108));
        ap_int<32> wgt_factor_in;
        wgt_factor_in = ((ap_int<32>)raw_insn(127, 118));
        ap_int<32> gemm_core;
        if (reset_reg == 1) {
          ap_int<32> mutate1;
          for (ap_int<32> i3 = 0; i3 < ((ap_uint<32>)iter_out); ++i3) {
            for (ap_int<32> j = 0; j < ((ap_uint<32>)iter_in); ++j) {
              for (ap_int<32> k = 0; k < ((ap_uint<32>)(uop_end - uop_bgn)); ++k) {
                ap_uint<16> scalar2;
                scalar2 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](10, 0));
                ap_uint<16> scalar3;
                scalar3 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](21, 11));
                ap_uint<16> scalar4;
                scalar4 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](31, 22));
                ap_int<32> mutate2;
                for (ap_int<32> col5 = 0; col5 < 16; ++col5) {
                  acc_mem[(col5 + ((((ap_int<32>)scalar2) + ((j * dst_factor_in) + (i3 * dst_factor_out))) * 16))] = 0U;
                }
              }
            }
          }
        } else {
          ap_int<32> mutate3;
          for (ap_int<32> i4 = 0; i4 < ((ap_uint<32>)iter_out); ++i4) {
            for (ap_int<32> j1 = 0; j1 < ((ap_uint<32>)iter_in); ++j1) {
              for (ap_int<32> k1 = 0; k1 < ((ap_uint<32>)(uop_end - uop_bgn)); ++k1) {
                ap_uint<16> scalar5;
                scalar5 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn)](10, 0));
                ap_uint<16> scalar6;
                scalar6 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn)](21, 11));
                ap_uint<16> scalar7;
                scalar7 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn)](31, 22));
                ap_int<32> multiply_signed[16];
                for (ap_int<32> c1 = 0; c1 < 16; ++c1) {
                  ap_int<32> dot;
                  dot = 0;
                  for (ap_int<32> m = 0; m < 16; ++m) {
                    dot = ((ap_int<32>)(((ap_int<33>)(((ap_int<16>)((ap_int<8>)inp_mem[(m + ((((ap_int<32>)scalar6) + ((j1 * src_factor_in) + (i4 * src_factor_out))) * 16))])) * ((ap_int<16>)((ap_int<8>)wgt_mem[((m + (c1 * 16)) + ((((ap_int<32>)scalar7) + ((j1 * wgt_factor_in) + (i4 * wgt_factor_out))) * 256))])))) + ((ap_int<33>)dot)));
                  }
                  multiply_signed[c1] = dot;
                }
                ap_int<32> mutate4;
                for (ap_int<32> col6 = 0; col6 < 16; ++col6) {
                  acc_mem[(col6 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 16))] = ((ap_uint<32>)(((ap_int<33>)((ap_int<32>)acc_mem[(col6 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 16))])) + ((ap_int<33>)multiply_signed[col6])));
                }
                ap_int<32> mutate5;
                for (ap_int<32> col7 = 0; col7 < 16; ++col7) {
                  out_mem[(col7 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 16))] = ((ap_uint<8>)acc_mem[(col7 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 16))]);
                }
              }
            }
          }
        }

/*             ap_int<32> gemm;
            ap_int<32> reset_reg;
            reset_reg = ((ap_int<32>)raw_insn(7, 7));
            ap_int<32> uop_bgn1;
            uop_bgn1 = ((ap_int<32>)raw_insn(20, 8));
            ap_int<32> uop_end1;
            uop_end1 = ((ap_int<32>)raw_insn(34, 21));
            ap_int<32> iter_out1;
            iter_out1 = ((ap_int<32>)raw_insn(48, 35));
            ap_int<32> iter_in1;
            iter_in1 = ((ap_int<32>)raw_insn(62, 49));
            ap_int<32> dst_factor_out1;
            dst_factor_out1 = ((ap_int<32>)raw_insn(74, 64));
            ap_int<32> dst_factor_in1;
            dst_factor_in1 = ((ap_int<32>)raw_insn(85, 75));
            ap_int<32> src_factor_out1;
            src_factor_out1 = ((ap_int<32>)raw_insn(96, 86));
            ap_int<32> src_factor_in1;
            src_factor_in1 = ((ap_int<32>)raw_insn(107, 97));
            ap_int<32> wgt_factor_out;
            wgt_factor_out = ((ap_int<32>)raw_insn(117, 108));
            ap_int<32> wgt_factor_in;
            wgt_factor_in = ((ap_int<32>)raw_insn(127, 118));
            ap_int<32> gemm_core;
            if (reset_reg == 1) {
              ap_int<32> mutate6;
              for (ap_int<32> i7 = 0; i7 < ((ap_uint<32>)iter_out1); ++i7) {
                for (ap_int<32> j1 = 0; j1 < ((ap_uint<32>)iter_in1); ++j1) {
                  for (ap_int<32> k1 = 0; k1 < ((ap_uint<32>)(uop_end1 - uop_bgn1)); ++k1) {
                    ap_uint<16> scalar3;
                    scalar3 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn1)](10, 0));
                    ap_uint<16> scalar4;
                    scalar4 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn1)](21, 11));
                    ap_uint<16> scalar5;
                    scalar5 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn1)](31, 22));
                    ap_int<32> mutate7;
                    for (ap_int<32> col17 = 0; col17 < 16; ++col17) {
                      acc_mem[(((ap_uint<32>)col17) + ((((ap_uint<32>)scalar3) + ((ap_uint<32>)((j1 * dst_factor_in1) + (i7 * dst_factor_out1)))) * 16U))] = 0U;
                    }
                  }
                }
              }
            } else {
              ap_int<32> mutate8;
              for (ap_int<32> i8 = 0; i8 < ((ap_uint<32>)iter_out1); ++i8) {
                for (ap_int<32> j2 = 0; j2 < ((ap_uint<32>)iter_in1); ++j2) {
                  for (ap_int<32> k2 = 0; k2 < ((ap_uint<32>)(uop_end1 - uop_bgn1)); ++k2) {
                    ap_uint<16> scalar6;
                    scalar6 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](10, 0));s
                    ap_uint<16> scalar7;
                    scalar7 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](21, 11));
                    ap_uint<16> scalar8;
                    scalar8 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](31, 22));
                    ap_int<32> multiply_signed[16];
                    for (ap_int<32> c3 = 0; c3 < 16; ++c3) {
                      ap_int<32> dot;
                      dot = 0;
                      for (ap_int<32> m = 0; m < 16; ++m) {
                        dot = ((ap_int<32>)(((ap_int<33>)(((ap_int<16>)((ap_int<8>)inp_mem[(((ap_uint<32>)m) + ((((ap_uint<32>)scalar7) + ((ap_uint<32>)((j2 * src_factor_in1) + (i8 * src_factor_out1)))) * 16U))])) * ((ap_int<16>)((ap_int<8>)wgt_mem[(((ap_uint<32>)(m + (c3 * 16))) + ((((ap_uint<32>)scalar8) + ((ap_uint<32>)((j2 * wgt_factor_in) + (i8 * wgt_factor_out)))) * 256U))])))) + ((ap_int<33>)dot)));
                      }
                      multiply_signed[c3] = dot;
                    }
                    ap_int<32> mutate9;
                    for (ap_int<32> col18 = 0; col18 < 16; ++col18) {
                      acc_mem[(((ap_uint<32>)col18) + ((((ap_uint<32>)scalar6) + ((ap_uint<32>)((j2 * dst_factor_in1) + (i8 * dst_factor_out1)))) * 16U))] = ((ap_uint<32>)(((ap_int<33>)((ap_int<32>)acc_mem[(((ap_uint<32>)col18) + ((((ap_uint<32>)scalar6) + ((ap_uint<32>)((j2 * dst_factor_in1) + (i8 * dst_factor_out1)))) * 16U))])) + ((ap_int<33>)multiply_signed[col18])));
                    }
                    ap_int<32> mutate10;
                    for (ap_int<32> col19 = 0; col19 < 16; ++col19) {
                      out_mem[(((ap_uint<32>)col19) + ((((ap_uint<32>)scalar6) + ((ap_uint<32>)((j2 * dst_factor_in1) + (i8 * dst_factor_out1)))) * 16U))] = ((ap_uint<8>)acc_mem[(((ap_uint<32>)col19) + ((((ap_uint<32>)scalar6) + ((ap_uint<32>)((j2 * dst_factor_in1) + (i8 * dst_factor_out1)))) * 16U))]);
                    }
                  }
                }
              }
            }
 */
        // ap_int<32> gemm;
        // ap_int<32> reset_reg;
        // reset_reg = ((ap_int<32>)raw_insn(7, 7));
        // ap_int<32> uop_bgn;
        // uop_bgn = ((ap_int<32>)raw_insn(20, 8));
        // ap_int<32> uop_end;
        // uop_end = ((ap_int<32>)raw_insn(34, 21));
        // ap_int<32> iter_out;
        // iter_out = ((ap_int<32>)raw_insn(48, 35));
        // ap_int<32> iter_in;
        // iter_in = ((ap_int<32>)raw_insn(62, 49));
        // ap_int<32> dst_factor_out;
        // dst_factor_out = ((ap_int<32>)raw_insn(74, 64));
        // ap_int<32> dst_factor_in;
        // dst_factor_in = ((ap_int<32>)raw_insn(85, 75));
        // ap_int<32> src_factor_out;
        // src_factor_out = ((ap_int<32>)raw_insn(96, 86));
        // ap_int<32> src_factor_in;
        // src_factor_in = ((ap_int<32>)raw_insn(107, 97));
        // ap_int<32> wgt_factor_out;
        // wgt_factor_out = ((ap_int<32>)raw_insn(117, 108));
        // ap_int<32> wgt_factor_in;
        // wgt_factor_in = ((ap_int<32>)raw_insn(127, 118));
        // ap_int<32> gemm_core;
        // if (reset_reg == 1) {
        //   ap_int<32> mutate1;
        //   for (ap_int<32> i3 = 0; i3 < ((ap_uint<32>)iter_out); ++i3) {
        //     for (ap_int<32> j = 0; j < ((ap_uint<32>)iter_in); ++j) {
        //       for (ap_int<32> k = 0; k < ((ap_uint<32>)(uop_end - uop_bgn)); ++k) {
        //         ap_uint<16> scalar2;
        //         scalar2 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](10, 0));
        //         ap_uint<16> scalar3;
        //         scalar3 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](21, 11));
        //         ap_uint<16> scalar4;
        //         scalar4 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](31, 22));
        //         ap_int<32> mutate2;
        //         for (ap_int<32> col5 = 0; col5 < 16; ++col5) {
        //           wgt_mem[(col5 + ((((ap_int<32>)scalar2) + ((j * dst_factor_in) + (i3 * dst_factor_out))) * 256))] = (ap_uint<8>)0;
        //         }
        //       }
        //     }
        //   }
        // } else {
        //   ap_int<32> mutate3;
        //   for (ap_int<32> i4 = 0; i4 < ((ap_uint<32>)iter_out); ++i4) {
        //     for (ap_int<32> j1 = 0; j1 < ((ap_uint<32>)iter_in); ++j1) {
        //       for (ap_int<32> k1 = 0; k1 < ((ap_uint<32>)(uop_end - uop_bgn)); ++k1) {
        //         ap_uint<16> scalar5;
        //         scalar5 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn)](10, 0));
        //         ap_uint<16> scalar6;
        //         scalar6 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn)](21, 11));
        //         ap_uint<16> scalar7;
        //         scalar7 = ((ap_uint<16>)uop_mem[(k1 + uop_bgn)](31, 22));
        //         ap_int<32> multiply_signed[16];
        //         for (ap_int<32> c1 = 0; c1 < 16; ++c1) {
        //           ap_int<32> dot;
        //           dot = 0;
        //           for (ap_int<32> m = 0; m < 16; ++m) {
        //             dot = ((ap_int<32>)(((ap_int<33>)(((ap_int<16>)((ap_int<8>)acc_mem[(m + ((((ap_int<32>)scalar6) + ((j1 * src_factor_in) + (i4 * src_factor_out))) * 16))])) * ((ap_int<16>)((ap_int<8>)inp_mem[((m + (c1 * 16)) + ((((ap_int<32>)scalar7) + ((j1 * wgt_factor_in) + (i4 * wgt_factor_out))) * 16))])))) + ((ap_int<33>)dot)));
        //           }
        //           multiply_signed[c1] = dot;
        //         }
        //         ap_int<32> mutate4;
        //         for (ap_int<32> col6 = 0; col6 < 16; ++col6) {
        //           wgt_mem[(col6 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 256))] = ((ap_uint<8>)(((ap_int<33>)((ap_int<32>)wgt_mem[(col6 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 256))])) + ((ap_int<33>)multiply_signed[col6])));
        //         }
        //         ap_int<32> mutate5;
        //         for (ap_int<32> col7 = 0; col7 < 16; ++col7) {
        //           out_mem[(col7 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 16))] = wgt_mem[(col7 + ((((ap_int<32>)scalar5) + ((j1 * dst_factor_in) + (i4 * dst_factor_out))) * 256))];
        //         }
        //       }
        //     }
        //   }
        // }







      } else {
        if (opcode == 4) {
          ap_int<32> uop_bgn1;
          uop_bgn1 = ((ap_int<32>)raw_insn(20, 8));
          ap_int<32> uop_end1;
          uop_end1 = ((ap_int<32>)raw_insn(34, 21));
          ap_int<32> iter_out1;
          iter_out1 = ((ap_int<32>)raw_insn(48, 35));
          ap_int<32> iter_in1;
          iter_in1 = ((ap_int<32>)raw_insn(62, 49));
          ap_int<32> dst_factor_out1;
          dst_factor_out1 = ((ap_int<32>)raw_insn(74, 64));
          ap_int<32> dst_factor_in1;
          dst_factor_in1 = ((ap_int<32>)raw_insn(85, 75));
          ap_int<32> src_factor_out1;
          src_factor_out1 = ((ap_int<32>)raw_insn(96, 86));
          ap_int<32> src_factor_in1;
          src_factor_in1 = ((ap_int<32>)raw_insn(107, 97));
          ap_int<32> alu_opcode;
          alu_opcode = ((ap_int<32>)raw_insn(109, 108));
          ap_uint<1> use_imm;
          use_imm = ((ap_uint<1>)raw_insn(110, 110));
          ap_int<32> imm;
          imm = ((ap_int<32>)raw_insn(127, 111));
          ap_int<32> alu;
          ap_int<32> mutate6;
          for (ap_int<32> i5 = 0; i5 < ((ap_uint<32>)iter_out1); ++i5) {
            for (ap_int<32> j2 = 0; j2 < ((ap_uint<32>)iter_in1); ++j2) {
              for (ap_int<32> k2 = 0; k2 < ((ap_uint<32>)(uop_end1 - uop_bgn1)); ++k2) {
                ap_uint<16> scalar8;
                scalar8 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](10, 0));
                ap_uint<16> scalar9;
                scalar9 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](21, 11));
                ap_uint<16> scalar10;
                scalar10 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](31, 22));
                for (ap_int<32> i6 = 0; i6 < 16; ++i6) {
                  if (alu_opcode == 0) {
                    acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))] = ((ap_uint<32>)((((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])) < ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))])) ? ((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))])));
                  } else {
                    if (alu_opcode == 1) {
                      acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))] = ((ap_uint<32>)((((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))]) < ((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))]))) ? ((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))])));
                    } else {
                      if (alu_opcode == 2) {
                        acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))] = ((ap_uint<32>)(((ap_int<33>)((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))])) + ((ap_int<33>)((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])))));
                      } else {
                        if (alu_opcode == 3) {
                          acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))] = ((ap_uint<32>)((((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])) < 0) ? (((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))]) << (((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])) * -1)) : (((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))]) >> ((((ap_int<32>)use_imm) == 1) ? ((ap_int<32>)((ap_int<16>)imm)) : ((ap_int<32>)acc_mem[(i6 + ((((ap_int<32>)scalar9) + ((j2 * src_factor_in1) + (i5 * src_factor_out1))) * 16))])))));
                        }
                      }
                    }
                  }
                }
                ap_int<32> mutate7;
                for (ap_int<32> col8 = 0; col8 < 16; ++col8) {
                  out_mem[(col8 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))] = ((ap_uint<8>)acc_mem[(col8 + ((((ap_int<32>)scalar8) + ((j2 * dst_factor_in1) + (i5 * dst_factor_out1))) * 16))]);
                }
              }
            }
          }
        }
      }
    }
  }
  ap_int<32> stage3;
  if (((ap_int<128>)raw_insn(5, 5)) == (ap_int<128>)1) {
    //ap_int<32> update0;
    //g2l_dep_queue[0] = (ap_uint<1>)1;
    g2l_dep_queue.write(1);
  }
  ap_int<32> stage4;
  if (((ap_int<128>)raw_insn(6, 6)) == (ap_int<128>)1) {
    //ap_int<32> update1;
    //g2s_dep_queue[0] = (ap_uint<1>)1;
    g2s_dep_queue.write(1);
  }
}

    ap_uint<8> inp_mem[32768 * 1 * 16];
    ap_uint<8> wgt_mem[262144 * 16 * 16];
    ap_uint<8> out_mem[32768 * 1 * 16];

//
// Currently this part is directly from vta.cc
void vta_bsim_v1(
  uint32_t instr_count,
  ap_uint<128>* insns,
  ap_uint<8>* uops,
  ap_uint<8>* inputs,
  ap_uint<8>* weights,
  ap_uint<8>* biases,
  ap_uint<8>* outputs) {

    ap_int<32> instr_count1;
  instr_count1 = ((ap_int<32>)instr_count);

  // Instantiate temporary instruction queues (used for peeking)
  hls::stream<ap_uint<128>> tmp_load_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_load_queue)
  hls::stream<ap_uint<128>> tmp_gemm_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_gemm_queue)
  hls::stream<ap_uint<128>> tmp_store_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_store_queue)

  // Instatiate physical instruction queues
  hls::stream<ap_uint<128>> load_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=load_queue)
  hls::stream<ap_uint<128>> gemm_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=gemm_queue)
  hls::stream<ap_uint<128>> store_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=store_queue)

  // Dependence queues
  hls::stream<bool> l2g_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=l2g_dep_queue)
  hls::stream<bool> s2g_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=s2g_dep_queue)
  hls::stream<bool> g2l_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=g2l_dep_queue)
  hls::stream<bool> g2s_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=g2s_dep_queue)

  // Instantiate memories
//   bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO];
//   bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO];
//   bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO];


  // Push all instructions into the queues
  fetch(instr_count1, insns, tmp_load_queue, tmp_gemm_queue, tmp_store_queue);

  // Global done indicator
  ap_uint<1> done = 0;

  // Temporary instructions
  ap_uint<128> tmp_load;
  ap_uint<128> tmp_gemv;
  ap_uint<128> tmp_store;

  // Peeking status
  bool tmp_load_popped = false;
  bool tmp_gemm_popped = false;
  bool tmp_store_popped = false;
  int exit_counter = 0;

  // Main control loop
  while (true) {
    // First execute as many load instructions as possible
    while (!tmp_load_queue.empty() || tmp_load_popped == true) {
      // Pop the load instruction
      if (!tmp_load_popped) {
        tmp_load_queue.read(tmp_load);
        tmp_load_popped = true;
      }
      // Check dependences and invoke the load stage
      VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_load);
      if ((insn.pop_next_dep && !g2l_dep_queue.empty()) ||
          !insn.pop_next_dep) {
        // Push the instruction in the load queue
        load_queue.write(tmp_load);
        tmp_load_popped = false;
        load(inputs, weights, load_queue, g2l_dep_queue, l2g_dep_queue, inp_mem, wgt_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Next execute as many gemm instructions as possible
    while (!tmp_gemm_queue.empty() || tmp_gemm_popped == true) {
      // Pop the gemm instruction
      if (!tmp_gemm_popped) {
        tmp_gemm_queue.read(tmp_gemv);
        tmp_gemm_popped = true;
      }
      // Check dependences and invoke the load stage
      VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_gemv);
      if (
        (insn.pop_prev_dep && !l2g_dep_queue.empty() &&
         insn.pop_next_dep && !s2g_dep_queue.empty()) ||
        (!insn.pop_prev_dep && insn.pop_next_dep &&
         !s2g_dep_queue.empty()) ||
        (insn.pop_prev_dep && !l2g_dep_queue.empty() &&
        !insn.pop_next_dep) ||
        (!insn.pop_prev_dep && !insn.pop_next_dep)
      ) {
        // Push the instruction in the load queue
        gemm_queue.write(tmp_gemv);
        tmp_gemm_popped = false;
        compute(done, uops, biases, gemm_queue, l2g_dep_queue, s2g_dep_queue,
                g2l_dep_queue, g2s_dep_queue, inp_mem, wgt_mem, out_mem);
      } else {
        // Execution of load stage pending on completion of other stages,
        // so break here...
        break;
      }
    }
    // Finally execute as many store instructions as possible
    while (!tmp_store_queue.empty() || tmp_store_popped == true) {
      // Pop the load instruction
      if (!tmp_store_popped) {
        tmp_store_queue.read(tmp_store);
        tmp_store_popped = true;
      }
      // Check dependences and invoke the load stage
      VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_store);

      if ((insn.pop_prev_dep && !g2s_dep_queue.empty()) ||
          !insn.pop_prev_dep) {
        // Push the instruction in the load queue
        store_queue.write(tmp_store);
        tmp_store_popped = false;
        store(outputs, store_queue, g2s_dep_queue, s2g_dep_queue, out_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Check if we get a signal that we are done
    if (done) {
      break;
    }
    exit_counter++;
    if (exit_counter > 1000) {
      if (tmp_load_popped) {
        if (g2l_dep_queue.empty()) {
          printf("waiting on g2l\n");
        }
      }
      if (tmp_gemm_popped) {
        VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_gemv);
        if (l2g_dep_queue.empty() && insn.pop_prev_dep) {
          printf("waiting on l2g\n");
        }
        if (s2g_dep_queue.empty() && insn.pop_next_dep) {
          printf("waiting on s2g\n");
        }
      }
      if (tmp_store_popped) {
        if (g2s_dep_queue.empty()) {
          printf("waiting on g2s\n");
        }
      }
      break;
    }
  }

  // Ensure that the tokens are empty
  bool tmp_tok;
  int l2g_count = 0;
  int s2g_count = 0;
  int g2l_count = 0;
  int g2s_count = 0;
  while (l2g_dep_queue.read_nb(tmp_tok)) {
    l2g_count++;
  }
  while (s2g_dep_queue.read_nb(tmp_tok)) {
    s2g_count++;
  }
  while (g2l_dep_queue.read_nb(tmp_tok)) {
    g2l_count++;
  }
  while (g2s_dep_queue.read_nb(tmp_tok)) {
    g2s_count++;
  }

  assert(l2g_count == 0 && s2g_count == 0 && g2l_count == 0 && g2s_count == 0);
}
