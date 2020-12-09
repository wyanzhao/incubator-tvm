#include "vta_bsim.h"
#include <stdlib.h>
#include <stdio.h>
#include <ap_utils.h>
#include <ap_int.h>
#include <cstring>


void vta_bsim(
		uint32_t instr_count,
		volatile 	ap_uint<8> *insns,
		volatile 	ap_uint<8> *uops,
		volatile	ap_uint<8> *inputs,
		volatile	ap_uint<8> *weights,
		volatile	ap_uint<8> *biases,
		volatile	ap_uint<8> *outputs,
		volatile 	uint32_t &done
)
{
#pragma HLS INTERFACE m_axi depth=3616 port=insns offset=slave bundle=ins_port
#pragma HLS INTERFACE m_axi depth=65536 port=weights offset=slave bundle=data
#pragma HLS INTERFACE m_axi depth=65536 port=outputs offset=slave bundle=data
#pragma HLS INTERFACE m_axi depth=8192 port=uops offset=slave bundle=uop_port
#pragma HLS INTERFACE m_axi depth=262144 port=biases offset=slave bundle=biases
#pragma HLS INTERFACE m_axi depth=65536 port=inputs offset=slave bundle=data
#pragma HLS INTERFACE s_axilite port=done
#pragma HLS INTERFACE s_axilite port=instr_count
#pragma HLS INTERFACE s_axilite port=return

	static ap_uint<32> uop_mem[8192 * 4];
	static ap_uint<8> inp_mem[2048 * 16];
	static ap_uint<8> wgt_mem[1024 * 16 * 16];
	static ap_uint<32> acc_mem[2048 * 4 * 4];
	static ap_uint<8> out_mem[2048 * 16];

  ap_int<32> _top;
  ap_int<32> S;

  ap_int<32> instr_count1;
  instr_count1 = ((ap_int<32>)instr_count);
  done = 0;

  for (int32_t i = 0; i < instr_count1; ++i) {
	  done = 0;
    ap_uint<8> insn_bytes[16];
    memcpy(insn_bytes, (const ap_uint<8> *)insns + i * 16, 16);
    ap_uint<128> insn;
    ap_uint<128> insn_temp;
    insn_temp = (ap_uint<128>)0;
    for (ap_int<32> i1 = 0; i1 < 16; ++i1) {
      insn_temp(((i1 * 8) + 7), (i1 * 8)) = insn_bytes[i1];
    }
    insn = insn_temp;
    ap_int<32> opcode;
    opcode = ((ap_int<32>)insn(2, 0));
    ap_int<32> S1;
    if (opcode == 0) {
      ap_int<32> memory_type;
      memory_type = ((ap_int<32>)insn(8, 7));
      ap_int<32> sram_base;
      sram_base = ((ap_int<32>)insn(24, 9));
      ap_int<32> dram_base;
      dram_base = ((ap_int<32>)insn(56, 25));
      ap_int<32> y_size;
      y_size = ((ap_int<32>)insn(79, 64));
      ap_int<32> x_size;
      x_size = ((ap_int<32>)insn(95, 80));
      ap_int<32> x_stride;
      x_stride = ((ap_int<32>)insn(111, 96));
      ap_int<32> y_pad_0;
      y_pad_0 = ((ap_int<32>)insn(115, 112));
      ap_int<32> y_pad_1;
      y_pad_1 = ((ap_int<32>)insn(119, 116));
      ap_int<32> x_pad_0;
      x_pad_0 = ((ap_int<32>)insn(123, 120));
      ap_int<32> x_pad_1;
      x_pad_1 = ((ap_int<32>)insn(127, 124));
      ap_int<32> load;
      if (x_size == 0) {
      } else {
        if (memory_type == 0) {
          ap_int<32> load_uop;
          for (ap_int<32> x1 = 0; x1 < x_size; ++x1) {
            ap_uint<8> burst[4];
            memcpy(burst, (const ap_uint<8> *)uops + ((dram_base + x1) * 4), 4);

            ap_uint<32> uop;
            ap_uint<32> uop_temp;
            uop_temp = 0U;
            for (ap_int<32> i3 = 0; i3 < 4; ++i3) {
              uop_temp(((i3 * 8) + 7), (i3 * 8)) = burst[i3];
            }
            uop = uop_temp;
            uop_mem[(sram_base + x1)] = uop;
          }
        } else {
          if (memory_type == 1) {
            ap_int<32> load_2d;
            ap_int<32> pad_top;
            for (ap_int<32> y = 0; y < y_pad_0; ++y) {
              for (ap_int<32> x2 = 0; x2 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x2) {
                ap_int<32> pad_clear;
                for (ap_int<32> row = 0; row < 16; ++row) {
                  for (ap_int<32> col = 0; col < 16; ++col) {
                    wgt_mem[((col + (row * 16)) + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * y)) + x2) * 256))] = (ap_uint<8>)0;
                  }
                }
              }
            }
            ap_int<32> pad_left;
            for (ap_int<32> y1 = 0; y1 < y_size; ++y1) {
              for (ap_int<32> x3 = 0; x3 < x_pad_0; ++x3) {
                ap_int<32> pad_clear1;
                for (ap_int<32> row1 = 0; row1 < 16; ++row1) {
                  for (ap_int<32> col1 = 0; col1 < 16; ++col1) {
                    wgt_mem[((col1 + (row1 * 16)) + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y1))) + x3) * 256))] = (ap_uint<8>)0;
                  }
                }
              }
            }
            ap_int<32> load_data;
            for (ap_int<32> y2 = 0; y2 < y_size; ++y2) {
              for (ap_int<32> x4 = 0; x4 < x_size; ++x4) {
                ap_uint<8> compute0[256];
                for (ap_int<32> r = 0; r < 16; ++r) {
                memcpy(compute0 + (r * 16), (const ap_uint<8> *) weights + (((((dram_base + (y2 * x_stride)) + x4) * 16) + r) * 16), 16);
                }

                ap_uint<8> pack0[256];
                for (ap_int<32> indices = 0; indices < 16; ++indices) {
                  for (ap_int<32> temp = 0; temp < 16; ++temp) {
                    ap_uint<8> pack0_temp;
                    pack0_temp = (ap_uint<8>)0;
                    pack0_temp(7, 0) = compute0[(temp + (indices * 16))];
                    pack0[(temp + (indices * 16))] = pack0_temp;
                  }
                }
                ap_int<32> mutate0;
                for (ap_int<32> row2 = 0; row2 < 16; ++row2) {
                  for (ap_int<32> col2 = 0; col2 < 16; ++col2) {
                    wgt_mem[((col2 + (row2 * 16)) + ((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y2))) + x4) + x_pad_0) * 256))] = pack0[(col2 + (row2 * 16))];
                  }
                }
              }
            }
            ap_int<32> pad_right;
            for (ap_int<32> y3 = 0; y3 < y_size; ++y3) {
              for (ap_int<32> x5 = 0; x5 < x_pad_1; ++x5) {
                ap_int<32> pad_clear2;
                for (ap_int<32> row3 = 0; row3 < 16; ++row3) {
                  for (ap_int<32> col3 = 0; col3 < 16; ++col3) {
                    wgt_mem[((col3 + (row3 * 16)) + (((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y3))) + x5) + x_pad_0) + x_size) * 256))] = (ap_uint<8>)0;
                  }
                }
              }
            }
            ap_int<32> pad_bottom;
            for (ap_int<32> y4 = 0; y4 < y_pad_1; ++y4) {
              for (ap_int<32> x6 = 0; x6 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x6) {
                ap_int<32> pad_clear3;
                for (ap_int<32> row4 = 0; row4 < 16; ++row4) {
                  for (ap_int<32> col4 = 0; col4 < 16; ++col4) {
                    wgt_mem[((col4 + (row4 * 16)) + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * ((y_pad_0 + y_size) + y4))) + x6) * 256))] = (ap_uint<8>)0;
                  }
                }
              }
            }
          } else {
            if (memory_type == 3) {
              uint32_t base_addr = 0;
            {
              ap_int<32> load_2d1;
              ap_int<32> pad_top1;
              for (ap_int<32> y5 = 0; y5 < y_pad_0; ++y5) {
                for (ap_int<32> x7 = 0; x7 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x7) {
                  ap_int<32> pad_clear4;
                  for (ap_int<32> col5 = 0; col5 < 16; ++col5) {
                    acc_mem[(col5 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * y5)) + x7) * 16))] = 0U;
                  }
                }
              }
            }
            {
              ap_int<32> pad_left1;
              for (ap_int<32> y6 = 0; y6 < y_size; ++y6) {
                for (ap_int<32> x8 = 0; x8 < x_pad_0; ++x8) {
                  ap_int<32> pad_clear5;
                  for (ap_int<32> col6 = 0; col6 < 16; ++col6) {
                    acc_mem[(col6 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y6))) + x8) * 16))] = 0U;
                  }
                }
              }
            }
            {
                ap_int<32> load_data1;
              for (ap_int<32> y7 = 0; y7 < y_size; ++y7) {
                for (ap_int<32> x9 = 0; x9 < x_size; ++x9) {
                  ap_uint<8> compute1[64];
                  memcpy(compute1, (const ap_uint<8> *)biases + (((dram_base + (y7 * x_stride)) + x9) * 64), 64);

                  ap_uint<32> pack1[16];
                  for (ap_int<32> temp1 = 0; temp1 < 16; ++temp1) {
                    ap_uint<32> pack1_temp;
                    pack1_temp = 0U;
                    for (ap_int<32> i4 = 0; i4 < 4; ++i4) {
                      pack1_temp(((i4 * 8) + 7), (i4 * 8)) = compute1[((temp1 * 4) + i4)];
                    }
                    pack1[temp1] = pack1_temp;
                  }
                  ap_int<32> mutate1;
                  for (ap_int<32> col7 = 0; col7 < 16; ++col7) {
                    acc_mem[(col7 + ((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y7))) + x9) + x_pad_0) * 16))] = pack1[col7];
                  }
                }
              }
           }
            {
              ap_int<32> pad_right1;
              for (ap_int<32> y8 = 0; y8 < y_size; ++y8) {
                for (ap_int<32> x10 = 0; x10 < x_pad_1; ++x10) {
                  ap_int<32> pad_clear6;
                  for (ap_int<32> col8 = 0; col8 < 16; ++col8) {
                    acc_mem[(col8 + (((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y8))) + x10) + x_pad_0) + x_size) * 16))] = 0U;
                  }
                }
              }
            }
            {
              ap_int<32> pad_bottom1;
              for (ap_int<32> y9 = 0; y9 < y_pad_1; ++y9) {
                for (ap_int<32> x11 = 0; x11 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x11) {
                  ap_int<32> pad_clear7;
                  for (ap_int<32> col9 = 0; col9 < 16; ++col9) {
                    acc_mem[(col9 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * ((y_pad_0 + y_size) + y9))) + x11) * 16))] = 0U;
                  }
                }
              }
            }
            base_addr += 100;


            } else {
              if (memory_type == 2) {
                ap_int<32> load_2d2;
                ap_int<32> pad_top2;
                for (ap_int<32> y10 = 0; y10 < y_pad_0; ++y10) {
                  for (ap_int<32> x12 = 0; x12 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x12) {
                    ap_int<32> pad_clear8;
                    for (ap_int<32> col10 = 0; col10 < 16; ++col10) {
                      inp_mem[(col10 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * y10)) + x12) * 16))] = (ap_uint<8>)0;
                    }
                  }
                }
                ap_int<32> pad_left2;
                for (ap_int<32> y11 = 0; y11 < y_size; ++y11) {
                  for (ap_int<32> x13 = 0; x13 < x_pad_0; ++x13) {
                    ap_int<32> pad_clear9;
                    for (ap_int<32> col11 = 0; col11 < 16; ++col11) {
                      inp_mem[(col11 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y11))) + x13) * 16))] = (ap_uint<8>)0;
                    }
                  }
                }
                ap_int<32> load_data2;
                for (ap_int<32> y12 = 0; y12 < y_size; ++y12) {
                  for (ap_int<32> x14 = 0; x14 < x_size; ++x14) {
                    ap_uint<8> compute2[16];

                    memcpy(compute2, (const ap_uint<8>*)inputs + (((dram_base + (y12 * x_stride)) + x14) * 16), 16);


                    ap_uint<8> pack2[16];
                    for (ap_int<32> temp2 = 0; temp2 < 16; ++temp2) {
                      ap_uint<8> pack2_temp;
                      pack2_temp = (ap_uint<8>)0;
                      pack2_temp(7, 0) = compute2[temp2];
                      pack2[temp2] = pack2_temp;
                    }
                    ap_int<32> mutate2;
                    for (ap_int<32> col12 = 0; col12 < 16; ++col12) {
                      inp_mem[(col12 + ((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y12))) + x14) + x_pad_0) * 16))] = pack2[col12];
                    }
                  }
                }
                ap_int<32> pad_right2;
                for (ap_int<32> y13 = 0; y13 < y_size; ++y13) {
                  for (ap_int<32> x15 = 0; x15 < x_pad_1; ++x15) {
                    ap_int<32> pad_clear10;
                    for (ap_int<32> col13 = 0; col13 < 16; ++col13) {
                      inp_mem[(col13 + (((((sram_base + (((x_size + x_pad_0) + x_pad_1) * (y_pad_0 + y13))) + x15) + x_pad_0) + x_size) * 16))] = (ap_uint<8>)0;
                    }
                  }
                }
                ap_int<32> pad_bottom2;
                for (ap_int<32> y14 = 0; y14 < y_pad_1; ++y14) {
                  for (ap_int<32> x16 = 0; x16 < ((ap_uint<32>)(((ap_int<34>)(((ap_int<33>)x_size) + ((ap_int<33>)x_pad_0))) + ((ap_int<34>)x_pad_1))); ++x16) {
                    ap_int<32> pad_clear11;
                    for (ap_int<32> col14 = 0; col14 < 16; ++col14) {
                      inp_mem[(col14 + (((sram_base + (((x_size + x_pad_0) + x_pad_1) * ((y_pad_0 + y_size) + y14))) + x16) * 16))] = (ap_uint<8>)0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if (opcode == 1) {

        ap_int<32> sram_base1;
        sram_base1 = ((ap_int<32>)insn(24, 9));
        ap_int<32> dram_base1;
        dram_base1 = ((ap_int<32>)insn(56, 25));
        ap_int<32> y_size1;
        y_size1 = ((ap_int<32>)insn(79, 64));
        ap_int<32> x_size1;
        x_size1 = ((ap_int<32>)insn(95, 80));
        ap_int<32> x_stride1;
        x_stride1 = ((ap_int<32>)insn(111, 96));
        if (x_size1 == 0) {
        } else {
          ap_int<32> store;
          ap_int<32> store_data;
          for (ap_int<32> y15 = 0; y15 < y_size1; ++y15) {
            for (ap_int<32> x17 = 0; x17 < x_size1; ++x17) {
              ap_int<32> mutate3;
              memcpy((ap_uint<8> *)outputs + (((dram_base1 + (y15 * x_stride1)) + x17) * 16), out_mem + (((sram_base1 + (y15 * x_size1)) + x17) * 16), 16);


            }
          }
        }
      } else {
        if (opcode == 4) {
          ap_int<32> uop_bgn;
          uop_bgn = ((ap_int<32>)insn(20, 8));
          ap_int<32> uop_end;
          uop_end = ((ap_int<32>)insn(34, 21));
          ap_int<32> iter_out;
          iter_out = ((ap_int<32>)insn(48, 35));
          ap_int<32> iter_in;
          iter_in = ((ap_int<32>)insn(62, 49));
          ap_int<32> dst_factor_out;
          dst_factor_out = ((ap_int<32>)insn(74, 64));
          ap_int<32> dst_factor_in;
          dst_factor_in = ((ap_int<32>)insn(85, 75));
          ap_int<32> src_factor_out;
          src_factor_out = ((ap_int<32>)insn(96, 86));
          ap_int<32> src_factor_in;
          src_factor_in = ((ap_int<32>)insn(107, 97));
          ap_int<32> alu_opcode;
          alu_opcode = ((ap_int<32>)insn(109, 108));
          ap_uint<1> use_imm;
          use_imm = ((ap_uint<1>)insn(110, 110));
          ap_int<32> imm;
          imm = ((ap_int<32>)insn(126, 111));
          ap_int<32> alu;
          ap_int<32> mutate4;



          for (ap_int<32> i5 = 0; i5 < ((ap_uint<32>)iter_out); ++i5) {
            for (ap_int<32> j = 0; j < ((ap_uint<32>)iter_in); ++j) {
              for (ap_int<32> k = 0; k < ((ap_uint<32>)(uop_end - uop_bgn)); ++k) {
                ap_uint<16> scalar0;
                scalar0 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](10, 0));
                ap_uint<16> scalar1;
                scalar1 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](21, 11));
                ap_uint<16> scalar2;
                scalar2 = ((ap_uint<16>)uop_mem[(k + uop_bgn)](31, 22));
                {
                    
                for (ap_int<32> i6 = 0; i6 < 16; ++i6) {
                  if (alu_opcode == 0) {
                    acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = ((((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))]))) < ((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))])) ? (((ap_uint<32>)((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))]))))) : ((ap_uint<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))]));
                  } else {
                    if (alu_opcode == 1) {
                      acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = ((((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))]) < ((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))])))) ? (((ap_uint<32>)((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))]))))) : ((ap_uint<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))]));
                    } else {
                      if (alu_opcode == 2) {
                        acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = ((ap_uint<32>)(((ap_int<33>)((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))])) + ((ap_int<33>)((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))]))))));
                      } else {
                        if (alu_opcode == 3) {
                          if (0 <= ((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))])))) {
                            acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = ((ap_uint<32>)(((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))]) >> ((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))])))));
                          } else {
                            acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = ((ap_uint<32>)(((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))]) << (((((ap_uint<32>)use_imm) == 1U) ? (((ap_int<32>)((ap_int<16>)imm))) : (((ap_int<32>)acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar1) + ((ap_uint<32>)((j * src_factor_in) + (i5 * src_factor_out)))) * 16U))]))) * -1)));
                          }
                        } else {
                          acc_mem[(((ap_uint<32>)i6) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = 0U;
                        }
                      }
                    }
                  }
                }
                }
                {
                                  ap_int<32> mutate5;
                for (ap_int<32> col16 = 0; col16 < 16; ++col16) {
                  out_mem[(((ap_uint<32>)col16) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))] = ((ap_uint<8>)acc_mem[(((ap_uint<32>)col16) + ((((ap_uint<32>)scalar0) + ((ap_uint<32>)((j * dst_factor_in) + (i5 * dst_factor_out)))) * 16U))]);
                }
                }
              }
            }
          }
        } else {
          if (opcode == 3) {
        	  // // Set done flag if we reach a FINISH instruction
              	done = 1;
          }

          else {
          if (opcode == 2) {
            ap_int<32> gemm;
            ap_int<32> reset_reg;
            reset_reg = ((ap_int<32>)insn(7, 7));
            ap_int<32> uop_bgn1;
            uop_bgn1 = ((ap_int<32>)insn(20, 8));
            ap_int<32> uop_end1;
            uop_end1 = ((ap_int<32>)insn(34, 21));
            ap_int<32> iter_out1;
            iter_out1 = ((ap_int<32>)insn(48, 35));
            ap_int<32> iter_in1;
            iter_in1 = ((ap_int<32>)insn(62, 49));
            ap_int<32> dst_factor_out1;
            dst_factor_out1 = ((ap_int<32>)insn(74, 64));
            ap_int<32> dst_factor_in1;
            dst_factor_in1 = ((ap_int<32>)insn(85, 75));
            ap_int<32> src_factor_out1;
            src_factor_out1 = ((ap_int<32>)insn(96, 86));
            ap_int<32> src_factor_in1;
            src_factor_in1 = ((ap_int<32>)insn(107, 97));
            ap_int<32> wgt_factor_out;
            wgt_factor_out = ((ap_int<32>)insn(117, 108));
            ap_int<32> wgt_factor_in;
            wgt_factor_in = ((ap_int<32>)insn(127, 118));
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
                    scalar6 = ((ap_uint<16>)uop_mem[(k2 + uop_bgn1)](10, 0));
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
          }
        }
        }
      }
    }
  }

}
