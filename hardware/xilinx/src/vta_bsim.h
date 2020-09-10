#ifndef VTA_VTA_BSIM_H_
#define VTA_VTA_BSIM_H_

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <assert.h>
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

void vta_bsim(uint32_t instr_count,
		volatile 	ap_uint<8> *insns,
		volatile 	ap_uint<8> *uops,
		volatile	ap_uint<8> *inputs,
		volatile	ap_uint<8> *weights,
		volatile	ap_uint<8> *biases,
		volatile	ap_uint<8> *outputs,
		volatile 	uint32_t &done
);

#endif
