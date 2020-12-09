
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vta.h
 * \brief Type definitions and prototype for VTA HLS design.
 */
#ifndef VTA_VTA_BSIM_V1_H_
#define VTA_VTA_BSIM_V1_H_

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <assert.h>
#include <hls_stream.h>

#include <vta/hw_spec.h>

/*!
* Define HLS stream depth
*/
#define PRAGMA_SUB(x) _Pragma (#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)
#define STREAM_IN_DEPTH 8

void fetch(
  ap_uint<32> insn_count,
  ap_uint<128>* insns,
  hls::stream<ap_uint<128>> &load_queue,
  hls::stream<ap_uint<128>> &gemm_queue,
  hls::stream<ap_uint<128>> &store_queue);


void compute(ap_uint<1> &done, ap_uint<8>* uops, ap_uint<8>* biases, hls::stream<ap_uint<128>>& gemm_queue, hls::stream<bool> & l2g_dep_queue, hls::stream<bool> & s2g_dep_queue, 
hls::stream<bool> & g2l_dep_queue, hls::stream<bool> & g2s_dep_queue, ap_uint<8>* inp_mem, ap_uint<8>* wgt_mem, ap_uint<8>* out_mem);

void load(ap_uint<8>* inputs, ap_uint<8>* weights, hls::stream<ap_uint<128>> &load_queue, hls::stream<bool> & g2l_dep_queue, hls::stream<bool> & l2g_dep_queue, 
ap_uint<8>* inp_mem, ap_uint<8>* wgt_mem);

void store(ap_uint<8>* output, hls::stream<ap_uint<128>>& store_queue, hls::stream<bool> & g2s_dep_queue, hls::stream<bool> & s2g_dep_queue, ap_uint<8>* out_mem);

void vta_bsim_v1(
  uint32_t instr_count,
  ap_uint<128>* insns,
  ap_uint<8>* uops,
  ap_uint<8>* inputs,
  ap_uint<8>* weights,
  ap_uint<8>* biases,
  ap_uint<8>* outputs);


#endif  // VTA_VTA_H_
