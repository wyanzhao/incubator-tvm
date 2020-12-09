'''VTA Behavioral Simulation model using HeteroCL'''
from heterocl.compute_api import pack
import numpy as np
import heterocl as hcl
from .instructions import store, gemm, alu, load, finish
from . import config, state
from .config import env

VTA_OPCODE_LOAD = 0
# Mem ID constant, uop memory
VTA_MEM_ID_UOP = 0
# Mem ID constant, weight memory
VTA_MEM_ID_WGT = 1
# Mem ID constant, input memory
VTA_MEM_ID_INP = 2
# Mem ID constant, accumulator/bias memory
VTA_MEM_ID_ACC = 3



def store_module(outputs, store_queue, g2s_dep_queue, s2g_dep_queue, out_mem):
    raw_insn = hcl.scalar(store_queue[0], name="raw_insn", dtype=hcl.UInt(config.INS_WIDTH))[0]
    sram_base = hcl.scalar(raw_insn[25:9], name="sram_base")
    dram_base = hcl.scalar(raw_insn[57:25], name="dram_base")
    y_size = hcl.scalar(raw_insn[80:64], name="y_size")
    x_size = hcl.scalar(raw_insn[96:80], name="x_size")
    x_stride = hcl.scalar(raw_insn[112:96], name="x_stride")

    with hcl.if_(raw_insn[4:3] == True):
        hcl.scalar(g2s_dep_queue[0])
    
    with hcl.if_(x_size.v == 0):
        pass
    with hcl.else_():
        store.store_2d(sram_base, dram_base, x_size, y_size, x_stride, out_mem, outputs)
        
    # TODO: Better way to implement FIFO and QUEUE in heteroCL?
    with hcl.if_(raw_insn[6:5] == True):
        hcl.update(s2g_dep_queue, lambda *x: 1)



#@hcl.def_([inputs.shape, weights.shape, (1024, ), (1024, ), (1024, ), inp_mem.shape, wgt_mem.shape])
def load_module(inputs, weights, load_queue, g2l_dep_queue, l2g_dep_queue, inp_mem, wgt_mem):        
    raw_insn = hcl.scalar(load_queue[0], name="raw_insn", dtype=hcl.UInt(config.INS_WIDTH))[0]
    memory_type = hcl.scalar(raw_insn[9:7], name="memory_type").v
    sram_base = hcl.scalar(raw_insn[25:9], name="sram_base")
    dram_base = hcl.scalar(raw_insn[57:25], name="dram_base")
    is_min_pad_value = hcl.scalar(raw_insn[58:57], name="is_pad_min_value")
        # 6 bits empty
    y_size = hcl.scalar(raw_insn[80:64], name="y_size")
    x_size = hcl.scalar(raw_insn[96:80], name="x_size")
    x_stride = hcl.scalar(raw_insn[112:96], name="x_stride")
    y_pad_0 = hcl.scalar(raw_insn[116:112], name="y_pad_0")
    y_pad_1 = hcl.scalar(raw_insn[120:116], name="y_pad_1")
    x_pad_0 = hcl.scalar(raw_insn[124:120], name="x_pad_0")
    x_pad_1 = hcl.scalar(raw_insn[128:124], name="x_pad_1")

    with hcl.if_(raw_insn[5:4] == True):
        hcl.scalar(g2l_dep_queue[0])
        
    with hcl.if_(memory_type == VTA_MEM_ID_INP):
        load.load_2d(sram_base, dram_base, x_size, y_size, x_stride,
                y_pad_0, y_pad_1, x_pad_0, x_pad_1, inp_mem, inputs, is_min_pad_value)
    with hcl.if_(memory_type == VTA_MEM_ID_WGT):
        load.load_2d(sram_base, dram_base, x_size, y_size, x_stride,
                    y_pad_0, y_pad_1, x_pad_0, x_pad_1, wgt_mem, weights, is_min_pad_value)
        
    # TODO: Better way to implement FIFO and QUEUE in heteroCL?
    with hcl.if_(raw_insn[7:6] == True):
        hcl.update(l2g_dep_queue, lambda *x: 1)


#@hcl.def_([(), insns.shape, (1024, ), (1024, ), (1024, )])
def fetch_module(insn_count, insns, load_queue, gemm_queue, store_queue):
    with hcl.Stage("fetch"):
        with hcl.for_(0, insn_count, name="i") as i:
            insn = insns[i]
            opcode = hcl.scalar(insn[3:0], name='opcode', dtype=hcl.UInt(3))
            memory_type = hcl.scalar(insn[9:7], name='memory_type', dtype=hcl.UInt(2))
            with hcl.if_(opcode == store.VTA_OPCODE_STORE):
                store_queue[i] = insn
            with hcl.elif_(opcode == VTA_OPCODE_LOAD):
                with hcl.if_(memory_type == VTA_MEM_ID_INP):
                    hcl.update(load_queue, lambda *x: insn)
                    #load_queue[i] = insn
                with hcl.elif_(memory_type == VTA_MEM_ID_WGT):
                    #load_queue[i] = insn
                    hcl.update(load_queue, lambda *x: insn)
                with hcl.else_():
                    hcl.update(gemm_queue, lambda *x: insn)
                    #gemm_queue[i] = insn
            with hcl.else_():
                hcl.update(gemm_queue, lambda *x: insn)
                #gemm_queue[i] = insn

def compute_module(done, uops, biases, gemm_queue, l2g_dep_queue, s2g_dep_queue, g2l_dep_queue, g2s_dep_queue,
                   inp_mem, wgt_mem, out_mem):


    uop_mem = hcl.compute((env.UOP_BUFF_SIZE,),lambda *x: 0,
                          name='uop_mem', dtype=hcl.UInt(config.UOP_WIDTH))
    acc_mem = hcl.compute((env.ACC_BUFF_SIZE, env.BATCH, env.BLOCK_OUT,), lambda *x: 0,
                          name='acc_mem', dtype=hcl.UInt(config.LONG_DATA_WIDTH))

    raw_insn = hcl.scalar(gemm_queue[0], name="raw_insn", dtype=hcl.UInt(128))[0]

    with hcl.Stage():
        with hcl.if_(raw_insn[4:3] == True):
            hcl.scalar(l2g_dep_queue[0])
    with hcl.Stage():
        with hcl.if_(raw_insn[5:4] == True):
            hcl.scalar(s2g_dep_queue[0])
    
    hcl.update(done, lambda *x: 0, "done")
    opcode = hcl.scalar(raw_insn[3:0], name='opcode')
    with hcl.Stage("compute2"):
        with hcl.if_(opcode == finish.VTA_OPCODE_FINISH):
            hcl.update(done, lambda *x: 1, "done")
        with hcl.elif_(opcode == load.VTA_OPCODE_LOAD):
            memory_type = hcl.scalar(raw_insn[9:7], name="memory_type").v
            sram_base = hcl.scalar(raw_insn[25:9], name="sram_base")
            dram_base = hcl.scalar(raw_insn[57:25], name="dram_base")
            is_min_pad_value = hcl.scalar(raw_insn[58:57], name="is_pad_min_value")
            # 6 bits empty
            y_size = hcl.scalar(raw_insn[80:64], name="y_size")
            x_size = hcl.scalar(raw_insn[96:80], name="x_size")
            x_stride = hcl.scalar(raw_insn[112:96], name="x_stride")
            y_pad_0 = hcl.scalar(raw_insn[116:112], name="y_pad_0")
            y_pad_1 = hcl.scalar(raw_insn[120:116], name="y_pad_1")
            x_pad_0 = hcl.scalar(raw_insn[124:120], name="x_pad_0")
            x_pad_1 = hcl.scalar(raw_insn[128:124], name="x_pad_1")
            with hcl.Stage():
                with hcl.if_(x_size.v == 0):
                    pass
                with hcl.elif_(memory_type == VTA_MEM_ID_UOP):
                    load.load_uop(sram_base, dram_base, x_size, uop_mem, uops)
                with hcl.elif_(memory_type == VTA_MEM_ID_ACC):
                    load.load_2d(sram_base, dram_base, x_size, y_size, x_stride,
                    y_pad_0, y_pad_1, x_pad_0, x_pad_1, acc_mem, biases, is_min_pad_value)
        with hcl.elif_(opcode == gemm.VTA_OPCODE_GEMM):
            #gemm_module(raw_insn, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem)
            gemm.gemm(raw_insn, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem)
        with hcl.elif_(opcode == alu.VTA_OPCODE_ALU):
            #alu_module(raw_insn, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem)
            alu.alu(raw_insn, uop_mem, acc_mem, out_mem)
    
    with hcl.Stage():
        with hcl.if_(raw_insn[6:5] == True):
            hcl.update(g2l_dep_queue, lambda *x: 1)
    with hcl.Stage():
        with hcl.if_(raw_insn[7:6] == True):
            hcl.update(g2s_dep_queue, lambda *x: 1)
        
@hcl.def_([state.UOP_MEM.shape, state.ACC_MEM.shape],
[state.UOP_MEM.dtype, state.ACC_MEM.dtype])
def test_module(uop_mem, acc_mem):
        pass
        #scale = hcl.compute(uop_mem.shape, lambda *x:0, name="test")
        #hcl.update(acc_mem, lambda *x: 0)

def alu_module(insn_raw, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem):
    alu.alu(insn_raw, uop_mem, acc_mem, out_mem)

# @hcl.def_(shapes=[(), state.UOP_MEM.shape, state.ACC_MEM.shape, state.INP_MEM.shape, state.WGT_MEM.shape, state.OUT_MEM.shape]
# #dtypes=[hcl.UInt(128), state.UOP_MEM.dtype, state.ACC_MEM.dtype, state.INP_MEM.dtype, state.WGT_MEM.dtype, state.OUT_MEM.dtype])
# )
def gemm_module(insn_raw, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem):
    gemm.gemm(insn_raw, uop_mem, inp_mem, wgt_mem, acc_mem, out_mem)


def build_fetch(target=None):
    insn_count = hcl.placeholder((), name="insn_count", dtype=hcl.UInt(32))
    #load_queue = hcl.placeholder((), name="load_queue", dtype=hcl.UInt(128))
    #gemm_queue = hcl.placeholder((), name="load_queue", dtype=hcl.UInt(128))
    #store_queue = hcl.placeholder((), name="load_queue", dtype=hcl.UInt(128))
    load_queue =  hcl.compute((1,), lambda *x: 0,"load_queue", hcl.UInt(128))
    gemm_queue =  hcl.compute((1,), lambda *x: 0,"gemm_queue", hcl.UInt(128))
    store_queue =  hcl.compute((1,), lambda *x: 0,"store_queue", hcl.UInt(128))

    inputs = [insn_count, state.INSNS, load_queue, gemm_queue, store_queue]
    s = hcl.create_schedule(inputs, fetch_module)
    #s.to(load_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    #s.to(g2l_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    #s.to(l2g_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)

    m = hcl.build(s, target)
    return m


def build_alu(target=None):
    insn_raw = hcl.placeholder((), name="insn_raw", dtype=hcl.UInt(128))
    uop_mem = state.UOP_MEM
    acc_mem = state.ACC_MEM
    inp_mem = state.INP_MEM
    wgt_mem = state.WGT_MEM
    out_mem = state.OUT_MEM

    inputs = [insn_raw, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem]
    s = hcl.create_schedule(inputs, alu_module)
    # #s.to(load_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    # #s.to(g2l_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    # #s.to(l2g_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)

    m = hcl.build(s, target)
    return m



def build_gemm(target=None):
    insn_raw = hcl.placeholder((), name="insn_raw", dtype=hcl.UInt(128))
    uop_mem = state.UOP_MEM
    acc_mem = state.ACC_MEM
    inp_mem = state.INP_MEM
    wgt_mem = state.WGT_MEM
    out_mem = state.OUT_MEM

    inputs = [insn_raw, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem]
    s = hcl.create_schedule(inputs, gemm_module)
    # #s.to(load_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    # #s.to(g2l_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    # #s.to(l2g_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)

    m = hcl.build(s, target)
    return m



def build_compute(target=None):
    done = hcl.placeholder((1, ), name="done", dtype=hcl.UInt(1))
    uops = state.UOPS
    biases = state.BIASES
    gemm_queue = hcl.compute((1,), lambda *x: 0,"gemm_queue", hcl.UInt(128))
    l2g_dep_queue = hcl.compute((1,), lambda *x: 0,"l2g_dep_queue", hcl.UInt(1))
    s2g_dep_queue = hcl.compute((1,), lambda *x: 0,"s2g_dep_queue", hcl.UInt(1))
    g2l_dep_queue = hcl.compute((1,), lambda *x: 0,"g2l_dep_queue", hcl.UInt(1))
    g2s_dep_queue = hcl.compute((1,), lambda *x: 0,"g2s_dep_queue", hcl.UInt(1))
    inp_mem = state.INP_MEM
    wgt_mem = state.WGT_MEM
    out_mem = state.OUT_MEM
    print(inp_mem)
    print(wgt_mem)
    print(out_mem)
    inputs = [done, uops, biases, gemm_queue, l2g_dep_queue, s2g_dep_queue, g2l_dep_queue, g2s_dep_queue,
    inp_mem, wgt_mem, out_mem]
    s = hcl.create_schedule(inputs, compute_module)
    m = hcl.build(s, target)
    return m



def build_load(target=None):
    load_queue =  hcl.compute((1,), lambda *x: 0,"load_queue", hcl.UInt(128))
    g2l_dep_queue = hcl.compute((1,), lambda *x: 0,"g2l_dep_queue", hcl.UInt(1))
    l2g_dep_queue = hcl.compute((1,), lambda *x: 0,"l2g_dep_queue", hcl.UInt(1))
    inp_mem = state.INP_MEM
    wgt_mem = state.WGT_MEM
    inputs = [state.INPUTS,
              state.WEIGHTS, load_queue, g2l_dep_queue, l2g_dep_queue, inp_mem, wgt_mem]
    s = hcl.create_schedule(inputs, load_module)
    #s.to(load_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    #s.to(g2l_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    #s.to(l2g_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)

    m = hcl.build(s, target)
    return m


def build_store(target=None):
    store_queue =  hcl.compute((1,), lambda *x: 0,"store_queue", hcl.UInt(128))
    g2s_dep_queue = hcl.compute((1,), lambda *x: 0,"g2s_dep_queue", hcl.UInt(1))
    s2g_dep_queue = hcl.compute((1,), lambda *x: 0,"s2g_dep_queue", hcl.UInt(1))
    out_mem = state.OUT_MEM
    inputs = [state.OUTPUTS, store_queue, g2s_dep_queue, s2g_dep_queue, out_mem]
    s = hcl.create_schedule(inputs, store_module)
    #s.to(load_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    #s.to(g2l_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)
    #s.to(l2g_dep_queue, hcl.platform.zc706.xcel, mode=hcl.IO.FIFO)

    m = hcl.build(s, target)
    return m




