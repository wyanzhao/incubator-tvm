'''modeling GEMM instruction'''
from functools import partial
import heterocl as hcl
from ..config import env
from .. import state

# Opcode, GEMM encoding
VTA_OPCODE_GEMM = 2

ENCODING = {
    'name': 'gemm',
    'type': 'gemm',
    'opcode': VTA_OPCODE_GEMM
}

# gemm_struct = utils.make_struct(encoding)
def gemm(instr, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem):
    '''gemm instruction'''
    with hcl.Stage("gemm"):
        reset_reg = hcl.scalar(instr[8:7], name="reset_reg")
        uop_bgn = hcl.scalar(instr[21:8], name="uop_bgn")
        uop_end = hcl.scalar(instr[35:21], name="uop_end")
        iter_out = hcl.scalar(instr[49:35], name="iter_out")
        iter_in = hcl.scalar(instr[63:49], name="iter_in")
        dst_factor_out = hcl.scalar(instr[75:64], name="dst_factor_out")
        dst_factor_in = hcl.scalar(instr[86:75], name="dst_factor_in")
        src_factor_out = hcl.scalar(instr[97:86], name="src_factor_out")
        src_factor_in = hcl.scalar(instr[108:97], name="src_factor_in")
        wgt_factor_out = hcl.scalar(instr[118:108], name="wgt_factor_out")
        wgt_factor_in = hcl.scalar(instr[128:118], name="wgt_factor_in")

        #hcl.print(reset_reg, "- reset_reg = %d\n")

        gemm_core(reset_reg, iter_out, iter_in, uop_bgn, uop_end, dst_factor_out, dst_factor_in,
                  src_factor_out, src_factor_in, wgt_factor_out, wgt_factor_in,
                  uop_mem, inp_mem, wgt_mem, acc_mem, out_mem)

def decode_uop(uop):
    acc_idx = hcl.scalar(uop[11:0], dtype=hcl.UInt(16))
    inp_idx = hcl.scalar(uop[22:11], dtype=hcl.UInt(16))
    wgt_idx = hcl.scalar(uop[32:22], dtype=hcl.UInt(16))
    return acc_idx, inp_idx, wgt_idx

def gemm_core(reset_reg, iter_out, iter_in, uop_bgn, uop_end, dst_factor_out, dst_factor_in,
              src_factor_out, src_factor_in, wgt_factor_out, wgt_factor_in,
              uop_mem, inp_mem, wgt_mem, acc_mem, out_mem,
              batch=env.BATCH, blkin=env.BLOCK_IN, blkout=env.BLOCK_OUT):
    '''gemm core subblock'''
    def fmutate(i, j, k):  # i <--iter_out   j <-- iter_in   k <-- 0 to (uop_end-uop_bgn)
        k += uop_bgn
        uop = uop_mem[k]
        acc_idx, inp_idx, wgt_idx = decode_uop(uop)
        acc_idx += j * hcl.cast(hcl.UInt(16), dst_factor_in.v) + \
                   i * hcl.cast(hcl.UInt(16), dst_factor_out.v)
        inp_idx += j * hcl.cast(hcl.UInt(16), src_factor_in.v) + \
                   i * hcl.cast(hcl.UInt(16), src_factor_out.v)
        wgt_idx += j * hcl.cast(hcl.UInt(16), wgt_factor_in.v) + \
                   i * hcl.cast(hcl.UInt(16), wgt_factor_out.v)
        itensor, wtensor = inp_mem[inp_idx], wgt_mem[wgt_idx]

        m = hcl.reduce_axis(0, blkin, 'm')
        # Note: tensors are declared unsigned but math needs to be done signed.
        otensor = hcl.compute(
            (batch, blkout),
            lambda r, c: hcl.sum(
                hcl.cast(hcl.Int(8), itensor[r][m]) *
                hcl.cast(hcl.Int(8), wtensor[c][m]), m,
                name='dot', dtype=hcl.Int(32)),
            name='multiply_signed', dtype=hcl.Int(32))
        def accumulate_signed(row, col):
            acc_mem[acc_idx][row][col] = hcl.cast(hcl.Int(32), acc_mem[acc_idx][row][col]) + \
                                         otensor[row][col]
        hcl.mutate(otensor.shape, accumulate_signed)
        def fmutate_out(row, col):
            out_mem[acc_idx][row][col] = hcl.cast(out_mem.dtype, acc_mem[acc_idx][row][col])
        hcl.mutate(otensor.shape, fmutate_out)

    def fmutate_reset(i, j, k):
        k += uop_bgn.v
        uop = uop_mem[k]
        acc_idx, _, _ = decode_uop(uop)
        acc_idx += j * hcl.cast(hcl.UInt(16), dst_factor_in.v) + \
                   i * hcl.cast(hcl.UInt(16), dst_factor_out.v)
        def fmutate_out_0(row, col):
            acc_mem[acc_idx][row][col] = 0
        hcl.mutate((batch, blkout), fmutate_out_0)

    with hcl.Stage("gemm_core"):
        domain = (hcl.cast(hcl.UInt(32), iter_out.v), hcl.cast(hcl.UInt(32), iter_in.v),
                  (hcl.cast(hcl.UInt(32), uop_end.v-uop_bgn.v)))
        with hcl.if_(reset_reg.v == 1):
            hcl.mutate(domain, fmutate_reset, name="fmutate_reset")
        with hcl.else_():
            hcl.mutate(domain, fmutate, name="fmutate")

def customize_gemm_core(batch=env.BATCH, blkin=env.BLOCK_IN, blkout=env.BLOCK_OUT):
    return partial(gemm_core, batch=batch, blkin=blkin, blkout=blkout)


# VTA GEMM instruction
# GEMM instruction is implemented by executing a sequence of micro-operations
# that is read in the local micro-op memory, delimited by \a uop_bgn and
# \a uop_end. For improved storage-efficiency, the micro-operations can be
# executed in a 2-level nested loop as follows:
#   for (i = 0; i < iter_out; i++) {
#     for (j = 0; j < iter_in; j++) {
#       for (k = uop_bgn; k < uop_end; k++) {
#         // Read micro op
#         uop_T uop = uop_mem[k];
#         // Read in memory indices
#         acc_idx_T acc_idx = uop.dst_idx;
#         inp_idx_T inp_idx = uop.inp_idx;
#         wgt_idx_T wgt_idx = uop.wgt_idx;
#         // Update those indices with the following affine functions
#         acc_idx += iter_in * dst_factor_in + iter_out * dst_factor_out;
#         inp_idx += iter_in * src_factor_in + iter_out * src_factor_out;
#         wgt_idx += iter_in * wgt_factor_in + iter_out * wgt_factor_out;
#         // Perform GEMM operation
#         acc_mem[acc_idx] += dot(inp_mem[inp_idx], wgt[wgt_idx]);
#       }
#     }
#   }
