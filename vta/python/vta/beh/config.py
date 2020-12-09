'''configuration of bsim'''
from ..environment import get_env

env = get_env() # pylint: disable=invalid-name

LOG_DATA_WIDTH = 3
LOG_LONG_DATA_WIDTH = 5  # TODO: remove me

LOG_UOP_WIDTH = 5
UOP_WIDTH = 1 << LOG_UOP_WIDTH # 32 bits in 4 bytes: [0,2] acc_idx, [3:5] inp_idx, [6:10] wgt_idx

LOG_INS_WIDTH = 7
INS_WIDTH = 1 << LOG_INS_WIDTH

LOG_DRAM_SIZE = 27
DRAM_SIZE = 1 << LOG_DRAM_SIZE # large memory

DATA_WIDTH = 1 << LOG_DATA_WIDTH # 8 bits in 1 byte
LONG_DATA_WIDTH = 1 << LOG_LONG_DATA_WIDTH # 32 bits in 4 bytes

# INP_MATRIX_WIDTH = int(env.INP_WIDTH * env.BATCH * env.BLOCK_IN)
# WGT_MATRIX_WIDTH = int(env.WGT_WIDTH * env.BLOCK_OUT * env.BLOCK_IN)
# print("WGT_WIDTH " + str(env.WGT_WIDTH))
# print("BLOCK_OUT " + str(env.BLOCK_OUT))
# print("BLOCK_IN " + str(env.BLOCK_IN))
# print("DATA_WIDTH " + str(DATA_WIDTH))
# ACC_MATRIX_WIDTH = int(env.ACC_WIDTH * env.BATCH * env.BLOCK_OUT)
# OUT_MATRIX_WIDTH = int(env.OUT_WIDTH * env.BATCH * env.BLOCK_OUT)

# INP_MAT_AXI_RATIO = int(INP_MATRIX_WIDTH / DATA_WIDTH)
# WGT_MAT_AXI_RATIO = int(WGT_MATRIX_WIDTH / DATA_WIDTH)
# ACC_MAT_AXI_RATIO = int(ACC_MATRIX_WIDTH / DATA_WIDTH)
# OUT_MAT_AXI_RATIO = int(OUT_MATRIX_WIDTH /DATA_WIDTH)
# print("WGT_MAT_AXI_RATIO " + str(WGT_MAT_AXI_RATIO))
# WGT_ELEM_BYTES = (WGT_MATRIX_WIDTH / 8)
# print("WGT_ELEM_BYTES " + str(WGT_ELEM_BYTES))