'''behavorial simulation model'''
from . import config, utils, state
from .beh import DRAM_ARRAY, beh_model
from .instructions import load, store, gemm, alu
