from vta.beh.beh_v1 import *
from vta.beh.beh import *

if __name__ == "__main__":
    
    BUILD_BSIM_V0 = False
    BUILD_FETCH = False 
    BUILD_LOAD = False
    BUILD_STORE = False
    BUILD_ALU = False
    BUILD_GEMM = False
    BUILD_COMPUTE = False
    target = "vhls"

    if BUILD_FETCH:
        m = build_fetch(target)
        print(m)
    
    if BUILD_LOAD:
        m = build_load(target)
        print(m)
    
    if BUILD_STORE:
        m = build_store(target)
        print(m)

    if BUILD_ALU:
        m = build_alu(target)
        print(m)
    
    if BUILD_GEMM:
        m = build_gemm(target)
        print(m)

    if BUILD_COMPUTE:
        m = build_compute(target)
        print(m)
