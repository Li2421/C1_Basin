"""Record full-input and sufficient-state query duplicates, never skip a search."""
from experiment import *
import ctypes
import numpy as np

PTR=ctypes.POINTER(ctypes.c_double)
IPTR=ctypes.POINTER(ctypes.c_int)
LPTR=ctypes.POINTER(ctypes.c_longlong)
LIB=ctypes.CDLL(str(HERE/'pa_query_reuse_facts_probe_20261022.so'))
LIB.pa_query_facts.argtypes=[ctypes.c_double,ctypes.c_double,PTR,PTR,ctypes.c_int,IPTR,PTR,ctypes.c_int,PTR,ctypes.c_double,LPTR]
LIB.pa_query_facts.restype=ctypes.c_int
SEEN={}
FULL=set()
STATS=dict(queries=0,generalized_duplicates=0,full_input_duplicates=0,duplicate_native_seconds=0.,suffix_mismatches=0)

def key(grid,start,goal,lens,reserved):
    facts=np.zeros(5,np.int64)
    sep=2*grid.env.config.agent_radius+grid.env.config.agent_collision_margin
    mask=LIB.pa_query_facts(grid.res,sep,start.ctypes.data_as(PTR),goal.ctypes.data_as(PTR),len(lens),lens.ctypes.data_as(IPTR),reserved.ctypes.data_as(PTR),len(grid.walls),grid.walls.ctypes.data_as(PTR),grid.margin,facts.ctypes.data_as(LPTR))
    common=(float(grid.res).hex(),float(sep).hex(),grid.max_time,float(grid.margin).hex(),grid.walls.tobytes(),goal.tobytes(),lens.tobytes())
    suffix=tuple(np.ascontiguousarray(p[1:] if len(p)>1 else p,np.float64).tobytes() for p in grid.reserved)
    return common+(facts.tobytes(),mask,suffix),common+(start.tobytes(),reserved.tobytes())

def observe(keys,length,out,elapsed):
    sufficient,full=keys
    value=(int(length),out[1:length].tobytes() if length else b'')
    STATS['queries']+=1
    if full in FULL:STATS['full_input_duplicates']+=1
    FULL.add(full)
    if sufficient in SEEN:
        STATS['generalized_duplicates']+=1
        STATS['duplicate_native_seconds']+=elapsed
        if SEEN[sufficient]!=value:
            STATS['suffix_mismatches']+=1
            raise AssertionError('Equal sufficient-state keys produced different original search suffixes')
    else:SEEN[sufficient]=value
