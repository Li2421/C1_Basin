"""Bounded exact A* query memo; preserve the actual continuous first point.

For a reserved path with at least two points, its first point is used only by
the ten initial-node predicates. Later nodes use its unchanged suffix. A
one-point reservation is retained in full because it persists at all times.
The original native A* executes on every miss and every unsupported input.
"""
from experiment import *
import collections
import ctypes
import sys
import numpy as np

PTR=ctypes.POINTER(ctypes.c_double)
IPTR=ctypes.POINTER(ctypes.c_int)
LPTR=ctypes.POINTER(ctypes.c_longlong)
LIB=ctypes.CDLL(str(HERE/'pa_query_memo_facts_candidate_20261022.so'))
LIB.pa_query_facts.argtypes=[ctypes.c_double,ctypes.c_double,PTR,PTR,ctypes.c_int,IPTR,PTR,ctypes.c_int,PTR,ctypes.c_double,LPTR]
LIB.pa_query_facts.restype=ctypes.c_int
MAX_ENTRIES=8192
MAX_BYTES=32*1024*1024
CACHE=collections.OrderedDict()
BYTES=0
COUNTERS=collections.Counter()


def footprint(value):
    return sys.getsizeof(value)+(sum(footprint(v) for v in value) if isinstance(value,tuple) else 0)


def sufficient_key(grid,start,goal,lens,reserved):
    if not (np.isfinite(grid.res) and 0<grid.res<=1000 and
            np.all(np.isfinite(start)) and np.all(np.isfinite(goal)) and
            np.all(abs(start)<grid.res*90000) and np.all(abs(goal)<grid.res*90000) and
            np.all(lens>0) and np.all(np.isfinite(reserved)) and
            np.all(np.isfinite(grid.walls)) and np.isfinite(grid.margin)):
        return None
    facts=np.zeros(5,np.int64)
    sep=2*grid.env.config.agent_radius+grid.env.config.agent_collision_margin
    if not np.isfinite(sep):return None
    mask=LIB.pa_query_facts(grid.res,sep,start.ctypes.data_as(PTR),goal.ctypes.data_as(PTR),len(lens),lens.ctypes.data_as(IPTR),reserved.ctypes.data_as(PTR),len(grid.walls),grid.walls.ctypes.data_as(PTR),grid.margin,facts.ctypes.data_as(LPTR))
    suffix=tuple(np.ascontiguousarray(p[1:] if len(p)>1 else p,np.float64).tobytes() for p in grid.reserved)
    return (float(grid.res).hex(),float(sep).hex(),grid.max_time,float(grid.margin).hex(),
            grid.walls.tobytes(),goal.tobytes(),lens.tobytes(),facts.tobytes(),mask,suffix)


def search(grid,start,goal,lens,reserved,out,fn,args):
    global BYTES
    COUNTERS['queries']+=1
    try:key=sufficient_key(grid,start,goal,lens,reserved)
    except MemoryError:key=None
    if key is None:
        COUNTERS['bypass']+=1
        return fn(*args)
    cached=CACHE.get(key)
    if cached is not None:
        length,suffix,size=cached
        CACHE.move_to_end(key)
        if length:
            out[0]=start
            out[1:length]=np.frombuffer(suffix,np.float64).reshape(-1,2)
        COUNTERS['hits']+=1
        COUNTERS['negative_hits']+=int(length==0)
        return length
    COUNTERS['misses']+=1
    length=fn(*args)
    if not 0<=length<=len(out):return length
    try:
        suffix=out[1:length].tobytes() if length else b''
        size=footprint(key)+footprint((length,suffix,0))
        if size>MAX_BYTES:
            COUNTERS['oversize']+=1
            return length
        while CACHE and (len(CACHE)>=MAX_ENTRIES or BYTES+size>MAX_BYTES):
            _,(_,_,old_size)=CACHE.popitem(last=False)
            BYTES-=old_size
            COUNTERS['evictions']+=1
        CACHE[key]=(length,suffix,size)
        BYTES+=size
        COUNTERS['stored']+=1
    except MemoryError:
        COUNTERS['allocation_fallback']+=1
    return length


def snapshot():
    return dict(COUNTERS,entries=len(CACHE),estimated_payload_bytes=BYTES,
                max_entries=MAX_ENTRIES,max_payload_bytes=MAX_BYTES)
