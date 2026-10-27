"""Bounded per-process cache of the unchanged deterministic bootstrap kernel."""
from experiment import HERE
import collections,ctypes
import numpy as np

LIB=ctypes.CDLL(str(HERE/'bootstrap_fp_control_20261022.so'))
LIB.bootstrap_fp_control.argtypes=[ctypes.POINTER(ctypes.c_uint32)]
LIMIT=64*1024*1024
MAX_ENTRIES=16
CACHE=collections.OrderedDict()
BYTES=0
COUNTERS=collections.Counter()


def memoized_bootstrap(M,types,iterations,fn):
    global BYTES
    COUNTERS['calls']+=1
    # These private callers use immutable arrays in a single process. Callback
    # error modes have observable side effects and are deliberately bypassed.
    errors=tuple(sorted(np.geterr().items()))
    if np.geterrcall() is not None or any(v in ('call','log') for k,v in errors):
        COUNTERS['bypass']+=1;return fn(M,types,iterations)
    controls=(ctypes.c_uint32*3)();LIB.bootstrap_fp_control(controls)
    try:
        key=(fn,int(iterations),M.dtype.str,M.shape,M.strides,M.tobytes(),types.dtype.str,types.shape,types.strides,types.tobytes(),tuple(controls),errors)
    except MemoryError:
        COUNTERS['bypass']+=1;return fn(M,types,iterations)
    value=CACHE.get(key)
    if value is not None:
        CACHE.move_to_end(key);COUNTERS['hits']+=1
        return value[0].copy(order='K'),value[1].copy(order='K')
    COUNTERS['misses']+=1
    point,draws=fn(M,types,iterations)
    size=len(key[5])+len(key[9])+point.nbytes+draws.nbytes+4096
    if size>LIMIT:COUNTERS['oversize']+=1;return point,draws
    try:
        stored_point=point.copy(order='K');stored_draws=draws.copy(order='K')
        while CACHE and (len(CACHE)>=MAX_ENTRIES or BYTES+size>LIMIT):
            _,(_,_,old_size)=CACHE.popitem(last=False);BYTES-=old_size;COUNTERS['evictions']+=1
        CACHE[key]=(stored_point,stored_draws,size);BYTES+=size;COUNTERS['stored']+=1
    except MemoryError:COUNTERS['allocation_fallback']+=1
    return point,draws


def snapshot():return dict(COUNTERS,entries=len(CACHE),estimated_payload_bytes=BYTES,max_entries=MAX_ENTRIES,max_payload_bytes=LIMIT)
