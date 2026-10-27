"""Verify every draw byte with unchanged5000-draw kernel and real cache fallbacks."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import numpy as np,ctypes,time
from statistics_bootstrap_memo_reader_20261022 import bootstrap,bootstrap_uncached
from bootstrap_exact_memo_20261022 import snapshot


def equal(a,b):
    assert len(a)==len(b)==2
    for x,y in zip(a,b):assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});os.nice(10)
    verify_receipt('BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json')
    prefix='BOOTSTRAP_EXACT_MEMO_TEST_20261022';assert not (HERE/(prefix+'.json')).exists()
    rng=np.random.default_rng(2026102245);M=rng.integers(2,size=(44,32,6)).astype(np.float64)
    M[...,1::2]=np.maximum(M[...,::2],M[...,1::2]);M[::5,:2]=np.nan;types=np.asarray([f't{i%8}' for i in range(44)])
    started=time.monotonic();reference=bootstrap_uncached(M,types,5000);reference_seconds=time.monotonic()-started
    started=time.monotonic();first=bootstrap(M,types,5000);first_seconds=time.monotonic()-started;equal(reference,first)
    started=time.monotonic();cached=bootstrap(M,types,5000);hit_seconds=time.monotonic()-started;equal(reference,cached)
    cached[0][0]=123.;cached[1][0,0]=123.;equal(reference,bootstrap(M,types,5000))
    rows=[dict(scope='44controllers,8types,32states,unknown bounds and missing clusters',iterations=5000,all5000_draw_bytes_equal=True,reference_seconds=reference_seconds,first_miss_seconds=first_seconds,repeat_hit_seconds=hit_seconds)]
    small=M[:3,:7].copy();small_types=types[:3].copy()
    for layout in (small,np.asfortranarray(small),small.astype(np.float32),small[:,::-1]):
        expected=bootstrap_uncached(layout,small_types,37);equal(expected,bootstrap(layout,small_types,37));equal(expected,bootstrap(layout,small_types,37))
    # Exercise real LRU capacity on the same mathematical kernel.
    for count in range(2,24):equal(bootstrap_uncached(small,small_types,count),bootstrap(small,small_types,count))
    # The real oversized result is returned unchanged and deliberately unstored.
    large=np.zeros((1,1,1800),np.float64);large_types=np.asarray(['one'])
    expected=bootstrap_uncached(large,large_types,5000);equal(expected,bootstrap(large,large_types,5000));del expected
    libc=ctypes.CDLL(None);libc.fegetround.restype=ctypes.c_int;libc.fesetround.argtypes=[ctypes.c_int];old=libc.fegetround()
    try:
        assert libc.fesetround(0x400)==0
        equal(bootstrap_uncached(small,small_types,31),bootstrap(small,small_types,31))
        equal(bootstrap_uncached(small,small_types,31),bootstrap(small,small_types,31))
    finally:assert libc.fesetround(old)==0
    callback=np.geterrcall()
    try:
        np.seterrcall(lambda *args:None)
        equal(bootstrap_uncached(small,small_types,13),bootstrap(small,small_types,13))
    finally:np.seterrcall(callback)
    stats=snapshot();assert stats['hits']>=6 and stats['evictions']>0 and stats['oversize']>0 and stats['bypass']>0,stats
    assert stats['entries']<=16 and stats['estimated_payload_bytes']<=64*1024*1024
    write(HERE/(prefix+'.json'),dict(utc=utc(),passed=True,full5000_draws_bitwise_equal=True,scope_tests=rows,dtype_stride_order_tests=True,returned_arrays_do_not_alias_cache=True,rounding_control_key_tested=True,callback_mode_bypass_tested=True,actual_capacity_eviction_tested=True,actual_oversize_unstored_tested=True,cache_stats=stats,new_rollouts=0,new_training=0,statistics_definitions_unchanged=True))
    verify_receipt('BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json')
    print('PASS:every5000 draw byte and bounded fallback matches original',rows,stats,flush=True)


if __name__=='__main__':main()
