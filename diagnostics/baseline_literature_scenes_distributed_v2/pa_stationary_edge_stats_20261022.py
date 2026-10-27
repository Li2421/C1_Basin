from experiment import HERE
import ctypes
LIB=ctypes.CDLL(str(HERE/'astar_kernel_stationary_edge_candidate_20261022.so'))
LIB.pa_stationary_edge_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
def snapshot():
 a=(ctypes.c_uint64*8)();LIB.pa_stationary_edge_stats(a)
 return dict(zip(('queries','eligible_edges','edge_hits','reserved_segment_tests','stored_edges','capacity_fallbacks','allocation_fallbacks','maximum_entries'),list(a)))
