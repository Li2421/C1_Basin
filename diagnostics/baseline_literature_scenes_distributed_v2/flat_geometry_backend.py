"""Install the proved exact geometry-cache dictionary layout, before worker use."""
from experiment import *
from registry import verify_receipt
import ctypes

_library=None


def install():
    global _library
    verify_receipt('CBS_FLAT_CACHE_DEPLOYMENT.receipt.json')
    verify_receipt('CBS_FLAT_CACHE_EQUIVALENCE.receipt.json')
    import tiled_planning
    if _library is None:
        _library=ctypes.CDLL(str(HERE/'cbs_kernel_flat_cache.so'))
        _library.astar_constrained.argtypes=tiled_planning.CBS.astar_constrained.argtypes
        _library.astar_constrained.restype=tiled_planning.CBS.astar_constrained.restype
    tiled_planning.CBS=_library
    return dict(kernel='cbs_kernel_flat_cache.so',equivalence_receipt_sha256=sha(HERE/'CBS_FLAT_CACHE_EQUIVALENCE.receipt.json'),storage_only=True)
