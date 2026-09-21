"""Serialization helpers for opt-in, unresolved-risk C1 geometry diagnostics."""
import numpy as np


def summarize_geometry(geometry):
    """JSON-safe per-step summary; deliberately omits any cone-risk value."""
    return dict(
        active_count=int(len(geometry["active_indices"])),
        active_indices=geometry["active_indices"].tolist(),
        completely_deadlock_free=bool(geometry["completely_deadlock_free"]),
        cone_distance_defined=bool(geometry["cone_distance_defined"]),
        max_abs_cbf_residual=float(np.abs(geometry["cbf_residual"]).max()),
        min_h=float(geometry["h"].min()),
    )
