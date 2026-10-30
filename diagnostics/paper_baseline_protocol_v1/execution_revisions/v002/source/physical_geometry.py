"""Explicit geometry variants; the inherited plant step is unchanged."""
from types import SimpleNamespace
import numpy as np


def geometry(config):
    from bottleneck_family.scenario import Geometry
    # Validation of physical points and literal walls is performed separately.
    # Config's staging-capacity rule is for the Gap sampler, not arbitrary maps.
    return Geometry(SimpleNamespace(**config))


def install(runtime):
    original = runtime.SwapEnv
    if getattr(original, '_paper_geometry_adapter', False):
        return

    class PhysicalGeometryEnv(original):
        _paper_geometry_adapter = True

        def __init__(self, state, config):
            super().__init__(state, config)
            declared = state.get('paper_geometry')
            if declared is None:
                return
            revised = geometry(declared['config'])
            expected = np.asarray(declared['walls'], float).reshape(-1, 2, 2)
            assert revised.walls.shape == expected.shape
            assert revised.walls.tobytes() == expected.tobytes()
            assert revised.valid_points(self.positions).all()
            assert revised.valid_points(self.goals).all()
            self.instance.geometry = revised
            self.walls = revised.walls

    runtime.SwapEnv = PhysicalGeometryEnv
