"""Isolate initializer/chart shift; original goal reference and FF unchanged."""
import numpy as np
from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField


class CommonInitializerField(EndpointMatchedField):
    def __init__(self, base, family):
        super().__init__(base, family)
        n = int(self.agent.config['joint_act_dim']) // 2
        # Exact physical noise law used in the older shared-navigation control.
        # Do not inherit that control's different navigation target.
        self.mean = np.zeros((n, 2))
        self.scale = np.full((n, 2), self.config.max_speed)

    def to_world(self, action, env):
        return np.asarray(action)

    def to_local(self, action, env):
        return np.asarray(action)
