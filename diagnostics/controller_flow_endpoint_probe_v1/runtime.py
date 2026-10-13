"""Different FF response operators retaining the frozen Flow nominal plan."""
from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField,FAMILIES


class FlowEndpointMatchedField(EndpointMatchedField):
    def nominal(self,env,key,traced=False):
        self._reference_key=key
        return super().nominal(env,key,traced)

    def target(self,env):
        # One frozen policy evaluation at the current physical state; no
        # lookahead or physical simulation, and no updated Flow weights.
        return self.base.nominal(env,self._reference_key,traced=False)[1]
