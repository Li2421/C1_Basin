"""Frozen nominal algorithms; only static-scene wall geometry cost accessor differs."""
from reservation_certificate_v2_pa_memo_query_disk_candidate_spatial_mask_20261022 import ReservationNominal as OriginalNominal
from controllers_layout_user_fix_20261023 import Nominal as LayoutNominal
class Nominal(OriginalNominal):
    def action(self,env,key):
        if env.config.scene=='lit_swap8_obstacles' and self.kind not in ('prioritized_astar','conflict_based_search'):
            return LayoutNominal.action(self,env,key)
        return super().action(env,key)
