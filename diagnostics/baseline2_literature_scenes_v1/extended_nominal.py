from certified_planning import MemoNominal
from pilot_policies import PilotNominal

class ExtendedNominal(MemoNominal):
    def action(self,env,key):
        if self.kind in ('deterministic','mixture'):return PilotNominal.action(self,env,key)
        return super().action(env,key)
