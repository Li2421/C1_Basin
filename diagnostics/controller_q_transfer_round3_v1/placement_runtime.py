"""Frozen native Flow with placement or archived κ FF programs."""
from diagnostics.ring_state_selection_transfer_v1.c1.runtime import make as make_kappa


class PlacementProgram:
    def __init__(self, base, placement):
        assert placement in ('FF','TT','TF','FT')
        self.base=base
        self.placement=placement

    def __getattr__(self,key):return getattr(self.base,key)

    def action(self,env,key,eta,chain='field'):
        assert chain=='field'
        if self.placement=='FF':return self.base.action(env,key,eta,'field')
        if self.placement=='TT':return self.base.action(env,key,eta,'old')
        from validation_stage2.factorial_controller import action
        a,cert,calls,_=action(self.base,env,key,eta,self.placement)
        return a,cert,calls


def make_program(base,name):
    if name in ('FF','TT','TF','FT'):return PlacementProgram(base,name)
    assert name in ('rotate_negative','isotropic','rotate_positive')
    return make_kappa(base,name)
