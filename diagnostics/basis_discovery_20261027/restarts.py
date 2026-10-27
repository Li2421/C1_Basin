"""Recovery screen initialized at genuine failed B3 trajectory states.

Same physical model, nominal configuration, speed limits, safety and original
horizon. These are new matched future streams, not replay of the old RNG prefix.
"""
import json,copy
import numpy as np
from runner import OUT,inputs,dump,digest,make_case,old_oracle,historical
from discovery_design import freeze,singles
from screen import pool_plan


def freeze_restart():
    path=OUT/'RESTART_FROZEN.json'
    if path.exists():return json.load(path.open())
    scenes,ctls,_,_=inputs();states=json.load(open(OUT/'static_states.json'));old={r['rollout_uid']:r for r in historical()};tasks=[]
    for diagnostic in states:
        c=diagnostic['case']
        if not c['hard']:continue
        scene=copy.deepcopy(scenes[c['scene']]);state=copy.deepcopy(scene['states'][c['state_index']]);tr=np.load(diagnostic['source_trace']);t=diagnostic['step']
        state.update(positions=tr['positions'][t].tolist(),velocities=tr['velocities'][t].tolist(),goals=tr['goals'].tolist())
        scene['name']=c['scene']+'__restart'+str(c['state_index']);scene['states']=[state]
        scene['definition_sha256']=digest(dict(original=scene['definition_sha256'],source=diagnostic['source_trace'],step=t,physical=state))
        tasks.append(dict(scene=scene,controller=c['controller'],state_index=0,baseline_eta=c['baseline_eta'],
            original_case=c,source_trace=diagnostic['source_trace'],source_step=t,selected_speed=diagnostic['selected_speed']))
    x=dict(cases=tasks,seed=3,semantics='Reset at saved physical state; full original horizon, new common future stream. Nominal/safety/physics unchanged. Not strict deadlock labels.',
        library='All15 canonical columns receive same4-point single screen, only on residual B3@16 recovery failures.')
    dump(path,x);return x


def main():
    frozen=freeze_restart();ctls=inputs()[1];seed=frozen['seed'];labels={};todo=[]
    def make(c,sp):return make_case(c['scene'],next(x for x in ctls if x['type_code']==c['controller']),0,sp,seed)
    for c in frozen['cases']:
        for sp in old_oracle.b3_pool()[:16]:todo.append(make(c,dict(eta=sp['eta'][:3],coefficients={})))
    base=pool_plan('restart_b3',todo,None);failed=[c for c in frozen['cases']if not any(r['success']for r in base if r['scene']==c['scene']['name'])]
    todo=[]
    for c in failed:
        for sp in singles(c):
            # Four main configurations per class; recipe variants are a separate
            # previous experiment and are not silently counted at equal budget.
            if sp.get('t5_seed','tangent')!='tangent'or sp.get('clearance_side',1)!=1:continue
            cc=make(c,sp);todo.append(cc);name=next(iter(sp['coefficients']));labels.setdefault(name,[]).append(cc['rollout_uid'])
    dump(OUT/'RESTART_LABELS.json',dict(labels=labels,failed_after_B3=[c['scene']['name']for c in failed]))
    pool_plan('restart_single',todo,None)

if __name__=='__main__':main()
