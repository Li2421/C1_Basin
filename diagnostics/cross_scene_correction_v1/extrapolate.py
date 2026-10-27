"""Only five frozen selected actions on N10; no candidate-wide extrapolation search."""
from common import *
from scenes import make_new
from stages import read_result
from queueing import build
import numpy as np


def prepare():
    proof=read(HERE/'MODEL_FROZEN.receipt.json')
    for p,h in proof['files'].items():assert sha(p)==h,p
    scene=next(s for s in read(HERE/'SCENES.json')if s['family']=='extrapolation')
    selected=read(HERE/'SELECTED_CONTROLLERS.json')[scene['name']]
    choices=read(HERE/'EXTRAPOLATION_FROZEN_CHOICES.json')
    unique=sorted({choices[k]for k in ('ridge','knn','global_fixed','eta13','safety')});cases=[]
    for ctl in selected.values():
        if ctl is None:continue
        for state in scene['states']:
            for a in unique:cases.append(make_new(scene,ctl,state,actions()[a]))
    register('extrapolation',cases);build('extrapolation')
    write(HERE/'EXTRAPOLATION_ARMS.json',dict(methods={k:choices[k]for k in ('ridge','knn','global_fixed','eta13','safety')},
        unique_actions=unique,scene=scene['name'],states=12,seed=0,before_results=True))


def results():
    arms=read(HERE/'EXTRAPOLATION_ARMS.json');rows=[]
    for c in read(HERE/'extrapolation_CASES.json'):
        r=read_result('extrapolation',c)
        rows.append(dict(controller=c['controller_payload']['nominal']['controller'],type=c['controller_payload']['nominal']['type_code'],
            state=c['state']['index'],action=next(i for i,a in enumerate(actions())if etahex(a['eta_a'])==etahex(c['eta_a'])),
            success=r['success'],collision=r['collision'],timeout=r['timeout'],unknown=r['numerical_failure'],
            collision_types=r.get('collision_types',{}),rollout_uid=c['rollout_uid']))
    summary={};rng=np.random.default_rng(2026102414)
    for method,a in arms['methods'].items():
        selected=[r for r in rows if r['action']==a];ctls=sorted({r['controller']for r in selected})
        if not ctls:
            summary[method]=dict(status='NO_QUALIFIED_CONTROLLER');continue
        y=np.asarray([[next(r['success']for r in selected if r['controller']==ctl and r['state']==j)for j in range(12)]for ctl in ctls])
        u=np.asarray([[next(r['unknown']for r in selected if r['controller']==ctl and r['state']==j)for j in range(12)]for ctl in ctls])
        upper=np.minimum(1.,y+u)
        ci=rng.integers(len(ctls),size=(5000,len(ctls)));st=rng.integers(12,size=(5000,len(ctls),12))
        boot=y[ci[:,:,None],st].mean((1,2))
        upper_boot=upper[ci[:,:,None],st].mean((1,2))
        summary[method]=dict(action=a,eta=actions()[a]['eta_a'],rate=float(y.mean()),ci=np.percentile(boot,[2.5,97.5]).tolist(),
            success_upper_rate=float(upper.mean()),unknown_upper_ci=np.percentile(upper_boot,[2.5,97.5]).tolist(),
            collision_rate=float(np.mean([r['collision']for r in selected])),unknown=int(sum(r['unknown']for r in selected)),
            timeout=int(sum(r['timeout']for r in selected)),n=len(selected),
            collision_types={k:int(sum(r['collision_types'].get(k,0)for r in selected))for k in ('agent','static','moving')})
    write(HERE/'EXTRAPOLATION_RESULTS.json',dict(methods=summary,rows=rows,type_equal=True,only_selected_actions=True,
        unknown_bounds='Frozen selected policy; numerical failures lower0/upper1 with unchanged denominator; paired type/state draws for lower/upper intervals.',
        limitation='Gap1 fixed map N10 only; no assertion about other topologies or N100.'))


if __name__=='__main__':
    (prepare if sys.argv[1]=='prepare'else results)()
