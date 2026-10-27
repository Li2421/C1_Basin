"""Additional frozen-rule baselines, isolated cache namespace; original plant/filter."""
import json,time
from pathlib import Path
import numpy as np
import runner as R
from baseline_kernels import groups,clf_solve
from baseline_roundabout import Roundabout
BASE=Path(__file__).resolve().parent
OUT=BASE/'baselines';OUT.mkdir(exist_ok=True)
# Only this module's process changes runner globals. Existing dictionary source is untouched.
R.OUT=OUT;R.EXP='exp_basis_discovery_baselines_20261027'
anchor=';_,done,info=env.step(a)'
assert R.old_runtime.SOURCE.count(anchor)==1
SOURCE=R.old_runtime.SOURCE.replace('def run(c):','def run(c, post):').replace(anchor,
    '\n            a,second=post(env,a,ref,filters,t,second)\n            _,done,info=env.step(a)')
ns=dict(vars(R.RN));exec(compile(SOURCE,__file__,'exec'),ns)

def right(q,s=1.):return s*np.array([q[1],-q[0]])/max(np.linalg.norm(q),1e-30)

class Rule:
    def __init__(self,spec):
        self.P=spec['baseline'];self.hist={};self.until=None;self.mgr=None
        self.stats=dict(calls=0,seconds=0.,fired_steps=0,clf_problems=0,clf_failures=0,clf_errors={},clf_reproject=0,clf_max_reproject=0.,clf_delta_max=0.)
    def __call__(self,env,a,ref,filters,t,second):
        start=time.perf_counter();P=self.P;name=P['rule'];q=env.positions;g=env.goals;d=np.linalg.norm(g-q,axis=1);arr=d<=env.config.goal_tolerance
        W=P.get('W',40);self.hist[t]=d.copy();self.hist.pop(t-W-1,None)
        stuck=(self.hist[t-W]-d<P.get('thr',.05))&~arr if t>=W else np.zeros(len(q),bool)
        target={}
        if name in ('rhr_goal','rhr_centre','rhr_always'):
            if self.until is None:self.until=np.zeros(len(q),int)
            self.until[stuck]=t+P.get('H',1)
            active=(self.until>t)&~arr if name!='rhr_always' else ~arr
            for i in np.flatnonzero(active):target[i]=a[i]+right(q[i]if name=='rhr_centre'else g[i]-q[i],P['s'])
        elif name=='merry_heuristic':
            for comp in groups(q,arr,stuck):
                ctr=q[comp].mean(0)
                for i in comp:
                    z=q[i]-ctr
                    if np.linalg.norm(z)<1e-9:z=g[i]-q[i]
                    target[i]=a[i]+right(z,P['s'])
        elif name=='merry_paper_si':
            if self.mgr is None:self.mgr=Roundabout(env)
            target,_=self.mgr.update(env,arr)
        elif name=='clf_cbf':
            for i in np.flatnonzero(~arr):
                self.stats['clf_problems']+=1
                try:
                    x,delta=clf_solve(filters[i],ref[i],q[i]-g[i],P.get('lambda',1.),P['p'])
                    self.stats['clf_delta_max']=max(self.stats['clf_delta_max'],delta)
                except Exception as exc:
                    self.stats['clf_failures']+=1;key=type(exc).__name__+': '+str(exc);self.stats['clf_errors'][key]=self.stats['clf_errors'].get(key,0)+1
                    x=filters[i].project(ref[i:i+1])[0][0]
                target[i]=x
        else:raise ValueError(name)
        for i,x in target.items():
            second[i]=filters[i].project(np.asarray(x)[None]);a[i]=second[i][0][0]
            if name=='clf_cbf':
                diff=float(np.linalg.norm(x-a[i]));self.stats['clf_reproject']+=int(diff>1e-7);self.stats['clf_max_reproject']=max(self.stats['clf_max_reproject'],diff)
        self.stats['calls']+=1;self.stats['fired_steps']+=int(bool(target));self.stats['seconds']+=time.perf_counter()-start
        return a,second
    def final(self):
        return dict(self.stats,roundabout_created=self.mgr.created if self.mgr else 0,roundabout_invalid=self.mgr.invalid if self.mgr else 0,roundabout_exits=self.mgr.exits if self.mgr else 0)

def run(c):
    hook=Rule(c['spec']);r,tr=ns['run'](c,hook);cfg=c['scenario_payload']['config'];dt=cfg['dt']
    d=np.linalg.norm(tr['goals'][None]-tr['positions'],axis=-1).sum(-1);reached=np.flatnonzero(d<=d[0]-.1)
    r.update(completion_time=r['episode_length']*dt if r['success']else None,truncated_time=r['episode_length']*dt if r['success']else cfg['max_steps']*dt,
        horizon_time=cfg['max_steps']*dt,local_progress_time=float(reached[0]*dt)if len(reached)else None,baseline=hook.final())
    return R.clean(r),tr
R.run=run

def make(c,params,seed):
    scenes,ctls,*_=R.inputs();ctl=next(x for x in ctls if x['type_code']==c['controller'])
    cc=R.make_case(scenes[c['scene']],ctl,c['state_index'],dict(eta=[0.,0.,0.],coefficients={}),seed)
    cc['spec']['baseline']=params
    cc['controller_payload'].update(dictionary=cc['spec'],external_baseline=params,baseline_sources={p.name:R.file_sha(p)for p in [Path(__file__),BASE/'baseline_kernels.py',BASE/'baseline_roundabout.py']},correction_semantics='baseline_post_original_safety_v1')
    cc['controller_uid']=R.db.uid('ctl',dict(scenario=cc['scenario_uid'],payload=cc['controller_payload']))
    cc['rollout_uid']=R.db.uid('roll',dict(state=cc['state_uid'],eta=cc['eta_uid'],controller=cc['controller_uid'],seed=cc['seed_key']))
    return cc

def execute(cases,stage):
    mark=OUT/f'{stage}_execution.json'
    if mark.exists()and json.load(mark.open()).get('complete'):
        ids={c['rollout_uid']for c in json.load(open(OUT/f'{stage}_planned.json'))['exact_configs']}
        return [r for r in map(json.loads,open(OUT/'candidates.jsonl'))if r['rollout_uid']in ids]
    return R.execute(cases,stage)

def main():
    p=json.load(open(BASE/'PROTOCOL.json'));ctls=R.inputs()[1]
    methods={'RHR_A':dict(rule='rhr_goal',W=40,thr=.05,s=.35,H=1),'RHR_persistent':dict(rule='rhr_always',s=.1),
      'Merry_heuristic':dict(rule='merry_heuristic',W=40,thr=.05,s=.35),'Merry_paper_SI':dict(rule='merry_paper_si'),
      'CLF_CBF_p1':dict(rule='clf_cbf',p=1.,**{'lambda':1.}),'CLF_CBF_p100':dict(rule='clf_cbf',p=100.,**{'lambda':1.})}
    # Declaration precedes rollout; all tuning restricted to original 22 development cases.
    menu=[dict(rule=rule,W=w,thr=.05,s=s,H=h)for rule in ('rhr_goal','rhr_centre')for w,s,h in [(40,.35,1),(20,.35,1),(40,.5,1),(40,.35,40),(20,.5,40),(40,.15,40)]]
    menu += [dict(rule='rhr_always',s=s)for s in (.1,.35)]
    frozen=OUT/'DESIGN.json'
    design=dict(methods=methods,enhanced_menu=menu,development_cases=p['cases'],splits=dict(test=dict(states=[18,19],seed=3),confirm=dict(states=[20,21],seed=4)),scenes=p['test_scenes'],
       history='Pure functions reused from baseline_comparison_v1/code; old learned-controller rollouts incompatible with two frozen native nominals and new seeds. Legacy stopped coordinator was not imported or resumed.',
       selection='Enhanced RHR minimizes (-success,collisions,truncated_time,canonical_params) over22 development cases; one global setting, no heldout selection.')
    if frozen.exists():assert json.load(frozen.open())==R.clean(design)
    else:R.dump(frozen,design)
    # Small real smoke precedes expansion; these exact IDs remain part of full study.
    smoke=[make(p['cases'][0],sp,0)for sp in methods.values()]
    rr=execute(smoke,'smoke');assert not any(r['numerical_failure']for r in rr)
    todo=[make(c,sp,0)for c in p['cases']for sp in menu];rr=execute(todo,'enhanced_train')
    path=OUT/'ENHANCED_FROZEN.json'
    if path.exists():best=json.load(path.open())['params']
    else:
        ranking=[]
        for sp in menu:
            hits=[r for r in rr if r['spec']['baseline']==sp]
            ranking.append(dict(params=sp,success=sum(r['success']for r in hits),collision=sum(r['collision']for r in hits),truncated_time=float(np.mean([r['truncated_time']for r in hits]))))
        ranking.sort(key=lambda x:(-x['success'],x['collision'],x['truncated_time'],R.digest(x['params'])))
        best=ranking[0]['params'];R.dump(path,dict(params=best,ranking=ranking,no_heldout_selection=True))
    methods['RHR_enhanced']=best
    labels={};todo=[]
    for split,ds in design['splits'].items():
        labels[split]={}
        for sc in design['scenes']:
            for ctl in ctls:
                for si in ds['states']:
                    c=dict(scene=sc,controller=ctl['type_code'],state_index=si)
                    for name,sp in methods.items():
                        cc=make(c,sp,ds['seed']);todo.append(cc);labels[split].setdefault(name,[]).append(cc['rollout_uid'])
    R.dump(OUT/'LABELS.json',labels);execute(todo,'independent')

if __name__=='__main__':main()
