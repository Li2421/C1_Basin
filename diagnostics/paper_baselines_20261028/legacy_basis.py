"""Frozen selected G+C2 on original four checkpoint/TT/FF runtimes.

Original safety geometry supplies all its pair/wall terms; the Ring polygon
seam already existed in the archived runtime and is not replaced here.
"""
import sys,os,json,inspect,time,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'b5_oracle_20261027'))
import legacy as L
import numpy as np
L.OUT=HERE/'legacy';L.OUT.mkdir(exist_ok=True);L.EXP='exp_dictionary_legacy_20261028'

def clearance_column(env,safety):
    p=np.asarray(env.positions);geom=safety.geometry;speed=env.config.max_speed;safe_d=geom['d_safe'];tau=.25*safe_d
    out=np.zeros_like(p);D=0.
    for i in range(len(p)):
        terms=[]
        for j in range(len(p)):
            if i==j:continue
            delta=p[i]-p[j];dist=np.linalg.norm(delta);terms.append((dist-safe_d,delta/max(dist,1e-30),j))
        for h,n in zip(geom['wall_h'][i],geom['wall_gradient'][i]):terms.append((h,np.asarray(n),None))
        gaps=np.asarray([v[0]for v in terms]);m=gaps.min();e=np.exp(-(gaps-m)/tau);weights=e/e.sum();D+=m-tau*np.log(e.sum())
        for w,(_,normal,j)in zip(weights,terms):
            out[i]+=w*normal
            if j is not None:out[j]-=w*normal
    out*=speed;out*=np.minimum(1.,speed/np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-30))
    return out

source=inspect.getsource(L.work)
anchor="            elif spec['mode']=='b3':act,_=safety.project(b3)"
assert source.count(anchor)==1
source=source.replace(anchor,"            elif spec['mode']=='dictionary_tt':act,_=safety.project(b3+spec['C2']*clearance_column(env,safety))\n"+anchor)
anchor="                act,_=rt.integrate(env,obs,key,dd,safety)"
assert source.count(anchor)==1
source=source.replace(anchor,"                if spec['mode']=='ff_dictionary':dd+=spec['C2']*clearance_column(env,safety)\n"+anchor)
ns=dict(vars(L),clearance_column=clearance_column);exec(compile(source,str(HERE/'generated_legacy_dictionary.py'),'exec'),ns);CORE=ns['work']
def work(c):return CORE(c)

def plan():
    old=json.load(open(HERE.parent/'b5_oracle_20261027/LEGACY_FROZEN.json'))['tasks'];selected=json.load(open(HERE/'basis/FINAL_FROZEN.json'))['best']['spec'];assert set(selected['coefficients'])=={'C2'}
    tasks=[]
    for c in old:
        if c['method']!='B3':continue
        for mode in ('dictionary_tt','ff_dictionary'):
            cc=copy.deepcopy(c);cc['method']='GC2_'+('TT'if mode=='dictionary_tt'else'FF');cc['spec']=dict(mode=mode,eta=selected['eta']+[0.,0.],C2=selected['coefficients']['C2'])
            cc['payload'].update(method=cc['method'],spec=cc['spec'],dictionary_adapter_sha256=L.sha(__file__),selected_from='development60_fixed_configs_before_new_test',geometry='Original joint safety geometry; all original pair and wall rows; identical softmin C2 functional; Ring retains original48-gon seam')
            cc['controller_uid']=L.db.uid('ctl',cc['payload']);cc['eta_uid']=L.db.eta_identity(selected['eta'])[0];cc['rollout_uid']=L.db.uid('roll',dict(state=cc['state_uid'],eta=cc['eta_uid'],controller=cc['controller_uid'],seed=cc['seed_key']));tasks.append(cc)
    L.dump(L.OUT/'FROZEN.json',dict(tasks=tasks,evidence='Original B3/safety/FF_B3 seed16/17 outcomes already exist and will be reused. Missing only selected GC2 TT/FF transfer on identical frozen checkpoints/initial states/future seeds. No modifications to archived runtime or production solver.'))
    L.register(tasks);print('legacy planned',len(tasks))

def main():
    if '--plan'in sys.argv:plan();return
    tasks=json.load(open(L.OUT/'FROZEN.json'))['tasks'];missing=L.register(tasks);path=L.OUT/'candidates.jsonl';done={r['rollout_uid']for r in map(json.loads,path.open())}if path.exists()else set();todo=[c for c in tasks if c['rollout_uid']not in done]
    assert all(tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))in missing for c in todo)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp,subprocess
    with path.open('a')as f,ProcessPoolExecutor(max_workers=min(8,len(os.sched_getaffinity(0))),mp_context=mp.get_context('spawn'),initializer=L.init_worker)as pool:
        for fu in as_completed([pool.submit(work,c)for c in todo]):
            r=fu.result();f.write(json.dumps(r)+'\n');f.flush();print(r['scenario'],r['method'],r['outcome'],r['error'],flush=True)
    subprocess.run([sys.executable,'-m','shared_rollout_db.src.merger','--experiment',L.EXP],cwd=ROOT,check=True,stdout=(L.OUT/'merge.log').open('w'))
    L.dump(L.OUT/'execution.json',dict(complete=True,new=len(todo),requested=len(tasks)))
if __name__=='__main__':main()
