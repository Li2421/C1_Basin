"""Round19b R5 probe: 200 simulator steps without reading terminal labels."""
from __future__ import annotations
import argparse,datetime,hashlib,json,sys,time
from dataclasses import replace
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from adapter import FourFlowAdapter
from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from poc.field_controller import Runtime
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def checks():
    for g in ('G0','G1A','G2'):assert read(HERE/f'{g}_REPORT.json')['status']=='PASS'
    assert read(HERE/'STATE_FREEZE.receipt.json')['fresh_states_sha256']==sha(HERE/'FRESH_STATES.json')
    assert not (HERE/'STAGE2_LABELS.json').exists()

class SafetyTracker:
    def __init__(self,base):self.base=base;self.active=False
    def __getattr__(self,k):return getattr(self.base,k)
    def project(self,target):
        a,d=self.base.project(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
    def project_field_target(self,target):
        a,d=self.base.project_field_target(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
class ProbeController:
    def __init__(self,rt):self.rt=rt;self.last=None
    def __getattr__(self,k):return getattr(self.rt,k)
    def safety(self,env):self.last=SafetyTracker(self.rt.safety(env));return self.last
    def prepare(self,*args,**kwargs):return Runtime.prepare(self,*args,**kwargs)
    def action(self,env,key,eta):return Runtime.action(self,env,key,eta,'field')

def worker(index):
    checks();names=read(HERE/'G2_REPORT.json')['checks'];assert len(names)==25
    c=names[index];name=c['name']+'__'+c['flavor'];ck=Path(c['checkpoint_path']);assert sha(ck)==c['checkpoint_sha256']
    eta=read(ROOT/'diagnostics/controller_ff_factorial_repair_v1/protocol.json')['eta']
    states=[s for s in read(HERE/'FRESH_STATES.json') if s['study_split']=='probe'];assert len(states)==16
    rt=FlowEndpointMatchedField(FourFlowAdapter(ck),'isotropic');probe=ProbeController(rt)
    out=HERE/'stage1'/'raw'/name;out.mkdir(parents=True,exist_ok=True)
    count=0
    for state in states:
        for k in (10,13,15):
            path=out/f"state{state['study_index']:02d}_eta{k}.json"
            if path.exists():continue
            env=rt.make_env(state);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False)
            p0=env.positions.copy();positions=[p0];actions=[];active=[];error=None;t0=time.monotonic()
            for t in range(200):
                try:
                    action,_,_=probe.action(env,key_for(state,0,t),np.asarray(eta[k]))
                    env.step(action) # Return value intentionally ignored: no task label is read.
                    positions.append(env.positions.copy());actions.append(np.asarray(action).copy());active.append(bool(probe.last.active))
                except Exception as exc:
                    error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);break
            positions=np.asarray(positions);actions=np.asarray(actions)
            complete=len(actions)==200;stats=None
            if complete:
                initial=np.linalg.norm(env.goals-p0,axis=-1).sum();final=np.linalg.norm(env.goals-positions[-1],axis=-1).sum()
                distances=np.linalg.norm(positions[:,:,None,:]-positions[:,None,:,:],axis=-1)
                pairs=np.triu_indices(4,1);speeds=np.linalg.norm(actions,axis=-1)
                stats=[float((initial-final)/initial),float(np.mean(active)),float(distances[:,pairs[0],pairs[1]].min()),
                       float(speeds.mean()),float((speeds.mean(-1)<.01).sum())]
            trace=path.with_suffix('.npz')
            np.savez_compressed(trace,positions=positions,actions=actions,projection_active=np.asarray(active),goals=env.goals)
            write(path,dict(schema='round19b_unlabeled_R5_probe_v1',controller=name,checkpoint_sha256=c['checkpoint_sha256'],
                            state_uid=state['state_uid'],state_index=state['study_index'],eta_index=k,eta=eta[k],seed=0,
                            physical_steps=len(actions),complete=complete,statistics=stats,error=error,
                            trace_path=str(trace),trace_sha256=sha(trace),elapsed_seconds=time.monotonic()-t0,
                            terminal_labels_read=False,probe_code_sha256=sha(__file__)))
            count+=1
    write(HERE/'stage1'/f'worker_{index:02d}.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          controller=name,expected=48,new=count,complete=True,checkpoint_sha256=c['checkpoint_sha256']))
    print(json.dumps(dict(worker=index,controller=name,new=count,expected=48)),flush=True)

def freeze():
    checks();names=read(HERE/'G2_REPORT.json')['checks'];states=[s for s in read(HERE/'FRESH_STATES.json') if s['study_split']=='probe']
    order=[x['name']+'__'+x['flavor'] for x in names];features=[];files=[];unknown=[]
    for i,name in enumerate(order):
        assert read(HERE/'stage1'/f'worker_{i:02d}.json')['complete']
        row=[]
        for state in states:
            for k in (10,13,15):
                p=HERE/'stage1'/'raw'/name/f"state{state['study_index']:02d}_eta{k}.json";r=read(p)
                assert r['terminal_labels_read'] is False and not any(key in r for key in ('outcome','success','timeout','collision'))
                assert r['trace_sha256']==sha(r['trace_path'])
                assert r['state_uid']==state['state_uid'] and r['eta_index']==k and r['controller']==name
                row.extend(r['statistics'] if r['complete'] else [float('nan')]*5)
                if not r['complete']:unknown.append(dict(controller=name,state=state['study_index'],eta=k,error=r['error']))
                files.append(dict(path=str(p),sha256=sha(p),trace_sha256=r['trace_sha256']))
        assert len(row)==240;features.append(row)
    raw=np.asarray(features,dtype=np.float32);assert raw.shape==(25,240)
    mean=np.nanmean(raw.astype(np.float64),axis=0);assert np.isfinite(mean).all(), 'entire R5 column missing'
    filled=np.where(np.isfinite(raw),raw,mean).astype(np.float64)
    std=filled.std(axis=0);std=np.where(std<1e-6,1.,std)
    centered=(filled-filled.mean(axis=0))/std
    _,_,v=np.linalg.svd(centered,full_matrices=False);components=v[:2].copy()
    for j in range(2):
        if components[j,np.argmax(np.abs(components[j]))]<0:components[j]*=-1
    scores=centered@components.T
    assert np.isfinite(scores).all()
    dest=HERE/'R5_FROZEN.npz';np.savez_compressed(dest,raw=raw,imputation_mean=mean,scale=std,
        components=components,scores=scores,controller_names=np.asarray(order))
    write(HERE/'R5_RAW_INDEX.json',files)
    receipt=dict(status='FROZEN_BEFORE_ANY_NEW_SUCCESS_LABEL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 raw_index_sha256=sha(HERE/'R5_RAW_INDEX.json'),features_sha256=sha(dest),
                 probe_code_sha256=sha(__file__),adapter_sha256=sha(HERE/'adapter.py'),
                 states_sha256=sha(HERE/'FRESH_STATES.json'),shape=list(raw.shape),unknown_probes=unknown,
                 preprocessing='column mean imputation; population zscore sd<1e-6->1; PCA2 SVD largest absolute loading positive')
    write(HERE/'R5_FREEZE.receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],shape=list(raw.shape),unknown=len(unknown))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['worker','freeze']);ap.add_argument('--index',type=int,default=0);a=ap.parse_args()
    if a.command=='worker':worker(a.index)
    else:freeze()
