"""Unlabeled partial trajectories. Never inspect termination or success fields."""
from core import *
import dataclasses,time
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for

def execute(index,workers):
    prereg();gate();p=read(HERE/'protocol.json');allowed={p['probe_sha256']};allowed.add(read(HERE/'PROBE_IO_REPAIR.receipt.json')['after_sha256']) if (HERE/'PROBE_IO_REPAIR.receipt.json').exists() else None;assert sha(__file__) in allowed;assert sha(HERE/'core.py')==p['core_sha256'];receipt=read(HERE/'probe_PREFLIGHT.receipt.json');assert receipt['protocol_sha256']==sha(HERE/'protocol.json') and receipt['preflight_sha256']==sha(HERE/'probe_CACHE_PREFLIGHT.json')
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    import jax
    states={s['state_uid']:s for s in p['states']};bases={};runtimes={};rows=read(HERE/'probe_MISSING_CASES.json');rows=[rows[k] for k in range(index,len(rows),workers)];done=0
    write(HERE/'probe'/f'START_AUTHORIZATION_{index}.json',dict(utc=utc(),worker=index,preflight_receipt_sha256=sha(HERE/'probe_PREFLIGHT.receipt.json'),protocol_sha256=sha(HERE/'protocol.json')))
    for c in rows:
        path=HERE/'probe/raw'/f"{c['rollout_uid']}.json"
        if path.exists():continue
        sc=c['scene'];variant=c['variant'];v=p['base_profiles'][sc]['flow_isotropic']['payload'];x64=v['jax_enable_x64'];jax.config.update('jax_enable_x64',x64)
        if variant not in bases:
            base=native_runtime(sc);assert sha(base.checkpoint)==v['flow_checkpoint_sha256'];base.config=dataclasses.replace(base.config,**p['geometries'][variant]);base.config=dataclasses.replace(base.config,terminate_on_success=False,terminate_on_collision=False,**({'terminate_on_deadlock':False} if hasattr(base.config,'terminate_on_deadlock') else {}));bases[variant]=base
        if variant not in runtimes:runtimes[variant]=FlowEndpointMatchedField(bases[variant],'isotropic')
        rt=runtimes[variant];s=states[c['state_uid']];s_for_env=dict(s,metadata=dict(s['metadata'],regime=('clearly_asymmetric','weakly_asymmetric','near_symmetric')[s['study_index']%3]) if sc=='double_bottleneck' else s['metadata']);env=rt.make_env(s_for_env);n=len(env.positions);initial=np.linalg.norm(env.goals-env.positions,axis=1);streak=np.zeros(n,int);ever=np.zeros(n,bool);minpair=np.inf;error=None;t0=time.monotonic();steps=0
        for t in range(c['probe_steps']):
            try:
                a,_,_=rt.action(env,key_for(s,0,t),np.asarray(c['eta']),'field');_,_,_,_=env.step(a);steps+=1
                speed=np.linalg.norm(env.velocities,axis=1);streak=np.where(speed<.01,streak+1,0);ever|=streak>=40
                dp=env.positions[:,None]-env.positions[None,:];d=np.linalg.norm(dp,axis=-1);d[np.diag_indices(n)]=np.inf;minpair=min(minpair,float(np.min(d)))
            except Exception as ex:error=dict(step=t,type=type(ex).__name__,message=str(ex)[:300]);break
        progress=float(np.mean((initial-np.linalg.norm(env.goals-env.positions,axis=1))>0))
        write(path,dict(schema='round20_unlabeled_probe_v1',case=c,steps=steps,stall_fraction=float(np.mean(ever)),progress_fraction=progress,minimum_pair_distance=minpair if np.isfinite(minpair) else None,error=error,elapsed_seconds=time.monotonic()-t0,execution_code_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),contains_success_label=False,contains_termination_label=False))
        done+=1
        if done%8==0:print(json.dumps(dict(worker=index,done=done,total=len(rows))),flush=True)
    write(HERE/'probe'/f'worker_{index}.json',dict(utc=utc(),assigned=len(rows),new=done,complete=True))
if __name__=='__main__':execute(int(sys.argv[1]),int(sys.argv[2]))
