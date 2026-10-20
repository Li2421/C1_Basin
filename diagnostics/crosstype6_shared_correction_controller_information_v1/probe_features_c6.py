"""Outcome-free feature formulas. Full fixed-denominator aggregation, failure masks."""
from common import *
from progress import progress

PER_ARM=[*[f'progress_{t}' for t in range(100,701,100)],'medium_first_stall_step_over_700','medium_ever_stall','medium_stall_fraction','strict_freeze_whole','moving_no_progress_whole','strict_freeze_after_p85','moving_no_progress_after_p85','projection_active_fraction','mean_rewrite_over_max_speed','probe_failed','p85_never_reached']
NAMES=[f'{arm}:{n}' for arm in ('safety','frozen_mean') for n in PER_ARM]+[f'frozen_mean_minus_safety_progress_{t}' for t in (200,400,700)]
assert len(NAMES)==39

def define():
    path=HERE/'PROBE_FEATURE_DEFINITION.json'
    write(path,dict(utc=utc(),names=NAMES,per_arm=PER_ARM,dimension=39,progress='mean_i clip(1-d_i(t)/d_i(0),0,1)',stall='t=50..699 before action: p(t)-p(t-50)<.010; first=700 if absent, separate ever flag',strict_freeze='mean over robot-frame indicators [distance>goal_tolerance and speed<.05], frames20..700',moving_no_progress='mean [distance>goal_tolerance and speed>=.05 and d(t-round(1/dt))-d(t)<.05], t=max(20,round(1/dt))..700',tail='same indicators restricted to frames from first p(t)>=.85 inclusive; absent tail=0 and explicit p85_never_reached=1',rewrite='mean over time and robots of norm(safe_action-nominal_action)/max_speed',failed_probe='raw feature values absent; fixed zeros contribute to sum/16, probe_failed=1; neither conditional valid-only average nor mean imputation',aggregation='sum of valid per-probe values divided by fixed16, with failure fraction; missing raw entries retained explicitly',progress_difference='per-state paired probe difference, fixed denominator16, missing pair contributes0 with arm failure masks',stall_group='Safety arm: median first stall step among valid stalled probes; <200 early, >=200 late; no-stall if all valid probes never stall; no valid probe = gap',eta_distance='For selected online action, norm(concat((early-human13)/RADIUS,(late-human10)/RADIUS)); requery uses stored realized eta for q0/1 if available, otherwise unresolved distance gap; median controller mean distance divides groups with deterministic controller-name tie break; never inserts human eta into training',family_groups='Learning=Flow,gaussian,deterministic,mixture,diffusion,chunking,recurrent,cvae,ring_ippo,ring_ippo_conservative,transformer_bc; Search=prioritized_astar,conflict_based_search; remainder optimization',no_success_or_outcome_reads=True))
    seal('PROBE_FEATURE_DEFINITION.receipt.json',[path,HERE/'probe_features_c6.py'],before_probe_rollout=True)

def one(record,config):
    if not record.get('complete') or record.get('error'):
        return None,dict(failed=True,reason=record.get('error') or 'incomplete700',steps=record.get('physical_steps'))
    assert record['terminal_labels_read'] is False and not any(k in record for k in ('success','outcome','summary'))
    with np.load(record['trace_path'],allow_pickle=False) as z:
        p=z['positions'];v=z['velocities'];g=z['goals'];nom=z['nominal_actions'];safe=z['actions'];active=z['projection_active']
    assert len(p)==701 and len(safe)==700
    dist=np.linalg.norm(p-g,axis=-1);d0=dist[0];pr=np.asarray([progress(x,g,d0) for x in p]);speed=np.linalg.norm(v,axis=-1);remaining=dist>config['goal_tolerance'];stall=(pr[50:700]-pr[:650])<.010;at=np.flatnonzero(stall)+50;tail=np.flatnonzero(pr>=.85);cut=int(tail[0]) if len(tail) else None
    strict=remaining&(speed<.05);window=round(1/config['dt']);ts=np.arange(max(20,window),701);moving=remaining[ts]&(speed[ts]>=.05)&((dist[ts-window]-dist[ts])<.05)
    sv=float(strict[20:].mean());mv=float(moving.mean());st=float(strict[max(20,cut):].mean()) if cut is not None else 0.;mt=float(moving[ts>=cut].mean()) if cut is not None and np.any(ts>=cut) else 0.
    values=[*[float(pr[t]) for t in range(100,701,100)],float(at[0]/700) if len(at) else 1.,float(bool(len(at))),float(stall.mean()),sv,mv,st,mt,float(active.mean()),float(np.linalg.norm(safe-nom,axis=-1).mean()/config['max_speed']),0.,float(cut is None)]
    assert len(values)==18 and np.isfinite(values).all()
    return np.asarray(values),dict(failed=False,first_stall_step=int(at[0]) if len(at) else None,first_p85=cut)

def assemble():
    base=read(HERE/'BASE_PROTOCOL.json');cases=read(HERE/'probe_CASES.json');names=base['controller_order'];values={n:np.zeros((2,16,18)) for n in names};raw=[];fail=[]
    for c in cases:
        r=read(HERE/'probe/raw'/f"{c['rollout_uid']}.json");v,metadata=one(r,base['native_manifest']['scenario_config']);ni=0 if c['probe_arm']=='safety' else 1;si=c['state_index']
        if v is None:values[c['base_controller_name']][ni,si,-2]=1.;fail.append(dict(controller=c['base_controller_name'],state_index=si,arm=c['probe_arm'],**metadata))
        else:values[c['base_controller_name']][ni,si]=v
        raw.append(dict(controller=c['base_controller_name'],state_index=si,arm=c['probe_arm'],values=None if v is None else v,**metadata))
    features=[]
    for n in names:
        x=values[n];valid=(x[0,:,-2]==0)&(x[1,:,-2]==0);diff=[float(np.sum((x[1,:,t//100-1]-x[0,:,t//100-1])*valid)/16) for t in (200,400,700)];features.append([*x.mean(1).reshape(-1),*diff])
    write(HERE/'PROBE_FEATURE_RAW.json',raw);write(HERE/'PROBE_FAILURES.json',fail)
    np.savez_compressed(HERE/'CONTROLLER_FEATURES_V2.npz',names=names,types=[base['types'][n] for n in names],features=np.asarray(features,np.float64),feature_names=NAMES)
    seal('CONTROLLER_FEATURES_V2.receipt.json',[HERE/n for n in ('PROBE_FEATURE_RAW.json','PROBE_FAILURES.json','CONTROLLER_FEATURES_V2.npz','PROBE_FEATURE_DEFINITION.receipt.json')],no_outcomes_used=True)
    print('Features:',len(names),'controllers;',len(fail),'failed probes retained',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
