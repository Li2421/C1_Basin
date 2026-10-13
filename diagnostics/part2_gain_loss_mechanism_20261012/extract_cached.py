"""Read-only, paired seed-0 physical/monitor feature extraction."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREV=ROOT/'diagnostics/part2_closed_loop_sensitivity_validation_20261012'
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
PAIRS={'toy_give_way':('FT','FF'),'ring_exchange':('TT','TF')}
HORIZONS=(10,20,80)
SEP=.05

def loadcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def label(a,b):
    if 'UNKNOWN' in (a,b):return 'unknown'
    if a=='TRUE' and b=='FALSE':return 'loss'
    if a=='FALSE' and b=='TRUE':return 'gain'
    return 'both_success' if a=='TRUE' else 'both_failure'

def split(uid):
    k=int(uid[-2:]);return 'discovery' if k<=8 else 'validation' if k<=16 else 'heldout_test'

def monitor(pos,actions,goals,cfg):
    # Exact recorded-position replay of environment.py:195-219. There is no
    # simulator call or recomputed controller action.
    errors=np.linalg.norm(pos-goals[None],axis=2)
    dt=float(cfg['dt']);window=float(cfg['progress_window_seconds'])/dt
    since=None;first=None;longest=0.;first_deadlock=None;candidates=[]
    for k in range(1,len(pos)):
        s=k-window;ready=s>=-1e-10
        if ready:
            s=max(0.,s);lo=math.floor(s);hi=math.ceil(s);alpha=s-lo
            past=(1-alpha)*errors[lo]+alpha*errors[hi]
            progress=past-errors[k]
            candidate=(not np.all(errors[k]<=cfg['goal_tolerance'])
                and np.max(np.abs(progress))<cfg['progress_epsilon']
                and np.linalg.norm(actions[k-1],axis=1).max()<cfg['max_speed']*cfg['speed_epsilon_fraction'])
        else:candidate=False
        if candidate:
            if first is None:first=k
            if since is None:since=k
            timer=(k-since)*dt;longest=max(longest,timer)
            if first_deadlock is None and timer>=cfg['deadlock_hold_seconds']-1e-10:first_deadlock=k
            candidates.append(k)
        else:since=None
    return dict(first_candidate=first,max_stuck_timer=longest,first_deadlock=first_deadlock,
                candidate_steps=len(candidates),candidate_steps_first80=sum(k<=80 for k in candidates))

def measure(pos,actions,goals,h):
    t=min(h,len(pos)-1);p=pos[t];init=pos[0]
    err0=np.linalg.norm(init-goals,axis=1);err=np.linalg.norm(p-goals,axis=1)
    progress=err0-err
    a=actions[:min(h,len(actions))]
    speeds=np.linalg.norm(a,axis=2)
    # Goal-aligned action rate at each realized state, not final label.
    g=goals[None]-pos[:len(a)];unit=g/np.maximum(np.linalg.norm(g,axis=2,keepdims=True),1e-12)
    aligned=np.sum(a*unit,axis=2)
    pair=np.linalg.norm(pos[:t+1,0]-pos[:t+1,1],axis=1)
    out=dict(progress_min=float(progress.min()),progress_sum=float(progress.sum()),
        progress_asymmetry=float(np.abs(progress.max()-progress.min())),
        goal_error_max=float(err.max()),aligned_rate_min=float(aligned.mean(axis=0).min()) if len(a) else None,
        aligned_rate_sum=float(aligned.mean(axis=0).sum()) if len(a) else None,
        speed_min=float(speeds.mean(axis=0).min()) if len(a) else None,
        speed_asymmetry=float(np.ptp(speeds.mean(axis=0))) if len(a) else None,
        min_pair_distance=float(pair.min()))
    if pos.shape[1]==2:
        # Bay entry and yielding are geometric behavior markers, not outcomes.
        bay=(np.abs(pos[:t+1,:,0])<=.2)&(pos[:t+1,:,1]>.2)
        out.update(bay_entry_any=int(bay.any()),bay_entry_steps=int(bay.any(axis=1).sum()),
            max_y=float(pos[:t+1,:,1].max()),min_y=float(pos[:t+1,:,1].min()),
            x_order_reversed=int(np.any((pos[:t+1,0,0]-pos[:t+1,1,0])>0)))
    else:
        ang=np.unwrap(np.arctan2(pos[:t+1,:,1],pos[:t+1,:,0]),axis=0)
        adv=ang[-1]-ang[0]
        radii=np.linalg.norm(pos[:t+1],axis=2)
        out.update(angular_min_abs=float(np.abs(adv).min()),angular_sum_abs=float(np.abs(adv).sum()),
            angular_coherence=float(abs(adv.sum())/max(np.abs(adv).sum(),1e-12)),
            min_radius=float(radii.min()))
    return out

def main():
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    b15={(r['uid'],int(r['eta_index']),r['placement']):r for r in loadcsv(PREV/'b15_by_candidate.csv')}
    pairs=[r for r in loadcsv(PREV/'gain_loss_audit.csv') if r['seed']=='0' and (r['old'],r['new'])==PAIRS.get(r['scene'])]
    assert len(pairs)==768
    cfg={}
    # Toy monitor defaults are confirmed against frozen checkpoint provenance
    # by using the printed config fields; validation checks replay below.
    cfg['toy_give_way']=dict(dt=.05,progress_window_seconds=2.,deadlock_hold_seconds=5.,
        progress_epsilon=.01,max_speed=.5,speed_epsilon_fraction=.05,goal_tolerance=.08)
    rows=[];monitor_checks=[]
    for r in pairs:
        uid=r['uid'];sc=r['scene'];ei=int(r['eta_index']);old,new=PAIRS[sc]
        bs=b15[(uid,ei,old)];bn=b15[(uid,ei,new)]
        out=dict(scene=sc,uid=uid,eta_index=ei,split=split(uid),old=old,new=new,
            b15_old=bs['status'],b15_new=bn['status'],b15_outcome=label(bs['status'],bn['status']),
            old_successes=int(bs['successes']),new_successes=int(bn['successes']))
        out['initial_pair_distance']=float(np.linalg.norm(np.diff(np.asarray(states[uid]['physical']['positions']),axis=0))) if sc=='toy_give_way' else ''
        out['initial_goal_error_max']=float(np.linalg.norm(np.asarray(states[uid]['physical']['positions'])-np.asarray(states[uid]['physical']['goals']),axis=1).max())
        eta=np.asarray(json.loads(Path(r['old_result']).read_text())['eta'])
        out.update(eta_0=eta[0],eta_1=eta[1],eta_2=eta[2],eta_norm=float(np.linalg.norm(eta)))
        trajectories={}
        for ch,path in ((old,r['old_result']),(new,r['new_result'])):
            meta=json.loads(Path(path).read_text());out[ch+'_termination']=meta['termination'];out[ch+'_seed0_success']=int(meta['success'])
            out[ch+'_max_pre_terminal_violation']=meta.get('max_pre_terminal_violation')
            if meta['error'] is not None:
                out[ch+'_numerical_failure']=1;continue
            with np.load(Path(path).with_suffix('.npz')) as z:
                trajectories[ch]={k:z[k].copy() for k in ('positions','actions','goals','noise_keys','certificates')}
            d=trajectories[ch];out[ch+'_numerical_failure']=0
            out[ch+'_first_active_cbf_80']=next((i+1 for i,v in enumerate(d['certificates'][:80,0]) if v<1e-6),'')
            out[ch+'_active_cbf_fraction_80']=float(np.mean(d['certificates'][:80,0]<1e-6))
            out[ch+'_first_goal_step']=next((i for i,e in enumerate(np.linalg.norm(d['positions'][1:]-d['goals'][None],axis=2),start=1) if np.all(e<=(.08 if sc=='toy_give_way' else .09))),'')
            for h in HORIZONS:
                for k,v in measure(d['positions'],d['actions'],d['goals'],h).items():out[f'{ch}_H{h}_{k}']=v
            if sc=='toy_give_way':
                mon=monitor(d['positions'],d['actions'],d['goals'],cfg[sc])
                out.update({ch+'_'+k:v if v is not None else '' for k,v in mon.items()})
                ps=meta['plant_summary'];check=abs(mon['max_stuck_timer']-float(ps['max_stuck_timer']))<1e-8
                monitor_checks.append(check)
                out[ch+'_monitor_replay_match']=int(check)
        if old in trajectories and new in trajectories:
            a=trajectories[old];b=trajectories[new]
            assert np.array_equal(a['noise_keys'][0],b['noise_keys'][0])
            out['first_action_l2']=float(np.linalg.norm(a['actions'][0]-b['actions'][0]))
            g=a['goals']-a['positions'][0];unit=g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
            out['first_goal_alignment_delta']=float(np.sum((b['actions'][0]-a['actions'][0])*unit))
            out['first_goal_alignment_min_delta']=float(np.min(np.sum((b['actions'][0]-a['actions'][0])*unit,axis=1)))
            out['noise_keys_first80_match']=int(np.array_equal(a['noise_keys'][:min(80,len(a['noise_keys']),len(b['noise_keys']))],
                b['noise_keys'][:min(80,len(a['noise_keys']),len(b['noise_keys']))]))
            n=min(len(a['positions']),len(b['positions']))
            distance=np.linalg.norm((a['positions'][:n]-b['positions'][:n]).reshape(n,-1),axis=1)
            out['first_separation_step_5cm']=next((i for i,d in enumerate(distance) if d>SEP),'')
            for h in HORIZONS:
                t=min(h,n-1);out[f'H{h}_joint_position_separation']=float(distance[t])
                out[f'H{h}_joint_progress_delta']=out[f'{new}_H{h}_progress_sum']-out[f'{old}_H{h}_progress_sum']
                out[f'H{h}_worst_progress_delta']=out[f'{new}_H{h}_progress_min']-out[f'{old}_H{h}_progress_min']
                out[f'H{h}_aligned_rate_delta']=out[f'{new}_H{h}_aligned_rate_sum']-out[f'{old}_H{h}_aligned_rate_sum']
        rows.append(out)
    assert all(monitor_checks),(len(monitor_checks),sum(monitor_checks))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    for name,scene in [('toy_loss_analysis.csv','toy_give_way'),('ring_gain_analysis.csv','ring_exchange')]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(r for r in rows if r['scene']==scene)
    (HERE/'extraction_summary.json').write_text(json.dumps(dict(rows=len(rows),toy_monitor_checks=len(monitor_checks),
        toy_monitor_replay_all_match=all(monitor_checks),seed0_valid_pairs=sum('first_action_l2' in r for r in rows),
        source='cached trajectories only',new_rollouts=0),indent=2)+'\n')
    print('extracted',len(rows),'monitor checks',len(monitor_checks),'valid',sum('first_action_l2' in r for r in rows))

if __name__=='__main__':main()
