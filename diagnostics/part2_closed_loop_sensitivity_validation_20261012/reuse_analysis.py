"""Read-only Part 2 reuse analysis. No model call or simulator step."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARD = Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
BOUND = ROOT / 'diagnostics/placement_boundary_mechanism_audit_v1'
H4 = ROOT / 'diagnostics/placement_strong_hypotheses_v1'
CHAINS = ('TT', 'TF', 'FT', 'FF')
PAIRS = (('TT','TF'), ('FT','FF'), ('TT','FT'), ('TF','FF'), ('TT','FF'))
TOL = {'toy_give_way': 0.08, 'ring_exchange': 0.09}
HORIZON = {'toy_give_way': 850, 'ring_exchange': 700}


def read(path):
    return json.loads(Path(path).read_text())


def write_json(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write_csv(name, rows, fields):
    with (HERE/name).open('w', newline='') as fp:
        w=csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def completion_margin(positions, goals, tol, horizon):
    """Early-terminal positions are held constant through H; no imputed success."""
    if positions.ndim!=3 or positions.shape[1:]!=goals.shape or len(positions)<2:
        raise ValueError('invalid saved trajectory')
    if not np.isfinite(positions).all():
        raise ValueError('nonfinite trajectory')
    # The constant padding cannot improve a running minimum, so saved steps suffice.
    err=np.linalg.norm(positions[1:]-goals[None],axis=2)
    worst=err.max(axis=1)
    t=int(np.argmin(worst))
    i=int(np.argmax(err[t]))
    sorted_time=np.sort(worst)
    sorted_agent=np.sort(err[t])
    return dict(rho=float(tol-worst[t]),active_step=t+1,active_agent=i,
        time_gap=float(sorted_time[1]-sorted_time[0]) if len(sorted_time)>1 else None,
        agent_gap=float(sorted_agent[-1]-sorted_agent[-2]) if len(sorted_agent)>1 else None,
        final_worst=float(worst[-1]),steps=len(positions)-1,padded_steps=horizon-(len(positions)-1))


def main():
    HERE.mkdir(exist_ok=True)
    tasks=read(HARD/'confirm_q16_tasks.json')['tasks']
    expected=len(tasks)
    records={}; row_index={}; missing=[]; mismatched=[]
    for q in tasks:
        path=HARD/'results'/f"{q['id']}.json"
        key=(q['state']['uid'],int(q['eta_index']),int(q['seed']),q['chain'])
        if not path.exists():
            missing.append(q['id']);continue
        d=read(path)
        if any((d.get(k)!=q.get(k) for k in ('chain','eta_index','seed'))) or d.get('state_uid')!=q['state']['uid']:
            mismatched.append(q['id']);continue
        records[key]=(d,path,q)
        row_index[key]=q
    seeds=range(16)
    states=read(HARD/'confirmation_manifest.json')['states']
    pairs=[];case_outcomes=Counter();b15=[];seed0_margins={};margin_rows=[];margin_label=Counter()
    for st in states:
        uid=st['uid'];scene=st['scenario']
        for ei in range(16):
            for ch in CHAINS:
                succ=[];valid=0
                for seed in seeds:
                    entry=records.get((uid,ei,seed,ch))
                    if entry is None:continue
                    d=entry[0]
                    if d.get('error') is None and d.get('termination')!='numerical_failure':
                        valid+=1;succ.append(bool(d['success']))
                lo=sum(succ);hi=lo+16-valid
                status='TRUE' if lo>=15 else 'FALSE' if hi<15 else 'UNKNOWN'
                b15.append(dict(scene=scene,uid=uid,eta_index=ei,placement=ch,observed=valid,
                    successes=lo,lower=lo,upper=hi,status=status))
            for seed in seeds:
                for old,new in PAIRS:
                    a=records.get((uid,ei,seed,old));b=records.get((uid,ei,seed,new))
                    va=a is not None and a[0].get('error') is None and a[0].get('termination')!='numerical_failure'
                    vb=b is not None and b[0].get('error') is None and b[0].get('termination')!='numerical_failure'
                    sa=a[0]['success'] if va else None;sb=b[0]['success'] if vb else None
                    outcome='unknown' if sa is None or sb is None else 'gain' if not sa and sb else 'loss' if sa and not sb else 'both_success' if sa else 'both_failure'
                    case_outcomes[(scene,old,new,outcome)]+=1
                    pairs.append(dict(scene=scene,uid=uid,eta_index=ei,seed=seed,old=old,new=new,
                        old_success=sa,new_success=sb,outcome=outcome,
                        old_termination=a[0].get('termination') if a else 'missing',
                        new_termination=b[0].get('termination') if b else 'missing',
                        old_result=str(a[1]) if a else '',new_result=str(b[1]) if b else '',
                        old_rho='',new_rho='',delta_rho='',old_active_step='',new_active_step=''))
                # Read real trajectory only for seed 0; all 16-seed outcomes stay separate.
                if seed!=0:continue
                for ch in CHAINS:
                    a=records.get((uid,ei,seed,ch))
                    if a is None:continue
                    d,path,q=a
                    if d.get('error') is not None or d.get('termination')=='numerical_failure':
                        margin_label[(scene,ch,'numerical_failure')]+=1;continue
                    arrpath=path.with_suffix('.npz')
                    if not arrpath.exists():
                        margin_label[(scene,ch,'missing_array')]+=1;continue
                    with np.load(arrpath) as z:
                        positions=z['positions'];goals=z['goals']
                    m=completion_margin(positions,goals,TOL[scene],HORIZON[scene])
                    seed0_margins[(uid,ei,ch)]=m
                    flag='agree' if (m['rho']>0)==bool(d['success']) else 'disagree'
                    margin_label[(scene,ch,flag)]+=1
                    margin_rows.append(dict(scene=scene,uid=uid,eta_index=ei,placement=ch,seed=0,
                        rho=m['rho'],success=d['success'],termination=d['termination'],label_agreement=flag,
                        **{k:m[k] for k in ('active_step','active_agent','time_gap','agent_gap','padded_steps')},
                        result_path=str(path),array_path=str(arrpath)))
    for p in pairs:
        if p['seed']!=0:continue
        a=seed0_margins.get((p['uid'],p['eta_index'],p['old']))
        b=seed0_margins.get((p['uid'],p['eta_index'],p['new']))
        if a is None or b is None:continue
        p.update(old_rho=a['rho'],new_rho=b['rho'],delta_rho=b['rho']-a['rho'],
            old_active_step=a['active_step'],new_active_step=b['active_step'])
    write_csv('gain_loss_audit.csv',pairs,list(pairs[0]))
    write_csv('completion_margin_seed0.csv',margin_rows,list(margin_rows[0]))
    write_csv('b15_by_candidate.csv',b15,list(b15[0]))
    by_b15={(x['uid'],x['eta_index'],x['placement']):x for x in b15}
    robust=Counter()
    for st in states:
        for ei in range(16):
            for old,new in PAIRS:
                a=by_b15[(st['uid'],ei,old)]['status'];b=by_b15[(st['uid'],ei,new)]['status']
                outcome='unknown' if 'UNKNOWN' in (a,b) else 'gain' if a=='FALSE' and b=='TRUE' else 'loss' if a=='TRUE' and b=='FALSE' else 'both_true' if a=='TRUE' else 'both_false'
                robust[(st['scenario'],old,new,outcome)]+=1
    # Reuse boundary samples, recomputing the new completion margin from trajectories.
    normal_protocol=read(BOUND/'boundary_normal_protocol.json')
    boundary_rows=[];boundary_detail=[]
    for c in normal_protocol:
        uid,chain=c['uid'],c['chain'];st=next(s for s in states if s['uid']==uid)
        matched=[]
        for path in (BOUND/'boundary_normal_results').glob(f'{uid}_{chain}_*.json'):
            d=read(path)
            if d.get('error') is not None:continue
            with np.load(path.with_suffix('.npz')) as z: positions=z['positions'].copy()
            m=completion_margin(positions,np.asarray(st['physical']['goals']),TOL[st['scenario']],HORIZON[st['scenario']])
            matched.append((d,m,positions,path))
            boundary_detail.append(dict(uid=uid,chain=chain,kind=d['kind'],epsilon=d['epsilon'],coordinate=d['coordinate'],sign=d['sign'],
                eta=d['eta'],rho=m['rho'],old_rho=d['rho'],active_step=m['active_step'],active_agent=m['active_agent'],
                success=d['success'],termination=d['termination'],path=str(path)))
        center=next(((d,m,p,path) for d,m,p,path in matched if d['kind']=='center'),None)
        if center is None:continue
        cd,cm,cp,cpath=center
        for eps in (.002,.0005):
            fd=[];pred=[];switch=[];abserrs=[]
            for j in range(3):
                minus=next(((d,m,p) for d,m,p,_ in matched if d['kind']=='fd' and d['epsilon']==eps and d['coordinate']==j and d['sign']==-1),None)
                plus=next(((d,m,p) for d,m,p,_ in matched if d['kind']=='fd' and d['epsilon']==eps and d['coordinate']==j and d['sign']==1),None)
                if minus is None or plus is None:continue
                dm,mm,pm=minus;dp,mp,pp=plus
                direct=(mp['rho']-mm['rho'])/(2*eps)
                t=cm['active_step'];i=cm['active_agent'];position=cp[t,i];goal=np.asarray(st['physical']['goals'])[i]
                vector=(position-goal)/np.linalg.norm(position-goal)
                if t>=len(pm):pm=np.concatenate([pm,np.repeat(pm[-1:],t-len(pm)+1,axis=0)])
                if t>=len(pp):pp=np.concatenate([pp,np.repeat(pp[-1:],t-len(pp)+1,axis=0)])
                sensitivity=(pp[t,i]-pm[t,i])/(2*eps)
                chain_grad=-float(vector@sensitivity)
                fd.append(direct);pred.append(chain_grad);abserrs.append(abs(direct-chain_grad))
                switch.append(mm['active_step']!=cm['active_step'] or mp['active_step']!=cm['active_step'] or
                              mm['active_agent']!=i or mp['active_agent']!=i)
            if len(fd)!=3:continue
            fd=np.asarray(fd);pred=np.asarray(pred)
            boundary_rows.append(dict(scene=st['scenario'],uid=uid,chain=chain,epsilon=eps,
                center_rho=cm['rho'],old_center_rho=cd['rho'],center_success=cd['success'],
                active_step=cm['active_step'],active_agent=cm['active_agent'],time_gap=cm['time_gap'],agent_gap=cm['agent_gap'],
                gradient_direct=fd.tolist(),gradient_chain=pred.tolist(),gradient_norm=float(np.linalg.norm(fd)),
                gradient_absolute_error=float(np.linalg.norm(fd-pred)),
                gradient_relative_error=float(np.linalg.norm(fd-pred)/max(np.linalg.norm(fd),1e-12)),
                normal_cosine=float(fd@pred/max(np.linalg.norm(fd)*np.linalg.norm(pred),1e-12)),
                active_switch=any(switch),root_fraction=c['fraction'],center_result=str(cpath)))
    write_csv('boundary_validation.csv',boundary_rows,list(boundary_rows[0]))
    write_json('boundary_samples.json',boundary_detail)
    write_json('reuse_summary.json',dict(expected_tasks=expected,valid_result_records=len(records),missing_result_count=len(missing),
        mismatched_result_count=len(mismatched),gain_loss_counts={':'.join(k):v for k,v in sorted(case_outcomes.items())},
        b15_counts={':'.join(k):v for k,v in sorted(robust.items())},
        margin_label_counts={':'.join(k):v for k,v in sorted(margin_label.items())},
        boundary_rows=len(boundary_rows),new_rollouts=0,source_paths=[str(HARD),str(BOUND),str(H4),str(FIELD)],
        status='read_only_reanalysis'))
    print(json.dumps(dict(expected=expected,valid=len(records),missing=len(missing),mismatch=len(mismatched),
        gain_loss_rows=len(pairs),seed0_margin_rows=len(margin_rows),boundary_rows=len(boundary_rows))),flush=True)


if __name__=='__main__':main()
