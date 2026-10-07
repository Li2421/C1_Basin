#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,itertools,json
from pathlib import Path
import numpy as np

H=Path(__file__).parent
S=H.parent/'orthoflow3_shared_eta_codebook_v1'

def read(p): return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
    p=H/name; p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ['mode_id'])
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x): (H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def matrix(fn):
    rows=read(S/fn); states=sorted({r['state_id'] for r in rows}); modes=range(12)
    by={(r['state_id'],int(r['mode_id'])):r for r in rows}
    Q=np.array([[float(by[(sid,m)]['empirical_Q']) for m in modes] for sid in states])
    K=np.array([[int(by[(sid,m)]['successes']) for m in modes] for sid in states])
    return states,Q,K,by

def overlap(A,sub):
    vals=[]
    for a,b in itertools.combinations(sub,2):
        u=np.logical_or(A[:,a],A[:,b]).sum(); vals.append(np.logical_and(A[:,a],A[:,b]).sum()/u if u else 1.)
    return (float(np.mean(vals)),float(np.max(vals))) if vals else (1.,1.)

def main():
    cb=read(S/'codebook_eta.csv'); Z=np.array([[float(r[f'z{i}']) for i in (1,2,3)] for r in cb])
    states,Q,K,_=matrix('train_mode_counts.csv'); A=K>=15; prev=A.mean(0)
    mode_rows=[]
    for m in range(12):
        mode_rows.append(dict(mode_id=m,eta1=cb[m]['eta1'],eta2=cb[m]['eta2'],eta3=cb[m]['eta3'],strong_states=int(A[:,m].sum()),train_states=len(states),strong_prevalence=float(prev[m]),mean_Q16=float(Q[:,m].mean()),dominant_over_35pct=bool(prev[m]>.35),eligible_10_30pct=bool(.10<=prev[m]<=.30)))
    write('mode_train_prevalence.csv',mode_rows)
    selections=[]; exact=[]; band=[]
    for k in range(1,13):
        for sub in itertools.combinations(range(12),k):
            p=prev[list(sub)]; cov=float(A[:,sub].any(1).mean()); mo,xo=overlap(A,sub)
            ds=[np.linalg.norm(Z[a]-Z[b]) for a,b in itertools.combinations(sub,2)]
            row=dict(mode_ids=';'.join(map(str,sub)),cardinality=k,all_modes_10_30pct=bool(np.all((p>=.10)&(p<=.30))),union_70_85pct=bool(.70<=cov<=.85),union_coverage=cov,min_mode_prevalence=float(p.min()),max_mode_prevalence=float(p.max()),mean_mode_prevalence=float(p.mean()),mean_jaccard_overlap=mo,max_jaccard_overlap=xo,min_eta_distance=float(min(ds)) if ds else 0.,mean_eta_distance=float(np.mean(ds)) if ds else 0.)
            selections.append(row)
            # A singleton cannot test the mission's state-conditioned mode
            # selection hypothesis, even if its union prevalence is in band.
            if k>=2 and row['all_modes_10_30pct'] and row['union_70_85pct']: exact.append((sub,row))
            if k>=2 and row['union_70_85pct']: band.append((sub,row))
    write('sparse_codebook_selection.csv',selections)
    if exact:
        sub,row=min(exact,key=lambda z:(z[1]['cardinality'],z[1]['mean_jaccard_overlap'],-z[1]['min_eta_distance'],z[0])); status='EXACT_TARGET'
    elif band:
        sub,row=min(band,key=lambda z:(z[1]['cardinality'],z[1]['max_mode_prevalence'],z[1]['mean_mode_prevalence'],z[1]['mean_jaccard_overlap'],-z[1]['min_eta_distance'],z[0])); status='PARETO_FALLBACK_NO_LOW_PREVALENCE_MODES'
    else:
        sub,row=min([(tuple(map(int,r['mode_ids'].split(';'))),r) for r in selections],key=lambda z:(abs(z[1]['union_coverage']-.775),z[1]['cardinality'],z[1]['max_mode_prevalence'],z[0])); status='PARETO_FALLBACK_NO_TARGET_UNION'
    sub=list(sub)
    fixed=max(sub,key=lambda m:(prev[m],Q[:,m].mean(),-m))
    strict_dominant=[m for m in range(12) if prev[m]>.35]
    chosen=[]
    for local,m in enumerate(sub): chosen.append(dict(local_mode_id=local,original_mode_id=m,eta=[float(cb[m][f'eta{i}']) for i in (1,2,3)],z=[float(cb[m][f'z{i}']) for i in (1,2,3)],train_strong_prevalence=float(prev[m]),train_mean_Q16=float(Q[:,m].mean())))
    frozen={'status':status,'selection_uses':'TRAIN only','strong_definition':'successes >= 15/16','desired_mode_prevalence':[.10,.30],'dominant_exclusion_threshold':.35,'desired_union_coverage':[.70,.85],'minimum_modes_for_stress_test':2,'selected_original_mode_ids':sub,'M_sparse':len(sub),'modes':chosen,'best_fixed_original_mode':int(fixed),'train_union_coverage':float(A[:,sub].any(1).mean()),'train_mean_oracle_Q16':float(Q[:,sub].max(1).mean()),'all_12_modes_exceeded_35pct':len(strict_dominant)==12,'strict_dominant_modes':strict_dominant,'removed_original_mode_ids':[m for m in range(12) if m not in sub],'selection_order':'target union band; smallest multi-mode subset; lowest maximum then mean mode prevalence; lowest overlap; eta diversity; IDs','source_hashes':{f:sha(S/f) for f in ('codebook_eta.csv','train_mode_counts.csv','state_split.json','normalization.json')}}
    dump('sparse_codebook.json',frozen)
    # Only after the TRAIN-only decision is frozen, materialize sparse VAL/TEST matrices.
    for split,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv'),('test','test_mode_q64.csv')]:
        rr=[]
        for r in read(S/fn):
            om=int(r['mode_id'])
            if om in sub: rr.append(dict(r,original_mode_id=om,sparse_mode_id=sub.index(om)))
        write(f'sparse_{split}_counts.csv',rr)
    difficulty=[]
    for split,fn,n,cut in [('train','train_mode_counts.csv',16,15),('val','val_mode_counts.csv',32,30),('test','test_mode_q64.csv',64,63)]:
        ss,q,k,_=matrix(fn);B=k[:,sub]>=cut
        for j,m in enumerate(sub): difficulty.append(dict(split=split,metric='mode',original_mode_id=m,states=len(ss),strong_or_B63_states=int(B[:,j].sum()),prevalence=float(B[:,j].mean()),mean_Q=float(q[:,m].mean())))
        difficulty.append(dict(split=split,metric='union_oracle',original_mode_id='',states=len(ss),strong_or_B63_states=int(B.any(1).sum()),prevalence=float(B.any(1).mean()),mean_Q=float(q[:,sub].max(1).mean())))
    write('sparse_codebook_difficulty.csv',difficulty)
    a=np.load(S/'state_features.npz'); np.savez_compressed(H/'state_features.npz',**{k:a[k] for k in a.files})
    dump('working_state.json',{'status':'SPARSE_CODEBOOK_FROZEN','completed':['train_only_prevalence','train_only_subset_selection','sparse_codebook_freeze'],'next_action':'train baselines and new MLP seeds'})
    write('experiment_ledger.csv',[{'stage':'sparse_codebook_freeze','status':'complete','new_rollouts':0,'note':status}])
    print(json.dumps(frozen,indent=2))
if __name__=='__main__': main()
