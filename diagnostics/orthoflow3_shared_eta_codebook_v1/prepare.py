#!/usr/bin/env python3
"""Freeze source-isolated states and a TRAIN-only shared eta codebook."""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

ROOT=Path('/home/zhihan/research/Basin_C1'); D=ROOT/'diagnostics'; H=Path(__file__).parent
POINT=D/'orthoflow3_true_t0_point_learning_v1'; XFER=D/'orthoflow3_t0_eta_continuity_cross_transfer_v1'; GEN=D/'orthoflow3_general_basin_geometry_v1'
AFF=np.array([.875,0.,.375]); SCALE=np.array([.75,1.,.75])

def read(p): return list(csv.DictReader(open(p)))
def dump(name,x): (H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(name,rows,fields=None):
    fields=fields or (list(rows[0]) if rows else ['state_id'])
    with (H/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sh(s): return hashlib.sha256(s.encode()).hexdigest()
def eta_key(e): return np.asarray(e,dtype='<f8').tobytes().hex()

def state_inventory():
    final=json.load(open(POINT/'final_source_split.json'))['states']; pa=np.load(POINT/'point_learning_arrays.npz')
    eligible=json.load(open(POINT/'eligible_state_manifest.json'))['selected_states']; ea=np.load(POINT/'conditioning_features.npz')['features']
    fixed=[]; feature={}
    for st in final:
        q=dict(st); feature[q['state_id']]=np.asarray(pa['features'][int(q['dataset_index'])],float);fixed.append(q)
    used={q['source_group'] for q in fixed}
    pool=[]
    for st in eligible:
        if st['source_group'] in used: continue
        q=dict(st);feature[q['state_id']]=np.asarray(ea[int(q['feature_index'])],float);pool.append(q)
    pool.sort(key=lambda q:(sh(q['source_group']),q['state_id']))
    by={s:[q for q in fixed if q['split']==s] for s in ('train','val','test')}
    need={'train':128-len(by['train']),'val':32-len(by['val']),'test':32-len(by['test'])}
    pos=0
    for s in ('train','val','test'):
        take=pool[pos:pos+need[s]];pos+=need[s]
        for q in take:q['split']=s
        by[s]+=take
    if {s:len(by[s]) for s in by}!={'train':128,'val':32,'test':32}: raise RuntimeError('insufficient inventory')
    states=[]; feats=[]
    for s in ('train','val','test'):
        for q in by[s]:
            q=dict(q);q['dataset_index']=len(states);q['split']=s;states.append(q);feats.append(feature[q['state_id']])
    groups={s:{q['source_group'] for q in states if q['split']==s} for s in by}
    leak=(groups['train']&groups['val'])|(groups['train']&groups['test'])|(groups['val']&groups['test'])
    if leak: raise RuntimeError(('source leak',leak))
    for q in states:
        if not Path(q['state_file']).exists(): raise RuntimeError(('missing state',q['state_file']))
    X=np.stack(feats); tr=np.array([q['split']=='train' for q in states]);hm=X[tr].mean(0);hs=X[tr].std(0);hs=np.where(hs<1e-8,1.,hs)
    np.savez_compressed(H/'state_features.npz',features=X,x=((X-hm)/hs).astype(np.float32),state_ids=np.array([q['state_id'] for q in states]),splits=np.array([q['split'] for q in states]))
    dump('state_split.json',{'counts':{s:len(by[s]) for s in by},'source_leakage':False,'source_groups':{s:sorted(groups[s]) for s in groups},'selection':'original frozen 24/8/8 preserved; remaining states assigned by source_group SHA256 before new mode outcomes','states':states})
    dump('normalization.json',{'h_mean':hm.tolist(),'h_std':hs.tolist(),'fit_train_only':True,'eta_affine_center':AFF.tolist(),'eta_scale':SCALE.tolist()})
    return states

def codebook(states):
    sm={q['state_id']:q for q in states}; train_groups={q['source_group'] for q in states if q['split']=='train'}
    obs=defaultdict(lambda:defaultdict(list))
    def add(sid,e,b63,src):
        if sid not in sm or sm[sid]['source_group'] not in train_groups or not b63:return
        if np.linalg.norm(e)<1e-12:return
        obs[eta_key(e)][sid].append(src)
    for r in read(GEN/'exact_q64_inventory.csv'):
        if r['scenario']=='ToyGiveWay_2A' and r['trials']=='64':add(r['state_id'],np.array([float(r[f'eta{i}']) for i in (1,2,3)]),r['B63']=='True','general_exact')
    for r in read(POINT/'selected_eta_targets.csv'):
        add(r['state_id'],np.array([float(r[f'target_eta{i}']) for i in (1,2,3)]),float(r['target_Q64'])>=63/64,'selected_target')
    for r in read(POINT/'q64_promotions.csv'):
        if r.get('successes64'):add(r['state_id'],np.array([float(r[f'eta{i}']) for i in (1,2,3)]),r['B63']=='True','point_promotion')
    for fn in ('cached_cross_transfer.csv','cross_transfer_results.csv'):
        for r in read(XFER/fn):add(r['destination_state'],np.array([float(r[f'eta{i}']) for i in (1,2,3)]),r['B63']=='True','cross_transfer')
    E=np.array([np.frombuffer(bytes.fromhex(k),dtype='<f8') for k in obs]); keys=list(obs)
    Z=(E-AFF)/SCALE
    if len(E)>1:cl=fcluster(linkage(Z,method='complete'),t=.05,criterion='distance')
    else:cl=np.ones(len(E),int)
    candidates=[]
    for c in sorted(set(cl)):
        ix=np.flatnonzero(cl==c);best=None
        for j in ix:
            support=len(obs[keys[j]]);nobs=sum(len(v) for v in obs[keys[j]].values());rank=(-support,-nobs,keys[j])
            if best is None or rank<best[0]:best=(rank,j,support,nobs)
        _,j,support,nobs=best
        candidates.append(dict(cluster_id=int(c),representative_key=keys[j],eta1=E[j,0],eta2=E[j,1],eta3=E[j,2],z1=Z[j,0],z2=Z[j,1],z3=Z[j,2],train_state_B63_support=support,exact_B63_observations=nobs,cluster_members=len(ix)))
    candidates.sort(key=lambda r:(-r['train_state_B63_support'],-r['exact_B63_observations'],r['representative_key']))
    write('codebook_candidates.csv',candidates)
    # Support plus farthest-point diversity. Support has equal maximum influence
    # to normalized distance; deterministic hashes settle exact ties.
    selected=[];remaining=list(range(len(candidates)));maxd=max(np.linalg.norm(np.array([r['z1'],r['z2'],r['z3']]) for r in candidates) if False else [1])
    C=np.array([[r['z1'],r['z2'],r['z3']] for r in candidates]); global_d=float(pdist(C).max()) if len(C)>1 else 1.;maxsup=max(r['train_state_B63_support'] for r in candidates)
    while remaining and len(selected)<12:
        if not selected:j=remaining[0]
        else:
            def score(j):
                d=min(np.linalg.norm(C[j]-C[k]) for k in selected)/max(global_d,1e-12);sup=candidates[j]['train_state_B63_support']/maxsup
                return (sup+d,sup,d,-int(candidates[j]['representative_key'],16))
            j=max(remaining,key=score)
        selected.append(j);remaining.remove(j)
    rows=[]
    for m,j in enumerate(selected):rows.append(dict(mode_id=m,**candidates[j]))
    CB=np.array([[r['z1'],r['z2'],r['z3']] for r in rows]);dm=squareform(pdist(CB)) if len(CB)>1 else np.zeros((1,1))
    for i,r in enumerate(rows):r['nearest_codebook_distance']=float(np.min(np.delete(dm[i],i))) if len(rows)>1 else ''
    write('codebook_eta.csv',rows)
    dist=[]
    for i in range(len(rows)):
        for j in range(len(rows)):dist.append({'mode_i':i,'mode_j':j,'normalized_eta_distance':dm[i,j]})
    write('codebook_pairwise_distances.csv',dist)
    dump('codebook_manifest.json',{'M':len(rows),'constructed_from':'TRAIN source groups only','merge_threshold_normalized':.05,'selection':'equal-weight normalized TRAIN-state support plus farthest-point diversity','eta_zero_excluded':True,'effective_diversity':{'min_pairwise':float(dm[np.triu_indices(len(rows),1)].min()),'median_pairwise':float(np.median(dm[np.triu_indices(len(rows),1)])),'max_pairwise':float(dm.max())}})
    return rows

def main():
    states=state_inventory();cb=codebook(states)
    dump('working_state.json',{'status':'CODEBOOK_FROZEN','completed':['state_split','train_only_codebook'],'next_action':'construct state-mode plans','M':len(cb)})
    write('experiment_ledger.csv',[{'stage':'prepare','status':'complete','states':len(states),'modes':len(cb),'new_rollouts':0}])
    print(json.dumps({'states':len(states),'counts':{s:sum(q['split']==s for q in states) for s in ('train','val','test')},'M':len(cb)},indent=2))
if __name__=='__main__':main()
