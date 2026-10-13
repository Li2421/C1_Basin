"""Read selected existing full continuations. No environment/controller execution."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,itertools
import numpy as np
import pandas as pd
from analyze import HERE,PREV,PARENT,CACHE,DISPLAY,PAIRS,read,save,sha,stats
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    cells=pd.read_csv(HERE/'basin_action_family_pairs.csv')
    cloud=pd.read_csv(HERE/'sampled_family_distances.csv')
    family=cloud.groupby(['scene','state_uid','contrast']).agg(max_bank_action_distance=('matched_max','max'),
        mean_bank_action_distance=('matched_rms','mean')).reset_index()
    flips=cells[cells.group.isin(['loss','rescue'])].groupby(['scene','state_uid','contrast']).size().reset_index(name='flipped_cells')
    eq=family.merge(flips,on=['scene','state_uid','contrast'])
    save('initial_family_counterexamples.json',eq.sort_values('max_bank_action_distance').to_dict('records'))
    selection=[]
    for (scene,contrast),g in cells.groupby(['scene','contrast']):
        if contrast not in ['TT-TF','TT-FT','TT-FF']:continue
        for gr in ['loss','rescue']:
            subset=g[g.group==gr]
            if len(subset):selection.append(dict(selection='smallest_initial_action_flip',**subset.sort_values('action_shift').iloc[0].to_dict()))
        unchanged=g[g.group.isin(['both_success','both_failure'])]
        if len(unchanged):selection.append(dict(selection='largest_initial_action_same_label',**unchanged.sort_values('action_shift',ascending=False).iloc[0].to_dict()))
    # Two conditional safety comparisons explicitly requested by the factorial interpretation.
    for sc in sorted(cells.scene.unique()):
        g=cells[(cells.scene==sc)&(cells.contrast=='TF-FF')&cells.group.isin(['loss','rescue'])]
        if len(g):selection.append(dict(selection='smallest_TF_FF_flip',**g.sort_values('action_shift').iloc[0].to_dict()))
    save('sequence_selection.json',selection)
    cfg=read(PARENT/'screening_manifest.json');states={s['uid']:s for s in read(PARENT/'confirmation_manifest.json')['states']}
    journals={}
    for p in sorted((PARENT/'journals').glob('*.jsonl')):
        for line in p.read_text().splitlines():
            x=json.loads(line);journals[x['id']]=x
    cached={};hashes={};rows=[];examples=[];arrays={};replay=[];plant=[]
    tasks=read(CACHE/'tasks.json')['tasks'];initial={(t['state']['uid'],t['seed']):t['id'] for t in tasks if t['kind']=='initial'}
    def get(uid,j,seed,ch):
        key=(uid,j,seed,ch)
        if key not in cached:
            case=dict(phase='confirm',state=states[uid],eta_index=j,eta=cfg['eta'][j],seed=seed,chain=ch,pseudo_steps=10)
            cid=digest(case)[:24];path=PARENT/'results'/cid
            r=read(path.with_suffix('.json'))
            assert (r['state_uid'],r['eta_index'],r['seed'],r['chain'])==key
            for ext,k in [('.json','result_sha256'),('.npz','npz_sha256')]:
                h=sha(path.with_suffix(ext));assert h==journals[cid][k];hashes[str(path.with_suffix(ext))]=h
            with np.load(path.with_suffix('.npz')) as z:
                a=z['actions'];p=z['positions'];nk=z['noise_keys']
            assert len(nk)==len(a) and len(p)==len(a)+1
            plant.append(float(np.max(np.abs(np.diff(p,axis=0)-.05*a))) if len(a) else 0.)
            cached[key]=(r,a,p,nk)
        return cached[key]
    for ix,v in enumerate(selection):
        uid=v['state_uid'];j=int(v['eta_index']);a,b=v['contrast'].split('-');candidates=[]
        for seed in range(16):
            values={ch:get(uid,j,seed,ch) for ch in DISPLAY}
            rr={ch:values[ch][0] for ch in DISPLAY}
            # Independently check original t0 action against the matched same-state probe.
            task=initial[(uid,seed)]
            with np.load(CACHE/'results'/f'{task}.npz') as z:
                out=z['context0_exec_all'];d=out.shape[-1];idx=1+j*(1+6*d+18)
                for ci,ch in enumerate(('TT','FT','TF','FF')):
                    if len(values[ch][1]):replay.append(float(np.max(np.abs(out[ci,idx]-values[ch][1][0].ravel()))))
            for ch1,ch2 in PAIRS:
                r1,u1,p1,n1=values[ch1];r2,u2,p2,n2=values[ch2];nt=min(len(u1),len(u2))
                assert np.array_equal(n1[:nt],n2[:nt])
                assert np.max(np.abs(p1[0]-p2[0]))==0
                dist=np.linalg.norm((u2[:nt]-u1[:nt]).reshape(nt,-1),axis=1)
                pos=np.linalg.norm((p2[:nt+1]-p1[:nt+1]).reshape(nt+1,-1),axis=1)
                first=lambda tol:int(np.flatnonzero(dist>tol)[0]) if (dist>tol).any() else None
                rec=dict(selection_index=ix,selection=v['selection'],scene=v['scene'],state_uid=uid,eta_index=j,seed=seed,
                    contrast=ch1+'-'+ch2,common_steps=nt,steps1=len(u1),steps2=len(u2),
                    term1=r1['termination'],term2=r2['termination'],success1=r1['success'],success2=r2['success'],
                    unknown=bool(r1['error'] or r2['error']),
                    action_rms=float(np.sqrt(np.mean(dist**2))),action_first=float(dist[0]),action_max=float(dist.max()),
                    position_end_common=float(pos[-1]),position_max=float(pos.max()),
                    first_action_difference_1e8=first(1e-8),first_action_difference_1e5=first(1e-5),
                    noise_identical_common_prefix=True,id1=r1['id'],id2=r2['id'])
                rows.append(rec)
                if ch1==a and ch2==b:
                    candidates.append((bool(not rec['unknown'] and rec['success1']!=rec['success2']),seed,rec,dist,pos,values))
        # Prefer a real seed-level disagreement, lowest seed as tie break; selection is explicitly conditional.
        chosen=sorted(candidates,key=lambda v:(not v[0],v[1]))[0]
        _,seed,rec,dist,pos,values=chosen
        examples.append(dict(**rec,basin_group=v['group'],q1=[v['q1low'],v['q1high']],q2=[v['q2low'],v['q2high']],
                             initial_mean_action_shift=v['action_shift'],initial_mean_jacobian_difference=v['j_absolute_difference']))
        arrays[f'example{ix}_action_difference']=dist;arrays[f'example{ix}_position_difference']=pos
        for ch in DISPLAY:
            arrays[f'example{ix}_{ch}_positions']=values[ch][2]
            arrays[f'example{ix}_{ch}_actions']=values[ch][1]
        print('read full cached sequences',ix,uid,j,a,b,'seed',seed,rec['term1'],rec['term2'],flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'selected_sequence_metrics.csv',index=False)
    save('selected_sequence_examples.json',examples)
    np.savez_compressed(HERE/'selected_sequence_arrays.npz',**arrays)
    save('sequence_provenance.json',dict(unique_rollouts=len(cached),hashes_checked=len(hashes),hashes=hashes,
         new_rollouts=0,max_plant_residual=max(plant),max_initial_probe_replay_error=max(replay),
         paired_noise_checks=len(rows),all_paired_noise_identical=True,script_sha256=sha(__file__)))
    print('sequence reading complete',len(cached),'unique existing rollouts',flush=True)
if __name__=='__main__':main()
