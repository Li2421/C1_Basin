"""Check an outcome-preserving controller intervention's feature sensitivity.

Phase and motion-gated programs are different registered controllers. Observed
outcome equality is checked at each exact matched seed, never assumed or used
to merge their cache identities. It is an empirical invariance on this panel,
not a proof of global controller equivalence.
"""
from pathlib import Path
import numpy as np
from scipy.special import expit
from .motion_factorial_train import OUT as MATRIX,SOURCE,read,write,csvwrite,SEEDS
from .motion_freeze_train import OUT as FREEZE
from shared_rollout_db.src.rollout_db import connect,canonical


def main():
    out=MATRIX/'rest_invariance_audit';out.mkdir(exist_ok=True)
    old=read(MATRIX.parent/'phase_factorial_support/protocol.json')['profiles']
    new=read(MATRIX/'protocol.json')['profiles'];pairs=read(SOURCE/'pairs.json')
    seedrows=[]
    with connect(True) as db:
        for ci,(a,b) in enumerate(zip(old,new)):
            assert a['controller_uid']!=b['controller_uid']
            for pair in pairs:
                records=[]
                for c in (a,b):
                    rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(pair['state_uid'],pair['eta_uid'],c['controller_uid']))}
                    records.append(rr)
                for k in range(pair['target_seeds']):
                    sk=canonical({'future_index':k});r,t=(a[sk] for a in records)
                    assert r['compatibility_quality']==t['compatibility_quality']=='EXACT_REUSE'
                    assert not r['conflict_quarantined'] and not t['conflict_quarantined']
                    assert r['rollout_uid']!=t['rollout_uid'] and r['original_source_file']!=t['original_source_file']
                    row=dict(program_pair=ci,split=pair['split'],state_uid=pair['state_uid'],eta_uid=pair['eta_uid'],seed=k,
                        phase_rollout_uid=r['rollout_uid'],motion_rollout_uid=t['rollout_uid'],
                        phase_source=r['original_source_file'],motion_source=t['original_source_file'])
                    for key in ('success','deadlock','timeout','collision','numerical_failure','episode_length','j_def','min_wall_distance','min_agent_distance'):
                        row[key+'_equal']=r[key]==t[key]
                        if key in ('j_def','min_wall_distance','min_agent_distance'):
                            row[key+'_absolute_difference']=abs(r[key]-t[key]) if r[key] is not None and t[key] is not None else None
                    seedrows.append(row)
    csvwrite(out/'matched_seed_records.csv',seedrows)
    choice=read(FREEZE/'source_selection.json')['selection'];d=np.load(MATRIX/'dataset.npz');va=np.flatnonzero(d['split']=='validation')
    q=d['success'][:,va]/(d['success'][:,va]+d['failure'][:,va]);assert np.array_equal(q[12:14],q[14:16])
    g=(d['success'][[12,13]][:,va]>=15).reshape(2,16,2)
    rr=[]
    for arm in ('full_update','head_only','trunk_only'):
        step=choice[arm]['step']
        for fold in range(3):
            for seed in SEEDS:
                folder=(MATRIX/'cv'/f'fold{fold}'/'motion_intervention'/'rest_motion'/f'seed{seed}' if arm=='full_update' else FREEZE/'cv'/f'fold{fold}'/arm/f'seed{seed}')
                z=np.load(folder/'predictions.npz')[f'step{step}'];p=expit(z)
                a,b=p[[12,13]],p[[14,15]];ca,cb=a.reshape(2,16,2).argmax(-1),b.reshape(2,16,2).argmax(-1)
                ci,hi=np.arange(2)[:,None],np.arange(16)[None,:]
                rr.append(dict(arm=arm,fold=fold,seed=seed,step=step,constituents_seen=fold!=2,
                    mean_probability_change=float(abs(a-b).mean()),max_probability_change=float(abs(a-b).max()),
                    ranking_flips=int((ca!=cb).sum()),phase_B15=int(g[ci,hi,ca].sum()),motion_B15=int(g[ci,hi,cb].sum())))
    csvwrite(out/'model_sensitivity.csv',rr)
    equal={k:int(sum(r[k+'_equal'] for r in seedrows)) for k in ('success','deadlock','timeout','collision','numerical_failure','episode_length','j_def','min_wall_distance','min_agent_distance')}
    write(out/'audit.json',dict(matched_seed_pairs=len(seedrows),equality_count=equal,new_rollouts=0,
        controller_keys_distinct=True,source_journals_distinct=True,cache_identities_not_merged=True,
        feature_difference='Only static zero-velocity goal response differs; h,eta,H20/entity response and moving goal response are the same for phase versus motion programs.',
        scope='Observed panel only; not a proof of equivalence outside these states/seeds or a reason to relax exact reuse signatures.'))
    print(dict(matched=len(seedrows),equality=equal),flush=True)
    print([r for r in rr if r['constituents_seen']],flush=True)


if __name__=='__main__':main()
