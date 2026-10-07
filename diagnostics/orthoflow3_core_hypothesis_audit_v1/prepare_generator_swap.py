"""Outcome-blind frozen generator state-conditioning intervention and cache plan."""
import csv
import json
import hashlib
from pathlib import Path

from shared_rollout_db.src.rollout_db import eta_identity, uid, canonical
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import (
    SCENARIO, CORRECTION_CONFIG, MAC_CONFIG,
)

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FROZEN=ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/frozen_proposals.json'
FIRST=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/frozen_proposals.json'
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/candidate_cache_keys.csv'
FEATURES=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/cohort_features.npz'


def main():
    if (OUT/'generator_swap_manifest.json').exists():
        print('Frozen generator swap already planned; no redesign.');return
    frozen=json.loads(FROZEN.read_text());first=json.loads(FIRST.read_text())
    states={int(x['episode_index']):x for x in first['states']}
    extended={int(x['episode_index']):x for x in frozen['states']}
    assert sorted(states)==list(range(200))
    assert sorted(extended)==list(range(200))
    assert all(states[i]['initial_positions']==extended[i]['initial_positions'] for i in states)
    old={}
    with OLD.open() as f:
        for r in csv.DictReader(f):old[(int(r['episode_index']),r['kind'])]=r
    recipients=[i for i in range(200) if i%3==0]
    # Every donor is another source family; no outcome, critic score or geometry selected.
    donor_shift=73
    selected=[];requests=[];mapping=[];used=set()
    controller_uid=uid('ctl',CORRECTION_CONFIG)
    for ep in recipients:
        original=states[ep];donor=states[(ep+donor_shift)%200]
        assert donor['source_group']!=original['source_group']
        eta=donor['eta']['sample_0'];eid,eta,_=eta_identity(eta)
        physical={'initial_positions':original['initial_positions']}
        content_hash=hashlib.sha256(canonical(physical).encode()).hexdigest()
        state_uid=uid('state',{'scenario':SCENARIO,'content':content_hash})
        assert state_uid==old[(ep,'sample_0')]['state_uid']
        assert controller_uid==old[(ep,'sample_0')]['controller_uid']
        es={'sample_0':original['eta']['sample_0'],'donor_sample_0':eta}
        selected.append({**original,'eta':es})
        for kind in ('sample_0','donor_sample_0'):
            ee=eta_identity(es[kind])[0]
            mapping.append({'episode_index':ep,'kind':kind,'state_uid':state_uid,'eta_uid':ee,'controller_uid':controller_uid})
        safety=eta_identity([0.,0.,0.])[0]
        mapping.append({'episode_index':ep,'kind':'safety','state_uid':state_uid,'eta_uid':safety,'controller_uid':controller_uid})
        mapping.append({'episode_index':ep,'kind':'mac_only','state_uid':state_uid,'eta_uid':safety,'controller_uid':uid('ctl',MAC_CONFIG)})
        req=(state_uid,eid,controller_uid)
        assert req not in used;used.add(req)
        requests.append({'state_uid':state_uid,'eta_uid':eid,'controller_uid':controller_uid,
                         'seed_keys':[canonical({'future_index':j}) for j in range(16)]})
    OUT.joinpath('frozen_proposals.json').write_text(json.dumps({'states':selected,'K_max':1,'cohort':frozen['cohort'],
        'source_frozen_sha256':hashlib.sha256(FIRST.read_bytes()).hexdigest(),
        'donor_rule':'(episode_index+73)%200; same frozen sample_0; recipient episode_index%3==0'},indent=2)+'\n')
    OUT.joinpath('planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    with (OUT/'candidate_cache_keys.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    if not (OUT/'cohort_features.npz').exists():OUT.joinpath('cohort_features.npz').symlink_to(FEATURES)
    OUT.joinpath('generator_swap_manifest.json').write_text(json.dumps({'n_recipients':len(recipients),
        'recipient_rule':'episode_index%3==0','donor_rule':'(episode_index+73)%200',
        'donor_selection_outcome_blind':True,'eta_kind':'frozen sample_0, no new generator inference',
        'source_group_overlap_per_pair':0,'planned_seed_slots':16*len(requests),'checkpoint_unchanged':True,
        'expected_h_sha256_source':str(FEATURES),'interpretation':'Tests empirical marginal proposals from donor states on recipient states. It is not a learned optimized global prior.'},indent=2)+'\n')
    print(json.dumps({'recipients':len(recipients),'planned_slots':len(requests)*16}))

if __name__=='__main__':main()
