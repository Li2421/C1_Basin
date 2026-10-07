"""Read-only export of existing predictions/outcomes; never import a rollout runner."""
import csv, hashlib, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]; D=ROOT/'diagnostics'
MATCH=D/'orthoflow3_tt_ff_matched_learnability_v1'
PROBE=D/'orthoflow3_controller_information_probe_v1'
CAUSE=D/'orthoflow3_critic_rootcause_resolution_v1'
HASHES={}; AUDIT=[]; COHORTS=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def track(path):
    path=Path(path); HASHES[str(path)]=sha(path); return path
def read(path):return json.loads(track(path).read_text())
def npz(path):return dict(np.load(track(path),allow_pickle=False))
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def frozen_predictions(folder,name='frozen_predictions.npz',guard='prediction_freeze.json'):
    f=read(folder/guard); assert f.get('target_labels_read',False) is False
    assert sha(folder/name)==f['predictions_sha256'];return npz(folder/name)
def save(name,s,f,eta,states,scores,tier,shared=True,**extra):
    s=np.asarray(s,int); f=np.asarray(f,int); assert s.shape==f.shape
    n,k=s.shape; eta=np.asarray(eta,float)
    if eta.ndim==2:eta=np.broadcast_to(eta,(n,k,3)).copy()
    assert eta.shape==(n,k,3) and len(states)==n
    assert np.all(s>=0) and np.all(f>=0) and np.all(s+f<=16)
    for key,z in scores.items():assert np.asarray(z).shape==s.shape and np.isfinite(z).all(),(name,key)
    if shared:assert np.array_equal(eta,np.broadcast_to(eta[:1],eta.shape))
    meta=dict(name=name,tier=tier,states=list(states),N=n,K=k,shared_eta=shared,models=list(scores),**extra)
    np.savez_compressed(HERE/'inputs'/f'{name}.npz',success=s,failure=f,eta=eta,**{f'z::{key}':z for key,z in scores.items()})
    COHORTS.append(meta)

def db_check(pairs,controller,s,f):
    """Recheck exact immutable seed keys and count bounds against read-only global DB."""
    con=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
    con.row_factory=sqlite3.Row; uids=[]; numeric=0
    for ix,p in enumerate(pairs):
        rows={r['seed_key']:r for r in con.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],controller))}
        ss=ff=0
        for seed in range(16):
            r=rows[json.dumps({'future_index':seed},separators=(',',':'))]
            assert not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE'
            assert not r['collision'];uids.append(r['rollout_uid'])
            if r['numerical_failure']:numeric+=1
            elif r['success']:ss+=1
            else:ff+=1
        assert (ss,ff)==(int(s.flat[ix]),int(f.flat[ix])),(ix,ss,ff)
    con.close()
    AUDIT.append(dict(controller=controller,pairs=len(pairs),seed_records=len(uids),numerical=numeric,
                      exact_DB_recheck=True,uid_digest=hashlib.sha256('|'.join(uids).encode()).hexdigest()))

def matched():
    p=read(MATCH/'protocol.json'); data=npz(MATCH/'dataset.npz'); truth=npz(MATCH/'test_truth.npz')
    pred=frozen_predictions(MATCH,guard='test_predictions_frozen.json')
    source=read(MATCH/'models_frozen.json')
    for run in source['runs']:assert sha(track(run['checkpoint']))==run['checkpoint_sha256']
    archive=npz(D/'orthoflow3_tt_field_basin_reshape_v1/paired_snapshot.npz')
    assert np.array_equal(archive['primary'],truth['outcomes'])
    old=read(D/'orthoflow3_tt_field_basin_reshape_v1/verification.json');assert old['status']=='PASS'
    for scene in ['toy_give_way','ring_exchange']:
        splitgroups=[{p['states'][i]['family'] for i in range(len(p['states'])) if data['scene'][i]==scene and data['split'][i]==split} for split in ['train','validation','test']]
        assert all(not (a&b) for a,b in [(splitgroups[0],splitgroups[1]),(splitgroups[0],splitgroups[2]),(splitgroups[1],splitgroups[2])])
        ids=np.flatnonzero((data['scene']==scene)&(data['split']=='test'))
        ti=np.array([list(truth['indices']).index(i)for i in ids]); states=[p['states'][i]['state_uid'] for i in ids]
        assert all(p['states'][i]['physical']['timestep']==0 for i in ids)
        for ci,chain in enumerate(['TT','FF']):
            a=truth['outcomes'][ci,ti];s=(a[:,:,:,0]*(1-a[:,:,:,1])).sum(-1);f=((1-a[:,:,:,0])*(1-a[:,:,:,1])).sum(-1)
            scores={}
            for key,z in pred.items():
                bits=key.split('__')
                if bits[:2]==[scene,chain] and bits[-2]=='test' and bits[-1]!='indices':scores['__'.join([bits[2],bits[3],bits[-1]])]=z
            save(f'matched_{scene}_{chain}',s,f,p['eta'],states,scores,'primary',scene=scene,controller=chain,
                 true_t0=True,training_states=32,validation_states=8,split='fresh TEST',seeds=16,source=str(MATCH))
    AUDIT.append(dict(cohort='matched',families_disjoint=True,predictions_hash_verified=True,models=24,
                      truth_matches_previously_seed_audited_snapshot=True,prior_seed_audit=str(D/'orthoflow3_tt_field_basin_reshape_v1/verification.json')))

def held_v11():
    for sub,t0 in [('held_controller_true_t0_v1',True),('held_controller_robust_selection_v1',False)]:
        folder=PROBE/sub; frozen=read(folder/'frozen_predictions.json');pairs=read(folder/'pairs.json');p=read(folder/'protocol.json')
        ev=read(folder/('evaluation_stage1.json' if t0 else 'held_selection_detail.json'))
        rows=ev['candidate_outcomes']; n=len(rows)//16; pairs=pairs[:n*16]
        assert all((a['state_uid'],a['eta_uid'])==(b['state_uid'],b['eta_uid'])for a,b in zip(rows,pairs))
        ss=np.array([r['success']for r in rows]).reshape(n,16);ff=np.array([r['failure']for r in rows]).reshape(n,16)
        scores={}
        for m in frozen['models' if t0 else 'predictions']:
            ck=PROBE/'models'/'ring_exchange'/m['kind']/f'seed{m["seed"]}'/'checkpoint.msgpack'
            assert sha(track(ck))==m['checkpoint_sha256']
            prob=np.array(m['probabilities'][:n*16]).reshape(n,16); assert np.all((prob>0)&(prob<1))
            scores[f'{m["kind"]}__{m["seed"]}__{m["context"]}']=np.log(prob/(1-prob))
        controller=p.get('held_controller_uid',pairs[0].get('alternate_controller_uid'))
        db_check(pairs,controller,ss,ff)
        save('ring_v11_t0' if t0 else 'ring_v11_mid',ss,ff,np.array([r['eta']for r in pairs]).reshape(n,16,3),
             [r['state_uid']for r in pairs[::16]],scores,'primary' if t0 else 'supporting',true_t0=t0,scene='ring_exchange',controller='v11',
             split='independent TEST' if t0 else 'fresh family diagnostic',seeds=16,source=str(folder))

def confirmations():
    for ctl in range(88132,88138):
        folder=CAUSE/f'motion_independent_confirmation_{ctl}';d=npz(folder/'dataset.npz');p=read(folder/'protocol.json');pairs=read(folder/'pairs.json')
        scores=frozen_predictions(folder);n=64;k=2
        s=d['success'].reshape(n,k);f=d['failure'].reshape(n,k)
        db_check(pairs,p['profiles'][0]['controller_uid'],s,f)
        save(f'ring_k2_{ctl}',s,f,np.array([r['eta']for r in pairs]).reshape(n,k,3),[r['state_uid']for r in pairs[::k]],scores,
             'supporting prospective K2',true_t0=True,scene='ring_exchange',controller=str(ctl),split='fresh TEST',seeds=16,source=str(folder))
    folder=CAUSE/'held_controller_k16_v1';mf=read(folder/'models_frozen.json')
    for entry in mf['models']+mf['legacy_models']:
        path=Path(entry['path'])/entry.get('checkpoint','checkpoint.msgpack');assert sha(track(path))==entry['checkpoint_sha256']
    read(folder/'protocol.json');read(folder/'statistical_scope.json');read(folder/'stage1_eligibility.json')
    for ctl in [88138,88139]:
        path=folder/f'target_{ctl}';d=npz(path/'truth_stage1.npz');pred=frozen_predictions(path,'predictions.npz');pairs=read(path/'pairs.json')[:256]
        p=read(path/'protocol.json');states=read(path/'states.json')[:16];assert all(x['physical']['timestep']==0 for x in states)
        db_check(pairs,p['profiles'][0]['controller_uid'],d['success'],d['failure'])
        scores={k:v[:16]for k,v in pred.items()}
        fixed=np.zeros((16,16));fixed[:,2]=1;scores['fixed_source_common_eta__0__correct']=fixed
        save(f'ring_k16_{ctl}',d['success'],d['failure'],np.array([r['eta']for r in pairs]).reshape(16,16,3),[r['uid']for r in states],scores,
             'primary',true_t0=True,scene='ring_exchange',controller=str(ctl),split='fresh TEST',seeds=16,source=str(path),seen_eta_indices=list(range(8)),unseen_eta_indices=list(range(8,16)))

def toy_replication():
    folder=D/'orthoflow3_c1_generalization_v1/toy_replication';freeze=read(folder/'frozen_proposals.json');states=freeze['states'];assert freeze['selection_before_outcomes']
    cells={};n=0
    for path in sorted((folder/'raw').glob('shard*.jsonl')):
        for line in track(path).read_text().splitlines():
            r=json.loads(line);key=(r['episode_index'],r['kind'],r['future_index']);assert key not in cells
            assert r['scientific_outcome_valid'] and not r['numerical_failure'];cells[key]=r;n+=1
    s=np.array([[sum(cells[i,f'sample_{j}',seed]['success']for seed in range(16))for j in range(16)]for i in range(48)])
    scores={}
    for kind in ['critic','eta_only_kernel','eta_only_mlp']:
        z=np.array([r[f'{kind}_scores']for r in states])
        if kind=='eta_only_kernel':z=np.log(np.clip(z,1e-12,1-1e-12)/np.clip(1-z,1e-12,1))
        scores[f'{kind}__frozen__correct']=z
        assert all(int(np.argmax(z[i]))==r[f'{kind}_choice']for i,r in enumerate(states))
    save('toy_generator_rep48',s,16-s,np.array([[r['eta'][f'sample_{j}']for j in range(16)]for r in states]),[r['state_uid']for r in states],scores,
         'supporting independent TEST',False,true_t0=True,scene='toy_give_way',controller='legacy_TT',split='fresh TEST',seeds=16,source=str(folder))
    baseline=[]
    for kind in ['fixed_common','generator_mean','safety','mac_only']:
        counts=[sum(cells[i,kind,j]['success']for j in range(16))for i in range(48)]
        baseline.append(dict(kind=kind,successes=counts,not_same_pool=True))
    write(HERE/'inputs/toy_policy_baselines.json',baseline);AUDIT.append(dict(cohort='toy_rep48',raw_records=n,all_seed_sets_complete=True,all_scores_frozen_before_outcomes=True))

def db_transfer():
    folder=CAUSE/'db_transfer_v1';truth=read(folder/'cached_truth.json');inp=read(folder/'target_input_manifest.json')
    pred=npz(folder/'target_predictions.npz');guard=read(folder/'prediction_freeze.json');assert sha(folder/'target_predictions.npz')==guard['predictions_sha256']
    s=np.array([r['success']for r in truth]).reshape(24,16);f=np.array([r['failure']for r in truth]).reshape(24,16)
    assert [r['state_uid']for r in truth[::16]]==[r['state_uid']for r in inp]
    save('db_transfer24',s,f,np.array([r['eta']for r in inp]),[r['state_uid']for r in inp],pred,
         'supporting cached target diagnostic',False,true_t0=True,scene='double_bottleneck',controller='held_scene',split='previously opened target',seeds=16,source=str(folder))

def source_fields():
    folder=D/'orthoflow3_controller_intervention_generalization_v1';data=npz(folder/'second_variant/triplet_dataset_h20.npz');rows=read(folder/'rich_probe_rows.json')
    records=[]
    for scene in ['toy_giveway','double_bottleneck','four_way_intersection','ring_exchange']:
        ids=np.array([i for i,r in enumerate(rows)if r['scene']==scene]);rr=[rows[i]for i in ids];va=np.flatnonzero([r['split']=='validation'for r in rr])
        tr={r['state_uid']for r in rr if r['split']=='train'};vs={rr[i]['state_uid']for i in va};assert not tr&vs
        zall={};metadata=[]
        for path in sorted((PROBE/'models'/scene).glob('*/seed*/validation_predictions.npz')):
            if 'crossmatrix' in str(path):continue
            z=npz(path)['logits'];doc=read(path.parent/'summary.json');track(path.parent/'checkpoint.msgpack')
            key=f'{path.parent.parent.name}__{doc["seed"]}';zall[key]=z[:,va]
            metadata.append(dict(model=key,heldout_controller=doc.get('heldout_controller'),source=str(path)))
        counts={f'{c}_{v}':data[f'{c}_{v}'][ids][va]for c in ['base','alt','second']for v in ['s','f']}
        np.savez_compressed(HERE/'inputs'/f'source_field_{scene}.npz',eta=data['eta'][ids][va],**counts,**{f'z::{k}':v for k,v in zall.items()})
        records.append(dict(scene=scene,rows=[rr[i]for i in va],models=metadata,controllers=2 if scene=='toy_giveway' else 3,
                            warning='Checkpoint selected on these source VAL families; descriptive, partial counts; candidate panel source-selected; not independent TEST'))
    write(HERE/'inputs/source_fields.json',records)

def current_source_fields():
    folder=CAUSE/'db_transfer_v1';rows=read(folder/'pairs.json');scores={};ids=None
    for path in sorted((folder/'models').glob('*/seed*/validation_predictions.npz')):
        d=npz(path)
        if ids is None:ids=d['indices']
        else:assert np.array_equal(ids,d['indices'])
        for c in ['correct','wrong_controller','joint_state_context_shuffle']:
            scores[f'{path.parent.parent.name}__{path.parent.name}__{c}']=d[c]
    rr=[rows[i]for i in ids];assert all(r['split']=='validation'for r in rr)
    write(HERE/'inputs/current_source_field_rows.json',[{k:r[k]for k in ['state_uid','eta_uid','controller_uid','scenario','s16','f16']}for r in rr])
    np.savez_compressed(HERE/'inputs/current_source_fields.npz',**scores)

def main():
    (HERE/'inputs').mkdir(parents=True,exist_ok=True)
    matched();held_v11();confirmations();toy_replication();db_transfer();source_fields();current_source_fields()
    # Preserve current and historical evidence, including failures, rather than selecting winners.
    supporting=[]
    for folder in [PROBE,D/'orthoflow3_controller_intervention_generalization_v1',D/'orthoflow3_c1_generalization_v1',D/'orthoflow3_mode_free_k_sweep_latency_v1',CAUSE/'motion_replication_adjudication']:
        for name in ['final_report.md','adjudication.json','k_sweep.csv','metrics.csv','paired_uncertainty.csv']:
            path=folder/name
            if path.exists():track(path);supporting.append(str(path))
    write(HERE/'cohorts.json',COHORTS);write(HERE/'cache_audit.json',AUDIT);write(HERE/'source_hashes.json',HASHES)
    write(HERE/'supporting_sources.json',supporting)
    write(HERE/'snapshot_integrity.json',{str(p.relative_to(HERE)):sha(p)for p in (HERE/'inputs').iterdir()})
    print(json.dumps(dict(cohorts=len(COHORTS),hashes=len(HASHES),new_rollouts=0,DB_verified_seed_records=sum(a.get('seed_records',0)for a in AUDIT))))

if __name__=='__main__':main()
