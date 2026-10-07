"""Independent label/selection reconciliation, provenance and split checks."""
import csv, gzip, hashlib, json, platform, importlib.metadata
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics';CAUSE=D/'orthoflow3_critic_rootcause_resolution_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    for name,h in read(HERE/'snapshot_integrity.json').items():assert sha(HERE/name)==h
    for name,h in read(HERE/'source_hashes.json').items():assert sha(name)==h
    for entry in read(HERE/'evidence_inventory.json'):assert sha(entry['path'])==entry['sha256']
    rows=read(HERE/'summary.json')['models'];lookup={(r['cohort'],r['model']):r for r in rows};n=0
    figure_states=read(HERE/'figure_selection.json')
    for meta in read(HERE/'cohorts.json'):
        expected=min(range(meta['N']),key=lambda i:hashlib.sha256(('q-basin-audit-v1|'+meta['states'][i]).encode()).hexdigest())
        assert figure_states[meta['name']]==expected
        d=np.load(HERE/'inputs'/f'{meta["name"]}.npz');s=d['success'];f=d['failure'];positive=s>=15;negative=f>=2;eligible=positive.any(1)
        for key in meta['models']:
            z=d['z::'+key];idx=np.argmax(z,axis=1);i=np.arange(meta['N']);r=lookup[meta['name'],key]
            assert r['selected_B15']==sum(positive[i,idx]);assert r['selected_unknown']==sum(~positive[i,idx]&~negative[i,idx])
            assert r['ranking_failures']==sum(eligible&negative[i,idx]);assert r['proposal_failures']==sum(negative.all(1))
            assert r['selected_B15']+r['ranking_failures']+r['ranking_unknown']==r['oracle_eligible']
            assert r['proposal_failures']+r['oracle_unknown']+r['oracle_eligible']==meta['N'];n+=1
    # Reconcile every unchanged original model-selection outcome, including unsuccessful seeds.
    original=0
    for ctl in [88138,88139]:
        for r in csv.DictReader((CAUSE/'held_controller_k16_v1/metrics_stage1.csv').open()):
            if int(r['controller'])!=ctl or r['subset']!='K16':continue
            key='__'.join([r['arm'],r['seed'],r['condition']]);new=lookup[f'ring_k16_{ctl}',key]
            assert int(r['B15'])==new['selected_B15'] and int(r['unresolved'])==new['selected_unknown'];original+=1
    for ctl in range(88132,88138):
        for r in csv.DictReader((CAUSE/f'motion_independent_confirmation_{ctl}/metrics.csv').open()):
            key='__'.join([r.get('arm',r.get('variant')),r['seed'],r['condition']]);pair=(f'ring_k2_{ctl}',key)
            if pair not in lookup:continue # Original secondary ensembles were not added to this audit.
            assert int(r['B15'])==lookup[pair]['selected_B15'] and int(r['unresolved'])==lookup[pair]['selected_unknown'];original+=1
    # Repeated learning seeds and TT/FF are not new physical states.
    corpus=read(CAUSE/'db_transfer_v1/pairs.json');sourceuids={r['state_uid']for r in corpus};families={r['family']for r in corpus};controlleruids={r['controller_uid']for r in corpus}
    splits=[]
    for sub in [f'motion_independent_confirmation_{j}'for j in range(88132,88138)]+[f'held_controller_k16_v1/target_{j}'for j in [88138,88139]]:
        states=read(CAUSE/sub/'states.json');p=read(CAUSE/sub/'protocol.json')
        a=len({r['uid']for r in states}&sourceuids);b=len({r['source_group']for r in states}&families);c=p['profiles'][0]['controller_uid']in controlleruids
        assert a==b==0 and not c
        splits.append(dict(target=sub,states=len(states),source_state_overlap=a,source_family_overlap=b,source_controller_UID_overlap=c,
                           source_corpus=str(CAUSE/'db_transfer_v1/pairs.json')))
    source=read(D/'orthoflow3_controller_intervention_generalization_v1/rich_probe_rows.json');oldids={r['state_uid']for r in source}
    for name in ['ring_v11_t0','ring_v11_mid']:
        meta=next(r for r in read(HERE/'cohorts.json')if r['name']==name);assert not oldids&set(meta['states'])
        splits.append(dict(target=name,source_state_overlap=0))
    # Eta-only cannot jointly follow an actual reversal in a shared exact bank.
    for r in read(HERE/'summary.json')['state_reversals']:
        if r['model'].split('__')[0]in ['eta_only','controller_cv_eta_only','family_cv_eta_only','global_train_eta','eta_mle','fixed_source_common_eta']:
            assert r['both_correct']==0,(r['cohort'],r['model'])
    fieldrows=read(HERE/'inputs/current_source_field_rows.json')
    ring=next(r for r in read(HERE/'current_full_field_cases.json')if r['scene']=='ring_exchange')
    ctlids={fieldrows[i]['controller_uid']for a,b,sg in ring['changes']for i in [a,b]}
    ctlpath=CAUSE/'db_transfer_v1/controllers.json';configs=read(ctlpath)
    contractkeys=['rng','environment_sha256','safety_config','safety_projection_sha256','success_semantics','orthoflow3_sha256','committed_t0_flow_sha256','dt','horizon']
    for key in contractkeys:
        values={json.dumps(configs[c]['config'][key],sort_keys=True)for c in ctlids};assert len(values)==1,key
    doc=dict(status='PASS',new_rollouts=0,new_training=0,verified_model_conditions=n,original_result_rows_reconciled=original,
             snapshot_hashes_verified=True,source_hashes_verified=True,outcome_blind_figure_state_rule_verified=True,split_audits=splits,
             physical_states=len({x for c in read(HERE/'cohorts.json')for x in c['states']}),
             per_state_rows=sum(1 for _ in gzip.open(HERE/'per_state.csv.gz','rt'))-1,
             current_Ring_field_pair_contract=dict(controllers=len(ctlids),identical_keys=contractkeys,controller_config_sha256=sha(ctlpath)),
             caveat='UID/family checks supplement frozen controller-weight and checkpoint provenance; no universal audit of all historical TEST-driven design choices is claimed.',
             python=platform.python_version(),packages={p:importlib.metadata.version(p)for p in ['numpy','scipy']})
    (HERE/'verification.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({k:v for k,v in doc.items()if k!='split_audits'}))

if __name__=='__main__':main()
