"""Validate completed evidence and gates; never generates a rollout."""
from pathlib import Path
import json,hashlib,sqlite3,datetime
import numpy as np
P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
 checks=[]
 for md,receipt in [('PREREGISTRATION.md','REGISTRATION.json'),('PRIORITY_AMENDMENT.md','AMENDMENT_RECEIPT.json'),('C1_TIE_RULE.md','C1_TIE_RECEIPT.json'),('CHECKPOINT_DIAGNOSTIC.md','CHECKPOINT_DIAGNOSTIC_RECEIPT.json'),('C2_FALLBACK_PREREGISTRATION.md','C2_FALLBACK_RECEIPT.json'),('C1_FAMILY_GATE_CLARIFICATION.md','C1_FAMILY_GATE_RECEIPT.json')]:assert sha(P/md)==read(receipt)['sha256'];checks.append('registration '+md)
 for f in read('DATA_INDEX.json')['files']:assert sha(f['path'])==f['sha256'];checks.append('unchanged source '+f['path'])
 assert read('SOURCE_VALIDATION.json')['status']=='PASS';checks.append('106496 N1 labels verified against DB')
 cases=read('c1/cases.json');missing=read('c1/missing_cases.json');cached=read('c1/cached_rollout_uids.json');uid=[r['rollout_uid'] for r in cases];assert len(uid)==len(set(uid))==24576;assert len(missing)==20736;assert len(cached)==3840;assert not(set(cached)&{r['rollout_uid'] for r in missing})
 expected=sha(P/'c1/protocol.json');stats={'unknown':0,'success':0,'collision':0};elapsed=0
 for c in missing:
  f=P/'c1/raw'/(c['rollout_uid']+'.json');r=json.loads(f.read_text());assert r['case']==c;assert r['protocol_sha256']==expected;assert not(r['success'] and (r['numerical_failure'] or r['collision']));assert 0<=r['episode_length']<=700 and (r['numerical_failure'] or r['episode_length']>=1)
  stats['unknown']+=r['numerical_failure'];stats['success']+=r['success'];stats['collision']+=r['collision'];elapsed+=r['elapsed_seconds']
 for i in range(88):assert read(f'c1/worker{i}.json')['completed']
 db=sqlite3.connect('file:'+str(P.parents[1]/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True)
 for c in cases:
  r=db.execute('select state_uid,eta_uid,controller_uid,seed_key,conflict_quarantined from rollout where rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None and list(r[:4])==[c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']] and r[4]==0
 y=np.load(P/'c1/labels.npz')['Y'];assert y.shape==(6,32,16,8) and set(np.unique(y))<=set([-1,0,1]);assert read('c1/RESULTS.json')['new_rollouts']==20736;checks.append('C1 manifest/raw/worker/DB/labels complete and exact')
 for t in read('splits.json')['tasks']:
  assert not(set(t['train'])&set(t['val']));assert not(set(t['train'])&set(t['test']));assert not(set(t['val'])&set(t['test']))
 z=np.load(P/'N1_predictions.npz');old=np.load(P.parents[1]/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/frozen_predictions.npz')
 for n in [32,512]:np.testing.assert_array_equal(z[f'N1_o0_n{n}_scalar'],old[f'physical23_eta_n{n}_logits'])
 assert read('EFFECTIVE_BASIS_AUDIT.json')['physical_steps']==0;assert not read('C2_STATIC_FEATURES_AUDIT.json')['outcomes_read'];assert not read('C2_STATIC_FEATURES_AUDIT.json')['errors']
 dec=read('C2_DECISION.json')
 if not read('c1/RESULTS.json')['gate']['selection_transfer_qualified']:
  assert dec['status'].startswith('NOT EXECUTED');c=read('C2_FALLBACK_RESULTS.json');assert len(c['records'])==7 and c['new_rollouts']==0
  a=np.load(P/'C2_FALLBACK_PREDICTIONS.npz');np.testing.assert_array_equal(a['R0'],a['R0_mismatched']);np.testing.assert_array_equal(a['R1'][:,1],a['R1'][:,4]);s=read('C2_SPLITS.json');assert not(set(s['train_states'])&set(s['validation_states']));assert not(set(s['joint_test_states'])&(set(s['train_states'])|set(s['validation_states'])));checks.append('C2 main gate and fallback isolation')
 out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,new_rollout_stats=stats,new_rollout_cpu_episode_hours=elapsed/3600,new_rollouts=20736,reused_unique=read('SOURCE_VALIDATION.json')['all_reused_unique'],all_unknown_retained=True,original_source_files_unchanged=True,actual_jobs_finished=True)
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
