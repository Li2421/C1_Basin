"""Independent final manifest, leakage, model-selection and threshold audit."""
from pathlib import Path
import json,hashlib,datetime,itertools
import numpy as np
H=Path(__file__).resolve().parent

def read(n):return json.loads((H/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 c=read('COMPLETE.json');assert c['status']=='COMPLETE'
 for n,h in c['reports'].items():assert sha(H/n)==h,n
 for p,h in read('DELIVERY_CODE_SHA256.json').items():assert sha(p)==h,p
 for n in ['EXECUTION_FROZEN.receipt.json','STATISTICS_FROZEN.receipt.json','TRAINING_FROZEN.receipt.json']:
  for p,h in read(n)['files'].items():assert sha(p)==h,p
 allf=read('ALL_FOLDS_FROZEN.receipt.json');assert len(allf['files'])==allf['folds']==len(read('FOLDS.json'))
 checkpoints=set();first_training=[]
 tables=read('LABEL_TABLES.json');inputs=read('INPUTS_V2.json')['scene_inputs']
 for fold in read('FOLDS.json'):
  assert not(set(fold['sources'])&set(fold['targets']));folder=H/'folds'/fold['name'];p=json.loads((folder/'PRIORS.json').read_text());expected=np.mean([np.asarray(tables[n]['y']).mean((0,1))for n in fold['sources']],axis=0);assert np.array_equal(p['p0'],expected)
  with np.load(fold['source_labels_path'],allow_pickle=False)as z:assert {k.split('__')[0]for k in z.files}==set(fold['sources'])
  lam=json.loads((folder/'LAMBDA_SELECTION.json').read_text())['full'];assert lam['selected_lambda']==lam['lambda_order'][int(np.argmax(lam['mean_success_rates']))]
  for target in fold['targets']:
   assert set(lam['source_scenes'])==set(fold['sources'])and target not in {r['target']for r in lam['internal_heldout_scenes']}
  for mode in ['full','generator']:
   sel=json.loads((folder/mode/'SELECTED.json').read_text());rows=sel['all_seeds'];assert sel['selected']==min(rows,key=lambda x:(x['validation_loss'],x['seed']))
   assert {r['seed']for r in rows}=={17,23,41}
   for r in rows:
    assert sha(r['checkpoint'])==r['sha256']and not r['heldout_labels_used'];checkpoints.add(r['checkpoint']);first_training.append(r['utc'])
  choices=json.loads((folder/'CHOICES.json').read_text())
  for n,row in choices.items():
   assert len(row['state_indices'])==16 and len(row['controllers'])==3 and len(row['candidate_etas'])==54
   assert row['state_indices']==read('EVALUATION_STATES.json')['scenes'][n]
   assert all(m['selected_index'] is not None for m in row['methods'].values())
 for p,h in allf['files'].items():
  assert sha(p)==h
  for f,digest in json.loads(Path(p).read_text())['files'].items():assert sha(f)==digest,f
 phase1=read('HETEROGENEITY_CHECK.json');assert min(first_training)>phase1['utc']
 assert allf['utc']<=read('proposal_REGISTRATION.receipt.json')['utc']
 db=read('DB_POSTFLIGHT.json');assert db['passed']and db['all_requests_present']and db['no_unmerged_experiment_journals'];assert read('INTEGRITY_COMPARE.json')['passed']and read('METHOD_PARITY_CHECK.json')['passed']
 budget=read('FINAL_ROLLOUT_BUDGET.json');assert budget['new_total']<=20000 and budget['new_total']==c['new_execution_count']
 s=read('STATISTICS.json');d=s['decisions'];p=s['comparisons']['v2'];g=phase1['groups']['all8'];assert phase1['established']==(g['global_gap']>=.1 and g['ci_states'][0]>0)
 assert d['useful']==(p['difference_from_global']>0 and p['ci_states'][0]>0)
 assert d['not_harmful']==all(r['differences_from_global']['v2']>=-.1 for r in s['per_scene'].values())
 assert d['learned']==any(s['scene_matching'][m]['p_one_sided']<.05 and s['scene_matching'][m]['D']>s['scene_matching']['generic_input']['D']for m in ['v2_49','v2_49_no_fallback'])
 for m,r in s['scene_matching'].items():
  matrix=np.rint(np.asarray(r['matrix'])*48).astype(int);perms=np.asarray(list(itertools.permutations(range(len(matrix)))));traces=matrix[np.arange(len(matrix))[None],perms].sum(1);assert r['p_one_sided']==np.mean(traces>=np.trace(matrix))
 with np.load(H/'METHOD_BOOTSTRAP.npz',allow_pickle=False)as z:
  assert z['states'].shape==z['scene_states'].shape==(5000,8)
  methods=list(z['methods']);gi=methods.index('global_fixed')
  for i,m in enumerate(methods):
   assert np.array_equal(np.percentile(z['states'][:,i]-z['states'][:,gi],[2.5,97.5]),s['comparisons'][m]['ci_states'])
 receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,report_hashes_verified=len(c['reports']),all_fold_freezes_verified=True,outer_checkpoints_verified=len(checkpoints),no_target_label_leakage=True,source_only_model_and_lambda_selection_verified=True,heterogeneity_before_training_verified=True,all_folds_frozen_before_proposals_verified=True,exact_matching_pvalues_independently_recomputed=True,intervals_and_unrounded_thresholds_verified=True,new_rollouts=c['new_execution_count'],db_and_source_audits_passed=True)
 (H/'FINAL_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
