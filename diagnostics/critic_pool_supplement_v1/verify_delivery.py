"""Independent delivery checks: source isolation, fixed recipes, frozen ensembles, paired stats."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent

def read(n):return json.loads((H/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 complete=read('COMPLETE.json');assert complete['status']=='COMPLETE'
 for name,h in complete['reports'].items():assert sha(H/name)==h,name
 for p,h in read('DELIVERY_CODE_SHA256.json').items():assert sha(p)==h,p
 for name in ['EXECUTION_FROZEN.receipt.json','TRAINING_FROZEN.receipt.json','FROZEN.receipt.json']:
  for p,h in read(name)['files'].items():assert sha(p)==h,p
 frozen=read('FROZEN.receipt.json');access=read('VALIDATION_LABEL_ACCESS.receipt.json');assert frozen['utc']<=access['utc']and access['first_label_parse_for_scoring_after_freeze']
 pool={r['name']:r for r in read('TRAINING_POOL.json')};forbidden=('lit_swap8_obstacles','lit_swap8_free','lit_circle_five','xsc_swap_n6','xsc_swap_n10','xsc_het_')
 assert len([r for r in pool.values()if not r['new']])==14
 assert all(not n.startswith(forbidden)and not r.get('static_definition','').startswith(forbidden)for n,r in pool.items())
 for p in(H/'datasets').glob('*.json'):
  d=json.loads(p.read_text());assert not any(n.startswith(forbidden)for n in d['names']);C=np.array([pool[n]['c_prime']for n in d['names']]);assert np.allclose(C.mean(0),d['c_mean'],rtol=0,atol=1e-12)
  sd=C.std(0);assert np.allclose(np.where(sd<1e-6,1,sd),d['c_scale'],rtol=0,atol=1e-12);assert np.allclose(np.where(sd<1e-9,1,sd),d['distance_sigma'],rtol=0,atol=1e-12)
  assert all(n not in v['names']and set(v['names'])<=set(d['names'])and len(v['names'])==3 for n,v in d['near'].items())
  with np.load(d['path'],allow_pickle=False)as z:
   assert z['x'].shape==(d['rows'],27)and np.isfinite(z['x']).all()and np.isfinite(z['y']).all();assert np.all((z['y']>=0)&(z['y']<=1));assert np.allclose(np.bincount(z['scene'],weights=z['prob']),np.full(len(d['names']),1/len(d['names'])),rtol=0,atol=1e-12)
 tasks=read('TRAIN_TASKS.json');assert len(tasks)==read('TRAINING_PLAN.json')['fits'];assert all(t['kind']in ['neighbour','no_neighbour','original']for t in tasks)
 for t in tasks:
  folder=H/'fits'/t['id']/f"seed{t['seed']}";fit=json.loads((folder/'FIT.receipt.json').read_text());pred=json.loads((folder/'PREDICTIONS.receipt.json').read_text());expected=4000 if t['kind']=='original'else 3000
  assert fit['steps']==expected and fit['batch_size']==1024 and fit['no_early_stopping']and fit['no_seed_selection']and not fit['validation_labels_used'];assert fit['initialization_seed']==fit['sampling_seed']==t['seed'];assert sha(fit['checkpoint'])==fit['sha256']and sha(pred['path'])==pred['sha256'];assert not any(n.startswith(forbidden)for n in fit['source_scenes']);assert json.loads((folder/'history.json').read_text())[-1]['step']==expected
 choices=read('ALL_CHOICES.json')
 for run in read('RUNS.json'):
  mean={n:[]for n in run['targets']}
  for seed in run['seeds']:
   with np.load(H/'fits'/run['id']/f'seed{seed}'/'PREDICTIONS.npz',allow_pickle=False)as z:
    for n in mean:mean[n].append(z[n])
  with np.load(H/'predictions'/run['id']/'ENSEMBLE.npz',allow_pickle=False)as z:
   for n,v in mean.items():
    x=np.mean(v,axis=0).reshape(run['target_metadata'][n]['shape']);assert np.array_equal(x,z[n+'__mean_logits']);prob=np.exp(-np.logaddexp(0,-x));assert np.array_equal(prob,z[n+'__probability']);assert choices[run['id']][n]['selected_index']==int(prob.mean((0,1)).argmax());assert np.array_equal(choices[run['id']][n]['per_cell_indices'],prob.argmax(-1))
  if run['role']=='cv':
   sources=read('datasets/'+run['dataset']+'.json')['names'];assert not(set(sources)&set(run['targets']));crowded=[n for n in sources if pool[n]['source_version']=='lit_crowded_random_flow'];assert len(crowded)in [0,7]
  assert len(run['seeds'])==(3 if run['role']=='learning_curve'else 5 if run['role']=='class_addition'else 10)
 s=read('STATISTICS.json');d=s['decisions'];comparisons=s['comparisons'];g=comparisons['main']['comparisons']['global_fixed'];k=comparisons['main']['comparisons']['knn2'];o=comparisons['main']['comparisons']['old_pool'];near='明显超过'if k['difference']>0 and k['ci_states'][0]>0 else'持平或略好'if k['difference']>=0 else'不如'
 assert d['useful']==(g['ci_states'][0]>0);assert d['beats_neighbour']==near;assert d['supplement_useful']==(o['difference']>0 and o['ci_states'][0]>0);assert d['not_harmful']==all(v>=-.1 for v in g['per_scene'].values());assert d['critic_fixed_on_this_validation_set']==(d['useful']and d['supplement_useful']and d['not_harmful']and near!='不如')
 with np.load(H/'BOOTSTRAP.npz',allow_pickle=False)as z:
  methods=list(z['methods']);assert z['states'].shape==z['scene_states'].shape==(5000,len(methods))
  for i,m in enumerate(methods):
   for base in ['global_fixed','knn2','old_pool']:
    j=methods.index(base);assert np.array_equal(np.percentile(z['states'][:,i]-z['states'][:,j],[2.5,97.5]),comparisons[m]['comparisons'][base]['ci_states']);assert np.array_equal(np.percentile(z['scene_states'][:,i]-z['scene_states'][:,j],[2.5,97.5]),comparisons[m]['comparisons'][base]['ci_scene_states'])
 assert read('DB_POSTFLIGHT.json')['passed']and read('HISTORICAL_ROWS_END.receipt.json')['passed']and read('INTEGRITY_COMPARE.json')['passed'];assert complete['new_main_requests']<=32000;assert read('REPLAY_CHECK.json')['passed']and read('REPLAY_CHECK.json')['count']==2
 events=[json.loads(l)for l in(H/'SCHEDULER_EVENTS.jsonl').read_text().splitlines()]
 assert all(e['count']<=17 for e in events if e['event']=='submitted');assert all(e['count']<=14 for e in events if e['event']=='training_submitted_RUNNING_readback')
 receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,report_hashes_verified=len(complete['reports']),models_verified=len(tasks),source_only_normalization_and_leave_self_out_neighbours_verified=True,scene_equal_sampling_verified=True,all_frozen_ensembles_and_choices_recomputed=True,validation_label_parse_after_freeze_verified=True,fixed_training_steps_seeds_and_recipe_verified=True,paired_intervals_and_unrounded_decisions_verified=True,db_and_old_source_audits_passed=True,main_requests=complete['new_main_requests'],maximum_total_cpus=18,maximum_training_parallel=14)
 (H/'FINAL_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
