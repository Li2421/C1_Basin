"""Full saved-trajectory equivalence and original/fast paired certification."""
from dispatch import *
from acceleration import accelerated_case
KEYS=('positions','velocities','nominal','safe','actions','projection_rewrite','swept_clearance','clearance_types','route_guides','route_remaining')
OUTCOMES=('success','collision','timeout','numerical_failure','outcome','episode_length','constraint_violation_steps','sampled_speed_violation_steps','restoration_steps','minimum_swept_clearance','final_goal_distances')

def compare(a,b,new=False):
 with np.load(a['trace'])as x,np.load(b['trace'])as y:
  keys=KEYS+ (('admission_json',)if new else())
  arrays={k:x[k].shape==y[k].shape and x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes()for k in keys}
 outcomes={k:a.get(k)==b.get(k)for k in OUTCOMES}
 return dict(equal=all(arrays.values())and all(outcomes.values()),arrays=arrays,outcomes=outcomes,a=a['rollout_uid'],b=b['rollout_uid'])

def certify():
 from unit_checks import main as units
 units();scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
 old=read(HERE/'inputs/previous/REPLAY_INPUTS.json')['records'];records=[]
 for x in old:
  prior=x['case'];c=accelerated_case(scenes[prior['scene']],prior['controller_payload']['nominal'],prior['state_index'],prior['spec'],purpose='old_equivalence200');records.append(dict(case=c,reference=x['reference'],type_id=x['type_id']))
 assert len(records)==200 and len({x['type_id']for x in records})==17
 dump(HERE/'REPLAY_INPUTS.json',dict(records=records,count=200))
 if(HERE/'EQUIVALENCE.json').exists():
  if not read(HERE/'EQUIVALENCE.json')['passed']:raise RuntimeError('HARD_STOP old numerical equivalence failed')
 else:
  rows={r['rollout_uid']:r for r in execute([x['case']for x in records],'accel_paired_old200',worker_entry='worker_accel.py')}
  checks=[compare(x['reference'],rows[x['case']['rollout_uid']])for x in records if x['case']['rollout_uid']in rows]
  passed=len(checks)==200 and all(x['equal']for x in checks)
  dump(HERE/'EQUIVALENCE.json',dict(passed=passed,episodes=len(checks),checks=checks,route_modes=sorted({x['case']['scenario_payload']['route_guidance']for x in records})))
  if not passed:raise RuntimeError('HARD_STOP old numerical equivalence failed')
 if(HERE/'ACCELERATION_CERTIFICATE.json').exists():return read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']
 combos=read(HERE/'COMBINATIONS.json')['combinations'];selected=[e['scene']for e in read(HERE/'SCENES.json')['entries']if e['type_id']in(8,10,11,15,16)and read(HERE/'PASSAGES.json')['by_scene'][e['scene']['name']]['representative']]
 pairs=[]
 for si in range(4):
  for sc in selected:
   for ctl in controllers():
    sp=combos[len(pairs)%len(combos)]['candidate_pool'][len(pairs)%3]
    pairs.append((case(sc,ctl,si,sp,purpose='new_pair'),accelerated_case(sc,ctl,si,sp,purpose='new_pair')))
 pairs=pairs[:50];assert len(pairs)==50
 a={r['rollout_uid']:r for r in execute([a for a,b in pairs],'new_original50',worker_entry='worker_recorded.py')}
 b={r['rollout_uid']:r for r in execute([b for a,b in pairs],'accel_paired_new50',worker_entry='worker_accel.py')}
 checks=[compare(a[x['rollout_uid']],b[y['rollout_uid']],new=True)for x,y in pairs if x['rollout_uid']in a and y['rollout_uid']in b]
 hashes=all(sha(HERE/'accel_inputs'/Path(x['target']).name)==x['sha256']for x in read(HERE/'inputs/previous_acceleration_inputs.json')['files'])
 passed=len(checks)==50 and all(x['equal']for x in checks)and hashes
 dump(HERE/'ACCELERATION_CERTIFICATE.json',dict(passed=passed,old_replays=200,new_pairs=len(checks),checks=checks,acceleration_module_hashes_equal=hashes,enabled=['same CSC arrays','NumPy Threefry fold','far-wall cost pruning','far-wall local constraint prescreen']if passed else[],original_seconds=sum(r['wall_seconds']for r in a.values()),fast_seconds=sum(r['wall_seconds']for r in b.values()),prior_random_and_sparse_checks='inputs/previous/ACCELERATION_UNITS.json',fallback='Original numerical path if paired certificate fails'))
 if not passed:
  with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n新代码加速认证未通过，关闭加速，使用原代码继续。\n')
 return passed

def no_passage(make):
 path=HERE/'NO_PASSAGE_EQUIVALENCE.json'
 if path.exists():return read(path)['passed']
 ps=read(HERE/'PASSAGES.json')['by_scene'];refs=[x for x in read(HERE/'REPLAY_INPUTS.json')['records']if not ps[x['case']['scene']]['passages']][:20]
 scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']};menu=read(HERE/'COMBINATIONS.json')['combinations'][0]['candidate_pool'][0]['one_way']
 cases=[make(scenes[x['case']['scene']],x['case']['controller_payload']['nominal'],x['case']['state_index'],dict(x['case']['spec'],one_way=menu),purpose='no_passage20')for x in refs]
 rows={r['rollout_uid']:r for r in execute(cases,'no_passage20')};checks=[compare(x['reference'],rows[c['rollout_uid']])for c,x in zip(cases,refs)if c['rollout_uid']in rows];passed=len(checks)==20 and all(x['equal']for x in checks)
 dump(path,dict(passed=passed,pairs=len(checks),checks=checks));return passed
