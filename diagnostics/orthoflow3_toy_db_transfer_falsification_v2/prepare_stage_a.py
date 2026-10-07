#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,sqlite3
from pathlib import Path
import numpy as np

H=Path(__file__).parent;D=H.parent;V1=D/'orthoflow3_toy_db_transfer_falsification_v1';DB=D/'orthoflow3_db_shared_mode_transfer_v1';G=H.parents[1]/'shared_rollout_db'
FAMS=('raw_toy','translation','diagonal','rotation_anisotropic','regularized_affine')
def read(p):return list(csv.DictReader(open(p)))
def dump(n,x):(H/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(n,rows):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(stage):
 H.mkdir(parents=True,exist_ok=True);(H/'plans'/'stage_a').mkdir(parents=True,exist_ok=True);(H/'raw').mkdir(exist_ok=True)
 if stage=='prepare':
  folds=json.load(open(V1/'mode_cv_folds.json'));trs=read(V1/'leave_mode_out_transforms.csv');pred=read(V1/'heldout_mode_predictions.csv')
  dump('mode_cv_folds.json',{**folds,'reused_from':str(V1/'mode_cv_folds.json'),'sha256':sha(V1/'mode_cv_folds.json')});write('leave_mode_out_transforms.csv',trs)
  stats=[]
  for fam in FAMS:
   z=[r for r in pred if r['family']==fam]
   stats.append({'family':fam,'heldout_modes':len(z),'median_residual':float(np.median([float(x['target_residual']) for x in z])),
    'mean_residual':float(np.mean([float(x['target_residual']) for x in z])),'max_residual':max(float(x['target_residual']) for x in z),
    'median_nearest_robust_distance':float(np.median([float(x['nearest_independent_robust_distance']) for x in z])),
    'max_condition_number':max(float(x['condition_number']) for x in z)})
  promising_fams={r['family'] for r in stats if r['family'] not in ('raw_toy','translation') and r['median_residual']<=.20 and r['median_nearest_robust_distance']<=.10 and r['max_condition_number']<=10.01}
  # Frozen A3 point rule. It uses coordinates/robust geometry only, never held-out control outcomes.
  selected=[]
  for r in pred:
   good=r['family'] in promising_fams and float(r['target_residual'])<=.15 and float(r['nearest_independent_robust_distance'])<=.10
   q=dict(r,geometry_promising=good,control_validation_selected=good,selection_rule='family median residual<=0.20, robust distance<=0.10, cond<=10.01; point residual<=0.15 and robust distance<=0.10')
   selected.append(q)
  write('heldout_mode_predictions.csv',selected);write('geometry_family_summary.csv',stats)
  panel=json.load(open(V1/'stage_a_state_panel.json'));dump('stage_a_state_panel.json',{**panel,'reused_from':str(V1/'stage_a_state_panel.json'),'sha256':sha(V1/'stage_a_state_panel.json')})
  split=json.load(open(DB/'db_state_split.json')); sm={s['state_id']:s for s in split['states']}; con=sqlite3.connect(G/'rollout.sqlite');con.row_factory=sqlite3.Row
  sc=con.execute("select scenario_uid from scenario where name='DoubleBottleneck_4A'").fetchone()[0]
  ctl=con.execute("""select controller_uid from controller_config where scenario_uid=? and compatibility_quality='EXACT_PROFILE' and rng_semantics_version like '%2026092907%'""",(sc,)).fetchone()[0]
  req=[]
  for p in selected:
   if str(p['control_validation_selected']).lower()!='true':continue
   for ps in panel['states']:
    sidrow=con.execute('''select a.state_uid,count(r.rollout_uid) n from state_alias a join state s using(state_uid)
      left join rollout r on r.state_uid=a.state_uid and r.controller_uid=? where a.scenario_uid=? and a.alias=?
      and s.identity_quality in ('CONTENT_EXACT','CONDITIONING_EXACT','SOURCE_GROUP_STABLE') group by a.state_uid
      order by count(r.rollout_uid) desc,case s.identity_quality when 'CONTENT_EXACT' then 0 when 'CONDITIONING_EXACT' then 1 else 2 end''',(ctl,sc,ps['state_id'])).fetchall()
    if not sidrow:raise RuntimeError(('state alias',ps['state_id'],0))
    from shared_rollout_db.src.rollout_db import eta_identity,canonical,uid
    eta=[float(p[f'eta{i}']) for i in (1,2,3)];eid=eta_identity(eta)[0]
    req.append({'request_id':f"f{p['fold']}_{p['family']}_m{p['mode_id']}__{ps['state_id']}",'fold':int(p['fold']),'family':p['family'],'mode_id':int(p['mode_id']),
      'state_id':ps['state_id'],'state_uid':sidrow[0][0],'eta':eta,'eta_uid':eid,'controller_uid':ctl,'seed_keys':[canonical({'future_index':i}) for i in range(16)]})
  dump('planned_rollouts.json',{'task':'ORTHOFLOW3_TOY_DB_TRANSFER_FALSIFICATION_V2','robust_rule':'>=15/16','requests':req})
  ep=str(H.resolve());eu=uid('exp',{'path':ep});con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,metadata_json) VALUES(?,?,?,?,?)',(eu,H.name,ep,sha(__file__),json.dumps({'no_third_scenario':True})));con.commit();con.close();dump('experiment_uid.json',{'experiment_uid':eu})
  dump('working_state.json',{'status':'STAGE_A_PREFLIGHT_REQUIRED','promising_families':sorted(promising_fams),'selected_predictions':sum(str(x['control_validation_selected']).lower()=='true' for x in selected),'requests':len(req)*16})
  print(json.dumps({'promising_families':sorted(promising_fams),'selected_predictions':sum(str(x['control_validation_selected']).lower()=='true' for x in selected),'state_eta_requests':len(req),'continuations_requested':len(req)*16},indent=2))
 elif stage=='missing':
  plan=json.load(open(H/'planned_rollouts.json'));pre=json.load(open(H/'cache_preflight_stageA.json'));tasks=[];reuse=[]
  assert len(plan['requests'])==len(pre['details'])
  for r,d in zip(plan['requests'],pre['details']):
   reuse.append({**{k:r[k] for k in ('request_id','fold','family','mode_id','state_id','state_uid','eta','eta_uid','controller_uid')},'cache_status':d['status'],'reused':d.get('reused',0),'missing':len(d.get('missing_seeds',[]))})
   if d['status']=='AGGREGATE_REUSE':continue
   for sk in d.get('missing_seeds',[]):
    fi=json.loads(sk)['future_index'];tasks.append({'probe_id':f"{r['request_id']}__s{fi:02d}",'phase':'stage_a_b15','state_id':r['state_id'],'controller':f"A_{r['family']}",'family':r['family'],'fold':r['fold'],'mode_id':r['mode_id'],'eta':r['eta'],'future_index':fi})
  write('cache_request_detail.csv',reuse)
  for j in range(6):
   (H/'plans'/'stage_a'/f'shard{j}.jsonl').write_text(''.join(json.dumps(t,separators=(',',':'))+'\n' for i,t in enumerate(tasks) if i%6==j))
  dump('missing_plan_summary.json',{'new_continuations':len(tasks),'shards':6,'aggregate_reuse_requests':sum(x['cache_status']=='AGGREGATE_REUSE' for x in reuse),'exact_reused':sum(x['reused'] for x in reuse)})
  print(json.dumps(json.load(open(H/'missing_plan_summary.json')),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','missing'));a=ap.parse_args();main(a.stage)
