"""Read-only structural catalogue of all >=16-valid-seed cells and shared-eta panels."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,json,hashlib,datetime
HERE=Path(__file__).resolve().parent
DB=HERE.parents[1]/'shared_rollout_db/rollout.sqlite'
def write(name,x):(HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def read(s):return json.loads(s) if s else {}
def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def main():
 c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True);c.row_factory=sqlite3.Row
 # Group on the indexed state/eta/controller order; no success labels opened here.
 by=defaultdict(lambda:defaultdict(set));experiments=defaultdict(set);cells=0
 for r in c.execute('''SELECT state_uid,eta_uid,controller_uid,COUNT(*) n,
     GROUP_CONCAT(DISTINCT experiment_uid) experiments
     FROM rollout WHERE conflict_quarantined=0 AND numerical_failure=0
     GROUP BY state_uid,eta_uid,controller_uid HAVING n>=16'''):
  ctl,s,e=r['controller_uid'],r['state_uid'],r['eta_uid'];by[ctl][s].add(e);experiments[ctl].update(r['experiments'].split(','));cells+=1
 ctl={r['controller_uid']:dict(r) for r in c.execute('SELECT * FROM controller_config')}
 state={r['state_uid']:dict(r) for r in c.execute('SELECT * FROM state')}
 sc={r['scenario_uid']:r['name'] for r in c.execute('SELECT scenario_uid,name FROM scenario')}
 ex={r['experiment_uid']:{'name':r['name'],'path':r['path'],'protocol_hash':r['protocol_hash'],'code_hash':r['code_hash']} for r in c.execute('SELECT * FROM experiment')}
 panel_rows=[];ctl_rows=[];reasons=Counter()
 for cid,states in by.items():
  meta=ctl[cid];sets={s:frozenset(e) for s,e in states.items()};eta_support=defaultdict(set)
  for s,es in sets.items():
   for e in es:eta_support[e].add(s)
  source_sets=set(sets.values());common=frozenset.intersection(*sets.values()) if sets else frozenset()
  if len(common)>=2:source_sets.add(common)
  panels={}
  for es in source_sets:
   if len(es)<2:continue
   support=set.intersection(*(eta_support[e] for e in es))
   if len(support)<4:continue
   # Closure adds every eta present in the entire supported state cohort.
   closed=frozenset.intersection(*(sets[s] for s in support));assert es<=closed
   signature=(frozenset(support),closed);panels[signature]=True
  ctl_bad=[]
  if meta['compatibility_quality']!='EXACT_PROFILE':ctl_bad.append('controller_not_exact_profile')
  cond=(meta['conditioning_version'] or '').lower()
  if not any(t in cond for t in ['true_t0','true-t0','physical_t0','frozen_test_initial_state']):ctl_bad.append('not_true_t0_controller')
  names=Counter(sc.get(state[s]['scenario_uid'],'UNKNOWN') for s in states)
  qualities=Counter(state[s]['identity_quality'] for s in states)
  ctl_rows.append({'controller_uid':cid,'scene_names':dict(names),'Q16_cells':sum(len(v) for v in states.values()),'Q16_states':len(states),'eta_union':len(eta_support),'n_structural_panels':len(panels),'quality':meta['compatibility_quality'],'conditioning':meta['conditioning_version'],'horizon':meta['horizon'],'dt':meta['dt'],'success_semantics':meta['success_semantics_version'],'flow_checkpoint_sha256':meta['flow_checkpoint_sha256'],'safety_config_hash':meta['safety_config_hash'],'state_identity_qualities':dict(qualities),'experiment_uids':sorted(experiments[cid]),'experiment_sources':[{'uid':x,**ex[x]} for x in sorted(experiments[cid]) if x in ex],'controller_reasons':ctl_bad})
  for (ss,es) in panels:
   if len(ss)<12:continue
   bad=list(ctl_bad);idq=Counter();source=Counter();n_no_phys=0;n_nonzero_step=0;scenes=Counter()
   for s in ss:
    z=state[s];idq[z['identity_quality']]+=1;source[z['source_group'] or '']+=1;scenes[z['scenario_uid']]+=1
    if z['identity_quality'] in ['ALIAS_ONLY']:bad.append('alias_only_state')
    if not z['physical_state_json'] and not z['h0_json']:n_no_phys+=1
    try:
     ph=read(z['physical_state_json']);step=ph.get('timestep',ph.get('step',0))
     if isinstance(step,(int,float)) and step!=0:n_nonzero_step+=1
    except Exception:bad.append('unparseable_physical_state')
   if len(scenes)!=1 or next(iter(scenes))!=meta['scenario_uid']:bad.append('scenario_mismatch')
   if n_nonzero_step:bad.append('nonzero_physical_step')
   bad=sorted(set(bad))
   for x in bad:reasons[x]+=1
   sid=sorted(ss);eid=sorted(es);pid='panel_'+sha(json.dumps([cid,sid,eid],separators=(',',':')))[:20]
   panel_rows.append({'panel_uid':pid,'controller_uid':cid,'scene':sc.get(meta['scenario_uid'],'UNKNOWN'),'state_uids':sid,'eta_uids':eid,'n_states':len(sid),'n_eta':len(eid),'state_identity_qualities':dict(idq),'n_states_without_physical_or_h0_json':n_no_phys,'state_source_groups':dict(source),'eligibility_reasons':bad,'eligible_metadata':not bad,'controller_experiment_uids':sorted(experiments[cid]),'source':'maximal complete shared-eta cohort from Q16 coverage only; no success label used'})
 panel_rows.sort(key=lambda p:(not p['eligible_metadata'],-p['n_states']*p['n_eta'],p['controller_uid'],p['panel_uid']))
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'database_path':str(DB),'database_bytes':DB.stat().st_size,'valid_rollouts':c.execute('SELECT COUNT(*) FROM rollout WHERE conflict_quarantined=0 AND numerical_failure=0').fetchone()[0],'total_rollouts':c.execute('SELECT COUNT(*) FROM rollout').fetchone()[0],'Q16_cells':cells,'Q16_controllers':len(by),'structural_panels_ge12_state_Kge2':len(panel_rows),'metadata_eligible_panels':sum(p['eligible_metadata'] for p in panel_rows),'exclusion_reason_counts':dict(reasons),'controllers':ctl_rows,'panels':panel_rows,'eligibility_note':'Outcomes not used for panel discovery. Seed alignment and raw provenance must still be verified; eligibility is provisional.'}
 write('inventory.json',out)
 print({'cells':cells,'controllers':len(by),'panels':len(panel_rows),'metadata_eligible':out['metadata_eligible_panels'],'scenes':dict(Counter(p['scene'] for p in panel_rows))},flush=True)
 c.close()
if __name__=='__main__':main()
