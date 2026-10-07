from core import *
import subprocess
def main():
 state=json.load(open(H/'working_state.json'));ledger=read(H/'experiment_ledger.csv');costs={p.parent.name:json.load(open(p)) for p in H.glob('rounds/*/cost_estimate.json')}
 for r in ledger:
  name=r['experiment'];actual=next((batch for batch,cost in costs.items() if name in cost.get('source_stages',[])),name);done=list((H/f'raw/{actual}').glob('shard*_runtime.json'));expected=sum(1 for p in (H/f'plans/{actual}').glob('shard*.jsonl') if p.stat().st_size>0)
  if len(done)==expected and expected:
   r['status']='COMPLETE';r['result']=f'{r["planned_continuations"]} stage continuations; physical batch={actual}; complete shard outputs'
  elif name=='refine1':r['status']='RUNNING';r['job_id']='996';r['result']='Submission993 failed before execution (missing array);996 correctly submitted as array, no duplicate continuations'
  elif actual=='refine2_escape2':r['status']='RUNNING';r['job_id']='998';r['result']='Packed two frozen scientific stages into one worker batch; no coordinate or seed changes'
  elif actual=='refine3_escape3':r['status']='RUNNING';r['job_id']='1000';r['result']='Packed frozen final boundary bisections and remaining alternative escape routes'
  elif name=='normal_escape4':r['status']='RUNNING';r['job_id']='1002';r['result']='One diagnostic normal-direction exit; full design review recorded before approaching150 new eta'
 write('experiment_ledger.csv',ledger)
 state.update(completed=[r['experiment'] for r in ledger if r['status']=='COMPLETE'],next_action='Monitor1002 normal escape; aggregate/analyze/connectivity audit; final topology or residual ambiguity; independent checks and handoff.',jobs=dict(state.get('jobs',{}),refine1=996,refine2_escape2=998,refine3_escape3=1000,normal_escape4=1002),supplemental_exact_cache=3)
 dump('working_state.json',state)
 index=json.load(open(H/'evidence_index.json'))
 for name in ('supplemental_cache_audit.json','source_completeness_audit.json','slice_analysis_summary.json','connectivity_audit_summary.json','escape2_preview.json','escape3_preview.json','remaining_region_design_diagnostic.json'):
  p=H/name
  if p.exists():index[name]=dict(path=str(p),sha256=sha(p))
 for name in ('protocol.md','core.py','aggregate.py','analyze_slices.py','connectivity_audit.py','design_escape2.py','refine_lines.py','pack_batch.py','finish_audit.py','figures.py','run_shard.py','rollout.sbatch','verify_completion.py','analyze_corridor_lines.py'):
  p=H/name;index['inspected:'+name]=dict(path=str(p),sha256=sha(p),purpose='Current audit execution or independent verification; inspected once, reuse checkpoint')
 dump('evidence_index.json',index)
 conn=json.load(open(H/'connectivity_audit_summary.json'));sl=json.load(open(H/'slice_analysis_summary.json'))
 dump('hypothesis_status.json',[
  dict(hypothesis_id='H1_BOUNDARY_INTRUSION',mathematical_form='sampled internal non-B63 path to E_bridge boundary',current_status='PARTIAL_SUPPORT',key_support={'representatives':conn['representatives_with_sampled_exterior_paths'],'old_interpolation_failures':conn['old_interpolation_failures_with_paths']},key_failure='Some proposed routes interrupted by exact B63',unresolved_property='Remaining representative and historical failure escape connectivity',next_discriminating_test='Frozen escape3 routes'),
  dict(hypothesis_id='H2_ENCLOSED_HOLE',mathematical_form='bounded failure pocket with surrounding success',current_status='NOT_CONFIRMED',key_support=None,key_failure='No six-direction success enclosure; existing sampled external paths',unresolved_property='Unlinked failure pocket escape',next_discriminating_test='Frozen escape3 routes'),
  dict(hypothesis_id='H3_INTERLEAVING',mathematical_form='repeated exact robust alternation on independent local lines not explained by notch',current_status='NOT_INDEPENDENTLY_DEMONSTRATED',key_support='One historical long SFSFS segment only',key_failure=sl['transition_histogram'],unresolved_property='Fine-scale additional transitions',next_discriminating_test='Frozen refine3 exact midpoint checks')])
if __name__=='__main__':main()
