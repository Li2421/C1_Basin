"""Normalize final logical queries without altering raw records or trajectories."""
from core import *

def rebuild():
 combinations={c['code']:c for c in read(HERE/'COMBINATIONS.json')['combinations']};family=next(c for c in read(HERE/'inputs/previous/FAMILIES_FOLLOWUP.json')['families']if c['scheme']=='B')
 from corrections_analysis import legacy_spec
 combinations['legacy_b']=dict(candidate_pool=[legacy_spec(sp)for sp in family['candidate_pool']])
 original=HERE/'candidates.jsonl'
 if original.exists()and not(HERE/'execution_candidates.jsonl').exists():original.replace(HERE/'execution_candidates.jsonl')
 tmp=HERE/'candidates.jsonl.tmp';count=0
 with tmp.open('w')as output:
  for path in sorted(HERE.glob('SEARCH_*.json')):
   state=read(path);code=state['code'];scope=state['scope']
   for g in state['groups']:
    for q in g['queries']:
     row=dict(record_type='logical_oracle_query',combination=code,scope=scope,identity=g['identity'],index=q['index'],query_uid=q['query_uid'],source_rollout_uid=q['result']['rollout_uid'],spec=combinations[code]['candidate_pool'][q['index']-1],result=q['result'],official_row_is_source_only=True)
     output.write(json.dumps(clean(row),ensure_ascii=False)+'\n');count+=1
  for path in sorted(HERE.glob('FIXED_*.json')):
   state=read(path)
   for r in state['records']:
    output.write(json.dumps(clean(dict(record_type='logical_fixed_query',combination=state['code'],scope=state['scope'],plus=state['plus'],**r)),ensure_ascii=False)+'\n');count+=1
 tmp.replace(original)
 # A completely reused batch still has a concrete execution record; no
 # synthetic official result is inserted for a semantic alias.
 for path in HERE.glob('*_all_requests.json'):
  stage=path.name[:-len('_all_requests.json')];execution=HERE/f'{stage}_execution.json'
  if execution.exists():continue
  data=read(path);dump(execution,dict(complete=True,new=0,logical_requests=len(data['exact_configs']),reuse_records=data.get('records',[]),reason='All logical results resolved through exact preflight or certified semantic reuse.'))
 dump(HERE/'CANDIDATE_STREAM.json',dict(records=count,path=str(original),raw_execution_stream=str(HERE/'execution_candidates.jsonl'),identity='combination/scope/physicalcase/index; source and logical query UIDs distinct'))

if __name__=='__main__':rebuild()
