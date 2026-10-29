"""Matched old composite-admission reference, and eight prior bay regressions."""
from core import *
from screening import search,selected_entries
from evidence import register
from validation import compare

def legacy_spec(sp):
 c=sp['coordination']['c'];return dict(eta=sp['eta'],coefficients=sp['coefficients'],one_way=dict(R_e=.8,R_c=.5,rho='fcfs',K=None,c1=c,c2=c,c3=c),legacy_admission=True)

def run(make):
 if(HERE/'CORRECTIONS_ANALYSIS.json').exists():return
 from dispatch import execute
 from analysis_results import metrics,write_report
 family=next(f for f in read(HERE/'inputs/previous/FAMILIES_FOLLOWUP.json')['families']if f['scheme']=='B');old=read(HERE/'inputs/previous/FOLLOWUP_B_hard.json')['groups'];pool=[legacy_spec(sp)for sp in family['candidate_pool']]
 combo=dict(code='legacy_b',name='上一轮合成一列冻结定义',candidate_pool=pool,no_passage_preserves_four=False)
 scenes={(e['type_id'],e['level']):e['scene']for e in selected_entries()};ctls={c['type_code']:c for c in controllers()}
 certificate=HERE/'LEGACY_COMPOSITE_EQUIVALENCE.json'
 if not certificate.exists():
  buckets={}
  for g in old:
   for q in g['queries']:
    key=(g['identity'][0],g['identity'][2],q['result']['outcome']);buckets.setdefault(key,[]).append((g,q))
  chosen=[]
  for index in range(20):
   for key in sorted(buckets):
    if len(buckets[key])>index:chosen.append(buckets[key][index])
    if len(chosen)==20:break
   if len(chosen)==20:break
  cases=[]
  for g,q in chosen:
   t,l,ctl,si=g['identity'];cases.append(make(scenes[(t,l)],ctls[ctl],si,pool[q['index']-1],purpose='old_composite_certificate'))
  rows={r['rollout_uid']:r for r in execute(cases,'legacy_composite20')};checks=[compare(q['result'],rows[c['rollout_uid']],new=True)for (g,q),c in zip(chosen,cases)if c['rollout_uid']in rows];passed=len(checks)==20 and all(x['equal']for x in checks)
  dump(certificate,dict(passed=passed,pairs=len(checks),checks=checks,source_sha256=sha(HERE/'admission_legacy.py')))
 state_path=HERE/'SEARCH_legacy_b_screen.json'
 if not state_path.exists():
  oldmap={tuple(g['identity']):g for g in old};groups=[];links=[]
  for e in selected_entries():
   if e['type_id']not in(11,16):continue
   for si in range(8):
    for ctl in controllers():
     identity=[e['type_id'],e['level'],ctl['type_code'],si]
     if read(certificate)['passed']and tuple(identity)in oldmap:
      original=oldmap[tuple(identity)];g=json.loads(json.dumps(original));g.pop('hard',None)
      for q in g['queries']:
       target=make(e['scene'],ctl,si,pool[q['index']-1],purpose='legacy_comparison');links.append(dict(target_query_uid=target['rollout_uid'],source_rollout_uid=q['result']['rollout_uid'],kind='legacy_composite_certified_alias',certificate=certificate.name));q['query_uid']=target['rollout_uid']
      if g['success']:g['solution']=g['queries'][g['first_success']-1]
     else:g=dict(identity=identity,success=False,searched=0,first_success=None,queries=[],gap=None)
     g['scene']=e['scene']['name'];groups.append(g)
  # Source identities are looked up and preflighted exactly; archived aliases
  # stay distinct from new numerical execution identities.
  requests=[]
  with db.connect(True)as con:
   for uid in sorted({x['source_rollout_uid']for x in links}):
    row=con.execute('SELECT state_uid,eta_uid,controller_uid,seed_key FROM rollout WHERE rollout_uid=?',(uid,)).fetchone();assert row;requests.append({k:row[k]for k in('state_uid','eta_uid','controller_uid','seed_key')})
  path=HERE/'legacy_existing_planned.json';dump(path,dict(requests=requests,aliases=links));dump(HERE/'legacy_existing_preflight.json',planner.preflight(path))
  m=read(HERE/'REUSE_MANIFEST.json');m['records'].update({x['target_query_uid']:x for x in links});dump(HERE/'REUSE_MANIFEST.json',m)
  dump(state_path,dict(code='legacy_b',scope='screen',groups=groups,pending=None,completed_stages=[],complete64=False))
 # search() resumes the32 explicitly initialized groups; it does not add other
 # types merely because the standard screening scene list has17 entries.
 state=search(combo,64,make)
 ref={tuple(g['identity']):g for g in metrics('reference')['groups']};oldmap={tuple(g['identity']):g for g in state['groups']};by_type=[]
 for t in(11,16):
  keys=[k for k in oldmap if k[0]==t]
  by_type.append(dict(type_id=t,title=next(sc['title']for k,sc in scenes.items()if k[0]==t),cases=len(keys),old={str(k):sum(oldmap[key]['success']and oldmap[key]['first_success']<=k for key in keys)for k in(16,64)},new={str(k):sum(ref[key]['success']and ref[key]['first_success']<=k for key in keys)for k in(16,64)}))
 prior=read(HERE/'inputs/previous/REGRESSIONS.json')['cases'];harms=[]
 for r in prior:
  key=tuple(r['identity'])
  if key[0]!=11 or r['old_first_success']>16:continue
  g=ref[key];h=oldmap[key];harms.append(dict(identity=list(key),v3_first_success=r['old_first_success'],previous_three_term_success16=False,old_composite_success16=bool(h['success']and h['first_success']<=16),new_success16=bool(g['success']and g['first_success']<=16),new_success64=g['success'],new_first_success=g['first_success']))
 assert len(harms)==8
 dump(HERE/'CORRECTIONS_ANALYSIS.json',dict(complete=state['complete64'],legacy_equivalence=read(certificate)['passed'],by_type=by_type,eight_bay_harms=harms,causal_scope='New versus old composite uses matched original four coefficients and q, differing only the newly specified admission definition. The eight prior harms came from the three-term menu, a distinct historical comparator.'))
 write_report()
