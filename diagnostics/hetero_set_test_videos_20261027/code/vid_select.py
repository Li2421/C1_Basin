import json,collections
Z=json.load(open('FINAL_FROZEN.json'));TE=Z['test']
R={}
for f in ['test_menu.jsonl','test_props.jsonl','test8.jsonl']:
 for d in map(json.loads,open(f)):
  if 8<=d['state']<=23:R[(d['scene'],d['key'],d['controller'],d['state'])]=(d['success'],d['outcome'])
CID=['centralized_sampling','gradient_mpc','distributed_shooting','centralized_mppi','cross_entropy','distributed_cross_entropy','conflict_based_search','prioritized_astar']
plan=[];gaps=[]
for v in TE:
 pk=f"p{Z['choices'][v]['ens16']}"
 for ci in range(8):
  pick=None
  for corr,label in [(pk,'pipeline'),('menu:sE22','best_fixed_eta'),('menu:eta13','eta13')]:
   for si in range(8,24):
    s=R[(v,'menu:safety',ci,si)];c=R[(v,corr,ci,si)]
    if s[0]==0 and c[0]==1:pick=(si,corr,label,s[1]);break
   if pick:break
  if pick:
   si,corr,label,sout=pick
   eta=Z['choices'][v]['proposals'][Z['choices'][v]['ens16']] if corr.startswith('p') else None
   plan.append(dict(scene=v,controller=ci,controller_type=CID[ci],state=si,role='safety_failure',key='menu:safety',eta=None,expected_outcome=sout))
   plan.append(dict(scene=v,controller=ci,controller_type=CID[ci],state=si,role='correction_success',correction=label,key=corr,eta=eta,expected_outcome='success'))
  else:
   sf=[si for si in range(8,24) if R[(v,'menu:safety',ci,si)][0]==0]
   gaps.append(dict(scene=v,controller_type=CID[ci],safety_failures=len(sf),reason='no state with safety failure and a correction success' ))
json.dump(dict(plan=plan,gaps=gaps),open('VIDEO_PLAN.json','w'),indent=1)
print(len(plan),'videos');print(collections.Counter(p['correction'] for p in plan if p['role']=='correction_success'))
for g in gaps:print(g)
