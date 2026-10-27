import json,collections,csv
Z=json.load(open('FINAL_FROZEN.json'));TE=Z['test']
S={s['name']:s for s in json.load(open('scenes_test.json'))}
R=collections.defaultdict(list)
for f in ['test_menu.jsonl','test_props.jsonl','test_nominal.jsonl']:
 for l in open(f):
  d=json.loads(l)
  if 8<=d['state']<=23:R[(d['scene'],d['key'],d['controller'])].append(d['success'])
CT=['centralized sampling','gradient MPC','distributed shooting']
cols=lambda v:[('pipeline',f"p{Z['choices'][v]['ens16']}"),('global fixed eta sE22','menu:sE22'),('fixed eta13','menu:eta13'),('safety only','menu:safety'),('nominal only','nominal')]
rows=[]
for v in TE:
 for ci in list(range(3))+['all']:
  r=dict(scene=v,controller=CT[ci] if ci!='all' else 'all 3')
  for nm,k in cols(v):
   x=sum((R[(v,k,c)] for c in (range(3) if ci=='all' else [ci])),[]);assert len(x)==(48 if ci=='all' else 16),(v,k,ci,len(x));r[nm]=(sum(x),len(x))
  rows.append(r)
with open('/home/zhihan/research/Basin_C1/diagnostics/hetero_set_critic_selfcheck_v1/TEST_PER_SCENE_BY_CONTROLLER.csv','w') as f:
 w=csv.writer(f);names=[n for n,_ in cols(TE[0])];w.writerow(['scene','controller']+[f'{n} successes' for n in names]+['episodes'])
 for r in rows:w.writerow([r['scene'],r['controller']]+[r[n][0] for n in names]+[r[names[0]][1]])
for r in rows:print(r['scene'],r['controller'],*[f"{r[n][0]}/{r[n][1]}" for n,_ in cols(r['scene'])],sep=' | ')
for v in TE:print(v,S[v]['config']['num_agents'],S[v].get('params'),Z['choices'][v]['proposals'][Z['choices'][v]['ens16']])
