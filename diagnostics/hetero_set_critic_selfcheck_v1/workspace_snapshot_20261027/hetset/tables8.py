import json,collections,csv
Z=json.load(open('FINAL_FROZEN.json'));TE=Z['test']
R=collections.defaultdict(list)
for f in ['test_menu.jsonl','test_props.jsonl','test_nominal.jsonl','test8.jsonl']:
 for l in open(f):
  d=json.loads(l)
  if 8<=d['state']<=23:R[(d['scene'],d['key'],d['controller'])].append(d['success'])
CT=['集中式随机采样','梯度下降 MPC','分布式 MPC（打靶）','集中式 MPPI','交叉熵','分布式交叉熵','冲突搜索','优先级 A*']
NAMES=['流水线','全局固定 η','固定 η13','只加安全投影','不加安全投影']
keys=lambda v:[f"p{Z['choices'][v]['ens16']}",'menu:sE22','menu:eta13','menu:safety','nominal']
CSV='/home/zhihan/research/Basin_C1/diagnostics/hetero_set_critic_selfcheck_v1/TEST_PER_SCENE_8_CONTROLLERS.csv'
with open(CSV,'w') as f:
 w=csv.writer(f);w.writerow(['scene','controller']+[n+' 成功局数' for n in NAMES]+['局数'])
 for v in TE:
  print(f'\n#### {v}\n| controller | '+' | '.join(NAMES)+' |\n|---|'+'---:|'*5)
  for ci in list(range(8))+['all']:
   cs=range(8) if ci=='all' else [ci];row=[]
   for k in keys(v):
    x=sum((R[(v,k,c)] for c in cs),[]);assert len(x)==16*len(cs),(v,k,ci,len(x));row.append(sum(x))
   n=16*len(cs);nm='**8 个合计**' if ci=='all' else CT[ci]
   w.writerow([v,nm.strip('*')]+row+[n]);print('|',nm,'|',' | '.join(f'{100*s/n:.1f}' for s in row),'|')
 tot=[sum(sum(R[(v,k,c)]) for v in TE for c in range(8)) for k in range(5) for _ in [0]] if False else None
for ci in range(8):
 print(CT[ci],*[f"{100*sum(sum(R[(v,keys(v)[j],ci)]) for v in TE)/(16*12):.1f}" for j in range(5)],sep=' | ')
print('ALL',*[f"{100*sum(sum(R[(v,keys(v)[j],c)]) for v in TE for c in range(8))/(16*12*8):.1f}" for j in range(5)],sep=' | ')
