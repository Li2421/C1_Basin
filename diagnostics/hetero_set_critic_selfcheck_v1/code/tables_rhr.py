import json,collections,csv
Z=json.load(open('FINAL_FROZEN.json'));TE=Z['test']
R=collections.defaultdict(list)
for f in ['test_menu.jsonl','test_props.jsonl','test_nominal.jsonl','test8.jsonl','test_rhr.jsonl']:
 for d in map(json.loads,open(f)):
  if 8<=d['state']<=23:R[(d['scene'],d['key'],d['controller'])].append(d['success'])
CT=['集中式随机采样','梯度下降 MPC','分布式 MPC（打靶）','集中式 MPPI','交叉熵','分布式交叉熵','冲突搜索','优先级 A*']
USE=[0,1,2,4,5,6,7];NAMES=['流水线','最好固定 η','安全投影＋右手规则','只加安全投影','不加安全投影']
keys=lambda v:[f"p{Z['choices'][v]['ens16']}",'menu:sE22','rhr','menu:safety','nominal']
CSV='/home/zhihan/research/Basin_C1/diagnostics/hetero_set_critic_selfcheck_v1/TEST_PER_SCENE_7_CONTROLLERS_WITH_RIGHT_HAND.csv'
w=csv.writer(open(CSV,'w'));w.writerow(['scene','controller','passes capability gate']+[n+' 成功局数' for n in NAMES]+['局数'])
pct=lambda s,n:f'{100*s/n:.1f}'
def cnt(vs,k,cs):
 x=[s for v in vs for c in cs for s in R[(v,k(v) if callable(k) else k,c)]];assert len(x)==16*len(vs)*len(cs);return sum(x),len(x)
hdr='| controller | '+' | '.join(NAMES)+' |\n|---|'+'---:|'*5
print('### 12 个场景合计\n'+hdr)
for ci in USE+['all',3]:
 cs=USE if ci=='all' else [ci];nm='**7 个合计**' if ci=='all' else ('集中式 MPPI（未通过能力检查，不计入合计）' if ci==3 else CT[ci])
 print('|',nm,'|',' | '.join(pct(*cnt(TE,(lambda v,j=j:keys(v)[j]),cs)) for j in range(5)),'|')
print('\n### 每个场景的 7 个合计\n'+hdr.replace('controller','场景'))
for v in TE:print('|',v,'|',' | '.join(pct(*cnt([v],(lambda v,j=j:keys(v)[j]),USE)) for j in range(5)),'|')
for v in TE:
 print(f'\n**{v}**\n'+hdr)
 for ci in USE+['all']:
  cs=USE if ci=='all' else [ci];print('|','**7 个合计**' if ci=='all' else CT[ci],'|',' | '.join(pct(*cnt([v],(lambda v,j=j:keys(v)[j]),cs)) for j in range(5)),'|')
 for ci in range(8):w.writerow([v,CT[ci],ci!=3]+[cnt([v],(lambda v,j=j:keys(v)[j]),[ci])[0] for j in range(5)]+[16])
