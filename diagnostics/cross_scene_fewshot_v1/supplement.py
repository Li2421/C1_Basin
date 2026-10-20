"""Deterministic MC intervals and presentation; no new simulation."""
from pathlib import Path
import json,math,hashlib,datetime
p=Path(__file__).resolve().parent;q=json.load(open(p/'results.json'));z=1.959963984540054
for cohort in q.values():
 for unit in cohort['units']:
  for methods in unit['stats'].values():
   for row in methods.values():
    for key in ('prob_regret_le3pp','prob_oracle'):
     ph=row[key];n=1000;center=(ph+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n);row[key+'_MC_Wilson95']=[center-half,center+half]
(p/'results.json').write_text(json.dumps(q,indent=2)+'\n');text=(p/'REPORT.md').read_text().split('\n主/次菜单方法')[0];text+='\n主/次菜单方法的统一 k*：\n\n| menu | E | S | SE |\n| --- | --- | --- | --- |\n'
for name,v in q.items():text+=f"| {name} | {v['E_kstar'] or '>16'} | {v['S_kstar'] or '>16'} | {v['SE_kstar'] or '>16'} |\n"
text+='\n主菜单各单元 k=16 的安全方法：\n\n| unit | mean regret pp | p90 pp | P(regret≤3pp) | MC Wilson95 |\n| --- | ---: | ---: | ---: | --- |\n'
for u in q['M']['units']:
 r=u['stats']['16']['S'];text+=f"| {u['unit']} | {100*r['mean_regret']:.2f} | {100*r['p90_regret']:.2f} | {r['prob_regret_le3pp']:.3f} | {[round(x,3) for x in r['prob_regret_le3pp_MC_Wilson95']]} |\n"
text+='\n解释：DB等大效应单元较容易，但统一门还要求对小效应或相近臂的Toy/Ring单元可靠选对；在这些单元，保守后验门仍经常保留D0。F2尤其包含k=1，默认臂在部分单元本身损失很大，故仅“少切换”不能保证低遗憾。k*>16是本轮网格的下界，不能推出某个未测试预算必定成功。SE为次要对照；候选池耗尽时残余预算抽未耗尽的已淘汰候选，最终选择仍限幸存者，这个边界实现细节不影响主S检验。30单元嵌套在10场景/变体中，规定的单元bootstrap不能解释为30个独立拓扑的证据。\n';(p/'REPORT.md').write_text(text)
(p/'FINAL_AUDIT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':hashlib.sha256((p/'analyze.py').read_bytes()).hexdigest(),'supplement_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((p/'results.json').read_bytes()).hexdigest(),'units_each_menu':30,'repeats':1000,'no_new_rollout_or_DB_write':True,'posterior_formula_fixture_verified':True},indent=2)+'\n')
