from common import *
from capability_stats import summarize,bounds
from pipeline import lookup
import csv

def arrays(rows,stage,baseline=False):
 names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];lo=np.zeros((14,16,2,1));hi=lo.copy();filled=np.zeros_like(lo,bool);events={k:lo.copy() for k in ['collision','timeout','unknown']};refs=[]
 for row in rows:
  if baseline:
   c=row['case'];f=Path(row['raw_path']);assert sha(f)==row['raw_sha256'];i=read(HERE/'HISTORICAL_INVENTORY.json')['original_eval_states'].index(c['state_uid'])
  else:c=row;f=lookup(c,stage);i=c['state_index']
  q=read(f);ix=(names.index(c['base_controller_name']),i,['F10','S500'].index(c['arm']),0);assert not filled[ix];filled[ix]=True;lo[ix]=q['success'];hi[ix]=q['success']+q['numerical_failure'];events['collision'][ix]=q['collision'];events['timeout'][ix]=q['timeout'];events['unknown'][ix]=q['numerical_failure'];refs.append(dict(rollout_uid=c['rollout_uid'],raw_path=str(f),raw_sha256=sha(f)))
 return lo,hi,events,filled,refs

def main():
 assert (HERE/'A_NUMERICAL_COMPLETE.receipt.json').exists();names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];aa=arrays(read(HERE/'A1_CASES.json'),'A');assert aa[3].all();r1=summarize(aa[0],aa[1],202610163700);l,h=r1['median_max_success_pct'];status='GENERATOR_VALID' if l>=20 else 'GENERATOR_INVALID' if h<20 else 'INCONCLUSIVE';r1.update(status=status,round35_geometry_gate_preserved=status=='GENERATOR_VALID',round35_geometry_gate_invalidated=status=='GENERATOR_INVALID',grade='CONFIRMATORY positive-control endpoint')
 base=arrays(read(HERE/'A2_BASELINE_REFERENCES.json'),'reference',True);assert base[3].all();base_result=summarize(base[0],base[1],202610163701,(base[0],base[1]));base_result.update(status='HISTORICAL_PAIRED_BASELINE',grade='DEVELOPMENT exact R10 reanalysis');out=dict(GA1=r1,GA2={'0.9':base_result},GA3=dict(status='NOT_TRIGGERED',eligible_radii=[]),baseline_references=base[4]);curves=[];table=[];allrefs={'A1':aa[4],'baseline':base[4]}
 for label,data,res in [('A1_r0.90',aa,r1),('A2_r0.90',base,base_result)]:
  for j,n in enumerate(names):curves.append(dict(subaxis=label,r=.9,controller=n,F10_bounds_pct=res['arms_success_bounds_pct'][j][0],S500_bounds_pct=res['arms_success_bounds_pct'][j][1],max_bounds_pct=res['controller_max_success_bounds_pct'][j]))
  table.append(dict(subaxis=label,**res))
 for u,r in enumerate([.8,.85,.95,1.]):
  rows=[c for c in read(HERE/'A2_CASES.json') if c['r']==r];z=arrays(rows,'A');allrefs[str(r)]=z[4]
  if not z[3].all():res=dict(status='GAP_INVALID_RESET',missing=int((~z[3]).sum()))
  else:
   res=summarize(z[0],z[1],202610163702+u,(base[0],base[1]));res.update(status='DESCRIPTIVE',grade='CONFIRMATORY collected labels; descriptive posthoc max summaries');np.savez_compressed(HERE/f'A2_r{r}_LABELS.npz',lower=z[0],upper=z[1],controller_names=names,arms=['F10','S500'],**z[2])
   for j,n in enumerate(names):curves.append(dict(subaxis=f'A2_r{r:.2f}',r=r,controller=n,F10_bounds_pct=res['arms_success_bounds_pct'][j][0],S500_bounds_pct=res['arms_success_bounds_pct'][j][1],max_bounds_pct=res['controller_max_success_bounds_pct'][j]))
   if status=='GENERATOR_INVALID' and res['median_max_success_pct'][0]>=20:out['GA3']['eligible_radii'].append(r)
  out['GA2'][str(r)]=res;table.append(dict(subaxis=f'A2_r{r:.2f}',**res))
 np.savez_compressed(HERE/'A1_LABELS.npz',lower=aa[0],upper=aa[1],controller_names=names,arms=['F10','S500'],**aa[2]);np.savez_compressed(HERE/'A2_r0.9_LABELS.npz',lower=base[0],upper=base[1],controller_names=names,arms=['F10','S500'],**base[2]);out['GA3']['status']='REQUIRED_DEVELOPMENT' if out['GA3']['eligible_radii'] else 'NOT_TRIGGERED';write(HERE/'CONTROL_RESULTS.json',out);write(HERE/'A_CAPABILITY_TABLE.json',table);write(HERE/'A_RAW_PROVENANCE.json',allrefs);write(HERE/'CONTROLLER_RADIUS_CURVES.json',curves)
 with (HERE/'CONTROLLER_RADIUS_CURVES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in curves)
 text='| 子项 | 中位数 max(F10,S500) 成功率识别界% | 95%CI% | 与同state r.90配对差pp | 配对差95%CI | 判定 |\n|---|---|---|---|---|---|\n'
 for v in table:text+=f"| {v['subaxis']} | {v.get('median_max_success_pct','GAP')} | {v.get('median_max_95CI_pct','GAP')} | {v.get('paired_difference_of_medians_pp','N/A')} | {v.get('paired_difference_95CI_pp','N/A')} | {v['status']} |\n"
 (HERE/'A_CAPABILITY_TABLE.md').write_text(text+'\nGA1按识别界点估计20%阈值判定，不用CI代替阈值。GA2为描述性最大值；配对差主项是两个中位数之差，controller内差的中位数另在JSON报告。unknown保留上下界，旧r.90标签直接引用。\n');seal('GEOMETRY_ELIGIBILITY_FROZEN.receipt.json',[HERE/'CONTROL_RESULTS.json',HERE/'A_CAPABILITY_TABLE.json',HERE/'CONTROLLER_RADIUS_CURVES.json'],before_GA3_submission=True);print('GA1',status,r1['median_max_success_pct'],'GA3eligible',out['GA3']['eligible_radii'],flush=True)
if __name__=='__main__':main()
