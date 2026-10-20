"""Registered discrete <=2pp support: explicitly preserve gaps."""
from pathlib import Path
import json,csv
HERE=Path(__file__).resolve().parent

def components(points):
 out=[]
 for x in points:
  x=int(x)
  if out and x-out[-1][-1]==100:out[-1].append(x)
  else:out.append([x])
 return out
assert components([200,400])==[[200],[400]] and components([300,400,500])==[[300,400,500]]
rows=[]
for filename,stats in [('STATISTICAL_RESULTS.json',json.load(open(HERE/'STATISTICAL_RESULTS.json'))),('results.json',json.load(open(HERE/'results.json')))]:
 target=stats['statistics'] if filename=='results.json' else stats
 for tag,r in target.items():
  if not isinstance(r,dict) or not r.get('E'):continue
  e=r['E'];cc=[components(x) for x in e['platform_steps']];e['contiguous_components']=cc;e['widest_contiguous_span']=[max(g[-1]-g[0] for g in groups) for groups in cc]
  if filename=='results.json':
   names=json.load(open(HERE/'DATA_INVENTORY.json')).get(tag,{}).get('names')
   if names is None:names=[n for n in json.load(open(HERE/'DATA_INVENTORY.json'))['R17']['names'] if n.endswith('__arch512')==(tag=='R17N2')]
   for i,n in enumerate(names):rows.append(dict(dataset=tag,controller=n,components=json.dumps(cc[i]),widest_contiguous_span=e['widest_contiguous_span'][i],full_support_span=e['platform_span'][i]))
 (HERE/filename).write_text(json.dumps(stats,indent=2,allow_nan=(filename!='results.json'))+'\n')
with (HERE/'ANALYSIS_E_PLATFORM_COMPONENTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
