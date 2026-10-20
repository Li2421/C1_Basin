from common import *
G=guard()
def pilot():
 p=read(HERE/'pilot_protocol.json');cs=read(HERE/'pilot_CASES.json');rows={r['rollout_uid']:r for r in read(HERE/'pilot_DB_ROWS.json')};rates={};eligible=[];low=[];high=[]
 for n in p['controller_order']:
  rates[n]={}
  for a in ['safety baseline','S500']:
   rr=[rows[c['rollout_uid']] for c in cs if c['base_controller_name']==n and c['arm']==a];assert len(rr)==16;U=sum(r['numerical_failure'] for r in rr);S=sum(r['success'] and not r['numerical_failure'] for r in rr);rates[n][a]=dict(lower=S/16,upper=(S+U)/16,unknown=U)
  if all(r['upper']<.15 for r in rates[n].values()):low.append(n)
  elif all(r['lower']>.98 for r in rates[n].values()):high.append(n)
  else:eligible.append(n)
 write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),eligible=eligible,excluded_low=low,excluded_high=high,rates=rates,family_status={f:'ELIGIBLE' if sum(n.startswith(f) for n in eligible)>=4 else 'INCONCLUSIVE' for f in ['DM','AS','G']}));seal('CAPABILITY_FROZEN.receipt.json',[HERE/'CAPABILITY_GATE.json',HERE/'pilot_DB_ROWS.json'],before_main_labels=True)
def tune():
 p=read(HERE/'tune_protocol.json');cs=read(HERE/'tune_CASES.json');rows={r['rollout_uid']:r for r in read(HERE/'tune_DB_ROWS.json')};grid=read(HERE/'RHR_GRID.json');allrows=[];selected={};modes=['progress','slow','neighbor_right','always']
 for family in ['DM','AS','G']:
  results=[]
  for i,rule in enumerate(grid):
   rr=[rows[c['rollout_uid']] for c in cs if c['family']==family and c['arm']==f'RHR_grid{i:02}'];S=sum(r['success'] and not r['numerical_failure'] for r in rr);U=sum(r['numerical_failure'] for r in rr);r=dict(family=family,index=i,rule=rule,n=len(rr),success=S,unknown=U,lower=S/len(rr) if rr else None,upper=(S+U)/len(rr) if rr else None);results.append(r);allrows.append(r)
  valid=[r for r in results if r['n']]
  if valid:selected[family]=min(valid,key=lambda r:(-r['lower'],r['rule']['speed'],-r['rule']['steps'],r['rule'].get('distance',float('inf')),modes.index(r['rule']['mode']),r['index']))
 write(HERE/'RHR_GRID_RESULTS.json',allrows);write(HERE/'RHR_SELECTED.json',selected);seal('RHR_FROZEN.receipt.json',[HERE/'RHR_GRID_RESULTS.json',HERE/'RHR_SELECTED.json',HERE/'tune_DB_ROWS.json'],before_main_labels=True)
if __name__=='__main__':globals()[sys.argv[1]]()
