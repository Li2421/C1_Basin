from common import *
from design import case
from registry import register
import copy

def design(stage):
 p=read(HERE/f'{stage}_protocol.json');cap=read(HERE/f'{stage}_CAPABILITY.json');p['controller_order']=cap['eligible'];p['states']=read(HERE/'TUNING_STATES.json');grid=[x['rule'] for x in read(HERE/'FIXED_RIGHT_HAND_RECOMPUTATION.json')['table']];rows=[];profiles={}
 for name in p['controller_order']:
  for j,rule in enumerate(grid):
   for s in p['states']:
    c,cp=case(s,name,p['profiles'][name],p,'RHR',rule=rule);c['grid_index']=j;rows.append(c);profiles[cp['controller_uid']]=cp
 register('tune_'+stage,p,rows,profiles)

def select(stage):
 phase='tune_'+stage;p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json');db={c['rollout_uid']:read(HERE/phase/'raw'/f"{c['rollout_uid']}.json") for c in cases};out={}
 for family in dict.fromkeys(p['types'].values()):
  names=[n for n in p['controller_order'] if p['types'][n]==family];table=[]
  if not names:out[family]=dict(status='GAP',reason='No eligible instances');continue
  for j in range(20):
   cs=[c for c in cases if c['base_controller_name'] in names and c['grid_index']==j];assert len(cs)==len(names)*16;rr=[db[c['rollout_uid']] for c in cs];lo=sum(r['success'] for r in rr)/len(rr);hi=lo+sum(r['numerical_failure'] for r in rr)/len(rr);table.append(dict(grid_index=j,rule=cs[0]['rhr_rule'],lower=lo,upper=hi,controllers=names,episodes=len(rr)))
  best=min(table,key=lambda x:(-x['lower'],x['rule']['speed'],x['rule']['steps'],['always','slow','progress'].index(x['rule']['mode'])));out[family]=dict(table=table,selected=best,unknown_selection_ambiguous=any(x['upper']>best['lower'] for x in table if x['grid_index']!=best['grid_index']))
 write(HERE/f'{stage}_FAIR_RULES.json',out);seal(stage+'_FAIR_RULES_FROZEN.receipt.json',[HERE/f'{stage}_FAIR_RULES.json',HERE/f'{phase}_POSTFLIGHT.json'],before_evaluation=True)
 print({k:v.get('selected',{}).get('grid_index') for k,v in out.items()})
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
