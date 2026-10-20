from common import *
from design import profiles,case
from registry import register
prereg();frozen=read(HERE/'ALL_CHOICES_FROZEN.receipt.json')
for path,digest in frozen['files'].items():assert sha(path)==digest,path
choices=read(HERE/'FROZEN_CHOICES.json');families=list(dict.fromkeys(n.rstrip('0123456789') for n in choices['controller_order']));p=profiles(families);p['controller_order']=choices['controller_order'];p['states']=read(HERE/'EVAL_STATES.json');fixed=read(HERE/'FIXED_RIGHT_HAND_RECOMPUTATION.json')['selected']['rule'];rows=[];cps={}
for name in p['controller_order']:
 family=p['types'][name];fair=choices['fair_rules'][family]['selected']['rule']
 for arm in ['nominal_only','safety_baseline','F10','F13','S500','Sprog','RHR','RHR_fair']:
  rule=fixed if arm=='RHR' else fair if arm=='RHR_fair' else None
  for s in p['states']:
   for seed in [0,1]:
    c,cp=case(s,name,p['profiles'][name],p,arm,seed,rule);rows.append(c);cps[cp['controller_uid']]=cp
register('main',p,rows,cps);seal('MAIN_DESIGN_FROZEN.receipt.json',[HERE/'main_protocol.json',HERE/'main_CASES.json',HERE/'ALL_CHOICES_FROZEN.receipt.json',HERE/'episode.py',HERE/'worker.py',HERE/'reuse_inactive.py',HERE/'analysis.py',HERE/'analysis_statistics.py',HERE/'FAST_ASTAR_IMPLEMENTATION.receipt.json',HERE/'fast_astar.py',HERE/'astar_kernel.so'],before_evaluation_submission=True)
