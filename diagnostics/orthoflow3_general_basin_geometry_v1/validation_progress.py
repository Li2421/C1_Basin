#!/usr/bin/env python3
from audit import *
def main():
 groups=defaultdict(list)
 for p in (HERE/'targeted_probe_rounds').glob('*/gate.json'):
  g=json.load(open(p))
  if 'false_inclusions' not in g:continue
  name=p.parent.name;base=name[3:] if name.startswith('db_') else name;groups[base].append((p,g))
 out=[]
 for name,items in sorted(groups.items(),key=lambda x:min(p.stat().st_mtime for p,g in x[1])):
  states=set();false=total=0;complete=True
  for p,g in items:
   states.update(r['state_id'] for r in read(p.parent/'manifest.csv'));false+=g['false_inclusions'];total+=g['expected'];complete &= g['complete']
  out.append(dict(attempt=name,family=items[0][1]['family'],states=len(states),state_ids=json.dumps(sorted(states)),false_inclusions=false,points=total,false_fraction=false/total,complete=complete,accepted=False))
 write('validation_history.csv',out)
 state=json.load(open(HERE/'working_state.json'))
 if len(out)>=2:
  a,b=out[-2:];same=a['state_ids']==b['state_ids'];gain=a['false_fraction']-b['false_fraction'] if same else None
  state['last_comparable_validation']=dict(previous=a['attempt'],latest=b['attempt'],same_state_cohort=same,false_inclusion_improvement=gain,at_least_five_percentage_points=gain is not None and gain>=.05)
  if gain is not None and gain>=.05:state['saturation_status']='NOT_REACHED: latest same-cohort prospective false-inclusion improvement >=5percentage points. No arbitrary cycle cap.'
 dump('working_state.json',state);print('Prospective validation history updated; no automatic saturation declaration.')
if __name__=='__main__':main()
