#!/usr/bin/env python3
import json,itertools,time,shutil
from pathlib import Path
from collections import defaultdict
from revise_oracle import select
from shared_families import fit
from cross_validate import read,write,dump,make_cloud,metrics,state_score,gate_summary,hpkey
HERE=Path(__file__).resolve().parent
def main():
 panel={r['state_id'] for r in read(HERE/'scenario_state_inventory.csv')};inv=[r for r in read(HERE/'exact_q64_inventory.csv') if r['state_id'] in panel and r['in_E_bridge']=='True'];by=defaultdict(list)
 for r in inv:by[r['state_id']].append(r)
 keys={s:[r['eta_key'] for r in select(rr,48)] for s,rr in by.items()};folds=json.load(open(HERE/'cv_folds.json'))['folds'];cloud=make_cloud();hps=[dict(p=2,gamma=g,max_cuts=k,safety=.9) for g,k in itertools.product((.5,.6,.7,.8,.9),(2,3,4))];allm={}
 for hp in hps:
  for sid,rr in by.items():
   lookup={r['eta_key']:r for r in rr};m=fit('domain_minus_boundary_caps',[lookup[k] for k in keys[sid]],**hp);allm[sid,hpkey(hp)]=metrics(sid,rr[0]['scenario'],m,rr,set(keys[sid]),cloud,48,hp)
 out=[];fh=[]
 for f in folds:
  best=max(hps,key=lambda hp:state_score([allm[s,hpkey(hp)] for s in f['development']]));fh.append(dict(fold=f['fold'],hyperparameters=best,development_score=state_score([allm[s,hpkey(best)] for s in f['development']])))
  out += [dict(allm[s,hpkey(best)],fold=f['fold']) for s in f['heldout']]
 gate=gate_summary(out,'domain_minus_boundary_caps',fh);dump('oracle_heavy/domain_minus_boundary_caps_gate.json',gate);write('oracle_heavy/domain_minus_boundary_caps_48label_cv.csv',out)
 old=json.load(open(HERE/'oracle_heavy_diagnostic.json'));old['gates']=[g for g in old['gates'] if g['family']!='domain_minus_boundary_caps']+[gate];old['any_cached_screen_pass']=any(g['cached_screen_pass'] for g in old['gates']);old['erosion_subset_fix']=True;dump('oracle_heavy_diagnostic.json',old);print(json.dumps({'usable':gate['usable_states'],'recall':gate['median_independent_B63_recall'],'pass':gate['cached_screen_pass']}))
if __name__=='__main__':main()
