"""Saved single-robot evidence only, for the user's moving correction question."""
import bootstrap
from experiment import *
from collections import Counter

def main():
    counts=Counter();sources=[];overlaps=[]
    for p in (HERE/'capability/raw').glob('*.json'):
        r=read(p)
        if r['case']['state']['scene']!='lit_swap8_moving':continue
        counts['requests']+=1;counts[r['outcome']]+=1
        if not r['collision']:continue
        counts['collision_with_any_constraint_violation']+=r.get('constraint_violation_steps',0)>0
        counts['collision_with_any_restoration']+=r.get('restoration_steps',0)>0
        counts['collision_with_speed_excess']+=r.get('sampled_speed_violation_steps',0)>0
        counts['collision_without_constraint_violation']+=r.get('constraint_violation_steps',0)==0
        counts['collision_without_restoration']+=r.get('restoration_steps',0)==0
        for kind,v in r.get('collision_types',{}).items():counts['collision_'+kind]+=bool(v)
        if r['minimum_swept_clearance'] is not None:overlaps.append(-r['minimum_swept_clearance'])
        sources.append(dict(path=str(p),sha256=sha(p)))
    write(HERE/'MOVING_CAPABILITY_COLLISION_AUDIT.json',dict(utc=utc(),counts=dict(counts),collision_overlap_m=dict(minimum=min(overlaps),maximum=max(overlaps)),sources=sources,new_rollouts=0,correction_tested=False,interpretation='Failure of safety-only solo execution; not an empirical test of nonzero eta correction'))
    print(dict(counts),flush=True)
if __name__=='__main__':main()
