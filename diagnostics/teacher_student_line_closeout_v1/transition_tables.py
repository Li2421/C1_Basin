from core import *
from stats import *
import csv,collections
precision()
def main():
 rows=list(csv.DictReader((HERE/'EPISODE_METRICS.csv').open()));by={(r['group_code'],int(r['controller_index']),int(r['state']),int(r['seed']),r['arm_code']):r for r in rows};modes={'stationary':0,'CW-consistent':1,'CCW-consistent':2,'mixed':3};counts=[];summary={}
 for g in ['TD','V0','V1','V3','Z']:
  C=1 if g=='Z' else 14;summary[g]={}
  for a in ['F13','S500']:
   tally=collections.Counter();rescue=np.zeros((C,32,2));already=np.zeros_like(rescue);mixed_success=np.zeros_like(rescue);success=np.zeros_like(rescue)
   for i in range(C):
    for s in range(32):
     for q in range(2):
      b=by[g,i,s,q,'safety baseline'];r=by[g,i,s,q,a];mb=modes[b['mode']];mt=modes[r['mode']];tally[mb,mt,r['outcome']]+=1
      if int(b['unknown']) or int(r['unknown']):continue
      success[i,s,q]=int(r['success']);mixed_success[i,s,q]=int(r['success']) and mt==3;resc=int(b['timeout']) and int(r['success']);rescue[i,s,q]=resc;already[i,s,q]=resc and mb in [1,2]
   summary[g][a]=dict(successes=int(success.sum()),successful_mixed_count=int(mixed_success.sum()),mixed_among_success=ratio(mixed_success,success),baseline_timeouts_rescued=int(rescue.sum()),already_consistent_among_rescued=ratio(already,rescue))
   counts.extend(dict(controller=FULL[g],group=g,arm=ARM_FULL[a],arm_code=a,baseline_mode=mb,corrected_mode=mt,outcome=o,n=n,evidence_level='DEVELOPMENT') for (mb,mt,o),n in sorted(tally.items()))
 with (HERE/'MODE_OUTCOME_TRANSITIONS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
 write(HERE/'MODE_OUTCOME_TRANSITIONS.json',dict(summary=summary,mode_legend={v:k for k,v in modes.items()},definition='Prespecified paired circulation-label and failure-type cross-tabulations; no new trajectory or label.'));print('paired mode/outcome cross-tabs complete',flush=True)
if __name__=='__main__':main()
