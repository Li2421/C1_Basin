from core import *
from stats import *
precision()
def main():
 rows=read(HERE/'MANIFEST.json');fs={}
 for p in (HERE/'features').glob('shard*.jsonl'):
  for l in p.open():
   f=json.loads(l);fs[f['rollout_uid']]=f
 by={(r['group'],r['controller_index'],r['state_index'],r['seed'],r['arm']):r for r in rows};out={}
 for g in ['TD','V0','V1','V3','Z']:
  C=1 if g=='Z' else 14;out[g]={}
  for arm in ['F13','S500']:
   out[g][arm]={}
   for t in [100,200,300]:
    x={k:np.full((C,32,2),np.nan) for k in ['progress_difference','consistent_rate_difference','four_same_rate_difference']}
    for i in range(C):
     for s in range(32):
      for q in range(2):
       a=by[g,i,s,q,arm];b=by[g,i,s,q,'safety baseline']
       if a['unknown'] or b['unknown']:continue
       aa=fs[a['rollout_uid']]['early'][str(t)];bb=fs[b['rollout_uid']]['early'][str(t)];x['progress_difference'][i,s,q]=aa['progress']-bb['progress'];x['consistent_rate_difference'][i,s,q]=int(aa['mode'] in [1,2])-int(bb['mode'] in [1,2]);x['four_same_rate_difference'][i,s,q]=int(aa['four_same'])-int(bb['four_same'])
    out[g][arm][str(t)]={k:mean(v) for k,v in x.items()}
 # Exposure-normalized sensitivity, fixed at the observed full trace duration.
 exposure={}
 for v,b in [('V1','V0'),('V3','V1')]:
  xa=np.full((14,32,2),np.nan);xb=xa.copy()
  for i in range(14):
   for s in range(32):
    for q in range(2):
     a=by[v,i,s,q,'nominal only'];bb=by[b,i,s,q,'nominal only']
     if a['unknown'] or bb['unknown']:continue
     af=fs[a['rollout_uid']];bf=fs[bb['rollout_uid']]
     if af['T']<40 or bf['T']<40:continue
     xa[i,s,q]=af['moving_no_progress_robot_seconds']/((af['T']-39)*4*.05);xb[i,s,q]=bf['moving_no_progress_robot_seconds']/((bf['T']-39)*4*.05)
  exposure[v+'_minus_'+b]=difference(xa,xb)
 write(HERE/'PAIRED_EARLY_DIAGNOSTICS.json',dict(utc=utc(),early=out,observed_window_exposure_normalized_moving_no_progress_difference=exposure,precision=precision(),evidence_level='DEVELOPMENT',note='Exposure sensitivity is descriptive, conditional on both observed traces having40steps; no new hypothesis confirmation.'))
 # Algebraic sanity checks for the actual clustered statistics implementation.
 one=np.ones((3,32,2));zero=np.zeros_like(one);unknown=np.ones_like(one);assert difference(one,one)['CI95']==[0.,0.];assert paired_success(one,zero,zero,zero)['CI_unknown_envelope']==[1.,1.];assert paired_success(zero,unknown,zero,unknown)['CI_unknown_envelope']==[-1.,1.];assert ratio(zero,zero)['estimate'] is None
 write(HERE/'STATISTICS_TESTS.json',dict(utc=utc(),status='PASS',checks=['paired identical observations zero effect','certain unit success effect [1,1]','unknown versus unknown [-1,1]','zero conditional denominator undefined'],new_rollouts=0));print('paired diagnostics and statistics tests complete',flush=True)
if __name__=='__main__':main()
