"""Aggregate the already-preregistered four_same flag; no new metric definitions."""
from common import *
import stats as st
G=guard()
def main():
 records=read(HERE/'ANALYSIS_MANIFEST.json');res=read(HERE/'results.json');wanted={r['rollout_uid'] for r in records if not r['unknown']};features={}
 for path in [HERE/'EPISODE_METRICS.jsonl']+sorted((D/'teacher_student_line_closeout_v1/features').glob('shard*.jsonl')):
  for line in path.open():
   r=json.loads(line)
   if r['rollout_uid'] in wanted:features[r['rollout_uid']]=r
 out={}
 for group,g in res['groups'].items():
  names=g.get('controllers',[])
  if not names:continue
  ci={n:i for i,n in enumerate(names)};out[group]={}
  for a in ['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair']:
   x=np.full((len(names),32,2),np.nan);y=x.copy();unknown=0
   for r in records:
    if r['name'] not in ci or r['arm']!=a:continue
    if r['unknown']:unknown+=1;continue
    m=features[r['rollout_uid']];k=(ci[r['name']],r['state_index'],r['seed']);x[k]=m['four_same']
    if r['timeout']:y[k]=m['four_same']
   out[group][a]=dict(all_four_same_nonzero=st.mean(x),all_four_same_nonzero_given_timeout=st.mean(y),unknown=unknown)
 write(HERE/'STRICT_FOUR_DIRECTION.json',dict(definition='Existing round30g four_same flag, same strict +/-0.15 thresholds; no altered logic. Mode1/2 counts have at least one nonzero direction and no opposite sign, and may include zero-sign robots.',groups=out,guard=guard()))
 print('STRICT FOUR DIRECTION PASS')
if __name__=='__main__':main()
