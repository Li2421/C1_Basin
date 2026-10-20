from core import *
from metrics import features,collision_geometry,signature_codes
import time
precision()
def main(shard,workers):
 prereg();rows=read(HERE/'MANIFEST.json');unique={r['rollout_uid']:r for r in rows};items=sorted(unique.items())[shard::workers];out=HERE/'features';out.mkdir(exist_ok=True);dest=out/f'shard{shard}.jsonl';assert not dest.exists();errors=[];t0=time.time()
 with dest.open('x') as f:
  for n,(rid,r) in enumerate(items):
   try:
    assert sha(r['trace_path'])==r['trace_sha256'];assert sha(r['raw_path'])==r['raw_sha256']
    with np.load(r['trace_path'],allow_pickle=False) as z:
     p=z['positions'];a=z['actions'];g=z['goals'];v=z['velocities'] if 'velocities' in z.files else None;m=features(p,a,g,v)
     if r['collision']:
      dis=np.unwrap(np.arctan2(p[...,1],p[...,0]),axis=0);sign,_=signature_codes(dis-dis[0]);m['collision_geometry']=collision_geometry(p,a,sign)
      step=m['collision_geometry']['step'];m['positions_at_collision_step']=p[max(0,min(step or len(a),len(a)))].tolist()
     else:m['collision_geometry']=None
    m.update(rollout_uid=rid,source_trace_sha256=r['trace_sha256'],source_raw_sha256=r['raw_sha256'],velocities_reconstructed=v is None)
    f.write(json.dumps(clean(m),separators=(',',':'),allow_nan=False)+'\n')
   except Exception as e:errors.append(dict(rollout_uid=rid,type=type(e).__name__,reason=str(e)))
   if n%250==0:print(shard,n,len(items),'elapsed',time.time()-t0,'errors',len(errors),flush=True)
 write(out/f'shard{shard}_COMPLETE.json',dict(utc=utc(),requested=len(items),completed=len(items)-len(errors),errors=errors,feature_sha256=sha(dest),precision=precision(),elapsed=time.time()-t0));print('complete',shard,len(items),len(errors),flush=True)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
