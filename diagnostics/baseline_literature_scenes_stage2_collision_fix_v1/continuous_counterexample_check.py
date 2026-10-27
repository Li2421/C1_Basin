"""Independent scalar checks of frozen Euler counterexamples, no QP or rollout."""
from audit import HERE,write
import json
import numpy as np

def main():
    result=[]
    for scene in ('lit_narrow_passage','lit_crowded_random'):
        dt=.2 if scene=='lit_narrow_passage' else .15
        for f in sorted((HERE/'unit_snapshots').glob(scene+'*.json')):
            for e in json.loads(f.read_text())['counterexamples']:
                if e.get('reason')!='feasible_collision':continue
                p=np.array(e['p']);v=np.array(e['v']);a=np.array(e['a']);pairs=[]
                for i in range(len(p)):
                    for j in range(i+1,len(p)):
                        d=p[i]-p[j];step=dt*(v[i]-v[j]);s=float(np.clip(-(d@step)/max(step@step,1e-30),0,1))
                        q=d+s*step;bar=d+step
                        pairs.append(dict(i=i,j=j,start_distance=float(np.linalg.norm(d)),predicted_distance=float(np.linalg.norm(bar)),
                                          interval_minimum=float(np.linalg.norm(q)),fraction=s,physical_clearance=float(np.linalg.norm(q)-.3),
                                          next_next_distance=float(np.linalg.norm(d+2*step+dt*dt*(a[i]-a[j])))))
                result.append(dict(scene=scene,index=e['index'],source=str(f),pairs=pairs,
                                   all_original_constraints_satisfied=all(d['maximum_violation']<=1e-7 for d in e['diags']),
                                   proof_precondition_holds=all(x['predicted_distance']>=.303101 for x in pairs)))
    write(HERE/'INDEPENDENT_COUNTEREXAMPLE_CHECK.json',dict(results=result,minimum_clearance=min(x['physical_clearance'] for r in result for x in r['pairs']),
          interpretation='These arbitrary initial velocities need not be reachable from zero under a feasible invariant-preserving filter. They invalidate an unconditional feasible-snapshot guarantee, not the conditional support-halfspace proof.'))
    print('Verified',len(result),'preserved counterexamples')
if __name__=='__main__':main()
