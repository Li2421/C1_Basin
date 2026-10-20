"""Read-only diagnostics over existing traces; no controller or simulator calls."""
from core import *
from audit_utils import source_inventory
import csv,collections

def stall_stats(mask):
    counts=[];dur=[]
    for agent in range(4):
        d=np.diff(np.r_[False,mask[:,agent],False].astype(int));lengths=np.flatnonzero(d==-1)-np.flatnonzero(d==1);lengths=lengths[lengths>=40];counts.append(len(lengths));dur.extend(lengths.tolist())
    return dict(stalled_robots=sum(x>0 for x in counts),stall_runs=sum(counts),stall_total_seconds=sum(dur)*.05,stall_max_seconds=max(dur,default=0)*.05)
def main():
    rows=[];errors=collections.Counter();by=collections.defaultdict(list)
    for f in sorted((HERE/'main/raw').glob('*.json')):
        r=read(f);c=r['case'];z=np.load(r['trace_path']);ss=stall_stats(z['slow_mask']);assert len(z['positions'])==r['physical_steps']+1;assert sha(r['trace_path'])==r['trace_sha256'];assert not r['jax_enable_x64'];assert not r['deadlock']
        if r['success']:assert not r['collision'] and (not len(z['swept_clearance']) or z['swept_clearance'].min()>=0)
        q=dict(rollout_uid=c['rollout_uid'],controller=c['base_controller_name'],arm=c['arm'],state_index=c['state_index'],seed=c['seed'],outcome=r['outcome'],success=r['success'],collision=r['collision'],timeout=r['timeout'],unknown=r['numerical_failure'],physical_steps=r['physical_steps'],elapsed_seconds=r['elapsed_seconds'],minimum_swept_clearance=r['minimum_swept_clearance'],**ss);rows.append(q);by[c['arm']].append(q)
        if r['error']:errors[(c['arm'],r['error']['type'],r['error']['message'])]+=1
    with (HERE/'rollout_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    perarm={}
    for arm,rs in by.items():
        fail=[r['physical_steps']*.05 for r in rs if not r['success'] and not r['unknown']]
        perarm[arm]=dict(n=len(rs),collisions=sum(r['collision'] for r in rs),unknown=sum(r['unknown'] for r in rs),timeout=sum(r['timeout'] for r in rs),stall_episode_rate=float(np.mean([r['stalled_robots']>0 for r in rs])),mean_stalled_robots=float(np.mean([r['stalled_robots'] for r in rs])),mean_stall_total_seconds=float(np.mean([r['stall_total_seconds'] for r in rs])),max_stall_seconds=float(max(r['stall_max_seconds'] for r in rs)),failure_time_seconds=dict(mean=float(np.mean(fail)),median=float(np.median(fail)),min=float(min(fail)),max=float(max(fail))) if fail else None)
    write(HERE/'FAILURE_DIAGNOSTICS.json',dict(utc=utc(),arms=perarm,errors=[dict(arm=k[0],type=k[1],message=k[2],count=n) for k,n in errors.items()],stall_is_diagnostic_only=True,timeout_is_not_deadlock=True,unknown_segments='Preserved, truncated at numerical failure; future motion unknown'))
if __name__=='__main__':
    main()
    source_inventory('imports_metrics.json')
