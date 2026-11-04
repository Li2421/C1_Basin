"""Audit saved numeric evidence without generating new controller trajectories."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.sched_setaffinity(0,{23})
from clock_plant import *
import argparse

def check_case(c):
    r=read(D/'raw'/(c['rollout_uid']+'.json'));assert not r['error'],r['error'];assert r['outcome']!='unknown'
    assert sha(r['trace'])==r['trace_sha256']
    env=ClockPlant(c['scenario'],c['state'],c['payload']['control_period_s'],c['payload'].get('external_root'))
    grid,controls,scores,h=timeline(c['scenario'],env.config.dt)
    with np.load(r['trace']) as archive:
        # NPZ __getitem__ decompresses on every access. Decode each immutable
        # array once; the checks below only read these arrays or mutate copies.
        z={key:archive[key] for key in archive.files}
        n=len(z['actions']);assert z['times'].shape==(n+1,)
        assert z['positions'][0].tobytes()==env.positions.tobytes()
        assert np.asarray([float(t) for t in grid[:n+1]]).tobytes()==z['times'].tobytes()
        expected=[t for t in grid[:n] if t in controls]
        assert np.asarray([float(t) for t in expected]).tobytes()==z['control_times'].tobytes()
        assert len(expected)==r['control_calls'];assert r['simulated_seconds']==z['times'][-1]
        clears=[];control=0
        for k,u in enumerate(z['actions']):
            if grid[k] in controls:
                env.set_control();assert u.tobytes()==z['control_actions'][control].tobytes()
                clipped=z['raw_actions'][control].copy();norm=np.linalg.norm(clipped,axis=1)
                clipped*=np.minimum(1.,env.config.max_speed/np.maximum(norm,1e-30))[:,None]
                assert clipped.tobytes()==u.tobytes();control+=1
            else:assert u.tobytes()==z['actions'][k-1].tobytes()
            assert np.max(np.linalg.norm(u,axis=1))<=env.config.max_speed+1e-12
            info=env.advance(u,grid[k+1],grid[k+1] in scores);clears.append(info['swept_clearance'])
            assert env.positions.tobytes()==z['positions'][k+1].tobytes()
            assert bool(z['score_flags'][k])==(grid[k+1] in scores)
            assert not env.done or k==n-1
        assert np.asarray(clears).tobytes()==z['swept_clearance'].tobytes()
        assert env.termination==r['outcome'];assert r['horizon_time']==float(h)
        assert min(clears)==r['minimum_swept_clearance']
        if r['success']:assert grid[n] in scores and not env.collided
        if r['timeout']:assert grid[n]==h
        stats=r.get('stats',{})
        if c['payload']['method']=='circulation':assert stats['shadow_mismatch']==0
    return dict(uid=r['rollout_uid'],method=c['payload']['method'],steps=n,calls=r['control_calls'],outcome=r['outcome'],passed=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilots',action='store_true');args=parser.parse_args()
    cases=read(D/'ACTIVE_NATIVE_CASES.json');profile=read(D/'EXECUTION_SOURCE_PROFILE.json')
    for path,h in profile.items():assert sha(path)==h,('source changed',path)
    if args.pilots:
        ids=set(read(D/'PILOT_SELECTION.json')['pilots'].values());cases=[c for c in cases if c['rollout_uid'] in ids]
    result=[]
    for i,c in enumerate(cases):
        result.append(check_case(c))
        if i%10==0:dump(D/'AUDIT_PROGRESS.json',dict(completed=i+1,total=len(cases),steps=sum(r['steps'] for r in result)))
    out=dict(passed=True,cases=len(cases),source_hashes_verified=len(profile),steps=sum(r['steps'] for r in result),control_calls=sum(r['calls'] for r in result),checks=result,new_controller_calls=0)
    dump(D/('PILOT_CERTIFICATE.json' if args.pilots else 'NEW_TRACE_AUDIT.json'),out);print('audit passed',len(cases),out['steps'])

if __name__=='__main__':main()
