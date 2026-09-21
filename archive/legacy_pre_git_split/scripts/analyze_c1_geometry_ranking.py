"""Read-only trajectory reconstruction; no new policy/controller optimization."""
import json
import numpy as np
from audit_c1_joint_witness_risk import ROOT, LocalRisk, save
from audit_c1_completed_waiting_risk import series, scalar, completion, MAXR, KAPPA
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFConfig, barrier_constraints

BASE=ROOT/'results/c1_multistep_objective_audit'
OUT=ROOT/'results/c1_geometry_ranking_analysis'
DT=.05;K=850

def analyze(rid,seed,action):
    path=OUT/f'{rid:04d}_{seed}_{action}.npz';report=path.with_suffix('.json')
    if report.exists():return json.loads(report.read_text())
    z=np.load(BASE/'traces'/path.name);T=len(z['success']);env=GiveWayEnv(Config(corridor_half_length=1.3));cbf=CBFConfig()
    values=[]
    for t in range(T):
        x=z['positions_before'][t];v=z['candidate'][t];u=z['applied'][t]
        A,b,h=barrier_constraints(dict(positions=x,walls=env.walls,config=env.config.to_dict()),cbf)
        model=LocalRisk(A,b,np.full(2,.5));r=model.score(v)
        if not r['valid']:raise ValueError('Undefined risk in matched nonzero branch')
        g=r['risk']/MAXR;speed2=np.sum((v/.5)**2);S=KAPPA**2/(KAPPA**2+speed2)
        residual=A@u-b
        values.append(dict(B=r['B'],E=r['E'],M=r['M'],raw=r['risk'],g=g,S=S,G=float(completion(g,speed2)),
            Hgeom=(1-S)*g,weighted_B=(1-S)*r['B']/MAXR,weighted_E=(1-S)*.25*r['E']/MAXR,
            pair_h=h['pairwise_h'],min_wall_h=h['min_wall_h'],
            active_pair=float(residual[0]<=1e-7),active_walls=float(np.sum(residual[1:]<=1e-7)),
            active_right_ceiling_agent0=float(residual[1+4]<=1e-7),
            zero_constraint_rows=float(np.sum(b==0)),exact_block_cert=float(model.zero_certificate(v)),
            applied_near_zero=float(np.linalg.norm(u)<1e-8),
            candidate_speed=float(np.linalg.norm(v)),applied_speed=float(np.linalg.norm(u)),
            projection_change=float(np.linalg.norm(u-v)),
            relative_speed=float(np.linalg.norm(u[:2]-u[2:])),
            max_agent_speed=float(np.max(np.linalg.norm(u.reshape(2,2),axis=1)))))
    data={k:np.r_[[v[k] for v in values],np.zeros(K-T)] for k in values[0]}
    zz={k:z[k] for k in z.files};zz.update(idx=np.arange(T),risk=data['raw'][:T])
    arrays,_,_,done=series(zz,1);data['P']=arrays['progress']
    data['goal_distance']=np.concatenate([np.linalg.norm(z['positions_after']-z['goals'],axis=2),
        np.repeat(np.linalg.norm(z['positions_after'][-1]-z['goals'],axis=1)[None],K-T,axis=0)])
    data.update(positions_before=z['positions_before'],positions_after=z['positions_after'],candidate=z['candidate'],applied=z['applied'],
                deadlock=z['deadlock'],candidate_deadlock=z['candidate_deadlock'],stuck_timer=z['stuck_timer'])
    coarse,_idx,_T,_done=series(z,10)
    row=dict(rid=rid,seed=seed,action=action,seconds=T*DT,success=done,
        dense={k:float(np.mean(data[k])) for k in ['P','G','S','Hgeom','weighted_B','weighted_E','g','B','E']},
        coarse={k:float(np.mean(coarse[k])) for k in ['progress','geometry','stop_only']},
        exact_block_cert_frames=int(np.sum(data['exact_block_cert'])),near_zero_applied_frames=int(np.sum(data['applied_near_zero'])),
        first_deadlock_seconds=float((np.flatnonzero(z['deadlock'])[0]+1)*DT) if np.any(z['deadlock']) else None,
        stagnation_seconds=float(np.sum(z['candidate_deadlock'])*DT),
        bay_entry_seconds=[float((np.flatnonzero(z['positions_after'][:,a,1]>.2)[0]+1)*DT) if np.any(z['positions_after'][:,a,1]>.2) else None for a in range(2)])
    row['phases']={}
    for lo,hi in [(0,5),(5,7),(7,10),(10,15),(15,20),(20,25.45),(25.45,42.5)]:
        sl=slice(round(lo/DT),round(hi/DT));valid_end=min(T,round(hi/DT))
        phase={k:float(np.mean(data[k][sl])) for k in ['P','G','S','Hgeom','weighted_B','weighted_E']}
        if valid_end>round(lo/DT):
            live=slice(round(lo/DT),valid_end)
            phase.update({k:float(np.mean(data[k][live])) for k in ['candidate_speed','applied_speed','projection_change','active_pair','active_walls','active_right_ceiling_agent0','relative_speed']})
            phase['min_pair_h']=float(np.min(data['pair_h'][live]));phase['min_wall_h']=float(np.min(data['min_wall_h'][live]))
        row['phases'][f'{lo}-{hi}']=phase
    np.savez_compressed(path,**data);save(report,row);return row

def main():
    OUT.mkdir(exist_ok=True)
    cases=[(192,20260916,a) for a in range(9)]+[(192,20260915,a) for a in [1,2]]
    for rid,other in [(190,7),(261,2)]:cases.extend((rid,seed,a) for seed in [20260916,20260915] for a in [1,other])
    rows=[]
    for case in cases:
        r=analyze(*case);rows.append(r);print(case,r['dense'],flush=True)
    matches=[]
    for rid,other in [(192,2),(190,7),(261,2)]:
        for seed in [20260916,20260915]:
            a=next(r for r in rows if r['rid']==rid and r['seed']==seed and r['action']==1)
            b=next(r for r in rows if r['rid']==rid and r['seed']==seed and r['action']==other)
            matches.append(dict(rid=rid,seed=seed,comparison=f'action1_minus_action{other}',
                dense_delta={k:a['dense'][k]-b['dense'][k] for k in a['dense']},
                coarse_delta={k:a['coarse'][k]-b['coarse'][k] for k in a['coarse']},
                phases={phase:{k:a['phases'][phase][k]-b['phases'][phase][k] for k in ['P','G','S','Hgeom','weighted_B','weighted_E']} for phase in a['phases']}))
    plan=[r for r in rows if r['rid']==192 and r['seed']==20260916]
    rank={name:sorted([dict(action=r['action'],cost=sum(r['dense'][k] for k in fields)) for r in plan],key=lambda r:r['cost']) for name,fields in [('P',['P']),('GP',['P','G']),('PS',['P','S']),('PHgeom',['P','Hgeom'])]}
    save(OUT/'summary.json',dict(matches=matches,dense_ID192_rankings=rank,rows=rows,
        activity_note='Activity at projected u uses residual<=1e-7; zero-point rows use b==0 and NNLS certificate. Near-active is not exact blocking.'))

if __name__=='__main__':main()
