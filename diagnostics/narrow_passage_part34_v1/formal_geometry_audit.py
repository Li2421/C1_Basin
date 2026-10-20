"""CPU geometry-only reconciliation of formal A1--A5; no model or rollout."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
sys.path.insert(0, str(FROZEN / 'source'))
from single_integrator.environment import GiveWayEnv, Config
from single_integrator.environment import segment_distance, point_segment_distance
from single_integrator.cbf import CBFConfig, barrier_geometry


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    cfg = Config(corridor_half_length=1.3)
    cbf = CBFConfig()
    c = cfg.agent_radius + cfg.wall_radius + cfg.wall_collision_margin + cbf.separation_buffer
    Ds = 2*cfg.agent_radius + cfg.agent_collision_margin + cbf.separation_buffer
    delta = .01
    D = Ds + delta
    margins = [.005, .001, .0001, .00001, .000001, .00000001]
    rows = list(csv.DictReader((HERE/'trajectory_audit.csv').open()))
    primary = .001
    records = []
    sensitivity = []
    examples = []
    for m in margins:
        h = cfg.corridor_width/2-c-m
        a = cfg.corridor_half_length-c-m
        H = cfg.bay_top-c-m
        counts = dict(m=m, h=h, a_plus=a, b_plus=h, H=H,
                      A2_height_gap=H-h-D, A4_longitudinal_gap=a-h-D,
                      A3_trajectories=0, A3_frames=0, A3_success=0,
                      A3_deadlock=0, A3_timeout=0, A3_true_t0=0,
                      role_R_trajectories=0, role_L_trajectories=0)
        for row in rows:
            with np.load(row['trace']) as data:
                p = data['positions']; cert = data['certificates']
            # A3 reference start must admit an actual cached action; omit final endpoint.
            p = p[:-1]
            s = p[:,0,0]-p[:,1,0]
            dist = np.linalg.norm(p[:,0]-p[:,1],axis=1)
            safe_action = (cert[:,0]>=-cbf.feasibility_tol)&(cert[:,1]<=cbf.speed_tol)
            any_mask = np.zeros(len(p),dtype=bool)
            for role, one, two in [('R',0,1),('L',1,0)]:
                z = p[:,two,1]-p[:,one,1]
                member = ((np.abs(p[:,one,0])<=a)&(np.abs(p[:,one,1])<=h)
                          &(np.abs(p[:,two,0])<=h)&(p[:,two,1]>=-h)&(p[:,two,1]<=H))
                mask = member&(s<0)&(z>=0)&(z<D)&(dist>=Ds)&safe_action
                indices = np.flatnonzero(mask)
                any_mask |= mask
                counts['role_'+role+'_trajectories'] += int(len(indices)>0)
                if m == primary:
                    # Progress thresholds use the complete archived positions.
                    with np.load(row['trace']) as data:
                        fullp = data['positions']
                    fulls=fullp[:,0,0]-fullp[:,1,0]
                    fullz=fullp[:,two,1]-fullp[:,one,1]
                    yi=np.flatnonzero(fullz>=D); pi=np.flatnonzero(fulls>=D)
                    y=int(yi[0]) if len(yi) else -1
                    q=int(pi[0]) if len(pi) else -1
                    rec=dict(id=row['id'],state_uid=row['state_uid'],seed=int(row['seed']),
                             eta_name=row['eta_name'],termination=row['termination'],role=role,
                             A3_t0=bool(mask[0]),A3_local=bool(len(indices)),
                             first_A3_step=int(indices[0]) if len(indices) else -1,
                             A3_frames=int(len(indices)),formal_yield_step=y,formal_pass_step=q,
                             formal_ordered=bool(y>=0 and q>y),
                             formal_min_z_between=float(fullz[y:q+1].min()) if y>=0 and q>=y else None)
                    start=int(indices[0]) if len(indices) else -1
                    local_y=np.flatnonzero((np.arange(len(fullz))>=start)&(fullz>=D)) if start>=0 else []
                    ly=int(local_y[0]) if len(local_y) else -1
                    local_p=np.flatnonzero((np.arange(len(fulls))>ly)&(fulls>=D)) if ly>=0 else []
                    lp=int(local_p[0]) if len(local_p) else -1
                    rec.update(local_yield_step=ly,local_pass_step=lp,
                               local_min_z_between=float(fullz[ly:lp+1].min()) if ly>=0 and lp>=ly else None)
                    records.append(rec)
                    if len(indices) and len(examples)<12:
                        t=int(indices[0]); env=GiveWayEnv(cfg); env.positions=p[t].copy()
                        g=barrier_geometry(env.snapshot(),cbf)
                        examples.append(dict(record=rec,positions=p[t].tolist(),
                                             min_wall_h=g['min_wall_h'],pairwise_h=g['pairwise_h']))
            if any_mask.any():
                counts['A3_trajectories']+=1
                counts['A3_'+row['termination']]+=1
            counts['A3_frames']+=int(any_mask.sum())
            counts['A3_true_t0']+=int(any_mask[0])
        sensitivity.append(counts)
    with (HERE/'formal_state_coverage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=dict(protocol_sha256=hashlib.sha256((HERE/'FORMAL_PROTOCOL.md').read_bytes()).hexdigest(),
                 config=cfg.to_dict(),Ds=Ds,delta=delta,D=D,wall_center_safety=c,
                 A1=dict(supported=2*cfg.agent_radius<cfg.corridor_width<4*cfg.agent_radius,
                         lower=2*cfg.agent_radius,W=cfg.corridor_width,upper=4*cfg.agent_radius),
                 primary=next(x for x in sensitivity if x['m']==primary),
                 sensitivity=sensitivity,examples=examples,
                 A5=dict(external_robots=0,status='VACUOUS'),
                 progress_by_role={role:{term:dict(total=sum(r['role']==role and r['termination']==term for r in records),
                     ordered=sum(r['role']==role and r['termination']==term and r['formal_ordered'] for r in records),
                     maintained=sum(r['role']==role and r['termination']==term and r['formal_ordered']
                                    and r['formal_min_z_between']>=D for r in records))
                     for term in ['success','deadlock','timeout']} for role in ['R','L']})
    summary['A3_continuation_progress']={term:dict(total=sum(r['A3_local'] and r['termination']==term for r in records),
                yield_then_pass=sum(r['A3_local'] and r['termination']==term and r['local_pass_step']>=0 for r in records),
                maintain_lateral=sum(r['A3_local'] and r['termination']==term and r['local_pass_step']>=0
                                     and r['local_min_z_between']>=D for r in records))
                for term in ['success','deadlock','timeout']}
    old_witness=[r for r in records if r['id']=='2f142928c7c061e81d88']
    summary['previous_witness_formal_membership']=old_witness
    write_json('formal_geometry.json',summary)
    probe_file=HERE/'formal_local_probes.json'
    if probe_file.exists():
        probes=json.loads(probe_file.read_text())
        eligible=[p for p in probes if p['A3_record']['termination']=='success'
                  and p['perp_delta_dz']>1e-6 and p['ds_dz_full'][1]>0]
        if eligible:
            selected=eligible[0]
            row=next(x for x in rows if x['id']==selected['case_id'])
            case=next(x for x in json.loads((FROZEN/'validation_stage2/protocol.json').read_text())['cases']
                      if x['id']==row['id'])
            outcome=json.loads(Path(row['trace']).with_suffix('.json').read_text())
            with np.load(row['trace']) as data:
                p=data['positions'];cert=data['certificates']
            t=selected['timestep'];env=GiveWayEnv(cfg);env.positions=p[t].copy()
            initial=barrier_geometry(env.snapshot(),cbf)
            wall=segment_distance(p[:-1,:,None,:],p[1:,:,None,:],
                                  env.walls[None,None,:,0,:],env.walls[None,None,:,1,:])
            wall-=cfg.agent_radius+cfg.wall_radius
            pair=point_segment_distance(np.zeros(2),p[:-1,0]-p[:-1,1],p[1:,0]-p[1:,1])-2*cfg.agent_radius
            write_json('formal_witness.json',dict(probe=selected,case=case,outcome=outcome,
                       D_s=Ds,delta=delta,D=D,m_wall=primary,
                       initial_min_wall_h=initial['min_wall_h'],initial_pairwise_h=initial['pairwise_h'],
                       full_episode_min_swept_wall_surface=float(wall.min()),
                       full_episode_min_swept_pair_surface=float(pair.min()),
                       suffix_min_swept_wall_surface=float(wall[t:].min()),
                       suffix_min_swept_pair_surface=float(pair[t:].min()),
                       suffix_min_linear_residual=float(cert[t:,0].min()),
                       suffix_max_speed_excess=float(cert[t:,1].max()),
                       trace=row['trace'],local_uid_is_registered_database_state=False,
                       local_remaining_steps=cfg.max_steps-t,local_completion_steps=int(outcome['steps'])-t))
    print(json.dumps({k:summary[k] for k in ['Ds','delta','D','primary','progress_by_role','previous_witness_formal_membership']},indent=2))


if __name__=='__main__': main()
