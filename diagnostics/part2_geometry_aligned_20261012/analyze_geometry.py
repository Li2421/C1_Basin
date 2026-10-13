"""Read-only Q16 geometric decomposition of cached first-action Jacobians and actions."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
OLD=ROOT/'diagnostics/part2_gain_loss_mechanism_20261012'
PROBES=HARD/'transfer_mechanism_v2/results'
CHAINS=('TT','FT','TF','FF')

def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))

def writecsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def norm(x):return float(np.linalg.norm(x))
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def scalar(w,u):return float(np.dot(w,np.asarray(u).ravel()))

def ring_frames(p,g,safe):
    radius=np.linalg.norm(p,axis=1);t=np.stack((-p[:,1],p[:,0]),axis=1)/radius[:,None]
    radial=p/radius[:,None]
    gap=wrap(np.arctan2(g[:,1],g[:,0])-np.arctan2(p[:,1],p[:,0]))
    # A nearly antipodal goal is direction ambiguous; retain the signed convention
    # as a mathematical diagnostic, and mark it rather than calling it favorable.
    sign=np.sign(gap);nominal=np.sign(np.sum(t*safe,axis=1))
    def weight(q):return (q[:,None]*t/radius[:,None]/len(p)).ravel()
    return radius,t,radial,gap,weight(sign),weight(nominal),weight(np.ones(len(p)))

def ring_motion(u,p,t,radius,sign):
    a=np.sum(np.asarray(u).reshape(p.shape)*t,axis=1)/radius
    return dict(shortest=float(np.mean(sign*a)),ccw=float(np.mean(a)),
                angular_abs=float(np.sum(np.abs(a))),angular_min_abs=float(np.min(np.abs(a))),
                angular_same_sign=int(np.all(a>0) or np.all(a<0)))

def toy_motion(u,p,g):
    u=np.asarray(u).reshape(2,2)
    e=(g-p)/np.linalg.norm(g-p,axis=1,keepdims=True)
    goal=np.sum(e*u,axis=1)
    rel=p[0]-p[1];vrel=u[0]-u[1]
    return dict(goal_mean=float(goal.mean()),goal_min=float(goal.min()),
                clearance_hdot=float(2*np.dot(rel,vrel)),
                upward_capacity=float(max(0,u[0,1],u[1,1])),
                lateral_asymmetry=float(abs(u[0,1]-u[1,1])),
                longitudinal_closing=float(-np.sign(rel[0])*vrel[0]))

def initial_probes():
    output=[];files=list(PROBES.glob('*.json'))
    for file in files:
        x=json.loads(file.read_text())
        if x['kind']!='initial':continue
        with np.load(file.with_suffix('.npz')) as z:
            pre=z['context0_pre_all'];exe=z['context0_exec_all'];basis=z['context0_basis']
            d=z['context0_deltas'];safe=z['context0_safe_nominal']
            activity=np.unpackbits(z['context0_terminal_projection_activity_packed'],axis=-1)
            field_activity=np.unpackbits(z['context0_field_projection_activity_packed'],axis=-1)
            p=z['context0_positions'];g=z['context0_goals']
            sc=x['rows'][0]['scene'];uid=x['state_uid'];seed=x['seed']
            for row in x['rows']:
                j=int(row['eta_index']);idx=1+(1+3*(2*len(safe.ravel())+6))*j;eta=np.asarray(row['eta'],float)
                if not np.allclose(d[idx],basis@eta,atol=1e-9,rtol=0):raise AssertionError((file,j))
                v=eta/max(norm(eta),1e-12);M=z[f'context0_center{j}_M_exec'][:,2]
                J=z[f'context0_center{j}_J_eta_exec'][:,2]
                i0,i1=(0,2) if sc=='ring_exchange' else (1,3)
                a,b=exe[i0,idx],exe[i1,idx]
                a0,b0=exe[i0,0],exe[i1,0]
                pa,pb=pre[i0,idx],pre[i1,idx]
                pair=b-a;baseline=b0-a0;correction=(b-b0)-(a-a0)
                # Primary local derivative is independently cached J_eta. Verify
                # the chain rule against M B, but never call it a finite contrast.
                gd=(J[i1]-J[i0])@v
                mb=(M[i1]-M[i0])@basis@v
                out=dict(scene=sc,uid=uid,eta_index=j,seed=seed,eta_norm=norm(eta),
                    action_l2=norm(pair),baseline_offset_l2=norm(baseline),
                    correction_contrast_l2=norm(correction),
                    zero_plus_correction_identity_error=norm(pair-baseline-correction),
                    differential_chainrule_error=norm(gd-mb),
                    old_fd_stable=int(row['chains'][CHAINS[i0]]['fd_stable']),
                    new_fd_stable=int(row['chains'][CHAINS[i1]]['fd_stable']),
                    old_chain_rule_relative_error=float(row['chains'][CHAINS[i0]]['chain_rule_relative_error']),
                    new_chain_rule_relative_error=float(row['chains'][CHAINS[i1]]['chain_rule_relative_error']),
                    old_one_sided_mismatch=float(row['chains'][CHAINS[i0]]['one_sided_mismatch']),
                    new_one_sided_mismatch=float(row['chains'][CHAINS[i1]]['one_sided_mismatch']),
                    old_activity_switch_fraction=float(row['chains'][CHAINS[i0]]['terminal_activity_switch_fraction']),
                    new_activity_switch_fraction=float(row['chains'][CHAINS[i1]]['terminal_activity_switch_fraction']),
                    terminal_signature_changed=int(np.any(activity[i0,idx]!=activity[i1,idx])),
                    field_signature_changed=int(np.any(field_activity[idx]!=field_activity[0])) if sc=='toy_give_way' else 0,
                    old_terminal_projection_deformation=norm(a-pa),
                    new_terminal_projection_deformation=norm(b-pb))
                if sc=='ring_exchange':
                    radius,t,radial,gap,wshort,wnom,wccw=ring_frames(p,g,safe)
                    sign=np.sign(gap)
                    da=ring_motion(a,p,t,radius,sign);db=ring_motion(b,p,t,radius,sign)
                    dpa=ring_motion(pa,p,t,radius,sign);dpb=ring_motion(pb,p,t,radius,sign)
                    out.update(ambiguous_agents=int(np.sum(np.pi-np.abs(gap)<.05)),
                        shortest_signs_mixed=int(np.any(sign>0) and np.any(sign<0)),
                        shortest_action_difference=scalar(wshort,pair),
                        ccw_action_difference=scalar(wccw,pair),
                        cw_action_difference=-scalar(wccw,pair),
                        nominal_sign_action_difference=scalar(wnom,pair),
                        shortest_baseline_offset=scalar(wshort,baseline),
                        shortest_finite_correction=scalar(wshort,correction),
                        old_shortest_local_differential=scalar(wshort,J[i0]@v),
                        new_shortest_local_differential=scalar(wshort,J[i1]@v),
                        shortest_local_differential=scalar(wshort,gd),
                        shortest_local_linear_extrapolation=scalar(wshort,gd)*norm(eta),
                        ccw_local_differential=scalar(wccw,gd),
                        nominal_sign_local_differential=scalar(wnom,gd),
                        old_shortest=da['shortest'],new_shortest=db['shortest'],
                        old_angular_abs=da['angular_abs'],new_angular_abs=db['angular_abs'],
                        angular_abs_action_difference=db['angular_abs']-da['angular_abs'],
                        angular_same_sign_old=da['angular_same_sign'],
                        angular_same_sign_new=db['angular_same_sign'],
                        old_pre_angular_abs=dpa['angular_abs'],new_pre_angular_abs=dpb['angular_abs'],
                        old_projection_angular_change=da['angular_abs']-dpa['angular_abs'],
                        new_projection_angular_change=db['angular_abs']-dpb['angular_abs'],
                        projection_angular_change_advantage=(db['angular_abs']-dpb['angular_abs'])-(da['angular_abs']-dpa['angular_abs']),
                        radial_speed_difference=float(np.mean(np.sum((b-a).reshape(p.shape)*radial,axis=1))))
                else:
                    m0=toy_motion(a,p,g);m1=toy_motion(b,p,g)
                    goal=(g-p)/np.linalg.norm(g-p,axis=1,keepdims=True)
                    wgoal=(goal/2).ravel()
                    rel=p[0]-p[1];wclear=np.r_[2*rel,-2*rel]
                    lateral_gradient=np.array([0.,1.,0.,-1.])
                    old_lateral_sign=np.sign(a[1]-a[3]);new_lateral_sign=np.sign(b[1]-b[3])
                    out.update(goal_mean_difference=m1['goal_mean']-m0['goal_mean'],
                        goal_pre_action_difference=scalar(wgoal,pb-pa),
                        goal_projection_change_of_contrast=scalar(wgoal,pair)-scalar(wgoal,pb-pa),
                        goal_min_difference=m1['goal_min']-m0['goal_min'],
                        clearance_hdot_difference=m1['clearance_hdot']-m0['clearance_hdot'],
                        upward_capacity_difference=m1['upward_capacity']-m0['upward_capacity'],
                        lateral_asymmetry_difference=m1['lateral_asymmetry']-m0['lateral_asymmetry'],
                        longitudinal_closing_difference=m1['longitudinal_closing']-m0['longitudinal_closing'],
                        goal_baseline_offset=scalar(wgoal,baseline),
                        goal_finite_correction=scalar(wgoal,correction),
                        old_goal_local_differential=scalar(wgoal,J[i0]@v),
                        new_goal_local_differential=scalar(wgoal,J[i1]@v),
                        goal_local_differential=scalar(wgoal,gd),
                        goal_local_linear_extrapolation=scalar(wgoal,gd)*norm(eta),
                        clearance_local_differential=scalar(wclear,gd),
                        old_lateral_asymmetry_local=float(old_lateral_sign*np.dot(lateral_gradient,J[i0]@v)),
                        new_lateral_asymmetry_local=float(new_lateral_sign*np.dot(lateral_gradient,J[i1]@v)),
                        lateral_asymmetry_local_difference=float(new_lateral_sign*np.dot(lateral_gradient,J[i1]@v)-old_lateral_sign*np.dot(lateral_gradient,J[i0]@v)),
                        old_goal_mean=m0['goal_mean'],new_goal_mean=m1['goal_mean'],
                        old_clearance_hdot=m0['clearance_hdot'],new_clearance_hdot=m1['clearance_hdot'])
                output.append(out)
    return output

def aggregate(rows, labels):
    groups=defaultdict(list)
    for r in rows:groups[(r['scene'],r['uid'],r['eta_index'])].append(r)
    result=[]
    for (sc,uid,j),group in sorted(groups.items()):
        assert len(group)==16,(sc,uid,j,len(group))
        label=labels[(sc,uid,j)]
        z=dict(scene=sc,uid=uid,eta_index=j,split=label['split'],b15_outcome=label['b15_outcome'],
               old_successes=label['old_successes'],new_successes=label['new_successes'],noise_count=len(group))
        for k in group[0]:
            if k in z or k in ('seed','scene','uid','eta_index'):continue
            z[k]=float(np.mean([r[k] for r in group]))
        result.append(z)
    return result

def main():
    labels={}
    for sc,name in [('ring_exchange','ring_gain_analysis.csv'),('toy_give_way','toy_loss_analysis.csv')]:
        for r in readcsv(OLD/name):labels[(sc,r['uid'],int(r['eta_index']))]=r
    seed_rows=initial_probes();assert len(seed_rows)==12288,len(seed_rows)
    rows=aggregate(seed_rows,labels)
    ring=[r for r in rows if r['scene']=='ring_exchange']
    toy=[r for r in rows if r['scene']=='toy_give_way']
    writecsv(HERE/'ring_directional_mechanism.csv',ring)
    writecsv(HERE/'toy_yielding_mechanism.csv',toy)
    (HERE/'geometry_cache_summary.json').write_text(json.dumps(dict(seed_actions=len(seed_rows),
        candidates=len(rows),ring_ambiguous_candidates=sum(r['ambiguous_agents']>0 for r in ring),
        ring_mixed_shortest_sign_candidates=sum(r['shortest_signs_mixed']>0 for r in ring),
        max_chainrule_error=max(r['differential_chainrule_error'] for r in seed_rows),
        max_zero_plus_correction_identity_error=max(r['zero_plus_correction_identity_error'] for r in seed_rows)),indent=2)+'\n')
    print('seed actions',len(seed_rows),'candidate rows',len(rows))

if __name__=='__main__':main()
