"""Geometry, multiscale regularity, and nearby-state scan summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'placement_boundary_mechanism_audit_v1'
CACHE=Path('/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/state_dependence_control')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')

def sv(a):return np.linalg.svd(a,compute_uv=False)
def rank(s,relative=1e-3):return int(np.sum(s>max(s[0]*relative,1e-10))) if s[0]>0 else 0
def angle(a,b):
    u,s,_=np.linalg.svd(a,full_matrices=False);v,t,_=np.linalg.svd(b,full_matrices=False)
    r=min(rank(s),rank(t))
    if not r:return []
    return np.degrees(np.arccos(np.clip(np.linalg.svd(u[:,:r].T@v[:,:r],compute_uv=False),0,1))).tolist()
def top2_angle(a,b):
    u=np.linalg.svd(a,full_matrices=False)[0][:,:2]
    v=np.linalg.svd(b,full_matrices=False)[0][:,:2]
    return float(np.degrees(np.arccos(np.clip(np.linalg.svd(u.T@v,compute_uv=False)[-1],0,1))))
def med(v):return float(np.median(v)) if v else None
def quant(v):return np.quantile(v,[0,.25,.5,.75,1]).tolist() if v else None

def main():
    p=json.loads((HERE/'protocol.json').read_text());action_rows=[];state_rows=[];cached=[]
    all_states=json.loads((HARD/'confirmation_manifest.json').read_text())['states']
    for st in all_states:
        uid=st['uid'];z=np.load(CACHE/(uid+'.npz'))
        d=json.loads((CACHE/(uid+'.json')).read_text())['diagnostics']['chains']
        b=z['basis'];g=z['center0_J_eta_exec'][:,2]
        row=dict(uid=uid,scene=st['scenario'],B_sigma=sv(b).tolist(),
            B_rank=rank(sv(b)),B_near_null_ratio=float(sv(b)[-1]/sv(b)[0]),
            chains={})
        for chain,i in (('TT',0),('FF',3)):
            s=sv(g[i]);row['chains'][chain]=dict(sigma=s.tolist(),rank=rank(s),
                near_null_ratio=float(s[-1]/s[0]) if s[0] else 0.,
                condition=float(s[0]/s[-1]) if s[-1] else None,
                fd_stable=bool(d[chain]['fd_stable']),
                epsilon_relative_change=d[chain]['eta_epsilon_relative_change'],
                one_sided_mismatch=d[chain]['one_sided_mismatch'],
                field_activity_switch_fraction=d[chain]['field_activity_switch_fraction'],
                terminal_activity_switch_fraction=d[chain]['terminal_activity_switch_fraction'])
        row['principal_angles_deg']=angle(g[0],g[3]);row['top2_max_angle_deg']=top2_angle(g[0],g[3]);cached.append(row)
    for st in p['states']:
        uid=st['uid']
        for pt in p['action_points']:
            records={};matrices={}
            for chain in ('TT','FF'):
                path=HERE/'action_results'/f"{uid}_{pt['name']}_{chain}"
                x=json.loads((path.with_suffix('.json')).read_text())
                with np.load(path.with_suffix('.npz')) as q:G=q['G_epsilon'].copy();B=q['B'].copy()
                matrices[chain]=G[-1];sig=sv(G[-1]);bs=sv(B)
                stable=np.asarray(x['branch_stable_axes'])
                gap=np.asarray(x['one_sided_gap'])[-1]/np.maximum(np.linalg.norm(G[-1],axis=0),1e-12)
                gchange=x['relative_G_scale_change'][-2]
                hold=x['holdout'][-1]
                records[chain]=dict(uid=uid,scene=st['scenario'],eta_name=pt['name'],chain=chain,
                    eps=x['eps'],G_sigma=sig.tolist(),B_sigma=bs.tolist(),G_rank=rank(sig),B_rank=rank(bs),
                    near_null_ratio=float(sig[-1]/sig[0]) if sig[0] else 0.,
                    condition=float(sig[0]/sig[-1]) if sig[-1] else None,
                    rank_by_epsilon=[rank(sv(z)) for z in G],
                    rank_constant_over_epsilon=len({rank(sv(z)) for z in G})==1,
                    branch_stable_axis_count=[int(np.sum(z)) for z in stable],
                    G_change_vs_finest=x['relative_G_scale_change'],
                    one_sided_relative_gap_finest=gap.tolist(),
                    heldout=hold,regularity=dict(
                        active_signature_stable=bool(np.all(stable[-1])),
                        G_scale_stable_2pct=bool(gchange<.02),
                        G_scale_stable_5pct=bool(gchange<.05),
                        heldout_error_5pct=bool(hold['relative_error']<.05),
                        one_sided_5pct=bool(np.max(gap)<.05)))
                action_rows.append(records[chain])
            action_rows[-2]['principal_angles_deg']=angle(matrices['TT'],matrices['FF'])
            action_rows[-1]['principal_angles_deg']=action_rows[-2]['principal_angles_deg']
            action_rows[-2]['top2_max_angle_deg']=top2_angle(matrices['TT'],matrices['FF'])
            action_rows[-1]['top2_max_angle_deg']=action_rows[-2]['top2_max_angle_deg']
    for st in p['states']:
        if st['uid'] not in p['state_probe_uids']:continue
        uid=st['uid'];d=np.asarray(p['state_position_directions'][uid])
        for pt in p['state_probe_points']:
            for chain in ('TT','FF'):
                stem=f"{uid}_{pt['name']}_{chain}"
                zero=json.loads((HERE/'state_results'/(stem+'_h0_s0.json')).read_text())
                with np.load(HERE/'state_results'/(stem+'_h0_s0.npz')) as q:g0=q['G'].copy();b0=q['B'].copy()
                for h in p['state_radii']:
                    for sign in (-1,1):
                        path=HERE/'state_results'/(stem+f'_h{h:g}_s{sign}')
                        x=json.loads(Path(str(path)+'.json').read_text())
                        with np.load(Path(str(path)+'.npz')) as q:g=q['G'].copy();b=q['B'].copy()
                        dg=float(np.linalg.norm(g-g0));db=float(np.linalg.norm(b-b0))
                        state_rows.append(dict(uid=uid,scene=st['scenario'],eta_name=pt['name'],chain=chain,
                            radius=h,sign=sign,state_distance=float(np.linalg.norm(h*d)),
                            G_difference=dg,G_difference_relative=dg/max(float(np.linalg.norm(g0)),1e-12),
                            B_difference=db,G_quotient=dg/h,B_quotient=db/h,
                            active_signature_same=x['activity']['branch_hash']==zero['activity']['branch_hash'],
                            principal_angles_deg=angle(g0,g),
                            singular_values_base=sv(g0).tolist(),singular_values_perturbed=sv(g).tolist()))
    def summaries(rows):
        out={}
        for scene in ('ring_exchange','toy_give_way'):
            for chain in ('TT','FF'):
                aa=[r for r in rows if r['scene']==scene and r['chain']==chain]
                out[f'{scene}:{chain}']=dict(n=len(aa),branch_stable_fraction=float(np.mean([r['regularity']['active_signature_stable'] for r in aa])),
                    G_change_2pct_fraction=float(np.mean([r['regularity']['G_scale_stable_2pct'] for r in aa])),
                    G_change_5pct_fraction=float(np.mean([r['regularity']['G_scale_stable_5pct'] for r in aa])),
                    joint_regular_5pct_fraction=float(np.mean([
                        r['regularity']['active_signature_stable'] and r['regularity']['G_scale_stable_5pct']
                        and r['regularity']['heldout_error_5pct'] and r['regularity']['one_sided_5pct']
                        for r in aa])),
                    heldout_relative_error_quantiles=quant([r['heldout']['relative_error'] for r in aa]),
                    finest_one_sided_gap_quantiles=quant([max(r['one_sided_relative_gap_finest']) for r in aa]),
                    G_near_null_ratio_quantiles=quant([r['near_null_ratio'] for r in aa]),
                    rank3_fraction=float(np.mean([r['G_rank']==3 for r in aa])),
                    rank_constant_over_epsilon_fraction=float(np.mean([r['rank_constant_over_epsilon'] for r in aa])))
        return out
    cached_summary={}
    for scene in ('ring_exchange','toy_give_way'):
        rr=[r for r in cached if r['scene']==scene]
        cached_summary[scene]=dict(states=len(rr),B_rank3=sum(r['B_rank']==3 for r in rr),
            TT_rank3=sum(r['chains']['TT']['rank']==3 for r in rr),
            FF_rank3=sum(r['chains']['FF']['rank']==3 for r in rr),
            TT_fd_stable=sum(r['chains']['TT']['fd_stable'] for r in rr),
            FF_fd_stable=sum(r['chains']['FF']['fd_stable'] for r in rr),
            TT_near_null_below_005=sum(r['chains']['TT']['near_null_ratio']<.05 for r in rr),
            FF_near_null_below_005=sum(r['chains']['FF']['near_null_ratio']<.05 for r in rr),
            effective_rank3_by_relative_threshold={str(th):dict(
                B=sum(r['B_near_null_ratio']>th for r in rr),
                TT=sum(r['chains']['TT']['near_null_ratio']>th for r in rr),
                FF=sum(r['chains']['FF']['near_null_ratio']>th for r in rr))
                for th in (1e-4,1e-3,1e-2)},
            principal_max_angle_quantiles=quant([max(r['principal_angles_deg']) for r in rr]),
            leading_top2_max_angle_quantiles=quant([r['top2_max_angle_deg'] for r in rr]),
            principal_max_angle_both_rank3_quantiles=quant([max(r['principal_angles_deg']) for r in rr
                if r['chains']['TT']['rank']==3 and r['chains']['FF']['rank']==3]))
    state_summary={}
    for scene in ('ring_exchange','toy_give_way'):
        for chain in ('TT','FF'):
            aa=[r for r in state_rows if r['scene']==scene and r['chain']==chain]
            state_summary[f'{scene}:{chain}']=dict(n=len(aa),active_same_fraction=float(np.mean([r['active_signature_same'] for r in aa])),
                by_radius={str(h):dict(median_G_difference=med([r['G_difference'] for r in aa if r['radius']==h]),
                     median_G_quotient=med([r['G_quotient'] for r in aa if r['radius']==h]),
                     median_principal_max_angle_deg=med([max(r['principal_angles_deg']) for r in aa
                         if r['radius']==h and r['principal_angles_deg']]),
                     active_same_fraction=float(np.mean([r['active_signature_same'] for r in aa if r['radius']==h])))
                     for h in p['state_radii']})
    out=dict(cached_48=cached_summary,new_action=summaries(action_rows),nearby_state=state_summary,
             branch_tolerance=p['action_active_tolerance'],rank_relative_tolerance=1e-3,
             regularity_thresholds_descriptive=True)
    for name,value in [('cached_geometry_rows.json',cached),('action_rows.json',action_rows),
                       ('state_rows.json',state_rows),('action_state_summary.json',out)]:
        (HERE/name).write_text(json.dumps(value,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(12,3.4))
    for i,scene in enumerate(('ring_exchange','toy_give_way')):
        for chain,color in (('TT','#1f77b4'),('FF','#d95f02')):
            aa=[r for r in action_rows if r['scene']==scene and r['chain']==chain]
            axs[i].scatter([r['G_sigma'][0] for r in aa],[r['G_sigma'][-1] for r in aa],label=chain,alpha=.8,c=color)
        axs[i].set_title(scene);axs[i].set_xlabel('$\\sigma_1(G)$');axs[i].set_ylabel('$\\sigma_3(G)$');axs[i].legend()
    for chain,color in (('TT','#1f77b4'),('FF','#d95f02')):
        aa=[r for r in state_rows if r['chain']==chain]
        axs[2].scatter([r['state_distance'] for r in aa],[r['G_difference'] for r in aa],label=chain,alpha=.55,c=color)
    axs[2].set_xscale('log');axs[2].set_yscale('log');axs[2].set_xlabel('$||\\Delta s||$');axs[2].set_ylabel('$||\\Delta G||_F$');axs[2].legend()
    fig.tight_layout();fig.savefig(HERE/'figures/action_geometry_state_continuity.pdf');fig.savefig(HERE/'figures/action_geometry_state_continuity.png',dpi=180);plt.close(fig)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
