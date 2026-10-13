"""Locate missing informative joint support without generating data or fitting."""
import csv,json,hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MAT=ROOT/'controller_ff_factorial_repair_v1'
UNION=ROOT/'controller_cached_union_v1'
NEW=ROOT/'controller_native_mass_repair_v1'
SEEDS=(17,23,41)


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    p=dict(question='Is the informative crossed-controller branch already failing on unseen states within its source scene, and do its seed labels contain reproducible interaction signal?',
        history=[dict(path=str(MAT/'independent_half_signal.json'),reuse='Original40state signal across train+VAL; reuse verbatim. No need regenerate.'),
                 dict(path=str(NEW/'matrix_results.json'),reuse='AlljointVAL and allheldfold results; no new inference.'),
                 dict(path=str(NEW/'fits'),reuse='Saved SOURCE TRAIN predictions, histories and sampling audits;32crossedstates distinct from512native states.'),
                 dict(path=str(ROOT/'orthoflow3_critic_rootcause_resolution_v1/hypothesis_status.json'),reuse='TT/H20 state breadth/function breadth/goal-motion alias already tested. Different FF program/currentstaticmoving-goalinput; not relabelled as currentFF evidence.'),
                 dict(path=str(ROOT/'controller_program_input_probe_v1/input_artifact_audit.json'),reuse='Oldknownparameter/initializer tests; no repeat.')],
        missing='Old signal pooled TRAIN32 and VAL8; no current native20 model/source-split matched signal/learning audit. This is recomputation only, not an additional training experiment.',
        frozen_scope='Every four-scene goal/Flow reference, separate TRAIN32 and VAL8; allprefixedeta0,5,10,15. No scene/state/eta chosen by favorable result.',
        method='Independent seed halves, same existing censor-aware covariance estimator,20kphysicalstate bootstraps. Model train performance descriptive; sourceVAL was checkpoint-selected, so favorable sourceVAL is not independent confirmation.',
        reliability='Midpoint sensitivity:2*cross-half centered covariance/(halfAvariance+halfBvariance). Unknowns retained as bounds for signal; midpoint reliability cannot establish censor-robust proof.',
        noise_boundary='Cross-half signal/noise describes these labels; does not prove that all numerical censoring or label noise is harmless. Train-label model error is not an unbiased generalization estimate.',
        decisions=['If reproducible signal absent,prune blind expansion ofthatbranch;do not callmodel failure a learning proof.',
                   'If sourceTRAIN fit strong but same-program unseen-state interaction fails,prioritize jointstate-support/learning before claiming a purely cross-scene missing-controller issue.',
                   'Do not generate newdata merely because countsmall; require a controlled contrast with an already-audited cache gap. No architecture/loss/sampling sweep.'],
        literature=[dict(url='https://arxiv.org/abs/2002.11833',use='PolicyEvaluationNetworks train from policy-return supervision; fingerprinting and policy support are distinct. Same-MDPpolicygeneralization does not certify different-mapstate-conditionedselection.'),
                    dict(url='https://arxiv.org/abs/2207.01566',use='Learnedprobingstates are an existing method; no blindadditionofprobes until current support audit is complete.')],
        new_fits=0,new_queries=0,new_rollouts=0,code_sha256=sha(__file__))
    dest=HERE/'protocol.json'
    if dest.exists():assert read(dest)==p
    else:write(dest,p)


def run():
    from diagnostics.controller_flow_endpoint_probe_v1.analyze import energy,centered
    p=read(HERE/'protocol.json');assert p['code_sha256']==sha(__file__)
    d=dict(np.load(MAT/'dataset.npz'));raw=dict(np.load(MAT/'raw_labels.npz'))
    u=dict(np.load(UNION/'dataset.npz'));ur=np.flatnonzero(u['source']=='matrix_goal_and_flow32');ur=ur[np.argsort(u['source_row'][ur])]
    np.testing.assert_array_equal(u['state_uid'][ur],d['state_uid'])
    allq={a:np.mean([np.load(NEW/'fits/joint'/a/f'seed{s}/predictions.npz')['probability'][ur][:,[0,5,10,15]] for s in SEEDS],0)
          for a in ('eta_only','union_native_slots','uniform_eta_only','uniform_native_slots')}
    rows=[];arr={};cases={};coverage=[]
    sampling=read(NEW/'fits/joint/union_native_slots/seed17/sampling_audit.json')
    for sc in np.unique(u['scene']):
        mask=(u['scene']==sc)&(u['split']=='train');cs=np.unique(u['controller'][mask]);native='native_flow' in cs
        for ctl in cs:
            ii=np.flatnonzero(mask&(u['controller']==ctl));observed=(u['success']+u['failure']+u['unknown'])[ii]>0
            mass=.2 if native and ctl=='native_flow' else .8/(len(cs)-1) if native else 1/len(cs)
            coverage.append(dict(scene=str(sc),controller=str(ctl),physical_states=len(ii),observed_state_eta_pairs=int(observed.sum()),
                rollout_count=int(sum(u[k][ii].sum() for k in ('success','failure','unknown'))),eta_observed_per_state=np.unique(observed.sum(1)).tolist(),
                within_scene_sampling_mass=float(mass),expected_draws_per_state=float(6000*120*mass/len(ii)),
                expected_draws_per_observed_cell_approx=float(6000*120*mass/observed.sum())))
    for sc in sorted(set(d['scene'])):
        for ref in ('goal','flow'):
            for split in ('train','validation'):
                ids=np.flatnonzero((d['scene']==sc)&(d['kind']==ref)&(d['split']==split))
                ids=np.array(sorted(ids,key=lambda k:(d['state_uid'][k],d['controller'][k])))
                states=np.unique(d['state_uid'][ids]);n=len(states);shape=(n,5,4)
                ss,uu,oo=[raw[k][ids][:,[0,5,10,15]].reshape(shape+(4,)) for k in ('success','unknown','observed')]
                ns=np.unique(oo.sum(-1));assert len(ns)==1;nt=int(ns[0]);assert nt in (2,4)
                lo=np.stack([ss[...,:nt//2].mean(-1),ss[...,nt//2:nt].mean(-1)],-1)
                hi=np.stack([(ss+uu)[...,:nt//2].mean(-1),(ss+uu)[...,nt//2:nt].mean(-1)],-1)
                mid=(lo+hi)/2;truth=mid.mean(-1);truthlo=lo.mean(-1);truthhi=hi.mean(-1)
                boot=np.random.default_rng(2026102301+n).integers(n,size=(20000,n))
                pref=f'{sc}__{ref}__{split}';arr[pref+'__bootstrap_indices']=boot
                cases[pref+'__states']=states;cases[pref+'__lower_halves']=lo;cases[pref+'__upper_halves']=hi
                for label,axes in [('state_eta',[0,2]),('state_controller_eta',[0,1,2])]:
                    point,lower,upper=energy(lo,hi,axes);bl=[];bu=[];rel=[]
                    for start in range(0,len(boot),500):
                        ib=boot[start:start+500];v,l,h=energy(lo[ib],hi[ib],[a+1 for a in axes]);bl.append(l);bu.append(h)
                        aa=centered(mid[ib,...,0],[a+1 for a in axes]);bb=centered(mid[ib,...,1],[a+1 for a in axes]);den=(aa*aa+bb*bb).mean((1,2,3))
                        rel.append(np.divide(2*v,den,out=np.full_like(v,np.nan),where=den>1e-12))
                    bl,bu,rel=map(np.concatenate,(bl,bu,rel));key=pref+'__'+label
                    arr[key+'__energy_lower']=bl;arr[key+'__energy_upper']=bu;arr[key+'__reliability_midpoint']=rel
                    aa,bb=[centered(mid[...,h],axes) for h in (0,1)];den=np.mean(aa*aa+bb*bb)
                    row=dict(scene=str(sc),reference=ref,split=split,interaction=label,N=n,controllers=5,K=4,trials_per_cell=nt,
                        unknown=int(uu.sum()),cross_half_energy_midpoint=float(point),energy_identification_bounds=[float(lower),float(upper)],
                        energy_censor_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bu,.975))],
                        reliability_midpoint=float(2*point/den) if den>1e-12 else None,
                        reliability_midpoint_CI95=np.nanquantile(rel,[.025,.975]).tolist() if np.isfinite(rel).any() else None)
                    model=[]
                    for arm,fullq in allq.items():
                        q=fullq[ids].reshape(shape);x=centered(q,axes);target=centered(truth,axes)
                        low=2*x*truthlo-x*x;high=2*x*truthhi-x*x
                        mse=np.mean((q-truth)**2);imse=np.mean((x-target)**2)
                        corr=float(np.corrcoef(x.ravel(),target.ravel())[0,1]) if min(x.std(),target.std())>1e-7 else None
                        model.append(dict(arm=arm,Q_MSE_midpoint=float(mse),interaction_MSE_midpoint=float(imse),interaction_corr_midpoint=corr,
                            interaction_skill_identification_bounds=[float(np.minimum(low,high).mean()),float(np.maximum(low,high).mean())]))
                    row['models']=model;rows.append(row)
    # Complete histories retained; summarize overfit in the predefined Flow branch without selecting another checkpoint.
    training=[]
    for seed in SEEDS:
        z=read(NEW/'fits/joint/union_native_slots'/f'seed{seed}/complete.json')
        for split,m in [('train',z['source_train_metrics']),('validation',z['selected'])]:
            for group in m['groups']:
                training.append(dict(seed=seed,split=split,selected_step=z['selected']['step'],**group))
    write(HERE/'results.json',dict(rows=rows,coverage=coverage,source_metrics=training,new_fits=0,new_queries=0,new_rollouts=0,
        full40state_signal_reused=read(MAT/'independent_half_signal.json'),
        limits='SOURCE-only diagnostic; sourceVAL checkpoint optimism retained. Censored midpoint correlations/reliability are sensitivity values. Does not prove input sufficient or all remaining causes enumerated.'))
    np.savez_compressed(HERE/'bootstrap.npz',**arr);np.savez_compressed(HERE/'raw_halves_and_state_ids.npz',**cases)
    for name,rr in [('coverage.csv',coverage),('source_metrics.csv',training)]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    print('Saved source-split interaction reliability, current predictions, exact independent coverage. No new fit/query/rollout.')


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);args=a.parse_args();globals()[args.stage]()
