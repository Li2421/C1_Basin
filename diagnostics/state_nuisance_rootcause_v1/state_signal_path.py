"""Trace genuine state x eta signal through existing labels and predictions.

Retrospective cached-data diagnostic. No rollout, training, or TEST selection.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import json
import sqlite3
import hashlib
import csv
from itertools import combinations
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = HERE.parent / 'orthoflow3_tt_ff_matched_learnability_v1'
OUT = HERE / 'state_signal_path'
SEEDS = (17, 23, 41)


def expit(x):
    from scipy.special import expit as sigmoid
    return sigmoid(x)


def csvout(path, rows):
    from diagnostics.state_nuisance_rootcause_v1.analyze import csvout as save_csv
    save_csv(path, rows)


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collect():
    OUT.mkdir(exist_ok=True)
    write(OUT / 'protocol.json', dict(
        scope='Read-only retrospective signal-path diagnosis on previously opened matched Toy/Ring cohorts',
        new_rollouts=0, new_training=0, model_selection='None; frozen source-selected and all prior Ring FF checkpoints',
        seed_audit='Exact state/eta/controller/root/namespace; original TRAIN uses future indices0:4; extra4:16 never used by these fits',
        interaction='Remove additive state and eta effects descriptively within each split, never refit predictions',
        reversal='Both states prefer opposite eta with guaranteed empirical Q gap at least0.25; numerical/missing outcomes remain unknown',
        limitations=['Same-state extra seeds assess label memorization, not unseen-state generalization',
                     'Crossed reversal cases overlap and are not independent observations',
                     'No inference of the unique or universal root cause from one matched panel']))
    d = np.load(SOURCE / 'dataset.npz')
    p = read(SOURCE / 'protocol.json')
    records = read(SOURCE / 'canonical_source_keys.json')
    con = sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    # Last axis: success, numerical failure, collision. -1 is not present.
    outcomes = np.full((2, 112, 16, 16, 3), -1, np.int8)
    uids = []
    for r in records:
        ci = ('TT', 'FF').index(r['chain'])
        i, j = r['state_index'], r['eta_index']
        rr = con.execute('SELECT rollout_uid,seed_key,success,numerical_failure,collision,compatibility_quality,conflict_quarantined FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                         (r['state_uid'], r['eta_uid'], r['controller_uid'])).fetchall()
        requested = set(r['all_requested_rollout_uids'])
        found = set()
        for row in rr:
            key = json.loads(row['seed_key'])
            if key.get('future_root') != 2026100403 or key.get('rng_namespace') != p['states'][i]['rng_namespace']:
                continue
            k = key.get('future_index', -1)
            if not 0 <= k < 16 or row['conflict_quarantined'] or row['compatibility_quality'] != 'EXACT_REUSE':
                continue
            assert (outcomes[ci, i, j, k] == -1).all()
            outcomes[ci, i, j, k] = [row[x] for x in ('success', 'numerical_failure', 'collision')]
            uids.append(row['rollout_uid'])
            found.add(row['rollout_uid'])
        assert requested <= found, (i, j, ci)
    con.close()
    truth = np.load(SOURCE / 'test_truth.npz')
    outcomes[:, truth['indices']] = truth['outcomes']
    coverage = []
    for scene in ('toy_give_way', 'ring_exchange'):
        for ci, chain in enumerate(('TT', 'FF')):
            for split in ('train', 'validation', 'test'):
                ix = np.flatnonzero((d['scene'] == scene) & (d['split'] == split))
                for name, sl in [('original', slice(0, 4 if split == 'train' else 16)), ('extra12', slice(4, 16))]:
                    oo = outcomes[ci, ix, :, sl]
                    present = oo[..., 0] >= 0
                    coverage.append(dict(scene=scene, chain=chain, split=split, seeds=name,
                                         expected=int(present.size), present=int(present.sum()),
                                         complete_cells=int(present.all(-1).sum()),
                                         numerical=int((oo[..., 1] == 1).sum())))
                if split != 'test':
                    k = 4 if split == 'train' else 16
                    oo = outcomes[:, ix, :, :k]
                    valid = (oo[0, ..., 1] == 0) & (oo[1, ..., 1] == 0)
                    np.testing.assert_array_equal((oo[ci, ..., 0] * valid).sum(-1), d['success'][ci, ix])
                    np.testing.assert_array_equal(valid.sum(-1) - d['success'][ci, ix], d['failure'][ci, ix])
    np.savez_compressed(OUT / 'exact_outcomes.npz', outcomes=outcomes)
    csvout(OUT / 'cache_coverage.csv', coverage)
    write(OUT / 'cache_manifest.json', dict(coverage=coverage, queried_source_cells=len(records),
          accepted_DB_records=len(uids), accepted_uid_set_sha256=hashlib.sha256('\n'.join(sorted(uids)).encode()).hexdigest(),
          dataset_sha256=sha(SOURCE/'dataset.npz'), protocol_sha256=sha(OUT/'protocol.json'),
          source_key_sha256=sha(SOURCE/'canonical_source_keys.json'), original_count_parity=True))
    print(json.dumps([r for r in coverage if r['split']=='train' and r['seeds']=='extra12'],indent=2))


def counts(o, common=None):
    valid = (o[..., 1] == 0) & (o[..., 0] >= 0)
    if common is not None:
        valid &= common
    s = ((o[..., 0] == 1) & valid).sum(-1)
    f = ((o[..., 0] == 0) & valid).sum(-1)
    return s, f, o.shape[-2]-s-f


def residual(q, mask):
    ii, jj = np.nonzero(mask)
    x = np.concatenate([np.eye(q.shape[0])[ii], np.eye(q.shape[1])[jj]], 1)
    a = q[mask]
    return a - x @ np.linalg.lstsq(x, a, rcond=None)[0]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else None


def reversals(s, f, u, k):
    lo, hi = s/k, (s+u)/k
    cases=[]
    for i, j in combinations(range(len(s)), 2):
        for a, b in combinations(range(s.shape[1]), 2):
            if lo[i,a]-hi[i,b] >= .25 and lo[j,b]-hi[j,a] >= .25:
                cases.append((i,j,a,b,1))
            elif lo[i,b]-hi[i,a] >= .25 and lo[j,a]-hi[j,b] >= .25:
                cases.append((i,j,a,b,-1))
    return np.array(cases, dtype=int).reshape(-1,5)


def assess(z, s, f, u, k, cases, selection=True):
    pp=expit(np.asarray(z,float)); mask=s+f>0
    q=np.divide(s,s+f,out=np.zeros_like(s,dtype=float),where=mask)
    exact=u==0
    qr, pr=residual(q,exact),residual(pp,exact)
    n=s+f
    result=dict(NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/n.sum()),
                interaction_rho=corr(pr,qr),
                interaction_R2=float(1-((pr-qr)**2).mean()/np.mean(qr**2)),
                predicted_interaction_sd=float(np.std(pr)),observed_interaction_sd=float(np.std(qr)),
                eligible_reversal_cases=len(cases))
    if len(cases):
        i,j,a,b,sign=cases.T
        d1=(z[i,a]-z[i,b])*sign
        d2=(z[j,b]-z[j,a])*sign
        result.update(reversal_both_correct=float(((d1>0)&(d2>0)).mean()),
                      reversal_direction_accuracy=float(np.r_[d1>0,d2>0].mean()))
    result.update(evaluated_cells=int(mask.sum()),exact_cells=int(exact.sum()))
    if selection:
        chosen=z.argmax(1); ar=np.arange(len(s))
        result.update(selected_Q_lower=float((s[ar,chosen]/k).mean()),
                      selected_Q_upper=float(((s+u)[ar,chosen]/k).mean()),
                      distinct_selected_eta=len(np.unique(chosen)))
        if k==16:
            result['B15']=int((s[ar,chosen]>=15).sum())
    return result


def labels_and_predictions():
    d=np.load(SOURCE/'dataset.npz');outcomes=np.load(OUT/'exact_outcomes.npz')['outcomes']
    frozen=np.load(HERE/'replayed_logits.npz')
    late=np.load(HERE/'checkpoint_overfitting/test_logits.npz')
    rows=[];repeat=[];structure=[];casebank={}
    for scene in ('toy_give_way','ring_exchange'):
        for ci,chain in enumerate(('TT','FF')):
            for split in ('train','validation','test'):
                ix=np.flatnonzero((d['scene']==scene)&(d['split']==split))
                oo=outcomes[ci,ix]
                settings=[('train4',slice(0,4)),('train4_extra_subset',slice(0,4)),('extra12',slice(4,16))] if split=='train' else [(split+'16',slice(0,16))]
                for target,sl in settings:
                    o=oo[:,:,sl];k=o.shape[-2];s,f,u=counts(o)
                    if target=='train4_extra_subset':
                        present=(oo[:,:,4:,0]>=0).all(-1)
                        s=np.where(present,s,0);f=np.where(present,f,0);u=k-s-f
                    if not (s+f).any():
                        continue
                    cases=reversals(s,f,u,k)
                    key=f'{scene}__{chain}__{target}';casebank[key]=cases
                    complete=u==0
                    q=np.divide(s,s+f,out=np.zeros_like(s,dtype=float),where=(s+f)>0)
                    qr=residual(q,complete)
                    fixed=int(np.argmax(d['success'][ci,np.flatnonzero((d['scene']==scene)&(d['split']=='train'))].sum(0)/
                                         (d['success'][ci,np.flatnonzero((d['scene']==scene)&(d['split']=='train'))]+
                                          d['failure'][ci,np.flatnonzero((d['scene']==scene)&(d['split']=='train'))]).sum(0)))
                    st=dict(scene=scene,chain=chain,target=target,states=len(ix),complete_cells=int(complete.sum()),
                            reversal_cases=len(cases),reversal_state_pairs=len(set(map(tuple,cases[:,:2]))),
                            interaction_variance=float(np.mean(qr**2)),train_fixed_eta=fixed)
                    selection=target not in ('train4_extra_subset','extra12')
                    if selection:
                        st.update(fixed_Q_lower=float((s[:,fixed]/k).mean()),oracle_Q_lower=float((s/k).max(1).mean()),
                                  exact_best_ties_mean=float(((q==q.max(1,keepdims=True))&complete).sum(1).mean()))
                    if k==16:
                        st.update(fixed_B15=int((s[:,fixed]>=15).sum()),oracle_B15=int((s>=15).any(1).sum()),
                                  B15_coverage_by_eta=(s>=15).sum(0).tolist())
                    structure.append(st)
                    for seed in SEEDS:
                        for kind in ('eta_only','full_context'):
                            z=frozen[f'{scene}__{chain}__{kind}__{seed}__{split}']
                            rows.append(dict(scene=scene,chain=chain,split=split,target=target,method='original_'+kind,seed=seed,step='selected',
                                             **assess(z,s,f,u,k,cases,selection)))
                        if scene=='ring_exchange' and chain=='FF':
                            for arm in ('full_replay','constant_h','constant_context','constant_all'):
                                for step in range(100,2001,100):
                                    z=late[f'{arm}__{seed}__{step}'] if split=='test' else np.load(HERE/'fits'/arm/f'seed{seed}'/f'step{step}_source.npz')[split]
                                    rows.append(dict(scene=scene,chain=chain,split=split,target=target,method=arm,seed=seed,step=step,
                                                     **assess(z,s,f,u,k,cases,selection)))
                # Independent halves assess reliability of conditional interactions.
                halves=[('2+2',slice(0,2),slice(2,4)),('4+12',slice(0,4),slice(4,16))] if split=='train' else [('8+8',slice(0,8),slice(8,16))]
                for label,a,b in halves:
                    sa,fa,ua=counts(oo[:,:,a]);sb,fb,ub=counts(oo[:,:,b]);mask=(ua==0)&(ub==0)
                    if not mask.any():
                        continue
                    qa=sa/oo[:,:,a].shape[-2];qb=sb/oo[:,:,b].shape[-2]
                    ra,rb=residual(qa,mask),residual(qb,mask)
                    ca,cb=reversals(sa,fa,ua,oo[:,:,a].shape[-2]),reversals(sb,fb,ub,oo[:,:,b].shape[-2])
                    entry=dict(scene=scene,chain=chain,split=split,halves=label,complete_cells=int(mask.sum()),
                               interaction_correlation=corr(ra,rb),cross_half_covariance=float(np.mean(ra*rb)),
                               first_half_reversals=len(ca),second_half_reversals=len(cb))
                    if len(ca):
                        i,j,x,y,sgn=ca.T
                        valid=mask[i,x]&mask[i,y]&mask[j,x]&mask[j,y]
                        direction=(qb[i,x]-qb[i,y])*sgn
                        other=(qb[j,y]-qb[j,x])*sgn
                        entry.update(first_reversals_with_complete_second=int(valid.sum()),
                                     opposite_half_confirms_both_strict=float(((direction[valid]>0)&(other[valid]>0)).mean()),
                                     opposite_half_both_still_gap025=float(((direction[valid]>=.25)&(other[valid]>=.25)).mean()))
                    repeat.append(entry)
    csvout(OUT/'interaction_metrics.csv',rows)
    write(OUT/'label_repeatability.json',repeat)
    write(OUT/'data_structure.json',structure)
    np.savez_compressed(OUT/'reversal_cases.npz',**casebank)
    for target in ('train4','extra12','validation16','test16'):
        for step in (100,500,2000):
            rr=[r for r in rows if r['target']==target and r['method']=='full_replay' and r['step']==step]
            print(target,step,{k:round(float(np.mean([r[k] for r in rr])),4) for k in ['NLL','interaction_rho','interaction_R2','reversal_both_correct']})
    print('TRAIN repeatability',json.dumps([r for r in repeat if r['split']=='train'],indent=2))


def block_export():
    """Read-only model perturbations; preserve each eta column, shuffle states."""
    from diagnostics.state_nuisance_rootcause_v1.experiment import setup
    jax,jnp,serialization,model,d,e,_=setup()
    conditions=('correct','h_shuffle','H20_shuffle','goal_shuffle','context_shuffle','all_state_shuffle')
    write(OUT/'block_protocol.json',dict(scene='ring_exchange',chain='FF',steps=[100,500,2000],seeds=list(SEEDS),
          conditions=list(conditions),permutations=10,group_definitions={'H20':[0,40],'goal':[40,72]},
          design='Same donor state per receiver across all eta; deterministic ten permutations shared by model seeds and steps',
          caveat='Feature shuffling can break physical consistency. Sensitivity diagnostic, not proof that an input should be removed.',
          new_training=0,new_rollouts=0,checkpoint_selection='None'))
    fun=jax.jit(model('full_context').apply);result={};manifests=[]
    for split in ('train','validation','test'):
        ix=np.flatnonzero((d['scene']=='ring_exchange')&(d['split']==split))
        ii,jj=np.repeat(ix,16),np.tile(np.arange(16),len(ix))
        c=d['context'][1,ii,jj]
        rng=np.random.default_rng(202610079)
        perm=np.stack([rng.permutation(len(ix)) for _ in range(10)])
        result[f'permutation__{split}']=perm
        # Compile one batch shape for all treatments and splits.
        for seed in SEEDS:
            for step in (100,500,2000):
                ck=HERE/'fits/full_replay'/f'seed{seed}'/f'step{step}.msgpack'
                pp=serialization.msgpack_restore(ck.read_bytes())
                for condition in conditions:
                    values=[]
                    for permutation in (perm[:1] if condition=='correct' else perm):
                        di=np.repeat(ix[permutation],16)
                        cc=c.copy()
                        if condition in ('H20_shuffle','context_shuffle','all_state_shuffle'):
                            cc[:,:40]=d['context'][1,di,jj,:40]
                        if condition in ('goal_shuffle','context_shuffle','all_state_shuffle'):
                            cc[:,40:72]=d['context'][1,di,jj,40:72]
                        if condition in ('context_shuffle','all_state_shuffle'):
                            cc[:,72:]=d['context'][1,di,jj,72:]
                        hi=di if condition in ('h_shuffle','all_state_shuffle') else ii
                        z=[]
                        for start in range(0,len(ii),128):
                            sl=slice(start,start+128)
                            z.append(np.asarray(fun(pp,{k:jnp.asarray(v[hi[sl]]) for k,v in e.items()},
                                jnp.asarray(d['eta'][ii[sl],jj[sl]]),jnp.asarray(cc[sl]))))
                        values.append(np.concatenate(z).reshape(len(ix),16))
                    result[f'{split}__{seed}__{step}__{condition}']=np.stack(values)
                manifests.append(dict(split=split,seed=seed,step=step,checkpoint_sha256=sha(ck)))
                print('exported blocks',split,seed,step,flush=True)
    np.savez_compressed(OUT/'block_logits.npz',**result)
    write(OUT/'block_manifest.json',dict(backend=jax.default_backend(),checkpoints=manifests,
          dataset_sha256=sha(SOURCE/'dataset.npz'),script_sha256=sha(__file__),protocol_sha256=sha(OUT/'block_protocol.json')))


def block_analyze():
    d=np.load(SOURCE/'dataset.npz');o=np.load(OUT/'exact_outcomes.npz')['outcomes']
    z=np.load(OUT/'block_logits.npz');rows=[]
    baseline=np.load(HERE/'checkpoint_overfitting/test_logits.npz')
    for key in z.files:
        if key.startswith('permutation'):continue
        split,seed,step,condition=key.split('__');seed,step=int(seed),int(step)
        ix=np.flatnonzero((d['scene']=='ring_exchange')&(d['split']==split))
        k=4 if split=='train' else 16
        s,f,u=counts(o[1,ix,:,:k]);cases=reversals(s,f,u,k)
        correct=z[f'{split}__{seed}__{step}__correct'][0]
        if split=='test':
            assert abs(expit(correct)-expit(baseline[f'full_replay__{seed}__{step}'])).max()<.002
        for rep,zz in enumerate(z[key]):
            m=assess(zz,s,f,u,k,cases)
            m.update(mean_abs_probability_change=float(abs(expit(zz)-expit(correct)).mean()),
                     mean_abs_logit_change=float(abs(zz-correct).mean()),
                     top1_changed=int((zz.argmax(1)!=correct.argmax(1)).sum()))
            rows.append(dict(split=split,seed=seed,step=step,condition=condition,permutation=rep,**m))
    csvout(OUT/'block_metrics.csv',rows)
    for step in (100,500,2000):
        for condition in ('correct','h_shuffle','H20_shuffle','goal_shuffle','context_shuffle','all_state_shuffle'):
            rr=[r for r in rows if r['step']==step and r['condition']==condition and r['split']=='test']
            print(step,condition,{k:round(float(np.mean([r[k] for r in rr])),4) for k in ('NLL','interaction_rho','reversal_both_correct','B15','mean_abs_probability_change')})


def summarize():
    rows=list(csv.DictReader((OUT/'interaction_metrics.csv').open()))
    blocks=list(csv.DictReader((OUT/'block_metrics.csv').open()))
    def means(rr,keys):
        return {k:float(np.mean([float(r[k]) for r in rr])) for k in keys}
    keys=('NLL','interaction_rho','interaction_R2','reversal_both_correct','reversal_direction_accuracy')
    summary=dict(original_full={},ring_FF_trajectory={},block_sensitivity={},label_extremes=[],witnesses=[])
    for scene in ('toy_give_way','ring_exchange'):
        for chain in ('TT','FF'):
            summary['original_full'][scene+':'+chain]={target:means([r for r in rows if r['scene']==scene and r['chain']==chain and r['method']=='original_full_context' and r['target']==target],keys)
                for target in ('train4','validation16','test16')}
    for step in (100,500,2000):
        summary['ring_FF_trajectory'][str(step)]={target:means([r for r in rows if r['method']=='full_replay' and r['step']==str(step) and r['target']==target],keys)
            for target in ('train4','train4_extra_subset','extra12','validation16','test16')}
        summary['block_sensitivity'][str(step)]={condition:means([r for r in blocks if r['split']=='test' and r['step']==str(step) and r['condition']==condition],
              (*keys,'mean_abs_probability_change','B15'))
            for condition in ('correct','h_shuffle','H20_shuffle','goal_shuffle','context_shuffle','all_state_shuffle')}
    d=np.load(SOURCE/'dataset.npz');o=np.load(OUT/'exact_outcomes.npz')['outcomes']
    protocol=read(SOURCE/'protocol.json');zz=np.load(HERE/'checkpoint_overfitting/test_logits.npz')
    cal=list(csv.DictReader((HERE/'checkpoint_overfitting/temperature_fits.csv').open()))
    calibrated=[]
    for scene in ('toy_give_way','ring_exchange'):
        ix=np.flatnonzero((d['scene']==scene)&(d['split']=='train'))
        for ci,chain in enumerate(('TT','FF')):
            # These are the ACTUAL common-valid targets used for fitting.
            s,f=d['success'][ci,ix],d['failure'][ci,ix];n=s+f
            ss,ff,uu=counts(o[ci,ix,:,:4]);es,ef,eu=counts(o[ci,ix,:,4:]);extra=(eu==0)&(uu==0)
            summary['label_extremes'].append(dict(scene=scene,chain=chain,
                original_training_pairs=int((n>0).sum()),original_hard_0_or_1_targets=int((((s==0)|(f==0))&(n>0)).sum()),
                additional_complete_cells=int(extra.sum()),additional_states=int(extra.any(1).sum()),
                four_successes_with_extra=int(((ss==4)&extra).sum()),four_successes_later_failed=int(((ss==4)&extra&(ef>0)).sum()),
                four_failures_with_extra=int(((ff==4)&extra).sum()),four_failures_later_succeeded=int(((ff==4)&extra&(es>0)).sum())))
    for split in ('train','validation','test'):
        ix=np.flatnonzero((d['scene']=='ring_exchange')&(d['split']==split));k=4 if split=='train' else 16
        s,f,u=counts(o[1,ix,:,:k]);cases=reversals(s,f,u,k)
        for step in (100,500,2000):
            temperature=float(next(r['temperature'] for r in cal if r['arm']=='full_replay' and int(r['step'])==step))
            for seed in SEEDS:
                z=zz[f'full_replay__{seed}__{step}'] if split=='test' else np.load(HERE/'fits/full_replay'/f'seed{seed}'/f'step{step}_source.npz')[split]
                calibrated.append(dict(split=split,step=step,seed=seed,temperature=temperature,**assess(z/temperature,s,f,u,k,cases)))
    csvout(OUT/'calibrated_interaction_metrics.csv',calibrated)
    for scene,chain,inds,eta in [('ring_exchange','FF',[1,3],[1,4]),('toy_give_way','TT',[0,1],[3,4])]:
        ci=('TT','FF').index(chain);ix=np.flatnonzero((d['scene']==scene)&(d['split']=='test'))
        oo=o[ci,ix][inds][:,eta];assert (oo[...,1]==0).all()
        w=dict(scene=scene,chain=chain,state_indices=ix[inds].tolist(),state_aliases=[protocol['states'][int(i)]['uid'] for i in ix[inds]],
               eta_indices=eta,eta_values=[protocol['eta'][j] for j in eta],success_out_of16=oo[...,0].sum(-1).tolist(),predictions={})
        if scene=='ring_exchange':
            for step in (100,500,2000):
                temperature=float(next(r['temperature'] for r in cal if r['arm']=='full_replay' and int(r['step'])==step))
                logits=np.array([zz[f'full_replay__{seed}__{step}'][inds][:,eta] for seed in SEEDS],float)
                w['predictions'][str(step)]=dict(mean_probability=expit(logits).mean(0).tolist(),
                    VAL_calibrated_mean_probability=expit(logits/temperature).mean(0).tolist(),
                    both_correct_by_seed=((logits[:,0,0]<logits[:,0,1])&(logits[:,1,0]>logits[:,1,1])).tolist())
        summary['witnesses'].append(w)
    ix=np.flatnonzero((d['scene']=='ring_exchange')&(d['split']=='test'));s,f,u=counts(o[1,ix]);cases=reversals(s,f,u,16)
    i,j,a,b,sgn=cases.T
    correctness={}
    for step in (100,2000):
        correctness[step]=np.mean([((zz[f'full_replay__{seed}__{step}'][i,a]-zz[f'full_replay__{seed}__{step}'][i,b])*sgn>0)&
                                  ((zz[f'full_replay__{seed}__{step}'][j,b]-zz[f'full_replay__{seed}__{step}'][j,a])*sgn>0) for seed in SEEDS],0)
    # Resample whole physical families, never treat overlapping quadruples or model seeds as independent N.
    mult=np.random.default_rng(2026100710).multinomial(len(ix),np.full(len(ix),1/len(ix)),size=10000)
    weights=mult[:,i]*mult[:,j];denom=weights.sum(1);good=denom>0
    change=correctness[2000]-correctness[100]
    summary['test_reversal_change_bootstrap']=dict(independent_families=len(ix),overlapping_cases=len(cases),
        mean=float(change.mean()),family_bootstrap95=np.quantile((weights[good]@change)/denom[good],[.025,.975]).tolist(),
        method='State multiplicity product for distinct-state reversal cases; model seeds averaged before family resampling')
    summary['scope']=dict(no_new_training=True,no_new_rollouts=True,no_TEST_checkpoint_selection=True,
        input_frozen=True,interaction_NLL='Own-chain nonnumerical outcomes; slight difference from original common-valid TT/FF NLL',
        original_common_valid_NLL_reference='checkpoint_overfitting/summary.json',
        extra_seed_limit='Ring FF only64 cached cells across8 TRAIN states; Toy FF96 cells across12; historically selected subset, not a representative new sample',
        fixed_eta_limit='B15 robust-coverage ceiling on this candidate bank differs from maximizing continuous Q',
        interpretation='Real state dependence, partial learned generalization, and probability overconfidence coexist; do not infer that state is useless')
    write(OUT/'summary.json',summary)
    manifest=read(SOURCE/'dataset_frozen.json')
    source_runs=read(SOURCE/'models_frozen.json')['runs']
    checkpoints=read(OUT/'block_manifest.json')['checkpoints']
    original_models_unchanged=all(sha(r['checkpoint'])==r['checkpoint_sha256'] for r in source_runs)
    intervention_models_unchanged=all(sha(HERE/'fits/full_replay'/f"seed{r['seed']}"/f"step{r['step']}.msgpack")==r['checkpoint_sha256'] for r in checkpoints)
    verification=dict(source_dataset_hash_matches=sha(SOURCE/'dataset.npz')==read(OUT/'cache_manifest.json')['dataset_sha256'],
          source_entities_hash_matches=sha(SOURCE/'entities.npz')==manifest['entities_sha256'],
          original_models_unchanged=original_models_unchanged,intervention_models_unchanged=intervention_models_unchanged,
          original_counts_exactly_reconstructed=read(OUT/'cache_manifest.json')['original_count_parity'],
          input_model_not_modified=True,new_training=0,new_rollouts=0,
          script_sha256=sha(__file__),outputs={name:sha(OUT/name) for name in ('exact_outcomes.npz','interaction_metrics.csv','block_logits.npz','block_metrics.csv','summary.json')})
    assert all(verification[k] for k in ('source_dataset_hash_matches','source_entities_hash_matches','original_models_unchanged','intervention_models_unchanged'))
    write(OUT/'verification.json',verification)
    report='''State 与 eta 的真实关系，为什么没有充分体现在当前学习中

结论：state 的影响是真实的。当前证据指向 Ring 的 state×eta 映射泛化不足、概率过度自信与 checkpoint 选择偏差共同存在；不能归因为 state 本身只是噪声。

本轮只读取现有数据与 checkpoint，无新训练或 rollout。主诊断 Ring FF；Toy TT 是正对照。所有 TEST 都是已打开的回顾性诊断，未按 TEST 重选模型。

1. 先确认数据中有可靠的交互，而非只看总成功率。
   去除 state 难度与 eta 平均优劣后，Ring FF TEST 的两半8-seed标签交互相关约0.856；TRAIN两半2-seed相关0.734。
   一个直接例子（0起始eta索引）：TEST state945630001 的 eta1/eta4 =3/16、16/16；state945630003 为15/16、8/16。
   因此不看 state 的固定排序无法同时答对；这不等于两者必为全部16候选中的全局最优。

2. 模型能拟合训练集的反转，但跨 state 的迁移明显不足。
   强反转定义：两个 state 对同一对 eta 的偏好相反，两边 Q 差均至少0.25；数值异常按未知区间处理。
   指标要求两边都排对，三模型seed取均值。重叠反转组不是独立样本。
   Ring FF：step100 TRAIN41.9%、VAL8.0%、TEST10.3%；step500为93.8%、44.2%、48.0%；step2000为99.1%、57.2%、59.9%。
   对应 TEST 交互相关0.184→0.326；VAL-only温度校准后的step2000交互相关0.419。
   正对照 Toy TT，原选定模型在新 state 的强反转两边同时答对95.0%、交互相关0.855，说明当前方法并非普遍不能使用state。
   上述 Ring 具体反转，在step100/500/2000的全部三seed仍然两边都偏好eta4；模型晚期的错误更自信。

3. 区分记标签噪声与跨state泛化。
   原始 Ring FF 只有32个独立TRAIN state×16eta，512对；443对训练目标恰为0或1（每对最多4个有效seed）。
   缓存里额外12seed只覆盖其中8个state、64对，不能外推代表全体。这个子集原4seed与额外12seed的交互相关0.909。
   在同一子集，step100→500→2000的原4seed NLL为0.287→0.103→0.064，额外12seed为0.296→0.205→0.263。
   晚期确实开始拟合有限seed噪声，但在额外seed上的交互相关仍为0.899；换新state只有约0.326，主要缺口不能只用标签噪声解释。
   缓存子集24个原先4/4成功的输入对里，6个在额外seed中出现失败；因此4/4不意味着真实成功概率为1。

4. 模型主要通过 response context 获取state信息，显式物理state分支影响小。
   在固定step2000模型上，同eta跨state置换输入；10个预定置换、三seed平均：
   原输入强反转正确率59.9%；只换物理h为60.1%（预测概率平均变化0.25个百分点）；
   换H20短轨迹响应为51.2%；换目标附近控制响应为20.6%；把h与全部context一起换为17.2%。
   目标附近响应包含初态到目标的方向等state信息。它有真实作用，不能把它等同于纯噪声。
   置换会打破特征间物理一致性，只是依赖性诊断，不能据此直接决定删特征。此前配对重训固定h几乎无影响，与此结果一致。

5. NLL与选择表现不一致的机制已经更明确。
   原规则按原始VAL NLL选step100，此时交互/反转能力尚弱。后期排序持续改善，但预测过度自信使NLL恶化。
   按原共用有效seed口径，TEST NLL从0.363到0.891；只用VAL拟合温度后，step2000为0.325，排序完全不变。
   这说明proper NLL本身不是无效目标，而是有限数据下未校准的NLL不能单独代表top1/反转能力；改rank loss的历史实验也未保证改进。
   此外，Ring FF的TRAIN固定eta13已覆盖14/16个B15成功state，候选池内oracle也是14/16；当前top1成功率没有提升空间。
   但连续Q仍有空间：固定eta平均观测成功率下界0.945，oracle0.973。全局top1饱和不代表每一对eta的state依赖消失。

判断与下一步：
   已排除“state全固定或完全无用”作为解释。已经证实state交互存在、训练state拟合远好于新state、显式h贡献小、context承担主要条件化作用，以及概率过度自信影响checkpoint选择。
   尚不能把32个state的数量、编码器结构、短轨迹摘要或Q4标签精度中的任何一个宣布为唯一根因；历史更宽数据/更长probe/排名loss改动也未稳定修复。
   后续应在独立state验证集同时追踪交互/反转准确率、选择regret与校准NLL；以state为单位验证，而非随机拆state–eta行。需要新的、固定eta确实有改善空间的state来确认选择收益。
   本报告给出可定位的学习瓶颈与反例，不把已打开TEST的诊断当作新的泛化证明。

可复现入口：state_signal_path.py collect / labels_and_predictions / block_export / block_analyze / summarize / plot。
原始证据：cache_coverage.csv、label_repeatability.json、interaction_metrics.csv、block_metrics.csv、summary.json、verification.json。
本报告交互审计的NLL使用各链有效seed；原checkpoint NLL使用TT/FF共同有效seed，两者小幅差异已明确标注。
'''
    (OUT/'FINAL_REPORT.txt').write_text(report)
    print(json.dumps(dict(bootstrap=summary['test_reversal_change_bootstrap'],label_extremes=summary['label_extremes']),indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((OUT/'interaction_metrics.csv').open()))
    blocks=list(csv.DictReader((OUT/'block_metrics.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    steps=list(range(100,2001,100))
    for target,label,color in [('train4','TRAIN: 32 states','#269267'),('validation16','VAL: 8 new states','#d58b27'),('test16','TEST: 16 new states','#b74449')]:
        yy=[np.mean([float(r['reversal_both_correct']) for r in rows if r['method']=='full_replay' and r['step']==str(step) and r['target']==target]) for step in steps]
        axes[0].plot(steps,100*np.array(yy),label=label,color=color,lw=2)
    axes[0].axvline(100,color='gray',ls=':',lw=1)
    axes[0].set(title='Ring FF: learns reversals, transfers partly',xlabel='Training step',ylabel='Both reversal directions correct (%)',ylim=(0,105))
    axes[0].legend(fontsize=8,loc='lower right')
    for target,label,color in [('train4_extra_subset','Original 4 seeds, same 64 cells','#269267'),('extra12','Extra 12 seeds, same 64 cells','#d58b27')]:
        yy=[np.mean([float(r['NLL']) for r in rows if r['method']=='full_replay' and r['step']==str(step) and r['target']==target]) for step in steps]
        axes[1].plot(steps,yy,label=label,color=color,lw=2)
    axes[1].set(title='Finite-seed fitting is also present',xlabel='Training step',ylabel='NLL on 8 cached TRAIN states')
    axes[1].legend(fontsize=8)
    conditions=['correct','h_shuffle','H20_shuffle','goal_shuffle','all_state_shuffle']
    labels=['Correct inputs','Shuffle physical state','Shuffle H20 response','Shuffle goal response','Shuffle state + context']
    values=[100*np.mean([float(r['reversal_both_correct']) for r in blocks if r['split']=='test' and r['step']=='2000' and r['condition']==c]) for c in conditions]
    axes[2].barh(labels,values,color=['#269267','#789fa6','#d5a14e','#bd694c','#b74449'])
    axes[2].invert_yaxis();axes[2].set(title='Learned state dependence uses context',xlabel='TEST reversal accuracy at step2000 (%)',xlim=(0,75))
    for i,v in enumerate(values):axes[2].text(v+1,i,f'{v:.1f}',va='center',fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.18,axis='x' if ax==axes[2] else 'y');ax.set_axisbelow(True)
    fig.suptitle('State matters; finite-data generalization and calibration limit the critic',fontsize=14)
    fig.savefig(OUT/'signal_path.png',dpi=180)
    fig.savefig(OUT/'signal_path.pdf')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('collect','labels_and_predictions','block_export','block_analyze','summarize','plot'))
    globals()[p.parse_args().action]()
