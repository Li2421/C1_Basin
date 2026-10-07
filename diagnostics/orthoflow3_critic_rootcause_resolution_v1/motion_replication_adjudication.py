"""Completed frozen evidence only. No new fit or model selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from scipy.stats import binomtest
from .motion_confirmation_eval import read,write,sha,csvwrite,measures,CONDITIONS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'motion_replication_adjudication'
ARMS=('full_update','trunk_only','eta_only','full_update_matched400')
SEEDS=(17,23,41)


def paired(a,b,s,f):
    ca,cb=a.argmax(1),b.argmax(1)
    idx=np.arange(len(s));same=ca==cb
    good=s>=15;bad=f>=2
    al,au=good[idx,ca].astype(int),(~bad[idx,ca]).astype(int)
    bl,bu=good[idx,cb].astype(int),(~bad[idx,cb]).astype(int)
    lower,upper=al-bu,au-bl
    lower[same]=upper[same]=0
    known=lower==upper
    rescue=int(((al==1)&(bu==0)).sum());brk=int(((au==0)&(bl==1)).sum())
    rng=np.random.default_rng(20261005)
    boot=rng.integers(0,len(s),(5000,len(s)))
    lo=s/16;hi=(16-f)/16
    qlo=lo[idx,ca]-hi[idx,cb];qhi=hi[idx,ca]-lo[idx,cb]
    qlo[same]=qhi[same]=0
    return dict(rescue_confirmed=rescue,break_confirmed=brk,
        unknown_comparisons=int((~known).sum()),same_choice=int(same.sum()),
        net_B15_lower=int(lower.sum()),net_B15_upper=int(upper.sum()),
        family_bootstrap_B15_rate_lower=float(np.quantile(lower[boot].mean(1),.025)),
        family_bootstrap_B15_rate_upper=float(np.quantile(upper[boot].mean(1),.975)),
        exact_p= float(binomtest(rescue,rescue+brk,.5).pvalue) if known.all() and rescue+brk else None,
        paired_Q16_gain_lower=float(qlo.mean()),paired_Q16_gain_upper=float(qhi.mean()),
        family_bootstrap_Q_gain_lower=float(np.quantile(qlo[boot].mean(1),.025)),
        family_bootstrap_Q_gain_upper=float(np.quantile(qhi[boot].mean(1),.975)))


def main():
    OUT.mkdir(exist_ok=True);metrics=[];comparisons=[];audits=[]
    for target in (88132,88133,88134,88135):
        folder=ROOT/f'motion_independent_confirmation_{target}'
        guard=read(folder/'prediction_freeze.json');state=read(folder/'working_state.json')
        assert guard['predictions_sha256']==sha(folder/'frozen_predictions.npz')
        assert state['missing']==state['collision']==state['conflict']==0 and state['journal_merged']
        data=dict(np.load(folder/'dataset.npz'));scores=dict(np.load(folder/'frozen_predictions.npz'))
        s,f=[data[k].reshape(64,2) for k in ('success','failure')]
        assert np.all(s+f+data['numerical'].reshape(64,2)==16)
        for arm in ARMS:
            for condition in CONDITIONS:
                # Predeclared for88134/35; retrospective secondary for88132/33.
                pp=np.mean([expit(scores[f'{arm}__{seed}__{condition}']) for seed in SEEDS],0)
                scores[f'{arm}__ensemble__{condition}']=logit(pp)
        for arm in ARMS:
            for seed in (*SEEDS,'ensemble'):
                for condition in CONDITIONS:
                    zz=scores[f'{arm}__{seed}__{condition}']
                    metrics.append(dict(target=target,arm=arm,seed=seed,condition=condition,
                        secondary=seed=='ensemble' or arm=='full_update_matched400',
                        ensemble_preregistered=target in (88134,88135),**measures(zz,s,f)))
                if arm=='eta_only':continue
                aa=scores[f'{arm}__{seed}__correct']
                for control in ('eta_only','wrong_controller_alt','joint_state_context_shuffle','state_shuffle'):
                    key=f'eta_only__{seed}__correct' if control=='eta_only' else f'{arm}__{seed}__{control}'
                    comparisons.append(dict(target=target,arm=arm,seed=seed,versus=control,**paired(aa,scores[key],s,f)))
        audits.append(dict(target=target,state=state,audit=read(folder/'evaluation_audit.json'),
            dataset_sha256=sha(folder/'dataset.npz'),prediction_sha256=sha(folder/'frozen_predictions.npz')))
    csvwrite(OUT/'metrics.csv',metrics);csvwrite(OUT/'paired_uncertainty.csv',comparisons)
    write(OUT/'adjudication.json',dict(
        verdict='PREDICTION_GAIN_REPLICATED_ROBUST_SELECTION_REPAIR_PARTIAL',
        independent_replication_targets=[88134,88135],initial_confirmation_targets=[88132,88133],
        confirmation_new_attempts=sum(a['state']['new_continuations'] for a in audits),
        confirmation_numerical_preserved=sum(a['state']['numerical'] for a in audits),
        new_rollouts_this_analysis=0,all_merged=True,collisions=0,conflicts=0,
        target_labels_used_to_select_frozen_models=False,selection_on_opened_targets=False,
        user_scope_clarification='Superiority is not required on eta-only-saturated scenes. Do not discard unsaturated null results or choose subgroups by wins.',
        evidence='88132 strong gain;88133 prediction/state contrast gain without stable B15 improvement;88134 small stable B15 gain with only4eta-prior headroom;88135 has11headroom and23robust reversal states but no B15 gain. CorrectC probability benefit does not guarantee correct eta ordering.',
        explicit_h='Separate h shuffle largely does not change selection; this is not no state learning, because response C is state conditioned and wrong-state C changes predictions.',
        ensemble='Arithmetic3seedmean;secondary only;prospectively registered for88134/35,retrospective on88132/33. Does not rescue the negative second replication.',
        uncertainty='Family bootstrap is conditional on each held controller and existing Q16 evidence. Seeds are not independent cohorts. Numeric uncertainty retained; identical chosen proposals cancel in paired contrasts. Multiple comparisons descriptive; do not select significant rows.',
        scope='Ring geometry;unseenfuturecontrollerweights and true-t0families;two seenexacteta. Notcross-scene,unseeneta,K16,orproofallcontrollerinputsufficient.',
        next_step='Source-only non-neural kernel/local fit discriminator, no additional taskrollout or target-controller sampling.',artifacts=audits))
    print(dict(adjudication=str(OUT/'adjudication.json'),new_rollouts=0),flush=True)


if __name__=='__main__':main()
