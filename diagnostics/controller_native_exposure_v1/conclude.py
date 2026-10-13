"""Consolidate the pre-specified local causal contrasts; never choose a model."""
import csv,json
import numpy as np
from .study import HERE,BASE,OLD,ARMS,SEEDS,read,write,sha


def centered(x):
    return x-x.mean(-2,keepdims=True)-x.mean(-1,keepdims=True)+x.mean((-2,-1),keepdims=True)


def run():
    p=read(HERE/'evaluation_protocol.json');r=read(HERE/'results.json');d=dict(np.load(OLD/'dataset.npz'))
    ix=np.flatnonzero(d['split']=='test');archive=np.load(HERE/'test_predictions.npz')
    source=[]
    for arm in ARMS:
        for seed in SEEDS:
            z=read(HERE/'fits'/arm/f'seed{seed}/complete.json')
            source.append(dict(arm=arm,seed=seed,selected_step=z['selected']['step'],train_Q_MSE=z['source_train']['Q_MSE'],
                train_interaction_MSE=z['source_train']['state_eta_MSE'],val_Q_MSE=z['selected']['Q_MSE'],val_interaction_MSE=z['selected']['state_eta_MSE']))
    arrays={};contrasts=[]
    for panel in ['test_all','test_fresh','test_original']:
        ids=np.arange(len(ix)) if panel=='test_all' else np.flatnonzero(d['origin'][ix]==panel[5:])
        ids=np.asarray(sorted(ids,key=lambda i:d['state_uid'][ix[i]]));n=len(ids)
        lo=d['success'][ix[ids]]/16;hi=(d['success']+d['unknown'])[ix[ids]]/16
        boot=np.random.default_rng(2026101016+n).integers(n,size=(20000,n))
        for which in ['selected','final']:
            for first,second in p['comparisons']:
                qa=archive[which+'__'+first][:,ids].mean(0);qb=archive[which+'__'+second][:,ids].mean(0)
                def bound(a,b,l,u):
                    a,b=centered(a),centered(b);x,y=2*(a-b)*l-a*a+b*b,2*(a-b)*u-a*a+b*b
                    return np.minimum(x,y).mean((-2,-1)),np.maximum(x,y).mean((-2,-1))
                a,b=bound(qa,qb,lo,hi);al=[];au=[]
                for start in range(0,len(boot),500):
                    ib=boot[start:start+500];x,y=bound(qa[ib],qb[ib],lo[ib],hi[ib]);al.append(x);au.append(y)
                al,au=np.concatenate(al),np.concatenate(au);key='__'.join([which,panel,first,second])
                arrays[key+'__lower']=al;arrays[key+'__upper']=au
                contrasts.append(dict(variant=which,panel=panel,first=first,second=second,N=n,interaction_MSE_gain=dict(
                    identification_bounds=[float(a),float(b)],censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(au,.975))])))
    write(HERE/'paired_interaction_results.json',contrasts);np.savez_compressed(HERE/'paired_interaction_bootstrap.npz',**arrays)
    write(HERE/'source_metrics.json',source)
    def dump(path,rows):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(HERE/'source_metrics.csv',source)
    flat=[]
    for row in r['rows']:
        flat.append({k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in row.items() if k!='reversal_pairs'})
    dump(HERE/'summary.csv',flat)
    # Every state retained, not merely selected examples. All outcome channels and labels remain in old dataset.
    fail=[];qlo=d['success'][ix]/16
    for arm in ARMS:
        q=archive['selected__'+arm].mean(0);pick=q.argmax(-1);fixed=int(qlo.mean(0).argmax())
        for i,j in enumerate(ix):
            fail.append(dict(arm=arm,state_uid=d['state_uid'][j],origin=d['origin'][j],selected_eta=int(pick[i]),
                selected_success=float(qlo[i,pick[i]]),selected_unknown=int(d['unknown'][j,pick[i]]),
                oracle_success=float(qlo[i].max()),eval_posthoc_fixed_eta=fixed,eval_posthoc_fixed_success=float(qlo[i,fixed]),
                gap_to_oracle=float(qlo[i].max()-qlo[i,pick[i]]),
                selected_collision=int(d['raw_outcomes'][j,pick[i],:,2].sum())))
    dump(HERE/'per_state_cases.csv',fail)
    lines=['Stage19 — native Ring optimization/exposure/interference audit','',
        'All labels cached.15 fits,3 training seeds;12 isolated fits plus3 exact joint reconstructions. Two failed P-core reconstructions are preserved separately and excluded. Zero new rollouts;64 static feature queries only.',
        'CPU integrity: same initial parameters/inputs/draws gave different floating-point trajectories on P and E cores. Original E-core arithmetic reconstructs the old checkpoint; all3 accepted joint checkpoint hashes must match Stage17. This is an implementation-integrity finding, not the scientific cause of transfer failure.',
        'Scope:512 nativeRing TRAIN,32 SOURCE VAL,64 already-opened independent TEST physical states (48 fresh-origin,16 original). These are DEVELOPMENT evidence, not untouched final confirmation and not LOSO. No held-test labels used for fitting, normalization or checkpoint choice.',
        'Primary checkpoint: native SOURCE VAL selection regret, negative ranking, interaction MSE,NLL,earliest. Final-step is mandatory sensitivity. The3 training-seed predictions are averaged, with individual-seed metrics also retained.',
        'Arms:joint exact original samples;isolated_sparse same native samples at same6000steps with other tasks removed and original denominator480;isolated_dense12000x128 native draws. Field/zero arms have identical architecture, initialization, physical/API slots and normalization;zero removes only64 controller channels.',
        'Sparse versus joint isolates inclusion of other tasks under the same native sample exposure. Dense versus sparse changes native training budget (draws, batch size and steps together); it does not identify which optimizer micro-detail matters.',
        'No source labels or states were regenerated. Fair native eta-only and old successful physical-slot checkpoints/predictions are reused. The global original checkpoint is an additional reference, not the matched-checkpoint causal baseline.',
        'Statistics:20000 physical-state paired bootstraps, state means recentered per replicate; mismatch jointly resamples donors/recipients and retains200 derangements; all16 fixed policies including bootstrap-reselected best; reversal discovery8/evaluation8 with pre-fixed symmetric criteria. A one-controller panel cannot measure controller generalization.',
        'Safety:all per-seed collision/numerical flags retained. No new compatible Safety rollout or complete deadlock/timeout breakdown is fabricated in this local causal diagnostic.','',
        'Source mean error across3 fit seeds: arm TRAIN_Q TRAIN_interaction VAL_Q VAL_interaction']
    for arm in ARMS:
        vals=[z for z in source if z['arm']==arm]
        lines.append(arm+' '+' '.join(f'{np.mean([x[k] for x in vals]):.5f}' for k in ['train_Q_MSE','train_interaction_MSE','val_Q_MSE','val_interaction_MSE']))
    for which in ['selected','final']:
        lines.extend(['',which+' primary all64: arm Q_MSE S_eta_corr selected_success B15 matched-minus-mismatch_CI best-fixed_CI'])
        for z in r['rows']:
            if z['panel']=='test_all' and z['variant']==which:
                lines.append(f"{z['arm']} {z['Q_MSE_midpoint']:.5f} {z['state_eta_corr']} {z['success_bounds']} {z.get('B15_bounds')} {z['matched_minus_mismatch']['censor_CI95']} {z['minus_bootstrap_reselected_per_controller_fixed']['censor_CI95']}")
        lines.append('Pre-specified paired contrasts: first minus second; positive error gain means improvement')
        for z in r['comparisons']:
            if z['panel']=='test_all' and z['variant']==which and 'Q_MSE_gain' in z:
                extra=next(x for x in contrasts if all(x[k]==z[k] for k in ['variant','panel','first','second']))
                lines.append(f"{z['first']} - {z['second']}: QgainCI={z['Q_MSE_gain']['censor_CI95']};interactionGainCI={extra['interaction_MSE_gain']['censor_CI95']};selectionGainCI={z['selection_gain']['censor_CI95']}")
    lines.extend(['','Evidence boundary: These controls explain the source native branch only. A local repair neither proves nor repairs held-scene controller x state x eta transfer. No root-cause closure or successful LOSO verdict follows from this experiment alone.',
        'Artifacts:immutable protocols, exact draw archives, source/checkpoint IDs, replay-integrity probes, checkpoint files, test_predictions.npz, predictions_and_controls.npz, all bootstrap arrays, source_metrics.csv, summary.csv, per_state_cases.csv; raw per-state/eta/seed outcomes reused from ../controller_entity_coverage_check_v1/dataset.npz.'])
    for seed in SEEDS:assert read(HERE/'fits/joint_replay'/f'seed{seed}/replay_integrity.json')['byte_identical']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_LOCAL_DIAGNOSTIC_NOT_LOSO',fits=15,failed_integrity_attempts_preserved=2,
        new_rollouts=0,static_descriptors=64,test_states=64,training_seeds=SEEDS,all_joint_replay_hashes_exact=True,
        report_sha256=sha(HERE/'REPORT.txt'),new_method_confirmation=False))
    print('Saved REPORT, source/test/paired tables, all-state cases and bootstrap outputs',flush=True)

if __name__=='__main__':run()
