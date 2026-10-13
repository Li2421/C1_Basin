"""All-fold paired local mechanism review; no terminal-Q claim."""
import csv
import numpy as np
from .study import HERE,BASE,FOLDS,read,write,sha


def run():
    rows=[];paired=[];boot={};sources=[]
    for fold in FOLDS:
        new=read(HERE/f'review_{fold}.json')['rows']
        old=read(BASE/f'review_{fold}.json')['rows']
        a=np.load(HERE/f'bootstrap_{fold}.npz');b=np.load(BASE/f'bootstrap_{fold}.npz')
        rows.extend(new)
        sources.extend(r for r in new if r['split']=='source_validation' and r['metric']=='state_eta')
        for r in new:
            if r['split']!='held_all' or r['metric'] not in ('state_eta','state_controller_eta'):continue
            tag=r['scene']+'__held_all';np.testing.assert_array_equal(a[tag+'__indices'],b[tag+'__indices'])
            o=next(z for z in old if z['split']=='held_all' and z['metric']==r['metric'])
            k=tag+'__'+r['metric'];gain=a[k+'__skill']-b[k+'__skill']
            pa=a[k+'__true_energy'];pb=b[k+'__true_energy']
            valid=(pa>1e-15)&(pb>1e-15)
            norm=np.full(len(gain),np.nan)
            norm[valid]=a[k+'__skill'][valid]/pa[valid]-b[k+'__skill'][valid]/pb[valid]
            point=(r['interaction_skill']/r['true_interaction_energy']-o['interaction_skill']/o['true_interaction_energy']) if min(r['true_interaction_energy'],o['true_interaction_energy'])>0 else None
            paired.append(dict(fold=fold,metric=r['metric'],native=o['interaction_skill'],
                common=r['interaction_skill'],common_CI95=r['skill_CI95'],
                paired_change=r['interaction_skill']-o['interaction_skill'],
                paired_CI95=np.quantile(gain,[.025,.975]).tolist(),
                native_true_energy=o['true_interaction_energy'],common_true_energy=r['true_interaction_energy'],
                common_true_energy_CI95=r['true_energy_CI95'],
                normalized_skill_change=point,
                normalized_change_CI95_conditional_positive_power=np.quantile(norm[valid],[.025,.975]).tolist() if valid.any() else None,
                ratio_undefined_fraction=float(1-valid.mean()),
                per_seed_skill=r['per_seed_skill']))
            boot[fold+'__'+r['metric']+'__skill_change']=gain
            boot[fold+'__'+r['metric']+'__normalized_change']=norm
    qualifying=[r['fold'] for r in paired if r['metric']=='state_eta' and r['common_CI95'][0]>0 and r['paired_CI95'][0]>0]
    source_ok=all(r['skill_CI95'][0]>0 for r in sources)
    result=dict(rows=rows,paired=paired,qualifying_local_folds=qualifying,
        source_state_eta_positive_CI_count=sum(r['skill_CI95'][0]>0 for r in sources),source_checks=len(sources),
        source_gate_passed=source_ok,new_fits=24,reused_baseline_fits=24,new_terminal_rollouts=0,
        static_action_queries=16384,root_cause_closed=False,
        boundary='Local action diagnostic only; not Q or selector success. Common-law intervention changes target power/difficulty as well as source-target program shift, so a positive difference alone does not uniquely prove confounding as root. Native-to-common performance is not evaluated on identical outcome targets. Opened DEVELOPMENT, source-only fits.',
        next_decision='Positive local gate: inspect all interaction and mismatch controls before deciding a Q experiment.' if qualifying else 'No local repair gate: common initialization alone does not establish transferable state_eta learning. Do not run new Q labels or mean/scale sweeps.')
    write(HERE/'results.json',result);np.savez_compressed(HERE/'paired_bootstrap.npz',**boot)
    with (HERE/'summary.csv').open('w') as f:
        fields=sorted(set().union(*(r.keys() for r in rows)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['COMMON INITIALIZER / NATIVE INITIALIZER — LOCAL TRANSFER CONTROL',
        f'SOURCE positive state_eta CI: {result["source_state_eta_positive_CI_count"]}/{len(sources)}',
        result['next_decision'],result['boundary'],'',
        'Held scene | metric | native skill | common skill [95%CI] | paired change [95%CI] | true common power [95%CI]']
    for r in paired:
        lines.append(f'{r["fold"]} | {r["metric"]} | {r["native"]:.8g} | {r["common"]:.8g} {r["common_CI95"]} | {r["paired_change"]:.8g} {r["paired_CI95"]} | {r["common_true_energy"]:.8g} {r["common_true_energy_CI95"]}')
    lines.extend(['','All old160naturalstates reused,32TRAIN+8VAL/scene,4operators,K4,all3fitseeds. No held fitting/normalization/checkpoint selection. Raw labels, input descriptors, first actions, seeds, source identities and predictions saved.',
        '20k pairedphysicalstatebootstrap with interaction recentering and mismatch donorpool resampling. Normalized skill/power sensitivity retains undefined ratio draws asNaN and reports theirfraction.',
        'Common law0/vmax world Cartesian changes entireinitializer/chart only; oldgoalreference/M/FF/safety unchanged. Exact physicalarrays/nominal14 reused, updatedCfeatures necessary.',
        '16384staticqueries,0numericalfailures,0Env.step/terminalrollouts. No safety/terminalsuccess claim possible from this static panel.',
        'Prior sharednavigation control changedreference; incompatible targets. OldStage35native24fits reused. Anisotropic feature adapter correction matches oldpipeline; four jobs stopped beforeactions, other12reused.',
        'Related primary literature reviewed before protocol: https://arxiv.org/abs/1702.02453; varying trainingprograms and conditioning are separate choices, not a transfer theorem for this task.'])
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_LOCAL_INITIALIZER_CONTROL',root_cause_closed=False,
        qualifying_local_folds=qualifying,new_terminal_rollouts=0,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(source_CI_positive=result['source_state_eta_positive_CI_count'],qualifying=qualifying))
    for r in paired:print(r['fold'],r['metric'],r['common_CI95'],r['paired_CI95'])


if __name__=='__main__':run()
