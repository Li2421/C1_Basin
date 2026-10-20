"""One frozen evaluation: endpoints, coverage, paired transitions and raw detail."""
from common import *
from statistics_pipeline import evaluate,best
import csv


def coverage(data,fixed):
    labels=data['labels'];lo=(labels==1).mean(-1);hi=(labels!=0).mean(-1)
    g=lo[:,:,:10].max(-1);gu=hi[:,:,:10].max(-1);base=lo[:,:,12];bu=hi[:,:,12];types=data['types']
    result={};candidate=read(HERE/'CANDIDATES.json')['states'][48:]
    identities=[]
    for row in fixed['pipeline']:
        identities.append([repr((candidate[s]['actions'][a]['switch_mode'],candidate[s]['actions'][a]['safety_baseline'],
            tuple(float(x).hex() for x in candidate[s]['actions'][a]['eta_a']),
            tuple(float(x).hex() for x in candidate[s]['actions'][a]['eta_b']))) for s,a in enumerate(row)])
    identities=np.asarray(identities)
    def pair_disagreement(rows):
        return np.asarray([1-np.sum(np.unique(row,return_counts=True)[1]*(np.unique(row,return_counts=True)[1]-1))/(24*23) for row in rows])
    index_pair_difference=pair_disagreement(fixed['pipeline']);identity_pair_difference=pair_disagreement(identities)
    for group in [*sorted(set(types)),'combined']:
        mask=types==group if group!='combined' else np.ones(len(types),bool)
        def avg(values,te=True):
            per=np.asarray(values)[mask].mean(1);tt=types[mask]
            return float(np.mean([per[tt==t].mean() for t in sorted(set(tt))])) if te else float(per.mean())
        reports={}
        for te in (True,False) if group=='combined' else (True,):
            choices=fixed['pipeline'];dist={str(a):avg(choices==a,te) for a in range(13)}
            varies=np.asarray([len(set(r))>1 for r in choices])
            within=np.asarray([len(set(r))/len(r) for r in choices])
            reports['type_equal' if te else 'controller_equal']=dict(
                empirical_nonworse_fraction=avg(g>=base,te),empirical_strict_better_fraction=avg(g>base,te),
                nonworse_unknown_bounds=[avg(g>=bu,te),avg(gu>=base,te)],
                strict_better_unknown_bounds=[avg(g>bu,te),avg(gu>base,te)],
                selected_generated_fraction=avg(choices<10,te),selected_anchor_fraction=avg(choices>=10,te),
                selected_action_distribution=dist,
                controllers_with_state_varying_choice_fraction=avg(np.repeat(varies[:,None],24,1),te),
                state_pair_candidate_index_disagreement_fraction=avg(np.repeat(index_pair_difference[:,None],24,1),te),
                state_pair_full_eta_schedule_disagreement_fraction=avg(np.repeat(identity_pair_difference[:,None],24,1),te),
                mean_distinct_actions_per_24_states=24*avg(np.repeat(within[:,None],24,1),te))
        result[str(group)]=reports
    return result


def transitions(data):
    # Episode-wise nominal collisions followed by projected baseline outcomes.
    mask=data['collision'][:,:,13,:].astype(bool);y=data['labels'][:,:,10,:]
    timeout=data['timeout'][:,:,10,:].astype(bool);types=data['types'];out={}
    for t in [*sorted(set(types)),'combined']:
        selected=np.ones(len(types),bool) if t=='combined' else types==t
        m=mask[selected];v=y[selected];tm=timeout[selected];n=int(m.sum())
        outcomes=dict(success=(v==1)&m,timeout=tm&m&(v>=0),unknown=(v<0)&m,
            collision=data['collision'][selected,:,10,:].astype(bool)&m&(v>=0))
        counts={k:int(value.sum()) for k,value in outcomes.items()}
        out[str(t)]=dict(nominal_collision_episodes=n,**{'projected_'+k:v for k,v in counts.items()},
            fractions_given_nominal_collision={k:v/n if n else None for k,v in counts.items()},
            other_termination_episodes=int((m&~np.logical_or.reduce(list(outcomes.values()))).sum()),
            note='Paired episode transitions; unknown takes precedence over known events; timeout is not classified as deadlock')
    return out


def details(data,fixed):
    labels=data['labels'];names=data['names'];states=data['state_uids'];candidates=read(HERE/'CANDIDATES.json')['states'][48:]
    fields=['controller','type','state_uid','candidate_index','candidate_name','early_eta_goal','early_eta_perp','early_eta_rel',
        'late_eta_goal','late_eta_perp','late_eta_rel','switch_mode','safety_baseline','selected_by','seed0','seed1',
        'successes','known_seeds','unknown_seeds','rollout_uid_seed0','rollout_uid_seed1',
        'pipeline_success_lower','state_in_sample_oracle_success_lower','state_crossfit_oracle_success_lower',
        'pipeline_in_sample_regret_pp','pipeline_crossfit_regret_pp']
    with (HERE/'CONTROLLER_STATE_DETAILS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,name in enumerate(names):
            for s,state in enumerate(states):
                row=labels[i,s];lo=(row==1).astype(float);pick=fixed['pipeline'][i,s]
                inside=float(lo[:13].mean(-1).max())
                cross=.5*(lo[best(lo[:13,0]),1]+lo[best(lo[:13,1]),0]);actual=float(lo[pick].mean())
                for a in range(14):
                    action=candidates[s]['actions'][a] if a<13 else dict(eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='nominal_only',safety_baseline=False)
                    writer.writerow(dict(controller=str(name),type=str(data['types'][i]),state_uid=str(state),candidate_index=a,
                        candidate_name=candidates[s]['actions'][a]['name'] if a<13 else 'nominal_only',
                        early_eta_goal=action['eta_a'][0],early_eta_perp=action['eta_a'][1],early_eta_rel=action['eta_a'][2],
                        late_eta_goal=action['eta_b'][0],late_eta_perp=action['eta_b'][1],late_eta_rel=action['eta_b'][2],
                        switch_mode=action['switch_mode'],safety_baseline=action['safety_baseline'],
                        selected_by='|'.join(k for k,v in fixed.items() if v[i,s]==a),seed0=int(row[a,0]),seed1=int(row[a,1]),
                        successes=int((row[a]==1).sum()),known_seeds=int((row[a]>=0).sum()),unknown_seeds=int((row[a]<0).sum()),
                        rollout_uid_seed0=data['uids'][i,s,a,0],rollout_uid_seed1=data['uids'][i,s,a,1],
                        pipeline_success_lower=actual,state_in_sample_oracle_success_lower=inside,state_crossfit_oracle_success_lower=float(cross),
                        pipeline_in_sample_regret_pp=100*(inside-actual),pipeline_crossfit_regret_pp=100*(cross-actual)))


def main():
    assert not (HERE/'CONFIRMATORY_RESULTS.json').exists()
    for receipt in ('NEW_EVAL_BUNDLE.receipt.json','NEW_EVAL_CHOICES.receipt.json','STATISTICS_TESTS.receipt.json'):
        for path,digest in read(HERE/receipt)['files'].items():assert sha(path)==digest,path
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'NEW_EVAL_CHOICES.npz',allow_pickle=False) as z:
        assert np.array_equal(data['names'],z['names']) and np.array_equal(data['state_uids'],z['state_uids'])
        fixed={k:z[k] for k in z.files if k not in ('names','types','state_uids')}
    summaries=evaluate(data,fixed,5000);combined=summaries['combined']['type_equal'];p=combined['methods']['pipeline']
    contrasts=combined['paired_comparisons'];endpoints=dict(
        endpoint1=dict(passed=p['gain_CI95_pp'][0]>0,gain_pp=p['gain_vs_progress_pp'],CI95_pp=p['gain_CI95_pp']),
        endpoint2=dict(passed=p['state_crossfit_regret_unknown_bounds_pp'][1]<=2 and p['harm_possible']==0,
            crossfit_regret_pp=p['state_crossfit_regret_pp'],unknown_bounds_pp=p['state_crossfit_regret_unknown_bounds_pp'],harm_possible=p['harm_possible']),
        endpoint3={k:contrasts['pipeline__minus__'+k] for k in ('no_controller','no_state')},
        endpoint4=contrasts['pipeline__minus__original_pipeline'])
    report=dict(utc=utc(),evidence_level='CONFIRMATORY',summaries=summaries,endpoints=endpoints,
        endpoint5_generator_coverage_and_choices=coverage(data,fixed),nominal_collision_to_safety_transitions=transitions(data),
        all_nine_type_model_reference_evidence='Descriptive: types seen during training; states unseen',
        warm_start_pretraining_scope='Original critic previously trained on original native Flow; outer exclusion refers to this round fine tuning')
    details(data,fixed);write(HERE/'CONFIRMATORY_RESULTS.json',report)
    seal('CONFIRMATORY_RESULTS.receipt.json',[HERE/'CONFIRMATORY_RESULTS.json',HERE/'CONTROLLER_STATE_DETAILS.csv',
        HERE/'confirmatory_statistics.py',HERE/'statistics_pipeline.py',HERE/'NEW_EVAL_BUNDLE.receipt.json',HERE/'NEW_EVAL_CHOICES.receipt.json'],
        replicates=5000,paired_type_controller_state_bootstrap=True,oracle_reselected_in_each_draw=True)
    print('Frozen confirmatory endpoints complete',flush=True)


if __name__=='__main__':main()
