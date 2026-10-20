"""Held-type sparse-menu development and the predeclared switch fallback."""
from common import *
from train_critic import load_data
from critic_model import metrics
from statistics_pipeline import evaluate_group,equivalence
from design import action_case
from shared_rollout_db.src.rollout_db import uid
OLD=D/'crosstype1b_generator_critic_pipeline_v1'
def common_average(values,types):return float(np.mean([values[types==t].mean() for t in sorted(set(types))]))
def main():
    data=load_data();types=data['types'];names=data['names'];nc=len(names)
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k].copy() for k in z.files if k not in ('names','types')}
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:p=z['full'];pn=z['no_controller']
    ranking=metrics(p,data['labels'],types,sorted(set(types)),data['candidate_indices']);timing_ranks=[r['ranking']['same_pair_different_time']['accuracy'] for r in ranking.values() if r['ranking']['same_pair_different_time']['accuracy'] is not None];timing_rank=float(np.mean(timing_ranks)) if timing_ranks else None
    score=lambda choice:common_average((data['labels'][np.arange(nc)[:,None],np.arange(48)[None],choice,0]==1).mean(1),types)*100
    self_success=score(methods['self_schedule']);fixed_success=score(methods['fixed500']);learned=self_success>=fixed_success-.5-1e-12 and timing_rank is not None and timing_rank>.5
    primary='self_schedule' if learned else 'fixed500';methods['pipeline']=methods[primary].copy();methods['generator_mean']=np.zeros((nc,48),np.int32)
    from select_models import choose_sampled
    methods['no_controller']=choose_sampled(pn,data['candidate_indices'])[primary]
    if primary=='fixed500':
        fixed=(data['candidate_indices']//40==data['candidate_indices']//5%8)|(data['candidate_indices']%5==3);score=np.where(fixed,p.mean(0)-p.std(0),-np.inf);maximum=score.max(-1,keepdims=True);ids=np.where(fixed&(score==maximum),data['candidate_indices'],np.iinfo(np.int32).max).min(-1);methods['conservative']=np.argmax(data['candidate_indices']==ids[...,None],-1)
    labels=np.full((nc,48,32,1),-2,np.int8);uids=np.full(labels.shape,'',dtype='U96');events={k:np.zeros_like(labels) for k in ('collision','timeout','unknown')};labels[:,:,:26]=data['labels'];uids[:,:,:26]=data['uids']
    for k in events:events[k][:,:,:26]=data[k]
    with np.load(HERE/'TRAINING_REFERENCES.npz',allow_pickle=False) as z:
        for k in ('labels','uids',*events):
            target=labels if k=='labels' else uids if k=='uids' else events[k];target[:,:,26:29]=z[k][:,:,:,:1]
    with np.load(OLD/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:old={k:z[k] for k in ('labels','uids','collision','timeout','unknown')}
    with np.load(OLD/'DEVELOPMENT_CHOICES_V2B.npz',allow_pickle=False) as z:old_choice=z['pipeline']
    for k in ('labels','uids',*events):
        target=labels if k=='labels' else uids if k=='uids' else events[k];target[:,:,29,0]=old[k][np.arange(nc)[:,None],np.arange(48)[None],old_choice,0]
    base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');reference_presence={}
    with rollout_db.connect(True) as db:
        for slot,name,eta in [(30,'F10',base['eta'][10]),(31,'nominal_only',[0.,0.,0.])]:
            count=0
            for i,n in enumerate(names):
                for s,state in enumerate(states):
                    a=dict(name=name,eta_a=eta,eta_b=eta,switch_mode='fixed',switch_step=700);case,_=action_case(base,str(n),state,a,0,s,-1);r=db.execute('SELECT success,collision,timeout,numerical_failure FROM rollout WHERE rollout_uid=?',(case['rollout_uid'],)).fetchone()
                    if r is None:continue
                    count+=1;labels[i,s,slot,0]=-1 if r['numerical_failure'] else r['success'];uids[i,s,slot,0]=case['rollout_uid']
                    for k in events:events[k][i,s,slot,0]=r['numerical_failure'] if k=='unknown' else r[k]
            reference_presence[name]=dict(exact_records=count,total=nc*48,new_rollouts=0)
    for key,slot in [('all_safety_baseline',26),('all_S500',27),('all_Sprog',28),('legacy_pipeline',29),('all_F10',30),('nominal_only',31)]:methods[key]=np.full((nc,48),slot,np.int32)
    non_diag=np.zeros((nc,48,32),bool);non_diag[:,:,:26]=data['candidate_indices']//40!=data['candidate_indices']//5%8;non_diag[:,:,27:29]=True;non_diag[:,:,29]=(old_choice%2==1)|(old_choice>=11);available=uids[:,:,:,0]!='';oracle_mask=available.copy();oracle_mask[:,:,31]=False
    packed=dict(names=names,types=types,labels=labels,uids=uids,uid_class=equivalence(uids),non_diagonal=non_diag,oracle_mask=oracle_mask,**events);reports={};unavailable={}
    for group in [*sorted(set(types)),'combined']:
        mask=np.ones(nc,bool) if group=='combined' else types==group;subset={k:v[mask] for k,v in packed.items()};selected={};missing={}
        for key,choice in methods.items():
            chosen=labels[np.arange(nc)[:,None],np.arange(48)[None],choice]
            if np.any(chosen[mask]==-2):missing[key]='Some exact historical records absent; no added rollout or missing-as-failure imputation'
            else:selected[key]=choice[mask]
        reports[str(group)]={'type_equal':evaluate_group(subset,selected,True,5000)};unavailable[str(group)]=missing
        if group=='combined':reports[str(group)]['controller_equal']=evaluate_group(subset,selected,False,5000)
    with np.load(OLD/'DEVELOPMENT_PROBABILITIES_V2.npz',allow_pickle=False) as z:old_p=z['full']
    from legacy_critic_model import metrics as old_metrics
    old_ranking_original=old_metrics(old_p,old['labels'],types,sorted(set(types)))
    old_raw=metrics(old_p,old['labels'],types,sorted(set(types)),np.broadcast_to(np.arange(13),(nc,48,13)));old_ranking={t:dict(nll=r['nll'],ranking_accuracy=r['ranking_accuracy'],support=r['ranking']['all']) for t,r in old_raw.items()}
    now_rank=float(np.mean([r['ranking_accuracy'] for r in ranking.values() if r['ranking_accuracy'] is not None]));old_rank=float(np.mean([r['ranking_accuracy'] for r in old_ranking.values() if r['ranking_accuracy'] is not None]));s500=reports['combined']['type_equal']['methods']['all_S500']['success_lower_pct'];target_success=reports['combined']['type_equal']['methods']['pipeline']['success_lower_pct']
    targets=dict(ranking_not_below_old13=now_rank>=old_rank,success_within1pp_human500=target_success>=s500-1-1e-12)
    decision=dict(utc=utc(),primary_method=primary,switch_time_learned_on_development=learned,self_schedule_success_pct=self_success,fixed500_success_pct=fixed_success,self_minus_fixed_pp=self_success-fixed_success,timing_ranking_accuracy=timing_rank,timing_supported_types=[t for t,r in ranking.items() if r['ranking']['same_pair_different_time']['accuracy'] is not None],criterion='Self success >= fixed500 minus0.5pp and identifiable same-pair/different-time ranking >0.5; absent ranking is not success',no_new_eval_outcomes_read=True)
    write(HERE/'MAIN_METHOD_SELECTION.json',decision);seal('MAIN_METHOD_SELECTION.receipt.json',[HERE/'MAIN_METHOD_SELECTION.json',HERE/'DEVELOPMENT_CHOICES.receipt.json',HERE/'development_statistics.py'],before_freeze=True)
    write(HERE/'DEVELOPMENT_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT_HELD_TYPE_SAMPLED_ACTIONS',reports=reports,unavailable_methods=unavailable,ranking=ranking,legacy13_ranking=old_ranking,legacy13_original_clipped_ranking=old_ranking_original,ranking_summary=dict(new26=now_rank,old13=old_rank,limitation='Different menus and identifiable-pair support; old benchmark uses its original observed-seed counts. Both ranking benchmarks use raw probabilities; original clipped legacy diagnostics also retained'),targets=targets,all_targets_met=all(targets.values()),primary_method_decision=decision,historical_reference_coverage=reference_presence,new_evaluation_labels_read=False))
    np.savez_compressed(HERE/'DEVELOPMENT_EVALUATION_BUNDLE.npz',**packed);np.savez_compressed(HERE/'DEVELOPMENT_ANALYSIS_CHOICES.npz',names=names,types=types,**methods)
    seal('DEVELOPMENT_RESULTS.receipt.json',[HERE/'DEVELOPMENT_RESULTS.json',HERE/'MAIN_METHOD_SELECTION.receipt.json',HERE/'DEVELOPMENT_EVALUATION_BUNDLE.npz',HERE/'DEVELOPMENT_ANALYSIS_CHOICES.npz',HERE/'development_statistics.py',HERE/'statistics_pipeline.py'],sparse_label_scoring_only=True,reference_labels_excluded_from_training=True)
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n开发版本v1完整结果：'+json.dumps(dict(targets=targets,decision=decision),ensure_ascii=False)+'。全部外折与完整曲线保留，不按外层效果选checkpoint。\n')
    print(json.dumps(dict(targets=targets,primary=primary,timing_rank=timing_rank),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
