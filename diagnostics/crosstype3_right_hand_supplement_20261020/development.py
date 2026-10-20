"""Prespecified16 known-type folds determine only the main schedule fallback."""
from common import *
from train_critic import load_data
from critic_model import metrics
from select_models import choose_sampled
from statistics_pipeline import evaluate_group,equivalence
from families import family
def equal(v,types):return float(np.mean([v[types==t].mean() for t in sorted(set(types))]))
def main():
    data=load_data();types=data['types'];nc=len(types)
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:p=z['full'];pn=z['no_controller']
    ranking=metrics(p,data['labels'],types,sorted(set(types)),data['candidate_indices']);r=[v['ranking']['same_pair_different_time']['accuracy'] for v in ranking.values() if v['ranking']['same_pair_different_time']['accuracy'] is not None];timing=float(np.mean(r)) if r else None
    def rate(choice):return equal((data['labels'][np.arange(nc)[:,None],np.arange(48)[None],choice,0]==1).mean(1),types)*100
    self_rate=rate(methods['self_schedule']);fixed_rate=rate(methods['fixed500']);learned=self_rate>=fixed_rate-.5-1e-12 and timing is not None and timing>.5;primary='self_schedule' if learned else 'fixed500'
    decision=dict(utc=utc(),primary_method=primary,self_schedule_success_pct=self_rate,fixed500_success_pct=fixed_rate,self_minus_fixed_pp=self_rate-fixed_rate,timing_ranking_accuracy=timing,timing_supported_types=[t for t,v in ranking.items() if v['ranking']['same_pair_different_time']['accuracy'] is not None],criterion='Self success >= fixed500 minus0.5pp and identifiable same-pair/different-time ranking >0.5',unseen_outcomes_read=False)
    write(HERE/'MAIN_METHOD_SELECTION.json',decision);seal('MAIN_METHOD_SELECTION.receipt.json',[HERE/'MAIN_METHOD_SELECTION.json',HERE/'DEVELOPMENT_CHOICES.receipt.json',HERE/'development.py'],before_freeze=True)
    methods['pipeline']=methods[primary].copy();methods['no_controller']=choose_sampled(pn,data['candidate_indices'])[primary]
    with np.load(HERE/'LEGACY_DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:
        assert np.array_equal(z['names'],data['names']);methods['legacy_pipeline']=z['legacy_pipeline']
    if primary=='fixed500':
        valid=(data['candidate_indices']%5==3);score=np.where(valid,p.mean(0)-p.std(0),-np.inf);best=np.where(valid&(score==score.max(-1,keepdims=True)),data['candidate_indices'],np.iinfo(np.int32).max).min(-1);methods['conservative']=np.argmax(data['candidate_indices']==best[...,None],-1)
    with np.load(HERE/'TRAINING_REFERENCES.npz',allow_pickle=False) as z:refs={k:z[k] for k in ('labels','uids','collision','timeout','unknown')}
    packed={k:np.concatenate((data[k],refs[k]),axis=2) for k in refs};packed.update(names=data['names'],types=types)
    packed['non_diagonal']=np.concatenate((data['candidate_indices']//40!=data['candidate_indices']//5%8,np.broadcast_to([False,True,True],(nc,48,3))),axis=2);packed['oracle_mask']=packed['uids'][...,0]!='';packed['uid_class']=equivalence(packed['uids'])
    for j,key in enumerate(('all_safety_baseline','all_S500','all_Sprog')):methods[key]=np.full((nc,48),26+j,np.int32)
    reports={};unavailable={}
    for group in [*sorted(set(types)),'combined','learned','optimization','search']:
        mask=np.asarray([group=='combined' or t==group or family(t)==group for t in types]);selected={};missing={}
        for key,v in methods.items():
            chosen=packed['labels'][np.arange(nc)[:,None],np.arange(48)[None],v]
            if np.any(chosen[mask]==-2):missing[key]=dict(reason='Exact historical reference coverage incomplete; no added rollout',available=int((chosen[mask]!=-2).sum()),total=int(chosen[mask].size))
            else:selected[key]=v[mask]
        # The paired evaluator requires safety; report sparse choices directly
        # when historical safety coverage is incomplete, rather than impute it.
        subset={k:v[mask] for k,v in packed.items()}
        if 'all_safety_baseline' in selected:
            reports[group]=dict(type_equal=evaluate_group(subset,selected,True,5000))
            if group=='combined':reports[group]['controller_equal']=evaluate_group(subset,selected,False,5000)
        else:
            from marginal_statistics import evaluate
            reports[group]=dict(type_equal=evaluate(subset,selected,True,5000),paired_reference_statistics='N/A: incomplete exact historical safety coverage')
            if group=='combined':reports[group]['controller_equal']=evaluate(subset,selected,False,5000)
        unavailable[group]=missing
    write(HERE/'DEVELOPMENT_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT_KNOWN16_TYPES',reports=reports,unavailable_methods=unavailable,ranking=ranking,primary_method_decision=decision,training_crossfit_oracle='N/A: only seed0; no extra training labels',legacy_c1c_comparator='Frozen c1c on identical sparse labels can be inferred separately; new-state comparator required and evaluated',human_references_excluded_from_critic_training=True))
    np.savez_compressed(HERE/'DEVELOPMENT_EVALUATION_BUNDLE.npz',**packed);np.savez_compressed(HERE/'DEVELOPMENT_ANALYSIS_CHOICES.npz',names=data['names'],types=types,**methods)
    seal('DEVELOPMENT_RESULTS.receipt.json',[HERE/n for n in ('DEVELOPMENT_RESULTS.json','MAIN_METHOD_SELECTION.receipt.json','DEVELOPMENT_EVALUATION_BUNDLE.npz','DEVELOPMENT_ANALYSIS_CHOICES.npz','development.py','statistics_pipeline.py')],no_unseen_correction_labels_read=True)
    print('Frozen development fallback:',primary,flush=True)
if __name__=='__main__':main()
