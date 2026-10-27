"""Freeze source-selected fits, scene-level proposals and all prescribed methods."""
from common import *
from predict_v2 import load,model_definition,raw_scores,scored,whole_scene,scene_proposals
from anchored import proposal_priors,sigmoid
from names_v2 import METHOD_NAMES
import numpy as np

def select_models():
    selected={}
    for task in read(HERE/'TRAIN_OUTER_TASKS.json'):
        key=(task['fold'],task['mode'])
        if key in selected:continue
        folder=HERE/'folds'/task['fold']/task['mode'];rows=[read(folder/f'seed{seed}'/'FIT.receipt.json')for seed in(17,23,41)]
        for r in rows:assert sha(r['checkpoint'])==r['sha256']and r['source_only_early_stopping']and not r['heldout_labels_used']
        best=min(rows,key=lambda r:(r['validation_loss'],r['seed']));write(folder/'SELECTED.json',dict(selected=best,all_seeds=rows,tie_rule='Minimum source-internal validation loss, then smallest seed.'))
        selected[key]=best
    return selected

def choice(index,menu,keys,C,S,**extra):
    if index is None:return dict(indices=None,applicable=False,**extra)
    a=np.full((C,S),index,int)if np.asarray(index).ndim==0 else np.asarray(index,int)
    assert a.shape==(C,S)
    unique,counts=np.unique(a,return_counts=True)
    return dict(indices=a.tolist(),selected_index=int(a[0,0])if np.all(a==a[0,0])else None,applicable=True,
        selected_candidates=[dict(index=int(i),key=keys[i],eta=menu[i].tolist(),cells=int(count))for i,count in zip(unique,counts)],**extra)

def main():
    assert(HERE/'train_outer_COMPLETE.json').exists();assert(HERE/'LAMBDA_SELECTION_SUMMARY.json').exists()
    selections=select_models();inputs=read(HERE/'INPUTS_V2.json');eta=np.asarray(inputs['etas'],float);keys=inputs['eta_keys'];guards=read(HERE/'GUARD.json')['folds']
    tables=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/LABEL_TABLES.json');receipts=[]
    for fold in read(HERE/'FOLDS.json'):
        folder=HERE/'folds'/fold['name'];receipt=HERE/'FOLD_FROZEN'/f"{fold['name']}.receipt.json"
        if receipt.exists():
            for p,h in read(receipt)['files'].items():assert sha(p)==h,p
            receipts.append(receipt);continue
        p=read(folder/'PRIORS.json');ls=read(folder/'LAMBDA_SELECTION.json');lam=ls['full']['selected_lambda'];full,fullfit=load(fold['name'],'full');raw_params=none_params=gparams=None
        if fold['generator']:
            raw_params,_=load(fold['name'],'raw');none_params,_=load(fold['name'],'no_prior');gparams,gfit=load(fold['name'],'generator')
        choices={};proposals={};ranking={}
        for name in fold['targets']:
            data=inputs['scene_inputs'][name];C,S=len(data['controllers']),len(data['h']);guard=guards[fold['name']][name]['scale_independent'];raw_guard=guards[fold['name']][name]['raw']
            prop=scene_proposals(gparams,gfit,fold,data,name)if fold['generator']else dict(etas=[])
            proposals[name]=prop;menu=np.concatenate([eta,np.asarray(prop['etas']).reshape(-1,3)]);candidate_keys=keys+(['proposal_'+str(i)for i in range(5)]if fold['generator']else[])
            p0=np.r_[p['p0'],proposal_priors(prop['etas'],eta,p)if fold['generator']else[]]
            raw=raw_scores(full,fold,data,menu);best,logits,scene_score=whole_scene(raw,p0,lam,'full',guard)
            b49,l49,s49=whole_scene(raw[:,:,:49],p['p0'],lam,'full',guard)
            methods={};scores={}
            methods['v2_49']=choice(b49,menu,candidate_keys,C,S,shrinkage=lam,scene_fallback=guard['scene_out_of_range']);scores['v2_49']=s49.tolist()
            gi=p['best_index'];methods['global_fixed']=choice(gi,menu,candidate_keys,C,S)
            ranking[name+'__full_selected']=l49
            if fold['generator']:
                methods['v2']=choice(best,menu,candidate_keys,C,S,shrinkage=lam,scene_fallback=guard['scene_out_of_range']);scores['v2']=scene_score.tolist()
                for method,state_fallback,scene_fallback,clip in [('v2_49_no_fallback',False,False,False),('v2_49_clip',False,False,True)]:
                    z=raw_scores(full,fold,data,eta,clip=True)if clip else raw[:,:,:49]
                    b,l,ss=whole_scene(z,p['p0'],lam,'full',guard,state_fallback,scene_fallback);methods[method]=choice(b,menu,candidate_keys,C,S,shrinkage=lam);scores[method]=ss.tolist()
                b,l,ss=whole_scene(raw[:,:,:49],p['p0'],1,'full',guard);methods['v2_49_lambda1']=choice(b,menu,candidate_keys,C,S,shrinkage=1);scores['v2_49_lambda1']=ss.tolist();ranking[name+'__full_lambda1']=l
                per_cell=l49.argmax(-1);per_controller=np.repeat(sigmoid(l49).mean(1).argmax(-1)[:,None],S,axis=1)
                if lam==0:per_cell[:]=gi;per_controller[:]=gi
                if guard['scene_out_of_range']:per_cell[:]=gi;per_controller[:]=gi
                else:per_cell[:,np.asarray(guard['states_out_of_range'],bool)]=gi
                methods['v2_49_per_cell']=choice(per_cell,menu,candidate_keys,C,S,shrinkage=lam)
                methods['v2_49_per_controller']=choice(per_controller,menu,candidate_keys,C,S,shrinkage=lam)
                rawdef=model_definition(fold['name'],'raw');rr=raw_scores(raw_params,rawdef,data,eta);lr=ls['raw']['selected_lambda']
                br,ll,ss=whole_scene(rr,p['p0'],lr,'raw',raw_guard);methods['v2_49_raw_description']=choice(br,menu,candidate_keys,C,S,shrinkage=lr,scene_fallback=raw_guard['scene_out_of_range']);scores['v2_49_raw_description']=ss.tolist();ranking[name+'__raw_selected']=ll
                rn=raw_scores(none_params,fold,data,eta);bn,ll,ss=whole_scene(rn,p['p0'],1,'no_prior',guard);methods['v2_49_no_prior']=choice(bn,menu,candidate_keys,C,S,scene_fallback=guard['scene_out_of_range']);scores['v2_49_no_prior']=ss.tolist();ranking[name+'__no_prior']=ll
                methods['generator_only']=choice(None if guard['scene_out_of_range']else 49,menu,candidate_keys,C,S,descriptive_only=True)
                for method,key in [('eta_A','sE29'),('knn',fold['baselines'][name]['knn']),('eta13','eta13'),('safety','safety')]:methods[method]=choice(keys.index(key),menu,candidate_keys,C,S)
                y=np.asarray(tables[name]['y'],float);weights=[]
                for s in range(S):
                    q=y[:,np.delete(np.arange(S),s)].mean((0,1));ties=q==q.max();weights.append((ties/ties.sum()).tolist())
                methods['scene_xstate']=dict(applicable=True,not_zero_shot=True,weights_by_state=weights,tie_rule='Average held-out-state outcomes across all tied maxima selected on other states.')
                methods['v1_pipeline']=dict(applicable=True,source='Frozen completed v1 per-cell results; no rerun.')
                assert set(methods)==set(METHOD_NAMES)
            ranking[name+'__prior']=np.broadcast_to(np.asarray(p['p0']),(C,S,49)).copy()
            # For lambda0 and guarded targets, enforce the exact prescribed
            # equivalence using unrounded indices, independently of scoring.
            if fold['generator']and(lam==0 or guard['scene_out_of_range']):assert methods['v2']['selected_index']==gi
            choices[name]=dict(controllers=data['controllers'],controller_types=data['controller_types'],state_indices=data['state_indices'],candidate_etas=menu.tolist(),candidate_keys=candidate_keys,
                methods=methods,scene_scores=scores,prior_probabilities=p0.tolist(),shrinkage=lam,guard=guard,raw_guard=raw_guard)
        write(folder/'PROPOSALS.json',proposals);write(folder/'CHOICES.json',choices);np.savez_compressed(folder/'RANKING_SCORES.npz',**ranking)
        paths=[folder/n for n in('definition.json','PRIORS.json','LAMBDA_SELECTION.json','PROPOSALS.json','CHOICES.json','RANKING_SCORES.npz','source_labels.npz')]
        paths.extend(HERE/n for n in('INPUTS_V2.json','GUARD.json','TRAINING_FROZEN.receipt.json','freeze_choices_v2.py','predict_v2.py','anchored.py','descriptors_v2.py','networks.py'))
        for mode in ['full']+(['raw','no_prior','generator']if fold['generator']else[]):
            paths.append(folder/mode/'SELECTED.json')
            for seed in(17,23,41):paths.extend(folder/mode/f'seed{seed}'/n for n in('FIT.receipt.json','history.json','checkpoint.msgpack'))
        if fold['generator']:paths.extend([folder/'raw_definition.json',folder/'source_labels_raw.npz'])
        for mode,links in fold['inner_links'].items():
            for link in links.values():
                nf=HERE/'folds'/link['training_node'];paths.extend(nf/n for n in('definition.json','PRIORS.json','source_labels_raw.npz'if mode=='raw'else'source_labels.npz'))
                paths.extend(nf/mode/'seed17'/n for n in('FIT.receipt.json','history.json','checkpoint.msgpack'))
        paths=list(dict.fromkeys(paths));seal(str(receipt.relative_to(HERE)),paths,fold=fold['name'],sources=fold['sources'],targets=fold['targets'],
            source_only_models_priors_lambda_and_normalization=True,new_rollouts_before_freeze=0,all_methods_and_scene_proposals_frozen=True)
        receipts.append(receipt);print('FROZEN',fold['name'],flush=True)
    assert len(receipts)==31
    write(HERE/'ALL_FOLDS_FROZEN.receipt.json',dict(utc=utc(),folds=31,files={str(p):sha(p)for p in receipts},new_rollouts_before_all_folds_frozen=0,models_priors_bandwidth_lambdas_guards_proposals_and_all_choices_frozen=True))

if __name__=='__main__':main()
