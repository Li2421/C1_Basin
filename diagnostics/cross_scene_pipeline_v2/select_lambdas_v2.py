"""Outer-fold shrinkage chosen using only held-in scene leave-one-out fits."""
from common import *
from predict_v2 import load,model_definition,raw_scores,whole_scene
from anchored import LAMBDAS
import numpy as np

def main():
    assert(HERE/'train_inner_COMPLETE.json').exists()
    inputs=read(HERE/'INPUTS_V2.json');etas=np.asarray(inputs['etas']);tables=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/LABEL_TABLES.json')
    cache={};summaries={}
    for fold in read(HERE/'FOLDS.json'):
        result={}
        for mode,links in fold['inner_links'].items():
            rows=[];rates=[]
            for target,link in links.items():
                node=link['training_node'];key=(node,mode,target)
                if key not in cache:
                    params,fit=load(node,mode,seed=17);d=model_definition(node,mode);p=read(HERE/'folds'/node/'PRIORS.json')
                    raw=raw_scores(params,d,inputs['scene_inputs'][target],etas);cache[key]=(raw,p,fit)
                raw,p,fit=cache[key];choices=[];rr=[]
                for shrinkage in LAMBDAS:
                    best,_,scores=whole_scene(raw,p['p0'],shrinkage,mode,link['guard'])
                    value=float(np.asarray(tables[target]['y'])[:,:,best].mean());choices.append(dict(shrinkage=shrinkage,candidate_index=best,candidate_key=inputs['eta_keys'][best],eta=etas[best].tolist(),success_rate=value));rr.append(value)
                rows.append(dict(target=target,source_training_node=node,checkpoint=fit['checkpoint'],checkpoint_sha256=fit['sha256'],guard=link['guard'],choices=choices));rates.append(rr)
            averages=np.asarray(rates).mean(0);which=int(np.argmax(averages));result[mode]=dict(selected_lambda=LAMBDAS[which],lambda_order=list(LAMBDAS),mean_success_rates=averages.tolist(),internal_heldout_scenes=rows,
                source_scenes=fold['sources'],external_targets_never_used=fold['targets'],tie_rule='Smallest lambda among exact maximum source-only success rates.')
        write(HERE/'folds'/fold['name']/'LAMBDA_SELECTION.json',result);summaries[fold['name']]={m:r['selected_lambda']for m,r in result.items()}
        print('LAMBDA',fold['name'],summaries[fold['name']],flush=True)
    write(HERE/'LAMBDA_SELECTION_SUMMARY.json',summaries)

if __name__=='__main__':main()
