"""Reanalyse existing crossed-support predictions, no training or rollout."""
import argparse
from pathlib import Path
import numpy as np
from diagnostics.controller_cross_scene_root_v2.crossed_support import HERE as OLD,ARMS,SCENE
from diagnostics.controller_cached_union_v1.study import read,write,freeze,sha,SEEDS
from diagnostics.controller_cached_union_v1.interaction_audit import center,gain
from diagnostics.controller_program_input_probe_v1.review import Data,analyze_panel
from diagnostics.controller_ff_factorial_repair_v1.review import selection_difference,mse_gain,intervals

HERE=Path(__file__).resolve().parent
FINAL=HERE.parent/'controller_q_diversity_v1/data/final/dataset.npz'
PAIRS=[('crossed','more_source'),('crossed','crossed_wrong_controller'),('crossed','crossed_wrong_eta'),('more_source','source')]


def prepare():
    old=read(OLD/'results.json');assert all('matched_minus_mismatch' not in row for row in old['rows'])
    artifacts={str(FINAL):sha(FINAL)}
    for fold in SCENE:
        for arm in ARMS:
            for seed in SEEDS:
                root=OLD/'fits'/fold/arm/f'seed{seed}';done=read(root/'complete.json')
                assert done['not_LOSO'] and done['checkpoint_rule']=='fixed final step; no selection'
                assert sha(root/'checkpoint.msgpack')==done['checkpoint_sha256']
                artifacts[str(root/'predictions.npz')]=sha(root/'predictions.npz')
            path=OLD/'data'/fold/arm/'dataset.npz';assert sha(path)==read(path.parent/'manifest.json')['dataset_sha256'];artifacts[str(path)]=sha(path)
    freeze(HERE/'protocol.json',dict(question='Did the previously observed benefit of8targetstates restore state-conditioned controller/eta interaction or merely calibrate target global priors?',
        reason='Stage19 identifies a local mixed-training penalty. Historical single-source transfer also failed, so that local cause cannot alone explain allLOSO. Reuse the existing equal-label-budget source/target intervention before launching another optimizer experiment.',
        history_audit='Original crossed_support.py/results contain Qerror, controllercontrast error and eta-centered correlation but no across-state demeaning, within-scene state mismatch or all-fixed falsification. No re-fit is required; all30prediction sets already exist.',
        old_input='Historical state_contact_bilinear;this reanalysis does NOT silently substitute currentcompact64/API-slotinputs.',
        treatment='Reused source64x8C plus either8additional SOURCE states or8 TARGET states x4originalCs x16eta Q1;exactly512extra labels. Wrong-C and wrong-eta associations alreadyfit asnegativecontrols.',
        isolation='All models fixed6000steps and originalSOURCE-only normalization;no target checkpointselection. Crossed arms DO train on8targetstates;NOTzero-shotLOSO.',
        panels='All old16VALstates/scenewith original0..3 andunseen4..5Cs;all opened32states/scene x unseen6..7Cs Q16. Keep source/target separate. Existingconfirmation is nowDEVELOPMENT.',
        primary='State x eta and state x controller x eta skill versuszero andpairedcontrol;projectoutstateandeta marginalsinside everybootstrapreplicate. These exclude perfectper-controller fixedeta priors.',
        secondary='Correct-vs-same-scene-state-mismatch,wrongC,all16fixedpolicies/bootstrapreselectedbest,selectionregret,8/8reversal onQ16and2/2onQ4,collision/numerical censorbounds.',
        reference='Empirical SOURCE-TRAIN eta mean is descriptive only,not a newlytrained or fair pairedeta-onlybaseline. No new baseline performanceclaim.',
        statistics='20000physicalstatepairedbootstrap;samestates/donors jointlyresampled for mismatch;allqualifyingreversalpaircases, symmetric3/8threshold,atleast3stateseachside. Three existingtrainingseedpredictions averaged;individualseedstats retained.',
        current_stage_stop='Do not calltargetQ_MSEgain a state-adaptationrepair unless interaction/mismatch controls supportit;no modelchanges ornewrolloutsinthisstage.',
        literature=[dict(url='https://proceedings.mlr.press/v119/raileanu20a.html',use='PD-VF useslimitedtargetinteraction toinferenvironmentembedding;itsadaptation claimisnotstrictzero-shot. Here targetlabels arediagnostic only.')],
        artifacts=artifacts,code_sha256=sha(__file__),new_fits=0,new_rollouts=0))
    print('Frozen reuse of30fits;no training/query/rollout',flush=True)


def run(fold):
    p=read(HERE/'protocol.json');assert sha(__file__)==p['code_sha256'];source_scene=SCENE[fold]
    datasets={arm:Data(**dict(np.load(OLD/'data'/fold/arm/'dataset.npz'))) for arm in ARMS};final=Data(**dict(np.load(FINAL)))
    predictions={arm:{k:np.asarray([np.load(OLD/'fits'/fold/arm/f'seed{s}/predictions.npz')[k] for s in SEEDS]) for k in ['source','opened_confirmation']} for arm in ARMS}
    source=datasets['source'];tr=source['split']=='train';prior=(source['success'][tr]/np.maximum(source['success'][tr]+source['failure'][tr],1)).mean(0)
    panels=[('source',sc,seen) for sc in SCENE.values() for seen in (True,False)]+[('opened_confirmation',sc,False) for sc in SCENE.values()]
    rows=[];skills=[];comparisons=[];arrays={};saved={};overlap=[]
    for panel,scene,seen in panels:
        d=final if panel=='opened_confirmation' else source
        mask=d['scene']==scene
        if panel=='source':mask&=(d['split']=='validation')&np.isin(d['controller'],[f'analytic{k}' for k in (range(4) if seen else (4,5))])
        ids=np.flatnonzero(mask);ids=np.asarray(sorted(ids,key=lambda i:(d['state_uid'][i],d['controller'][i])))
        keys=[(d['state_uid'][i],d['controller'][i]) for i in ids]
        dd=Data(**{k:v[ids] for k,v in d.items() if k!='eta'});qq={}
        for arm in ARMS:
            a=final if panel=='opened_confirmation' else datasets[arm];lookup={(s,c):i for i,(s,c) in enumerate(zip(a['state_uid'],a['controller']))}
            jj=np.asarray([lookup[k] for k in keys]);qq[arm]=predictions[arm][panel][:,jj]
            ts=set(datasets[arm]['state_uid'][datasets[arm]['split']=='train']);evaluation=set(dd['state_uid']);assert not ts&evaluation
            overlap.append(dict(fold=fold,panel=panel,scene=scene,seen=seen,arm=arm,train_eval_state_overlap=0))
        name=panel+('_seen' if seen else '_unseen');meta=dict(variant=fold,panel=name,scene=scene,controller_group='all')
        pp=dict(qq);pp['state_field_compact']=pp.pop('source');pp['eta_only']=np.broadcast_to(prior,(3,len(ids),16))
        rr=analyze_panel(dd,np.arange(len(ids)),pp,meta,arrays,saved)
        for r in rr:
            if r['arm']=='state_field_compact':r['arm']='source'
            if r['arm']=='eta_only':r['arm']='empirical_source_eta_prior'
            r['target_scene']=scene!=source_scene;r['not_LOSO']=r['arm'].startswith('crossed')
            r['minus_source']=r.pop('minus_state_field_compact');r['minus_empirical_source_eta_prior']=r.pop('minus_eta_only')
            rows.append(r)
        states=np.unique(dd['state_uid']);cs=np.unique(dd['controller']);n,nc=len(states),len(cs);shape=(n,nc,16)
        trials=dd['trials'].reshape(n,nc,1);lo=dd['success'].reshape(shape)/trials;hi=(dd['success']+dd['unknown']).reshape(shape)/trials
        q={arm:a.mean(0).reshape(shape) for arm,a in qq.items()};boot=np.random.default_rng(2026101020+n).integers(n,size=(20000,n));prefix='__'.join([fold,name,scene])
        arrays[prefix+'__mechanism_bootstrap_indices']=boot
        for first,second in PAIRS:
            a,b=selection_difference(lo,hi,q[first].argmax(-1),q[second].argmax(-1));sel=intervals(a.mean(1),b.mean(1),boot,prefix+first+second+'selection',arrays)
            a,b=mse_gain(q[second],q[first],lo,hi);err=intervals(a.mean((1,2)),b.mean((1,2)),boot,prefix+first+second+'Qgain',arrays)
            comparisons.append(dict(**meta,first=first,second=second,N=n,Q_MSE_gain=err,selection_gain=sel))
        a,b=np.triu_indices(nc,1)
        for mode in ['S_eta','S_C_eta']:
            lower,upper=(lo,hi) if mode=='S_eta' else (lo[:,a]-hi[:,b],hi[:,a]-lo[:,b])
            pred=q if mode=='S_eta' else {arm:v[:,a]-v[:,b] for arm,v in q.items()}
            pairs=[(arm,'zero') for arm in ARMS]+PAIRS;dist={pair:[[],[]] for pair in pairs}
            x={arm:center(v[None],True) for arm,v in pred.items()};x['zero']=np.zeros_like(x['source'])
            points={pair:gain(x[pair[0]],x[pair[1]],lower[None],upper[None]) for pair in pairs}
            for start in range(0,len(boot),250):
                ib=boot[start:start+250];x={arm:center(v[ib],True) for arm,v in pred.items()};x['zero']=np.zeros_like(x['source'])
                for pair in pairs:
                    al,au=gain(x[pair[0]],x[pair[1]],lower[ib],upper[ib]);dist[pair][0].append(al);dist[pair][1].append(au)
            for pair in pairs:
                al,au=[np.concatenate(v) for v in dist[pair]];key=prefix+'__'+mode+'__'+pair[0]+'__minus__'+pair[1];arrays[key+'__lower']=al;arrays[key+'__upper']=au;pl,pu=points[pair]
                skills.append(dict(**meta,N=n,controllers=nc,target_scene=scene!=source_scene,interaction=mode,first=pair[0],second=pair[1],
                    identification_bounds=[float(pl[0]),float(pu[0])],censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(au,.975))]))
    write(HERE/f'{fold}_results.json',dict(rows=rows,skills=skills,comparisons=comparisons,overlap=overlap,new_fits=0,new_rollouts=0,not_final_confirmation=True,
        engine_aliases='Saved engine state_field_compact=source;eta_only=empirical_SOURCE_TRAIN_eta_prior. The latterisnot learnedQeta.'))
    np.savez_compressed(HERE/f'{fold}_bootstrap.npz',**arrays);np.savez_compressed(HERE/f'{fold}_predictions_and_controls.npz',**saved)
    print(fold,'completed all six original panels',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--fold',choices=list(SCENE));a=ap.parse_args()
    prepare() if a.stage=='prepare' else run(a.fold)
