"""Fill missing crossed-controller labels on existing natural SOURCE states."""
import argparse,copy,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from diagnostics.controller_crossed_precision_v1.study import read,write,freeze,sha,SEEDS
from diagnostics.controller_ff_factorial_repair_v1 import study as original_matrix
from diagnostics.controller_response_families_v1 import study as engine
from diagnostics.controller_q_diversity_v1 import experiment as original
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'controller_crossed_precision_v1'
FOLDS={'joint':None}
ARMS=('eta_only','union_native_slots')
EXPERIMENT='exp_controller_ring_crossed_128_Q4_v1'


def prepare():
    assert read(BASE/'completion.json')['status']=='COMPLETE_SOURCE_PRECISION_DIAGNOSTIC'
    d=dict(np.load(BASE/'dataset.npz'));old=read(original_matrix.HERE/'protocol.json')
    ring=read(ROOT/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    matrix=[s for s in old['states'] if s['scenario']=='ring_exchange' and s['study_split']=='train'];assert len(matrix)==32
    old_ids={s['state_uid'] for s in matrix}
    native={str(s):i for i,s in enumerate(d['state_uid']) if d['scene'][i]=='ring_exchange' and d['controller'][i]=='native_flow' and d['split'][i]=='train' and d['source'][i]=='native_ring512'}
    assert len(native)==512 and old_ids<=set(native)
    # Original generator order was frozen before all controller outcomes.
    chosen=[copy.deepcopy(s) for s in ring['states'] if s['state_uid'] in native and s['state_uid'] not in old_ids][:96]
    assert len(chosen)==96
    for j,s in enumerate(chosen):s.update(study_split='train',study_index=32+j)
    held_hash=set(d['physical_hash'][d['split']=='validation'])
    assert not held_hash&{d['physical_hash'][native[s['state_uid']]] for s in chosen}
    p=copy.deepcopy(old);p['states']=chosen
    p['controllers']={'ring_exchange':{k:v for k,v in old['controllers']['ring_exchange'].items() if k.startswith('flow_')}}
    p.update(schema='crossed_ring_source128_Q4_v1',seeds=[0,1,2,3],train_seeds=[0,1,2,3],
        state_selection='First96originalgenerator-order nativeRingTRAIN states outsideold32crossedTRAIN. No outcome/controller/critic filtering. Allalreadyregisteredphysicalstates; sameglobalnaturaldistribution. No newphysicalstategeneration.',
        source_state_coverage='32->128independentstatesPERcrossedFlowcontroller. GlobaldistinctphysicalTRAINstatesunchanged:the96statesalreadyhadnativeFlowlabels. This isolates missingjointS/Ccoverage,notnewglobalphysicalstates.',
        purpose='AfterQ2->Q4alonepartlyimprovedcalibrationbutfailedinteractionrepair, testfourfoldcrossedstatecoverage atfixedQ4precision.',
        scope='SOURCE DEVELOPMENT,actualFF,allfiveFlowreferenceoperators,unchangedfourprefixedeta. No K16/B15orLOSOclaim.',
        expected_new_outcomes=7680,expected_reuse_outcomes=0,
        prior_audit=['controller_crossed_support_audit_v1','controller_crossed_seed_replication_v1','controller_crossed_precision_v1',
                     'controller_cached_union_v1','controller_cross_scene_root_v2/UNION_CACHE_COMPATIBILITY_AUDIT.json'],
        compatibility='SharedDBexactcontrollerUIDsupportaudited;othernative/analytic/TTlabels retainedbutcannotrelabeltheir outcomes as these5FFoperators. Everycompatiblelabel mustbereused.',
        learning='Samecurrentphysical/API-slot+compact64,6000steps,native20%sampler,sourceVALrule. FreezeoldSOURCE TRAINnormalizationtoavoidmixingnormalizationchange withcoverage. Fair eta-onlysameexpandedlabels/sampling.',
        primary='IndependentSOURCEVAL8Ringstates(allfiveC) error/interaction versusQ4N32;other3SOURCEscenesretained. ExistingVALsource-selected,soanypositivegateisnotfreshconfirmation.',
        gate='Prospective source-screen:meanRingVALQ_MSE andSxeta_MSE each<=.85ofN32Q4;>=2/3fitseedsimproveboth; globalSOURCEVALregret<=old+.01. Ifpass,independentheld-physicalstateconfirmationrequired;ifnot,doNOTblindlyexpandto512orretunearchitecture.',
        secondary='FreezeN32Q4modelsbeforelabels;inferon96new(S,C)combinationsbeforetraining. TheyareunseencombinationsbutphysicalstateswerealreadyseenwithnativeC. Saveforcausaldiagnosis;aftertrainingtheyareTRAIN,notTEST.',
        no_automatic_large_evaluation=True,
        literature=dict(url='https://arxiv.org/abs/2002.11833',use='Valueevaluationrequiresstates/policysupervision;precisionandcoveragecontrolledseparately. Doesnotguaranteeourcross-sceneextension.'),
        bootstrap='20000pairedphysical-stateunits;retainnumericalbounds/mismatch/allfixed/reversal. No rolloutpseudoN.',
        new_code_sha256=sha(__file__))
    for name,src in [('runtime.py',original_matrix.HERE/'runtime.py'),('static_operator_audit.json',original_matrix.HERE/'static_operator_audit.json')]:
        f=HERE/name
        if not f.exists():f.symlink_to(src.resolve())
    # Reuse the exact training implementation; the single preprocessing change
    # freezes the old SOURCE normalization rather than refitting it on added rows.
    code=(BASE/'learn.py').read_text()
    oldnorm="values=features[tr][mask[tr]];mean=values.mean(0);scale=np.maximum(values.std(0),.05)"
    newnorm="frozen_norm=read(BASE/'fits/joint'/job['arm']/f\"seed{job['seed']}\"/'normalization.json');mean=np.asarray(frozen_norm['mean'],np.float32);scale=np.asarray(frozen_norm['scale'],np.float32)"
    assert code.count(oldnorm)==1
    code=code.replace('from .study import HERE,FOLDS,ARMS,SEEDS,read,write,freeze,sha','from .study import HERE,BASE,FOLDS,ARMS,SEEDS,read,write,freeze,sha').replace(oldnorm,newnorm)
    anchor='    pools={};coverage=[];sampling_weights={}'
    assert code.count(anchor)==1
    addition="    old_checkpoint=BASE/'fits/joint'/job['arm']/f\"seed{job['seed']}\"/'checkpoint.msgpack'\n    old_params=serialization.from_bytes(params,old_checkpoint.read_bytes())\n    pre=infer(old_params,np.arange(len(d['scene'])))\n    np.savez_compressed(out/'pre_fit_predictions.npz',probability=1/(1+np.exp(-np.clip(pre,-80,80))),state_uid=d['state_uid'])\n"
    code=code.replace(anchor,addition+anchor)
    f=HERE/'learn.py'
    if f.exists():assert f.read_text()==code
    else:f.write_text(code)
    p['experiment']=dict(code_sha256=sha(f),intervention='OnlycrossedS/CsupportQ4N32->N128;normalizationfrozenfromN32sourceTRAIN')
    training=read(BASE/'protocol.json')
    for k in ('training','native_sampling_mass','eta_input_center','eta_input_radius','arms','seeds'):
        if k=='seeds':continue
        p[k]=copy.deepcopy(training[k])
    p['training_seeds']=SEEDS;p['folds']=FOLDS
    frozen=[]
    for arm in ARMS:
        for seed in SEEDS:
            f=BASE/'fits/joint'/arm/f'seed{seed}/checkpoint.msgpack';frozen.append(dict(arm=arm,seed=seed,checkpoint=str(f),sha256=sha(f)))
    freeze(HERE/'models_frozen.json',frozen);p['frozen_models_sha256']=sha(HERE/'models_frozen.json')
    freeze(HERE/'protocol.json',p)
    cases=[original.case(p,s,c,j,k) for s in chosen for c in p['controllers']['ring_exchange'] for j in p['eta_indices'] for k in [0,1,2,3]]
    cases.sort(key=lambda c:(c['state_index'],c['controller_name'],c['eta_index'],c['seed']))
    counts=Counter();missing=[];groups=defaultdict(list);support=[]
    with connect(True) as db:
        profiles=list(db.execute('SELECT controller_uid,config_json FROM controller_config'))
        for name,c in p['controllers']['ring_exchange'].items():
            row=next(x for x in profiles if x['controller_uid']==c['controller_uid']);assert json.loads(row['config_json'])==c['payload']
            a=db.execute('SELECT count(*) n,count(distinct state_uid) s FROM rollout WHERE controller_uid=?',(c['controller_uid'],)).fetchone()
            support.append(dict(controller=name,controller_uid=c['controller_uid'],outcomes=a['n'],states=a['s']))
        for c in cases:
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if a is None:missing.append(c)
            else:
                assert not a['conflict_quarantined'] and all(a[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if a['numerical_failure'] else 'exact_reuse']+=1
            groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
    counts['genuinely_missing']=len(missing)
    freeze(HERE/'cases.json',cases);write(HERE/'missing_cases.json',missing)
    write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    write(HERE/'cache_history_audit.json',dict(profiles_scanned=len(profiles),exact_profiles=support,
        physical_states_reused=96,new_physical_states=0,new_crossed_outcomes=len(missing),state_val_overlap=0,
        feature_reuse='Nativephysicalentityarrays andnominal14areexactcache-reuse;onlymissingstaticaffinefieldresponsesqueried. NoEnv.step.',
        old_global_state_count_unchanged=True))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Ring source crossed coverage128Q4',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=24,new_rollouts_max=7680))))
    print(dict(counts,new_physical_states=0,new_crossed_states=96),flush=True)


def worker(index,workers):
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    p=read(HERE/'protocol.json');assert all(sha(Path(f))==h for f,h in p['source_code_hashes'].items())
    engine.HERE=HERE;engine.EXPERIMENT=EXPERIMENT
    engine.EndpointMatchedField=lambda base,name:FlowEndpointMatchedField(base,name.removeprefix('flow_'))
    engine.worker(index,workers)


def merge():
    engine.HERE=HERE;engine.EXPERIMENT=EXPERIMENT;engine.merge()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','worker','merge']);a.add_argument('--index',type=int,default=0);a.add_argument('--workers',type=int,default=144);q=a.parse_args()
    worker(q.index,q.workers) if q.stage=='worker' else globals()[q.stage]()
