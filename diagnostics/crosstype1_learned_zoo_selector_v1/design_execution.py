"""Exact execution identities and official registration/preflight after stage gates."""
from common import *
from registry import register,normalized_state
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy

ARMS=('safety_baseline','F10','F13','S100','S300','S500','Sprog','RHR')
RULE=dict(mode='progress',steps=40,speed=.35)
LOCKED=('implicit','ring_ippo')


def type_of(name,cp):
    kind=cp['payload'].get('controller_class','Flow')
    return 'gaussian' if kind=='G' else kind


def profiles():
    p=read(HERE/'HISTORICAL_BASE_PROTOCOL.json')
    for cfg in read(HERE/'TRAINING_CONFIGS.json') if (HERE/'TRAINING_CONFIGS.json').exists() else []:
        folder=HERE/'policies'/cfg['name']
        receipt=folder/'COMPLETE.receipt.json'
        if not receipt.exists():continue
        for path,digest in read(receipt)['files'].items():assert sha(path)==digest
        source=copy.deepcopy(p['profiles'][cfg['source_instance']]['payload'])
        source.update(controller_class=cfg['family'],flow_checkpoint_path=str(folder/'best.msgpack'),
              flow_checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),
              policy_config_sha256=sha(folder/'CONFIG.json'),observation_frame=cfg['frame'],
              adapter_manifest_path=cfg['manifest'],source_Flow_recipe=cfg['recipe'],
              conditioning='nominal_TT_project_correct_reproject',execution_semantics='crosstype1_learned_TT_v1',
              placement='TT',chain='TT',nominal_implementation_sha256=sha(HERE/'zoo_runtime.py'),
              model_implementation_sha256=sha(HERE/'zoo_models.py'),jax_enable_x64=False)
        p['profiles'][cfg['name']]=dict(controller_uid=uid('ctl',source),payload=source)
    if (HERE/'IPPO_TRAINING_CONFIGS.json').exists():
        for cfg in read(HERE/'IPPO_TRAINING_CONFIGS.json'):
            folder=HERE/'policies'/cfg['name']
            if not (folder/'CHECKPOINT_SELECTED.receipt.json').exists():continue
            for path,digest in read(folder/'CHECKPOINT_SELECTED.receipt.json')['files'].items():assert sha(path)==digest
            source=copy.deepcopy(p['profiles'][cfg['source_instance']]['payload'])
            source.update(controller_class='ring_ippo',flow_checkpoint_path=str(folder/'best.msgpack'),
                   flow_checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),
                   policy_config_sha256=sha(folder/'CONFIG.json'),observation_frame=cfg['frame'],
                   adapter_manifest_path=cfg['manifest'],placement='TT',chain='TT',jax_enable_x64=False,
                   conditioning='nominal_TT_project_correct_reproject',execution_semantics='crosstype1_ring_ippo_TT_v1',
                   nominal_implementation_sha256=sha(HERE/'zoo_runtime.py'),model_implementation_sha256=sha(HERE/'ippo.py'))
            p['profiles'][cfg['name']]=dict(controller_uid=uid('ctl',source),payload=source)
    p['types']={name:type_of(name,cp) for name,cp in p['profiles'].items()}
    return p


def case(state,name,cp,p,arm,seed=0,probe=False):
    v=copy.deepcopy(cp['payload'])
    eta=[0.,0.,0.] if arm in ('nominal_only','safety_baseline','RHR') else p['eta'][10 if arm=='F10' else 13]
    after=p['eta'][10] if arm in ('S100','S300','S500','Sprog') else eta
    switch=int(arm[1:]) if arm in ('S100','S300','S500') else 700
    if arm=='nominal_only':
        v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round38_nominal_only_v1')
    elif arm in ('S100','S300','S500'):
        v['eta_schedule']=dict(eta_a=eta,eta_b=after,switch_step=switch,action_index_rule='a if index<t else b')
    elif arm=='Sprog':
        v['progress_schedule']=dict(pstar=.9681620884830543,fallback=600,implementation_sha256=sha(HERE/'progress.py'))
    elif arm=='RHR':
        v['continuing_right_hand']=dict(rule=RULE,implementation_sha256=sha(HERE/'right_hand.py'),trigger_sha256=sha(HERE/'trigger.py'))
    if probe:
        v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=300,
                 terminate_on_success=False,terminate_on_collision=False,
                 probe_code_sha256=sha(HERE/'rollout_runtime.py'))
    cid=uid('ctl',v);eid=eta_identity(eta)[0];sk=canonical(dict(future_index=seed))
    c=dict(rollout_uid=uid('probe' if probe else 'roll',dict(state=state['state_uid'],eta=eid,controller=cid,seed=sk)),
           state_uid=state['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,
           family=p['types'][name],arm=arm,seed=seed,eta_a=eta,eta_b=after,switch_step=switch,
           placement='FF' if p['types'][name]=='Flow' else 'TT',state_index=state.get('study_index',0),
           rhr_rule=RULE if arm=='RHR' else None)
    return c,dict(controller_uid=cid,payload=v)


def design(stage,cohort,split,selection='all'):
    prereg();gates=read(HERE/'REPRODUCTION_GATE.json')['type_passed']
    p=profiles();names=[]
    for name,cp in p['profiles'].items():
        kind=p['types'][name]
        if (cohort=='locked')!=(kind in LOCKED):continue
        key='base_u' if kind=='Flow' else 'G' if kind=='gaussian' else kind
        if key in gates and not gates[key]:continue
        if selection=='new_only' and kind not in ('chunking','recurrent','cvae','implicit','ring_ippo'):continue
        names.append(name)
    assert selection in ('all','new_only')
    if cohort=='locked' and split in ('probe','evaluation','tuning'):
        frozen=read(HERE/'FROZEN_PIPELINE.receipt.json')
        for path,digest in frozen['files'].items():assert sha(path)==digest
        if split in ('evaluation','tuning'):
            chosen=read(HERE/'LOCKED_CHOICES.receipt.json')
            for path,digest in chosen['files'].items():assert sha(path)==digest
    if split!='capability':
        cap=read(HERE/f'{cohort.upper()}_CAPABILITY.json')
        names=[name for name in names if name in cap['eligible']]
    else:
        # Historical instances keep their already established eligibility.
        names=[name for name in names if p['types'][name] in ('chunking','recurrent','cvae','implicit','ring_ippo')]
    statefile={'capability':'PILOT_STATES.json','probe':'PROBE_STATES.json',
               'evaluation':'EVAL_STATES.json','tuning':'TUNING_STATES.json'}[split]
    p['states']=[s if 'transport_physical' in s else normalized_state(s) for s in read(HERE/statefile)]
    p['controller_order']=names;p['phase']=split
    armnames={'capability':('safety_baseline','S500','F10'),'probe':('safety_baseline','F13','F10'),
              'evaluation':('nominal_only',)+ARMS,'tuning':ARMS}[split]
    seeds=(0,1) if split=='evaluation' else (0,)
    old_candidates={}
    if split in ('evaluation','tuning'):
        from history_audit import signature
        source_states={}
        for r in read(HERE/'HISTORY_REUSE_CANDIDATES.json'):
            if r['split']==split:
                c=r['source']['case'];path=r['source']['protocol']
                if path not in source_states:
                    source_states[path]={s['state_uid']:s for s in read(path)['states']}
                original=source_states[path][c['state_uid']]
                # Historical DB state aliases can differ while physical state
                # and the original RNG token are identical. Reuse their exact
                # registered request; never relabel or execute the alias again.
                old_candidates[(r['controller'],signature(original),r['arm'],c['seed'])]=r['source']
    rows=[];cps={};reuse=[]
    with rollout_db.connect(True) as db:
        for name in names:
            for si,s in enumerate(p['states']):
                for arm in armnames:
                    for seed in seeds:
                        source=old_candidates.get((name,signature(s),arm,seed)) if split in ('evaluation','tuning') else None
                        if source:
                            c=copy.deepcopy(source['case']);c['state_index']=si
                            assert db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is not None
                            old=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()
                            assert old is not None
                            cp=dict(controller_uid=c['controller_uid'],payload=json.loads(old[0]))
                            c['placement']='FF' if p['types'][name]=='Flow' else 'TT'
                            reuse.append(source)
                        else:
                            c,cp=case(s,name,p['profiles'][name],p,arm,seed,split=='probe')
                            c['state_index']=si
                        rows.append(c);cps[cp['controller_uid']]=cp
    identity=HERE/'EXPERIMENT_REGISTERED_IDENTITY.json'
    if not identity.exists():
        write(identity,dict(experiment_uid=EXP,name='Cross-type learned zoo selector 1',path=str(HERE),
              protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(HERE/'registry.py'),
              metadata_json=canonical(dict(series='crosstype',round=1))))
    write(HERE/f'{stage}_HISTORY_REUSE.json',reuse)
    register(stage,p,rows,cps)
    from execution_receipt import freeze
    freeze(stage,cohort=cohort,split=split)

if __name__=='__main__':design(*sys.argv[1:])
