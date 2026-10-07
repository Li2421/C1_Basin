"""Freeze source-VAL choices, then open the untouched64-family confirmation."""
import argparse,copy,json,os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import numpy as np
from . import state_support as source
from . import seed_replication as engine
from .support_train import KINDS,SIZES,SEEDS
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
from shared_rollout_db.src.rollout_db import connect,transaction,canonical
OUT=source.OUT.parent/'state_support_confirmation'
EXP='exp_orthoflow3_rootcause_source_confirmation_v1'
read=source.read;write=source.write

def freeze():
    import csv
    if (source.OUT/'models_frozen.json').exists():raise FileExistsError('Already frozen')
    models=[];scores={}
    all_sizes=(*SIZES,*(s+'_pair_equal' for s in SIZES))
    for size in all_sizes:
      for kind in (*KINDS,'native_state_controller'):
        values=[]
        for seed in SEEDS:
            p=source.OUT/'models'/size/kind/f'seed{seed}';done=read(p/'complete.json')
            assert not done['confirmation_labels_used']
            model=dict(size=size,kind=kind,seed=seed,path=str(p),checkpoint_sha256=sha(p/'best.msgpack'),
                normalization_sha256=sha(p/'normalization.json'),source_VAL_NLL=done['best_source_VAL_NLL'])
            models.append(model);values.append(done['best_source_VAL_NLL'])
        scores[f'{size}/{kind}']=float(np.mean(values))
    choices={size:min(KINDS[1:],key=lambda k:scores[f'{size}/{k}']) for size in all_sizes}
    doc=dict(models=models,source_validation_scores=scores,primary_source_VAL_choices=choices,
        primary='Compare each size arm source-VAL-selected context formulation with its eta-only; compare matched sizes for every prespecified model.',
        secondary='Native ordered Markov state + known source controller ID is privileged diagnostic only; not a deployment or cross-scene model.',
        confirmation='All64 previously frozen true-t0 test families, two fixed source eta10/15, both known controllers, standard16 seeds. No exclusions based on outcome.',
        conclusion_rules=['Q prediction gain alone is not robust selection gain','Report absolute B15 oracle headroom',
            'Family-cluster paired bootstrap for NLL and selection; preserve numerical unknown outcomes',
            'Context/state replacement is diagnostic and not model selection','No test-driven architecture, threshold or checkpoint changes'],
        no_target_LOSO_labels=True,old_seed_replication_now_source_TRAIN=True,
        dataset_sha256=sha(source.OUT/'dataset.npz'),protocol_sha256=sha(source.OUT/'protocol.json'))
    doc['pre_confirmation_weighting_control']='Observed-count versus pair-equal pure NLL; two exact eta per family, so pair equality also equalizes family mass. Same observations, same batch draws, same optimizer and steps; both selected by the same full16-seed source VAL NLL.'
    write(source.OUT/'models_frozen.json',doc);print(json.dumps({'selected':choices,'VAL':scores},indent=2))

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Confirmation already registered')
    frozen=read(source.OUT/'models_frozen.json')
    for m in frozen['models']:assert sha(Path(m['path'])/'best.msgpack')==m['checkpoint_sha256']
    states=copy.deepcopy([s for s in read(source.OUT/'states.json') if s['split']=='test'])
    assert len(states)==64
    mapping={s['uid']:i for i,s in enumerate(states)}
    for i,s in enumerate(states):s['index']=i;s['repair_index']=i
    pairs=copy.deepcopy(read(source.OUT/'pairs_confirmation.json'))
    for p in pairs:p['state_index']=mapping[p['state_uid']]
    protocol=copy.deepcopy(read(source.OUT/'protocol.json'))
    protocol.update(experiment_uid=EXP,experiment='sealed_source_confirmation',requested_seed_count=4096,
        models_frozen_sha256=sha(source.OUT/'models_frozen.json'),target_labels_used=False,
        scope='Previously sealed new source families; same known controllers and exact two eta, not LOSO or unseen eta.')
    write(OUT/'protocol.json',protocol);write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    requests=[]
    for c in protocol['profiles']:
        part=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[canonical({'future_index':k}) for k in range(16)]) for p in pairs]
        write(OUT/f'planned_{c["name"]}.json',{'requests':part});requests+=part
    write(OUT/'planned_rollouts.json',{'requests':requests})
    with connect() as db,transaction(db):
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'sealed_rootcause_source_confirmation',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'models_frozen_sha256':protocol['models_frozen_sha256']})))
    write(OUT/'working_state.json',dict(phase='prepared_after_model_freeze',families=64,new_rollouts=0))

def configure():engine.OUT=OUT;engine.EXP=EXP

def physical():
    from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
    configure();engine.configure();native.physical()

def features(index):
    from . import support_inputs
    # Reuse exact instrument and feature code, with a copied pair split only
    # for this input-only function; no outcomes can enter the features.
    import jax
    from .agent_response import instrument
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
    assert jax.default_backend()=='gpu'
    c=read(OUT/'protocol.json')['profiles'][index];target=OUT/f'inputs_{c["name"]}.npz'
    if target.exists():raise FileExistsError('Confirmation inputs already frozen')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20};rt=rc.RichRuntime('ring_exchange',c['path']);trace=instrument(rt)
    states=read(OUT/'physical.json');pairs=read(OUT/'pairs.json');n=len(pairs)
    context=np.zeros((n,24),np.float32);response=np.zeros((n,4,16),np.float32);valid=np.zeros(n,bool);errors=[]
    for i,p in enumerate(pairs):
        assert states[p['state_index']]['state_uid']==p['state_uid'];trace['runs']=[]
        try:
            f=rt.features(states[p['state_index']]['physical'],p['eta']);assert len(trace['runs'])==4 and all(len(r)==20 for r in trace['runs'])
            r=np.array([np.array(run).mean(0) for run in trace['runs']]);context[i]=f['mean'];response[i]=np.concatenate([r[[0,2]].mean(0),r[[1,3]].mean(0)],-1)
            assert np.isfinite(context[i]).all() and np.isfinite(response[i]).all();valid[i]=True
        except Exception as exc:errors.append(dict(pair=i,error=f'{type(exc).__name__}: {exc}'))
    np.savez_compressed(target,context=context,agent_response=response,valid=valid)
    write(OUT/f'input_audit_{c["name"]}.json',dict(invalid=errors,outcome_labels_used=False,models_frozen_sha256=sha(source.OUT/'models_frozen.json')))

def dataset():
    if (OUT/'dataset.npz').exists():raise FileExistsError('Confirmation table frozen')
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    p=read(OUT/'pairs.json');n=len(p);s=np.zeros((2,n),np.float32);f=np.zeros_like(s);num=np.zeros_like(s);keys=[]
    with connect(True) as db:
      for ci,c in enumerate(read(OUT/'protocol.json')['profiles']):
       for i,pair in enumerate(p):
        records=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(pair['state_uid'],pair['eta_uid'],c['controller_uid'])).fetchall()
        assert len(records)==16 and {r['seed_key'] for r in records}=={canonical({'future_index':k}) for k in range(16)}
        assert all(r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined'] for r in records)
        valid=[r for r in records if not r['numerical_failure']];s[ci,i]=sum(r['success'] for r in valid);f[ci,i]=len(valid)-s[ci,i];num[ci,i]=16-len(valid)
        keys.append(dict(state_uid=pair['state_uid'],eta_uid=pair['eta_uid'],controller_uid=c['controller_uid'],rollout_uids=[r['rollout_uid'] for r in records]))
    inputs=[np.load(OUT/f'inputs_{c}.npz') for c in ('alt','second')]
    np.savez_compressed(OUT/'dataset.npz',success=s,failure=f,numerical=num,standard_success=s,standard_failure=f,
        context=np.stack([r['context'] for r in inputs]),agent_response=np.stack([r['agent_response'] for r in inputs]),valid=np.stack([r['valid'] for r in inputs]),
        eta=np.array([r['eta'] for r in p],np.float32),eta_index=np.array([r['eta_index'] for r in p]),state_index=np.array([r['state_index'] for r in p]))
    write(OUT/'dataset_DB_keys.json',keys)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','prepare','physical','filter','run','features','dataset'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action in ('freeze','prepare','physical','dataset'):globals()[a.action]()
    elif a.action=='features':features(a.index)
    else:
        configure()
        if a.action=='filter':engine.filter_numerical()
        else:engine.run(engine.CONTROLLERS[a.index//6],a.index%6)
