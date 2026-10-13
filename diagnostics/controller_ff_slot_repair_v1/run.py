"""Reuse all factorial evidence; intervene only on native agent-slot retention."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import read,write,freeze,sha

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'controller_ff_factorial_repair_v1'
FOLDS=('joint','held_toy','held_db','held_four','held_ring')
SEEDS=(17,23,41)
JOBS=[dict(fold=f,condition='encounter_native_slots',seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    freeze(HERE/'intervention_protocol.json',dict(
        hypothesis='The controller-conditioned value need not be invariant to active reindexing of a fixed ordered controller API. Current entity pooling can discard this information.',
        historical_reuse_audit=dict(
            prior='controller_entity_symmetry_probe_v1/development_results.json',
            matched_control='controller_ff_factorial_repair_v1/fits',
            new_evidence_gap='No same-architecture slot intervention on the four-scene actualFF crossed-controller panel existed. Reuse all75oldfits,15360outcomes andcachedfeatures; train only15newslot fits.'),
        change='Set four unused zero nominalFlow physical-entity channels3:7 to one-hot native agent slot, masked at padding. Controllerresponse83, all actual physicalfeatures, parameter shapes, initialization, sampling, source-only normalizers,optimizer,steps and checkpoint rules unchanged.',
        input_boundary='Diagnostic nativeAPIrole input, no sceneID, no extra controller queries. Reservedzerochannel repurposing is local to this arm; existing featurefiles stay unchanged. Doesnot promote slotIDs as sufficient transferable controller encoding.',
        data_sha256=sha(OLD/'dataset.npz'),raw_sha256=sha(OLD/'raw_labels.npz'),
        training_rule_sha256=sha(OLD/'analysis_protocol.json'),
        source_code_sha256={n:sha(OLD/n) for n in ('learn.py','review.py','fit_diagnostics.py')},
        wrapper_sha256=sha(__file__),jobs=JOBS,new_rollouts=0,new_features=0,
        baseline_refits=0,opened_states_role='DEVELOPMENT',
        gate='Require improved Q/state interaction and correct-S versus same-scenewrong-S; evaluate all4LOSO folds and joint. Fixedeta/Cswap controls retained. Successalone insufficient.',
        caveat='This is adaptive scientific diagnosis onopeneddata. No heldlabels enter eachfoldfit/normalization/checkpoint, but no independentfinalclaim without newuntouchedstates.',
        literature=['https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html',
                    'https://proceedings.mlr.press/v97/wagstaff19a.html']))
    for name in ('dataset.npz','raw_labels.npz','dataset_audit.json','analysis_protocol.json','evaluation_protocol.json'):
        path=HERE/name
        if not path.exists():path.symlink_to(OLD/name)
    for fold in FOLDS:
        dest=HERE/'fits'/fold;dest.mkdir(parents=True,exist_ok=True)
        for source in (OLD/'fits'/fold).iterdir():
            if source.is_dir() and not (dest/source.name).exists():(dest/source.name).symlink_to(source,target_is_directory=True)
    snap=HERE/'code_snapshots';snap.mkdir(exist_ok=True)
    for name in ('learn.py','review.py','fit_diagnostics.py'):
        target=snap/name
        if not target.exists():shutil.copy2(OLD/name,target)


def fit(index):
    from diagnostics.controller_ff_factorial_repair_v1 import learn
    p=read(HERE/'intervention_protocol.json')
    assert sha(OLD/'learn.py')==p['source_code_sha256']['learn.py']
    original_load=learn.load
    def role_inputs():
        d=original_load()
        assert (d['entity_agents'][...,3:7]==0).all()
        am=d['entity_agent_mask'];assert am.shape[1]==4
        d['entity_agents'][...,3:7]=np.eye(4)[None]*am[...,None]
        return d
    learn.HERE=HERE;learn.load=role_inputs;learn.JOBS=JOBS
    learn.fit(index)
    j=JOBS[index];out=HERE/'fits'/j['fold']/j['condition']/f"seed{j['seed']}"
    control=OLD/'fits'/j['fold']/'encounter_same_all_labels'/f"seed{j['seed']}"
    assert read(out/'normalization.json')==read(control/'normalization.json')
    a,b=read(out/'job.json'),read(control/'job.json')
    for k in ('source_train_state_ids','source_val_state_ids','dataset_sha256','rule_sha256'):
        assert a[k]==b[k],k
    write(out/'intervention_audit.json',dict(wrapper_sha256=sha(__file__),protocol_sha256=sha(HERE/'intervention_protocol.json'),
        exact_source_split_and_normalization_parity=True,new_rollouts=0,unchanged_network_shapes_and_seed=True))


def review():
    from diagnostics.controller_ff_factorial_repair_v1 import review as r,fit_diagnostics as f
    conditions=tuple(r.CONDITIONS)+('encounter_native_slots',)
    original_corr=r.corr
    def corr(a,b):
        if min(np.asarray(a).std(),np.asarray(b).std())<1e-7:return None
        return original_corr(a,b)
    r.HERE=HERE;r.CONDITIONS=conditions;r.corr=corr;r.run()
    f.HERE=HERE;f.CONDITIONS=conditions;f.corr=corr;f.run()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='fit':fit(a.index)
    else:review()
