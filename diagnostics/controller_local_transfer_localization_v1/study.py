"""Source-only local execution LOSO after the cached noise repair passed.

This resumes the previously pruned local-vs-return localization with adequate
source supervision. No new rollouts, controller queries, or return fits.
"""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_noise_coupling_v1'
EVAL=ROOT/'controller_local_to_return_probe_v1/vector_control'
FOLDS={'toy':'toy_give_way','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
ARMS=('motion_complete','motion_eta_only')
RUNS=[dict(fold=f,seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    prior=read(BASE/'results.json');assert prior['gate_passed']
    d=dict(np.load(BASE/'dataset.npz'));tr=d['split']=='train'
    targets=[dict(np.load(BASE/'targets'/f'motion_pair{i}_common_complete.npz')) for i in range(6)]
    ss=targets[0]['success'].copy();ff=targets[0]['failure'].copy()
    # Every one of the four keys occurs in exactly three of the six pairs.
    ss[tr]=sum(t['success'][tr] for t in targets)/3
    ff[tr]=sum(t['failure'][tr] for t in targets)/3
    active=np.repeat(d['entity_agent_mask'].astype(bool),2,-1)
    np.testing.assert_allclose((ss+ff)[tr][:,[0,5,10,15]],np.broadcast_to(4*active[tr,None,:],(int(tr.sum()),4,8)),atol=1e-12,rtol=0)
    for t in targets:
        np.testing.assert_array_equal(t['success'][~tr],ss[~tr]);np.testing.assert_array_equal(t['failure'][~tr],ff[~tr])
    path=HERE/'dataset.npz'
    if not path.exists():path.symlink_to((BASE/'dataset.npz').resolve())
    out=HERE/'targets';out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for arm in ARMS:
        np.savez_compressed(out/(arm+'.npz'),success=ss,failure=ff)
        hashes[arm]=sha(out/(arm+'.npz'))
    assert hashes[ARMS[0]]==hashes[ARMS[1]]
    # Reuse the exact Stage30 trainer, net, optimization and source-VAL rule.
    code=(BASE/'learn.py').read_text()
    (HERE/'learn.py').write_text(code)
    p=copy.deepcopy(read(BASE/'protocol.json'))
    p['schema']='cached_local_FF_execution_LOSO_after_noise_repair_v1'
    p['arms']=ARMS;p['folds']=FOLDS
    p['experiment']=dict(
        code_sha256=sha(HERE/'learn.py'),
        question='Can the complete physical+field+known-initializer input learn transferable LOCAL state/controller/eta action effects after source local supervision was repaired? This distinguishes local mapping failure from failure emerging only in long-return composition.',
        history='Stage29 local source gate failed3scenes, so72heldfits correctly never run. Stage30 all6paired-noise panels now validate source state_eta learning inDB/Four/Ring. No previous heldscene local target fits using this repaired supervision. Old action-as-Qinput experiments are a DIFFERENT question.',
        cache='Reuse all4 alreadycached static action keys; pooled objective equals average ofall6previous two-key panel objectives. No newcontroller queries/rollouts. Original UID action labels reserved as independent-noise scientific evaluation; SOURCEVAL uses original UID labels only onsource scenes.',
        inputs='Exact existing72 physical/APIrole+compact64+initializer8; no response preview/sceneID, local actions are TARGET only.',
        checkpoint='SourceVAL local state_eta MSE, completeaction MSE, NLL thenearliest. Same6000steps/architecture/budget andTRAINlabels for eta-only. No heldscene fitting, normalization or selection.',
        primary='All40oldnaturalheldstates per scene, separately32TRAIN-origin and8VAL-origin; all4controllers,K4,allactive8or4actioncomponents. IndependentUIDnoisetargets. Physicalstate bootstrap20000,recenter eachresample. state_eta/triple skill relative tozero-interactionnull, matched-vs-same-scene state permutation, correct-vs-mismatched C; sourceVAL learning separately checked.',
        interpretation='Only claim failureemergesafterlocalmapping for a fold ifsourceANDheld local state_eta skill positive and matched-vs-mismatch positive; controllertransfer additionally needs positive triple skill wheretruehalfsignal supported. Local failure despitepositive sourcelearning leaveslocal representation/source support unresolved. No local success substitutedforQ/eta-selection success.',
        boundary='OpenedDEVELOPMENT, notfreshconfirmation. FoursharedTRAINnoise keys: CIs conditionalonthatpanel, independentUIDevaluationnoise. Partialcontrollertriplepower mustbereported, no cherry-pickedscene.',
        literature=[dict(url='https://arxiv.org/abs/2011.03506',relevance='Valueequivalence distinguishes fitting local dynamics from fitting value-relevant structure. Local success neithernecessarynor sufficientforgeneralvalue transfer; this is empirical localization, notan importedguarantee.'),
                    dict(url='https://arxiv.org/abs/1606.05312',relevance='Priorlocal-versus-returnmotivation; its fixed-dynamics rewardtransfer theorem doesnotcoverourscenechange.')],
        hardware='Newfits allsameGPUclass, highestfloat32matmulprecision. Stage30CPUweightsnotretrained; no numericpairedcomparison againstoldCPUfits claimed.',
        new_fits=24,new_rollouts=0,new_queries=0)
    freeze(HERE/'protocol.json',p)
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(path),target_sha256=hashes,
        source_dataset_sha256=sha(BASE/'dataset.npz'),source_targets_sha256={str(i):sha(BASE/'targets'/f'motion_pair{i}_common_complete.npz') for i in range(6)},
        rows=len(d['scene']),physical_states=len(set(d['state_uid'])),new_rollouts=0,new_queries=0))
    freeze(HERE/'source_gate.json',dict(passed=True,authority=str(BASE/'results.json'),authority_sha256=sha(BASE/'results.json'),
        interpretation='PreviouslypassedpairedsourceLOCALgateauthorizeshelddiagnosticfits. Doesnotguaranteeeach3sourcefoldlearns; fold-specific SOURCEVALskill reviewedbeforeinterpretingheldfailure.'))
    freeze(HERE/'history_audit.json',dict(paths=[str(BASE),str(EVAL),str(ROOT/'controller_ff_action_mediation_v1')],
        evidence_gap='Source-only LOCAL-target transfer after noise coupling has never been fitted. PriorlearningwasJOINTall4scenes; oldheldtests predictedQ fromactions,notactions fromphysical/controller inputs.',
        reused_original_UID_label_sha256=sha(EVAL/'motion_raw.npz'),reused_static_label_sha256=sha(EVAL/'independent_static_actions.npz'),
        no_repeat_Q_fits=True,no_new_rollout=True))
    print(dict(new_fits=24,new_rollout=0,new_query=0))


def fit(index):
    from . import learn
    run=RUNS[index]
    for arm in ARMS:
        learn.fit(learn.JOBS.index(dict(**run,arm=arm)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
