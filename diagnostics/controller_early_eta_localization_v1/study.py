"""One-factor input-routing intervention, with unchanged network parameters."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_local_transfer_localization_v1.study import FOLDS,ARMS,SEEDS,EVAL,read,write,freeze,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_local_transfer_localization_v1'
RUNS=[dict(fold=f,arm='motion_complete',seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    d=np.load(BASE/'dataset.npz')
    assert np.max(abs(d['entity_agents'][...,10:13]))==0
    for name in ('dataset.npz','targets','dataset_audit.json','source_gate.json'):
        dest=HERE/name
        if not dest.exists():dest.symlink_to((BASE/name).resolve())
    for fold in FOLDS:
        dest=HERE/'fits'/fold/'motion_eta_only';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to((BASE/'fits'/fold/'motion_eta_only').resolve())
    p=copy.deepcopy(read(BASE/'protocol.json'));p['schema']='single_factor_early_eta_local_v1'
    p['experiment']=dict(code_sha256=sha(HERE/'learn.py'),net_sha256=sha(HERE/'net.py'),
        history='Stage35sourceVAL state_eta skill positive12/12 scene/folds; heldDB/Four skillnegative, Toy/Ringuncertain, all4triple skillnegative. Therefore future unknown Flow/long-return alone insufficient. Oldbilinearheads fuseetaafterphysicalpooling; no existing prepool eta-routing control found in historicalcode/ledger.',
        intervention='Copy existing normalized3Deta into three verified-zero per-agent physical reservedchannels10:13 beforetheUNCHANGEDencoder. Keeporiginal lateeta branch too. Sameinputinformation, allphysicalquantities/APIroles/knowninit8 retained; no additionalcontrollerquery.',
        invariants='Sameparameter shapes/initial weights, TRAINlabels, source-onlynormalization, draws,6000steps, optimizer, checkpointcriterion, GPUhighestprecision. Changedonly eta routing; nohyperparameterorarchitecturewidthsweep.',
        checkpoint='Same SOURCEVAL local state_etaMSE/fullactionMSE/NLL. Noheldfitting orselection. Baseline24fits exactreused;12newstatefits;eta-only unchanged reused.',
        primary='Predeclared paired early-minus-late state_eta/triple skill with physicalstatebootstrap inALL4heldscenes; correctstate/controller permutation controls sameasStage35. Needpositiveabsoluteskill aswellasimprovement to claimlocalrepair. Reportall3seeds. NoQtransferclaimfromlocalresults.',
        conditional_next='Only iflocalrelationshiprepairswithpositiveheldmechanismevidence considerapplyingthisSAMEroutingtoQ. Otherwise pruneasufficientrepair, donotsweepwidth/depth/lr.',
        scope='AllopenedDEVELOPMENT; heldresults motivatedtheglobalhypothesis,notfinalconfirmation. FreshuntouchedQconfirmation required forultimateclaim.',
        literature=[dict(url='https://arxiv.org/abs/1612.00222',relevance='InteractionNetworks motivateobject/relationconditionedcomputations beforepooling. Specificsingle-factor eta-routingtest isours, notthepaperalgorithm oratransferguarantee.')],
        new_fits=12,new_rollouts=0,new_queries=0)
    freeze(HERE/'protocol.json',p)
    freeze(HERE/'history_audit.json',dict(prior_source=str(BASE),prior_protocol_sha256=sha(BASE/'protocol.json'),
        no_existing_early_eta_control_found=True,old_bilinear_is_late_fusion=True,zero_channels_verified=True,
        baseline_fits_reused=24,new_rollouts=0,new_queries=0))


def fit(index):
    import jax,jax.numpy as jnp
    from diagnostics.controller_local_to_return_probe_v1.vector_net import Critic as Old
    from .net import Critic as New
    from . import learn
    job=RUNS[index];d=np.load(HERE/'dataset.npz')
    x={k[7:]:jnp.asarray(v[:1],jnp.float32) for k,v in d.items() if k.startswith('entity_')}
    x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    c=jnp.asarray(d['feature'][:1],jnp.float32);e=jnp.asarray(d['eta'][:1],jnp.float32)
    a=Old('state_field_compact',8).init(jax.random.PRNGKey(job['seed']),x,c,e)
    b=New('state_field_compact',8).init(jax.random.PRNGKey(job['seed']),x,c,e)
    aa,ta=jax.tree_util.tree_flatten(a);bb,tb=jax.tree_util.tree_flatten(b);assert ta==tb
    for u,v in zip(aa,bb):np.testing.assert_array_equal(u,v)
    freeze(HERE/f'initialization_audit{index}.json',dict(job=job,identical_parameter_arrays=True,
        parameter_count=sum(v.size for v in aa),devices=[str(x) for x in jax.devices()]))
    learn.fit(learn.JOBS.index(job))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
