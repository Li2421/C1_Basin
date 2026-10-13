"""Check that local action diagnostics are not failing due to the readout."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_local_transfer_localization_v1'
EVAL=ROOT/'controller_local_to_return_probe_v1/vector_control'
FOLDS={'toy':'toy_give_way','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
ARMS=('motion_complete','motion_eta_only')
RUNS=[dict(fold=f,seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    freeze(HERE/'history_audit.json',dict(
        database_inventory=str(ROOT/'controller_correction_transfer_local_v1/history_audit.json'),
        historical_tests=['controller_entity_symmetry_probe_v1/probe.py','controller_entity_symmetry_probe_v1/field_equivariance.py',
            'controller_local_to_return_probe_v1/vector_net.py','controller_early_eta_localization_v1','controller_local_transfer_localization_v1'],
        difference='Old nativeactivepermutationtests identifiedpolicy/APIroledependence. ThiskeepsAPIroles andallCfeatures; onlychanges localaction outputmapping tosharedper-agentreadout withretainedper-agentlatent. Passivepermutationmovesidentityalongwithagent, neverassumesnativepolicyactiveinvariance. Oldearlyeta changedroutingnotreadout. No prior suchlocaltargetreadoutfit found.',
        reuse='ExactStage35dataset/targetlabels/all24baselinefits; new12statefits only; no simulation/query/newstates.',
        question='Couldlocaldiagnosticfailurebeintroducedbyusinga pooledscalar-Qtrunkwithglobaleight-coordinate actionhead? MusttestbeforeequatinglocalfailurewithoriginalQroot.'))
    (HERE/'dataset.npz').symlink_to((BASE/'dataset.npz').resolve());(HERE/'targets').symlink_to((BASE/'targets').resolve())
    for f in FOLDS:
        out=HERE/'fits'/f;out.mkdir(parents=True,exist_ok=True)
        (out/'motion_eta_only').symlink_to((BASE/'fits'/f/'motion_eta_only').resolve())
    code=(BASE/'learn.py').read_text().replace('from diagnostics.controller_local_to_return_probe_v1.vector_net import Critic','from .net import Critic')
    (HERE/'learn.py').write_text(code);(HERE/'review.py').write_text((BASE/'review.py').read_text())
    p=copy.deepcopy(read(BASE/'protocol.json'));p['schema']='local_diagnostic_per_agent_readout_validity_v1'
    p['experiment']=dict(
        question='Can thesameinput/source data learnlocal FFinteraction usingaper-agentreadout appropriateforvectoractions? Thisauditsdiagnosticvalidity, notyetareplacementscalarQcritic.',
        change='Exactoldentityencoderweights/shapes includingglobal128latent; exposeexistingagent64latent. Shareddecoder peragentconcat[global128,agent64,eta32] ->100->64->2. Width100 chosenbeforedata byparametercount matching;290fewerparameters thanoldglobalhead. Noη-earlyrouting/primitive/oraclefeatures.',
        initialization='Verifyold/newencoderandetaencoderinitialweightsbitexact,parametercountsandpassivepermutationequivariancebeforefits. Decodernecessarilydifferent, so notsame-all-parameters intervention.',
        invariants='All72input/physical/APIroles/normalization/labels/source splits/sampling6000steps/checkpointcriterion/3seeds unchanged. All24oldfits reused,12newstatefits,faireta baselineunchanged.',
        gate='All4folds checked. Ifheldstate_eta skill positiveCIandpairedimprovementpositive, reviewtriple/mismatchandmodelseedreplicability. Iflocaltransferrepaired, priorlocalfailures cannotbeusedasfundamentalevidenceagainstdatainformation/controllerlearnability. Neitherpositive nornegativealoneprovesQroot.',
        boundary='Passiveequivariancekeepsphysical/C/identityassociation andoutputassociation. DoesNOTassertactivereindexing ofnativeFlowisvalid. Node-outputactiondiagnosticversusscalarQtaskdifferent; doNOTproposeper-agentactionheadasQrepairwithoutQarchitecture/test.',
        code_sha256=sha(HERE/'learn.py'),network_sha256=sha(HERE/'net.py'),preparation_sha256=sha(__file__),
        literature=[dict(url='https://arxiv.org/abs/1612.00222',relevance='InteractionNetworks useobject/relationcomputationfordynamicalpredictions. Motivatesmatchingactionoutputtoagents,notguaranteeofoursceneorQgeneralization.')])
    freeze(HERE/'protocol.json',p)
    freeze(HERE/'dataset_audit.json',read(BASE/'dataset_audit.json'))
    freeze(HERE/'source_gate.json',dict(passed=True,authority=str(BASE/'source_gate.json'),reason='Originalsourcegate12/12passed; testingdiagnosticreadoutvalidity,sourceperformancewillberechecked.'))
    print(dict(new_fits=12,reused_baselines=24,new_rollouts=0,new_queries=0))


def audit():
    import jax,jax.numpy as jnp
    from .net import Critic
    from diagnostics.controller_local_to_return_probe_v1.vector_net import Critic as Old
    d=np.load(HERE/'dataset.npz');p=read(HERE/'protocol.json');rows=[]
    # Two-agent paddedandfour-agentexamples, shapesmatchallfittingdatasets.
    ii=[int(np.flatnonzero(d['scene']==sc)[0]) for sc in FOLDS.values()]
    x={k[7:]:jnp.asarray(v[ii],jnp.float32) for k,v in d.items() if k.startswith('entity_')}
    x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    c=jnp.asarray(d['feature'][ii],jnp.float32);e=jnp.zeros((4,3),jnp.float32)
    new=Critic('state_field_compact',8);old=Old('state_field_compact',8)
    perm=np.array([2,0,3,1])
    xp=dict(x)
    for k in ('agents','agent_mask','obstacles'):xp[k]=x[k][:,perm]
    xp['pairs']=x['pairs'][:,perm][:,:,perm]
    for seed in SEEDS:
        key=jax.random.PRNGKey(seed);a=new.init(key,x,c,e);b=old.init(key,x,c,e)
        for name in ('physical_encoder','eta'):
            aa=jax.tree_util.tree_leaves(a['params'][name]);bb=jax.tree_util.tree_leaves(b['params'][name]);assert len(aa)==len(bb)
            for u,v in zip(aa,bb):np.testing.assert_array_equal(u,v)
        n=sum(v.size for v in jax.tree_util.tree_leaves(a));o=sum(v.size for v in jax.tree_util.tree_leaves(b));assert n-o==-290
        value=np.asarray(new.apply(a,x,c,e)).reshape(4,4,2)
        changed=np.asarray(new.apply(a,xp,c[:,perm],e)).reshape(4,4,2)
        err=float(abs(changed-value[:,perm]).max());assert err<2e-6
        rows.append(dict(seed=seed,new_parameters=n,old_parameters=o,difference=n-o,encoder_and_eta_initialization_bitexact=True,
            passive_equivariance_max_error=err))
    freeze(HERE/'architecture_audit.json',dict(rows=rows,passed=True,network_sha256=sha(HERE/'net.py')))
    print(rows)


def fit(index):
    assert read(HERE/'architecture_audit.json')['passed']
    from . import learn
    learn.fit(learn.JOBS.index(dict(**RUNS[index],arm='motion_complete')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    fit(a.index) if a.stage=='fit' else globals()[a.stage]()
