"""Source-only LOSO extension of the fixed-Q4 crossed coverage intervention."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_native_mass_repair_v1'
N32=ROOT/'controller_crossed_precision_v1';N128=ROOT/'controller_crossed_coverage_v1'
ORIGINAL=ROOT/'controller_ff_factorial_repair_v1';DIV=ROOT/'controller_q_diversity_v1'
MAT=HERE/'evaluation_data'
FOLDS={'held_toy':'toy_give_way','held_db':'double_bottleneck','held_four':'four_way_intersection'}
ALL_FOLDS={'joint':None,**FOLDS,'held_ring':'ring_exchange'}
ARMS=('eta_only','union_native_slots')
LEGEND={'eta_only':'N128_eta','union_native_slots':'N128_state','uniform_eta_only':'N32_eta','uniform_native_slots':'N32_state'}


def link(dest,src):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.resolve()==src.resolve(),str(dest)
    else:dest.symlink_to(src.resolve())


def prepare():
    assert read(N128/'completion.json')['gate_passed']
    # Same algorithm as Stage26; only use the corresponding SOURCE-fold frozen normalization.
    code=(N128/'learn.py').read_text()
    code=code.replace("BASE/'fits/joint'/job['arm']", "BASE/'fits'/job['fold']/job['arm']")
    start=code.index("    old_checkpoint=BASE/");end=code.index('    pools={}',start)
    code=code[:start]+code[end:]
    target=HERE/'learn.py'
    if target.exists():assert target.read_text()==code
    else:target.write_text(code)
    master=dict(schema='source_crossed_coverage_LOSO_Q4_control_v1',history=[str(x) for x in (BASE,N32,N128)],
        question='Does the SOURCE state/controller coverage repair also improve held-scene interaction, or is it only local?',
        prior_reuse='Stage22allfourfolds completed onN32Q2. Stage25onlyjointN32Q4,Stage26onlyjointN128Q4. Exactnew7680crossedlabels firstexistedStage26,so no earlierN128Flow5folds exist. Jointfitsallreused. HeldRing SOURCEdataidenticaltoStage22,so allsixheldRingfitsreusedwithouttraining.',
        contrasts='N32Q4 versusN128Q4,all3sourcefolds containingRing. Eachpairedfit runsequentiallyonsameallocatedCPU. Samearchitecture,input,6000steps,sampler,sourceVALselection. No test-based fitting or normalization.',
        isolation='ForheldToy/DB/Fouralltargetscene rows excludedfrom gradients,normalizationandcheckpointselection. Source-only normalizations fromStage22frozen per fold. HeldRingmodelunchanged:addingRingTRAINdata cannotchangeaRing-held fit.',
        scope='PreviouslyopenedDEVELOPMENT;no finalconfirmationclaim. SourceVALrepairdoesnotestablishLOSO. All4heldscenesreported;Ring is a no-treatment reference,notanindependent replicationofthecoverageintervention.',
        labels='No newrollout/query. AlloldK4labels reused;forRing40heldstates improveold32Q2 toQ4 withalreadycachedStage24seeds2/3. BothN32/N128selectors useexactlysame evaluationlabels. Fourfixedetapoliciesexamined,notall16.',
        checkpoint='SOURCEselectionregret,thenpairwiseranking,interactionMSE,NLL. Samefrozenoriginalrule. Three-seedmean fixed before evaluation.',
        next_decision='Do notexpandrolloutsautomatically. IfnoheldscenehaspositiveSxCxeta skillandpositive matched-state decisionvalue beyondfixedpriors,coveragealoneisnot sufficienttransferrepair;proceedtoinput-sufficiencyhistoryaudit. Anypositivecandidate needsuntouchednaturalstateconfirmation.',
        literature=dict(url='https://proceedings.mlr.press/v119/raileanu20a.html',review='Re-readprimaryPDVFbeforestage. Its policy/environmentvaluefactorization motivates separatinglocalcoveragefromheldenvironmenttransfer;targetdynamicsinteractiontheredoesnotestablishourzero-shotLOSO.'),
        new_fits=36,reused_joint_fits=12,reused_held_ring_fits=6,new_rollouts=0,code_sha256=sha(target),
        data={k:dict(path=str(p/'dataset.npz'),sha256=sha(p/'dataset.npz')) for k,p in [('n32',N32),('n128',N128)]},
        paired_CPU_policy='Eachof18jobs fitsN32thenN128 on sameallocatedcore;nodeconcurrency<=24 includingourstages,noCPUreservationwhenidle.')
    freeze(HERE/'protocol.json',master)
    for level,src in [('n32',N32),('n128',N128)]:
        dest=HERE/level;dest.mkdir(exist_ok=True)
        for file in ('dataset.npz','dataset_audit.json'):link(dest/file,src/file)
        p=copy.deepcopy(read(N128/'protocol.json'));p['folds']=FOLDS
        p['experiment']=dict(code_sha256=sha(target),level=level,intervention='OnlysourcecrossedcoverageatfixedQ4;source-foldnormalizationfrozen',parent_sha256=sha(HERE/'protocol.json'))
        freeze(dest/'protocol.json',p);freeze(dest/'source_gate.json',dict(passed=True,source=str(N128/'completion.json'),development_only=True,no_held_label_model_selection=True))
    link(HERE/'dataset.npz',N128/'dataset.npz')
    for fold in ALL_FOLDS:
        for alias in LEGEND:
            level='n32' if alias.startswith('uniform_') else 'n128';arm='eta_only' if alias.endswith('eta_only') else 'union_native_slots'
            if alias=='eta_only':arm='eta_only'
            for seed in SEEDS:
                origin=(BASE if fold=='held_ring' else (N32 if level=='n32' else N128) if fold=='joint' else HERE/level)
                dest=HERE/'fits'/fold/alias/f'seed{seed}';dest.parent.mkdir(parents=True,exist_ok=True)
                src=origin/'fits'/fold/arm/f'seed{seed}'
                if not dest.is_symlink():dest.symlink_to(src.resolve())
                else:assert dest.resolve()==src.resolve()
    # Original matrix rows, only enrich existing RingTRAIN seed0/1 by cachedseed2/3.
    MAT.mkdir(exist_ok=True);old=dict(np.load(ORIGINAL/'dataset.npz'));raw=dict(np.load(ORIGINAL/'raw_labels.npz'))
    rep=dict(np.load(ROOT/'controller_crossed_seed_replication_v1/raw_labels.npz'))
    lookup={(str(s),str(c)):i for i,(s,c) in enumerate(zip(old['state_uid'],old['controller']))}
    for si,s in enumerate(rep['state_uid']):
        for ci,c in enumerate(rep['controller']):
            row=lookup[str(s),str(c)]
            for ji,j in enumerate((0,5,10,15)):
                for key in raw:
                    if key in rep:raw[key][row,j]=rep[key][si,ci,ji]
                ss,uu,nn=[raw[k][row,j].sum() for k in ('success','unknown','observed')]
                old['success'][row,j]=ss;old['unknown'][row,j]=uu;old['failure'][row,j]=nn-ss-uu
    if not (MAT/'dataset.npz').exists():
        np.savez_compressed(MAT/'dataset.npz',**old);np.savez_compressed(MAT/'raw_labels.npz',**raw)
    link(MAT/'review.py',ORIGINAL/'review.py')
    freeze(HERE/'evaluation_protocol.json',dict(method_protocol_sha256=sha(HERE/'protocol.json'),
        panels='Allfourheldscenes natural40states. JointSOURCEVAL8 descriptive. K4,Q2/Q4(other3scenes),RingQ4 all40states withcachedindependentreplication.',
        seed_halves='Q2:1/1;Q4:2/2. Symmetricstrongreversal>=.5 and>=2physicalstatesineachdirection;includeallqualifyingpairs.',
        bootstrap='20000physical-statepaired;200derangements andcorrectdonor-recipientbootstrap;bestper-Cfixed reselectedwithinbootstrap.',
        computational_aliases=LEGEND,reused_engines={str(MAT/'review.py'):sha(MAT/'review.py')},
        evaluation_data_sha256=sha(MAT/'dataset.npz'),raw_sha256=sha(MAT/'raw_labels.npz'),code_sha256=sha(HERE/'review.py'),new_rollouts=0))
    print(dict(new_fits=36,new_rollouts=0,reused_joint=12,reused_held_ring=6),flush=True)


def fit(index):
    from . import learn
    # One allocated CPU for both sides; no rollout or unrelated job is touched.
    for level in ('n32','n128'):
        learn.HERE=HERE/level;learn.fit(index)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    fit(a.index) if a.stage=='fit' else prepare()
