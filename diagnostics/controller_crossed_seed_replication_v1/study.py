"""Independent seeds at the exact existing source states: no refit or search."""
import argparse,copy,json
from pathlib import Path
from collections import Counter,defaultdict
from diagnostics.controller_ff_factorial_repair_v1 import study as prior
from diagnostics.controller_response_families_v1 import study as engine
from diagnostics.controller_q_diversity_v1 import experiment as original
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXPERIMENT='exp_controller_crossed_source_independent_seed_v1'
read,write,freeze,sha=prior.read,prior.write,prior.freeze,prior.sha


def prepare():
    old=read(prior.HERE/'protocol.json');p=copy.deepcopy(old)
    p['states']=[s for s in old['states'] if s['scenario']=='ring_exchange' and s['study_split']=='train']
    assert len(p['states'])==32
    p['controllers']={'ring_exchange':{k:v for k,v in old['controllers']['ring_exchange'].items() if k.startswith('flow_')}}
    assert len(p['controllers']['ring_exchange'])==5
    p.update(schema='crossed_source_independent_seed_v1',purpose='Separate fitting stochastic TRAIN labels from learning reproducible state/controller/eta effects, before expanding independent states.',
        scope='SOURCE diagnostic on ALL existing32RingTRAINstates and5Flow-referencecontrollers,allprefixed4eta. No newphysicalstates/controllers/eta; no fitting or model selection.',
        discovery_seeds=[0,1],independent_evaluation_seeds=[2,3],seeds=[0,1,2,3],
        new_rollouts_maximum=1280,expected_reuse=1280,
        hypothesis='Low sourceTRAIN error versus unseen-state error may reflect Q2 label-noise fitting rather than only inadequate state coverage. Test frozen sourcepredictions on independentrollouts at the SAME physicalstates.',
        prior_reuse_audit=[str(ROOT/'controller_crossed_support_audit_v1/results.json'),str(prior.HERE/'independent_half_signal.json'),str(ROOT/'controller_native_mass_repair_v1/completion.json'),str(ROOT/'orthoflow3_critic_rootcause_resolution_v1/seed_reproducibility_summary.json')],
        compatibility='Last item is earlier TT/H20/controllerprogram evidence and cannot supply labels for these actualFFaffineprograms. SharedDBall244controllerprofiles scanned for exactcurrentUIDsupport; states/candidates/execution/seedidentitiesmustmatch. Newencodingdoesnotinvalidateoldlabels.',
        models='Frozen native20joint plus originaluniformjointstate and correspondingeta-only,3seedsaveraged. Hashes frozen in models_frozen.json before outcomes. No checkpointselection onnewseeds.',
        primary='Predictive squared-error gain over zero state×controller×eta interaction on independentseeds2/3; pairedphysicalstatebootstrap,censorintervals. Secondary state×eta,rawQ,errorrelativeoldempiricalQ2lookup,mismatchandallfixedK4.',
        label_noise='Use independentseed crossproducts to estimate noise-corrected Q/interaction squared error, without treating empiricalQ2 as exacttruth. Report possible negative unbiased estimates and censor bounds;no clippingorclaimof exactBayeserror.',
        decision='If sourceinteraction skillCIpositive on independentseeds but unseen-state remainsnegative, prioritize state-support contrast. If sourceinteractionskill alsofailsor worsenssubstantiallyversusfittedseeds,prioritize labelprecision/overfit before a largecoverageexpansion. Ambiguous results stayambiguous;no automaticexpansion.',
        bootstrap='20000physicalstateunits,allC/eta/seedsclustered;donorandrecipient jointlyresampled;200savedderangements;allfixedK4policies. No B15claimfromQ2.',
        no_final_claim='These are TRAINphysicalstates with newrolloutseeds,not fresh-state generalizationorLOSO. No targetscene labelsfortraining.',
        literature=dict(url='https://arxiv.org/abs/2002.11833',review='PVN explicitly estimates policy-return targets from multipleepisodes. ThismotivatesseparatingMonteCarlosupervisionerrorfromnew-stateerror;thepaperdoesnotproveourfailurecause.'),
        next_gate='Finishentire32x5x4x2independentpanel,thenreview;no newfits or automatic statecoverage expansion.',
        new_code_sha256=sha(__file__))
    for name,source in [('runtime.py',prior.HERE/'runtime.py'),('static_operator_audit.json',prior.HERE/'static_operator_audit.json')]:
        f=HERE/name
        if not f.exists():f.symlink_to(source.resolve())
    frozen=[]
    for a in ['eta_only','union_native_slots','uniform_eta_only','uniform_native_slots']:
        for seed in (17,23,41):
            f=ROOT/'controller_native_mass_repair_v1/fits/joint'/a/f'seed{seed}'
            frozen.append(dict(arm=a,seed=seed,checkpoint=str(f/'checkpoint.msgpack'),checkpoint_sha256=sha(f/'checkpoint.msgpack'),
                predictions=str(f/'predictions.npz'),prediction_sha256=sha(f/'predictions.npz')))
    freeze(HERE/'models_frozen.json',frozen);p['model_manifest_sha256']=sha(HERE/'models_frozen.json')
    freeze(HERE/'protocol.json',p)
    cases=[original.case(p,s,c,j,k) for s in p['states'] for c in p['controllers']['ring_exchange'] for j in p['eta_indices'] for k in p['seeds']]
    cases.sort(key=lambda r:(r['state_index'],r['controller_name'],r['eta_index'],r['seed']))
    missing=[];counts=Counter();groups=defaultdict(list);support=[]
    with connect(True) as db:
        profiles=list(db.execute('SELECT controller_uid,config_json FROM controller_config'))
        for name,c in p['controllers']['ring_exchange'].items():
            profile=next(r for r in profiles if r['controller_uid']==c['controller_uid'])
            assert json.loads(profile['config_json'])==c['payload']
            rr=db.execute('SELECT count(*) AS n,count(distinct state_uid) AS s,count(distinct eta_uid) AS e FROM rollout WHERE controller_uid=?',(c['controller_uid'],)).fetchone()
            support.append(dict(name=name,controller_uid=c['controller_uid'],outcomes=rr['n'],states=rr['s'],eta=rr['e']))
        for r in cases:
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if a is None:missing.append(r)
            else:
                assert not a['conflict_quarantined'] and all(a[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if a['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    assert all(r['seed'] in [2,3] for r in missing)
    counts['genuinely_missing']=len(missing);assert len(missing)<=1280
    write(HERE/'cache_history_audit.json',dict(controller_profiles_scanned=len(profiles),exact_execution_support=support,
        source_old_outcomes_read_only=True,old_TT_replication_incompatible=True,
        reason='Independent TRAINstate seed2/3 outcomes for exactcurrentFFcontrollers absent; do not retrain onnoisylabels toguessrepeatability.',new_code_sha256=sha(__file__)))
    freeze(HERE/'cases.json',cases);write(HERE/'missing_cases.json',missing)
    write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Source crossed-controller independent seed diagnosis',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=24,new_rollouts_max=1280,no_fitting=True))))
    print(dict(counts),flush=True)


def worker(index,workers):
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    p=read(HERE/'protocol.json');assert all(sha(Path(f))==h for f,h in p['source_code_hashes'].items())
    engine.HERE=HERE;engine.EXPERIMENT=EXPERIMENT
    engine.EndpointMatchedField=lambda base,name:FlowEndpointMatchedField(base,name.removeprefix('flow_'))
    engine.worker(index,workers)


def merge():
    engine.HERE=HERE;engine.EXPERIMENT=EXPERIMENT;engine.merge()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','worker','merge']);a.add_argument('--index',type=int,default=0);a.add_argument('--workers',type=int,default=72);q=a.parse_args()
    worker(q.index,q.workers) if q.stage=='worker' else globals()[q.stage]()
