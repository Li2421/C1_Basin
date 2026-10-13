"""Root repair: retain FF operator information and cross its training support."""
import argparse,copy,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from diagnostics.controller_response_families_v1 import study as parent
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_response_families_v1.runtime import FAMILIES
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'controller_response_families_v1'
OLD=HERE.parent/'controller_q_diversity_v1'
EXPERIMENT='exp_controller_operator_root_repair_v1'
read,write,freeze,sha=old.read,old.write,old.freeze,old.sha

def prepare():
    original=read(OLD/'protocol.json');prior=read(PARENT/'protocol.json')
    states=[s for s in original['states'] if (s['study_split']=='train' and s['study_index']<32) or (s['study_split']=='validation' and s['study_index']<8)]
    p=copy.deepcopy(prior);p['states']=states;p['schema']='controller_operator_root_repair_v1'
    p.update(scope='DEVELOPMENT root repair;existing old confirmation is never reused as pristine confirmation',
        parent_response_protocol_sha256=sha(PARENT/'protocol.json'),train_seeds=[0,1],validation_seeds=[0,1,2,3],
        changed='Controller function information and response-family coverage only; no optimizer or architecture search.',
        source_labels='Reuse full original high-coverage analytic/nativeFlow cache in training, plus endpoint-matched families.',
        label_gate='Operator repair must improve independent-state Q/controller and eta effects before final confirmation is generated.')
    freeze(HERE/'protocol.json',p)
    for name in ('runtime.py','static_operator_audit.json'):
        if not (HERE/name).exists():(HERE/name).symlink_to(PARENT/name)
    cases=[old.case(p,s,c,j,k) for s in states for c in FAMILIES for j in p['eta_indices'] for k in (p['train_seeds'] if s['study_split']=='train' else p['validation_seeds'])]
    cases.sort(key=lambda r:(r['split'],r['state_index'],r['controller_name'],r['eta_index'],r['seed'],r['scene']))
    freeze(HERE/'cases.json',cases)
    groups=defaultdict(list);missing=[];counts=Counter()
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Controller operator information root repair',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap_when_idle=24))))
        for r in cases:
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if a is None:missing.append(r)
            else:
                assert not a['conflict_quarantined'] and all(a[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if a['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing)
    write(HERE/'missing_cases.json',missing);write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    freeze(HERE/'root_comparison_protocol.json',dict(
        primary_question='Does providing the actual FF vector-field response repair the observed exact nominal-fingerprint alias on independent physical states?',
        root_evidence_sha256=sha(PARENT/'root_alias_result.json'),
        methods=['eta_only','nominal_contact_padded64','static_field_compact64'],
        architecture='Identical existing bilinear physical+eta head with64controller channels in both controller-conditioned arms.',
        labels='Identical old and newly crossed cache; original eta16 pool. New operator labels only at pre-fixed indices0,5,10,15; no eta search.',
        fit_scenes=['joint Toy+Ring','Toy only','Ring only'],
        objective='Identical binomial NLL; source-validation Q MSE plus0.5controller-contrast MSE; NLL not sole checkpoint rule.',
        source_selection='Source scene only for single-source fits; all target labels reserved for development diagnosis after freezing.',
        final_confirmation='Not generated until independent-state/controller/eta diagnostics support a root repair.',
        literature=['https://arxiv.org/abs/2002.11833','https://proceedings.mlr.press/v119/raileanu20a.html'],
        limitation='This is a fundamental input/data identifiability repair, not yet proof that all held-scene mechanisms can extrapolate.'))
    print(json.dumps(dict(counts)),flush=True)

def features(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.contact_features import measure
    from diagnostics.physical_state_critic_formal_v1 import physical_entities as phys
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    p=read(HERE/'protocol.json');bases={};runtimes={};count=0
    for s in p['states'][index::workers]:
        sc=s['scenario']
        if sc not in bases:bases[sc]=native_runtime(sc)
        jax.config.update('jax_enable_x64',p['controllers'][sc]['flow']['payload']['jax_enable_x64'])
        for name in FAMILIES:
            dest=HERE/'features'/s['state_uid']/(name+'.npz')
            if dest.exists():continue
            if (sc,name) not in runtimes:runtimes[sc,name]=EndpointMatchedField(bases[sc],name)
            rt=runtimes[sc,name]
            class QueryInterface:
                def __getattr__(self,key):return getattr(rt,key)
            nominal=controller_context(rt,s);field=field_context(QueryInterface(),s);contact,_=measure(rt,s)
            ent=phys.entities(sc,s['physical']);dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,nominal=nominal,field=field,contact=contact,**{'entity_'+k:v for k,v in ent.items()});count+=1
    write(HERE/'features'/f'worker{index}.json',dict(complete=True,measurements=count,simulator_steps=0,labels_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','features','merge']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=96);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='features':features(a.index,a.workers)
    else:
        parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT
        if a.action=='worker':parent.worker(a.index,a.workers)
        else:parent.merge()
