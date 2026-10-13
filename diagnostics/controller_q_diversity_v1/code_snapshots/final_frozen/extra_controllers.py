"""Independent controller-coverage expansion, preserving the frozen execution.

Only64 existing TRAIN states per scene are crossed with four new controllers.
No official VAL or FINAL labels choose their parameters or physical states.
"""
import argparse
import copy
from collections import Counter,defaultdict
import numpy as np
from . import experiment as e

EXTRA_EXPERIMENT='exp_controller_q_diversity_ff_extra_20261009_v1'


def protocol():
    p=e.read(e.HERE/'protocol.json')
    for scene,profiles in e.read(e.HERE/'extra_controller_profiles.json').items():
        p['controllers'][scene].update(profiles)
    if (e.HERE/'safety_profiles.json').exists():
        for scene,profiles in e.read(e.HERE/'safety_profiles.json').items():p['controllers'][scene].update(profiles)
    return p


def register():
    from scipy.stats import qmc
    from datetime import datetime
    lo=np.array([.55,.2,-.85,0.,0.,1.5]);hi=np.array([1.1,.9,.85,.6,.8,4.5])
    values=lo+qmc.LatinHypercube(6,seed=2026100941).random(4)*(hi-lo)
    p=e.read(e.HERE/'protocol.json');extra={}
    for scene,profiles in p['controllers'].items():
        extra[scene]={}
        for i,params in enumerate(values):
            v=copy.deepcopy(profiles['analytic0']['payload']);v['controller_parameters']=params.tolist()
            extra[scene][f'analytic{i+8}']=dict(controller_uid=e.uid('ctl',v),payload=v,role='train')
    e.freeze(e.HERE/'extra_controller_profiles.json',extra)
    design=e.HERE/'extra_controller_design.json'
    if not design.exists():
        e.freeze(design,dict(created_local=datetime.now().astimezone().isoformat(),
            parameters=values.tolist(),controller_names=[f'analytic{i}' for i in range(8,12)],
            source='Independent6D Latin hypercube seed2026100941; identical original global parameter ranges',
            states='First64 pre-frozen TRAIN physical states per scene; no directed state sampling',
            new_requested_rollouts=8192,rollouts_per_state_eta=1,eta_pool_sha256=e.sha(e.HERE/'protocol.json'),
            official_validation_labels_seen=False,final_labels_seen=False,
            motivation='TRAIN-only leave-controller-out probe transferred poorly despite good unseen-state fit; raw and compressed field encodings did not remove the gap',
            controlled_comparison='Original4 analytic controllers x128states versus8x64states, retaining the same nativeFlow128states; equal total training label budget. Also fit all compatible labels as deployment candidate.',
            cache_audit='Original full8database audit found no compatible analytic-family data outside this experiment; exact new controller profiles receive standard preflight',
            resource_cap=12,deadline='2026-10-09T09:00:00+08:00'))
    with e.connect() as db,e.transaction(db):
        # experiment.path is UNIQUE. The parent already owns HERE; silently
        # ignoring that conflict would leave shard journals with no FK parent.
        experiment_path=str(e.HERE/'batches/source_extra')
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?) ON CONFLICT(experiment_uid) DO NOTHING',
            (EXTRA_EXPERIMENT,'Controller coverage4to8, frozenFF',experiment_path,e.sha(design),e.sha(__file__),e.canonical(dict(parent=e.EXPERIMENT))))
        assert db.execute('SELECT path FROM experiment WHERE experiment_uid=?',(EXTRA_EXPERIMENT,)).fetchone()[0]==experiment_path
        for profiles in extra.values():
            for ctl in profiles.values():
                v=ctl['payload']
                prior=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert prior is None or prior[0]==e.canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],e.uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],e.canonical(v['rng']),e.canonical(v),'EXACT_PROFILE'))
    print('Registered four independent extra controllers per scene; no rollout labels read',flush=True)


def preflight():
    p=protocol();rows=[]
    for s in p['states']:
        if s['study_split']!='train' or s['study_index']>=64:continue
        for c in range(8,12):
            for j in range(16):rows.append(e.case(p,s,f'analytic{c}',j,0))
    rows.sort(key=lambda r:(r['state_index'],r['eta_index'],r['controller_name'],r['scene']))
    counts=Counter(requested=len(rows));missing=[];groups=defaultdict(list)
    with e.connect(True) as db:
        for r in rows:
            old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if old:
                assert not old['conflict_quarantined']
                assert all(old[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['numerical_unknown_not_rerun' if old['numerical_failure'] else 'exact_reuse']+=1
            else:missing.append(r)
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing);folder=e.HERE/'batches/source_extra'
    e.freeze(folder/'cases.json',rows);e.write(folder/'missing_cases.json',missing)
    e.freeze(folder/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=x,controller_uid=c,seed_keys=ss) for (s,x,c),ss in groups.items()]))
    e.write(folder/'exact_identity_preflight.json',dict(counts,protocol_sha256=e.sha(e.HERE/'protocol.json'),cases_sha256=e.sha(folder/'cases.json'),extra_profiles_sha256=e.sha(e.HERE/'extra_controller_profiles.json')))
    print(dict(counts),flush=True)


def features():
    import jax
    from .runtime import native_runtime,AnalyticField,controller_context
    from .features import field_context
    p=protocol();bases={};runtimes={};count=0
    for s in p['states']:
        if s['study_split']!='train' or s['study_index']>=64:continue
        scene=s['scenario']
        if scene not in bases:bases[scene]=native_runtime(scene)
        physical=np.load(e.HERE/'features'/s['state_uid']/'analytic0.npz')
        for i in range(8,12):
            name=f'analytic{i}';dest=e.HERE/'features'/s['state_uid']/(name+'.npz')
            if dest.exists():continue
            ctl=p['controllers'][scene][name]
            jax.config.update('jax_enable_x64',bool(ctl['payload']['jax_enable_x64']))
            if (scene,name) not in runtimes:runtimes[scene,name]=AnalyticField(bases[scene],ctl['payload']['controller_parameters'])
            rt=runtimes[scene,name]
            out={k:physical[k] for k in physical.files if k.startswith('entity_')}
            out.update(nominal=controller_context(rt,s),field=field_context(rt,s))
            temp=dest.with_suffix('.tmp.npz');np.savez_compressed(temp,**out);temp.replace(dest);count+=1
    e.write(e.HERE/'extra_features_complete.json',dict(measurements=count,simulator_steps=0,labels_read=False,extra_profile_sha256=e.sha(e.HERE/'extra_controller_profiles.json')))
    print('Extra input measurements complete',count,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','preflight','features','worker','merge']);ap.add_argument('--index',type=int);ap.add_argument('--workers',type=int,default=64);args=ap.parse_args()
    if args.action=='register':register()
    elif args.action=='preflight':preflight()
    elif args.action=='features':features()
    else:
        # Registry extension only. The frozen rollout worker and FF arithmetic
        # are reused verbatim, with a separate journal/experiment namespace.
        e.protocol=protocol;e.EXPERIMENT=EXTRA_EXPERIMENT
        if args.action=='worker':e.worker('source_extra',args.index,args.workers)
        else:e.merge()


if __name__=='__main__':main()
