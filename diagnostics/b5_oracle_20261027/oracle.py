"""Resumable finite-pool oracle. All new rollouts pass official cache preflight."""
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:18])
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import time
from runtime import OUT, ROOT, SNAP, H, RN, clean, digest, dump, make_case, run, SOURCE, ADAPTED
import numpy as np
from scipy.stats import qmc
from shared_control.correction_b5 import yield_menu
from shared_rollout_db.src import rollout_db as db, planner, cache_writer

EXP='exp_b5_oracle_20261027'
DEV_SCENES=['xsc_hs_plus_w025','xsc_hs_box_n09_r014_f25','xsc_hs_swap_n07','hs_obst_n08_R140_r045']
TEST_SCENES=['xsc_hs_plus_w030_te','xsc_hs_box_n09_r013_f25_te','xsc_hs_swap_n06_te','hs_obst_n08_R142_r050_te']


@lru_cache(None)
def inputs():
    scenes={s['name']:s for s in json.load(open(SNAP/'hetset/scenes_all2.json'))}
    controllers=json.load(open(ROOT/'diagnostics/cross_scene_correction_v1/SELECTED_CONTROLLERS.json'))['xsc_swap_n6']
    acts=json.load(open(ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json'))['lit_swap8_free']
    history={}
    p=SNAP/'hetset/fresh.jsonl'
    for line,l in enumerate(p.open(),1):
        r=json.loads(l);history[(r['scene'],r['controller'],r['state'],r['action'])]=(r,str(p),line)
    return scenes,[controllers['centralized_sampling'],controllers['gradient_mpc']],acts,history


def b3_pool(zero=False):
    acts=inputs()[2]
    specs=[dict(name='eta'+str(k),mode='b3',eta=acts['eta'+str(k)]['eta_a']+[0.,0.]) for k in range(16)] if not zero else []
    lo=[0. if zero else .5,-.5,0.];hi=[1.25,.5,.75]
    points=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102701).random_base2(4),lo,hi)
    points=np.concatenate([points,list(itertools.product(*zip(lo,hi)))])
    specs += [dict(name=('b3g0_' if zero else 'b3_')+str(k),mode='b3',eta=list(x)+[0.,0.]) for k,x in enumerate(points)]
    return specs


def b5_pool(scene,si,mode,recipe,zero=False):
    p=np.asarray(scene['states'][si]['positions']);d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(d,np.inf)
    i,j=np.unravel_index(np.argmin(d),d.shape)
    # Frozen, state-only pair selection. Both ordered roles and all 13 waypoints.
    roles=[asdict(r) for a,b in [(int(i),int(j)),(int(j),int(i))] for r in yield_menu(a,b)]
    base=[0. if zero else .875,0.,.375]
    specs=[]
    for sign in (-.75,.75):
        specs.append(dict(name=f'{mode}_{recipe}_T{sign}',mode=mode,recipe=recipe,eta=base+[sign,0.]))
    for ri,role in enumerate(roles):
        for sign in (0.,.75,-.75):
            specs.append(dict(name=f'{mode}_{recipe}_r{ri}_t{sign}',mode=mode,recipe=recipe,role=role,eta=base+[sign,1.]))
    lo=[0. if zero else .5,-.5,0.,-1.,0.];hi=[1.25,.5,.75,1.,1.]
    points=qmc.scale(qmc.Sobol(5,scramble=True,seed=2026102702).random_base2(4),lo,hi)
    for k,e in enumerate(points):
        specs.append(dict(name=f'{mode}_{recipe}_sob{k}',mode=mode,recipe=recipe,role=roles[k%len(roles)],eta=e.tolist()))
    if zero:
        specs=[dict(x,name=x['name']+'_g0') for x in specs]
    return specs


def register(cases,stage):
    with db.connect() as con, db.transaction(con):
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,metadata_json) VALUES(?,?,?,?)',
                    (EXP,EXP,str(OUT),db.canonical(dict(phase='finite_oracle_no_training'))))
        for c in cases:
            physical={k:c['state'][k] for k in ('positions','velocities','goals')}
            con.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',
                (c['scenario_uid'],c['scene'],c['scenario_uid'],db.canonical(c['scenario_payload'])))
            con.execute('INSERT INTO state(state_uid,scenario_uid,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?) ON CONFLICT(state_uid) DO NOTHING',
                (c['state_uid'],c['scenario_uid'],digest(physical),db.canonical(physical),db.canonical(c['scenario_payload']),db.canonical(dict(scene=c['scene'],state_index=c['state_index'],experiment=EXP)),'CONTENT_EXACT'))
            assert con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()[0]==db.canonical(physical)
            con.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,config_json,compatibility_quality) VALUES(?,?,?,?)',
                (c['controller_uid'],c['scenario_uid'],db.canonical(c['controller_payload']),'EXACT_PROFILE'))
            eid,e,hx=db.eta_identity(c['eta_a'])
            con.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*e,hx))
    path=OUT/f'{stage}_planned_rollouts.json'
    dump(path,dict(requests=[{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for c in cases],
                   exact_configs=cases,notes='Full eta5/role/recipe/schedule are in controller UID; DB eta remains original eta3.'))
    pf=planner.preflight(path);dump(OUT/f'{stage}_cache_preflight.json',pf)
    assert pf['summary']['ambiguous']==0
    return {(x['state_uid'],x['eta_uid'],x['controller_uid'],s)for x in pf['details'] for s in x['missing_seeds']}


def worker_init():
    cpu=sorted(os.sched_getaffinity(0))[(mp.current_process()._identity[0]-1)%len(os.sched_getaffinity(0))]
    os.sched_setaffinity(0,[cpu])


def work(c):
    t0=time.monotonic()
    if c['spec']['mode'].startswith('exact_rhr'):
        from rhr_exact import run as rr
        result,arrays=rr(c)
    else:result,arrays=run(c)
    trace=OUT/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(exist_ok=True)
    np.savez_compressed(trace,**arrays)
    row=dict(result,**{k:c[k]for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid','scene','state_index','seed','spec')},
        controller=c['controller_payload']['nominal']['type_code'],eta=c['eta_a'],future_index=c['seed'],
        cache_identity_schema='registered_exact_uid_v1',initial_positions=c['state']['positions'],
        initial_velocities=c['state']['velocities'],goals=c['state']['goals'],deadlock=0,
        source='new_rollout',trace=str(trace),worker_cpu=list(os.sched_getaffinity(0)),wall_seconds=time.monotonic()-t0)
    cache_writer.append_journal([row],EXP)
    return row


def old_row(c):
    ci=0 if c['controller_payload']['nominal']['type_code']=='centralized_sampling' else 1
    entry=inputs()[3].get((c['scene'],ci,c['state_index'],c['spec']['name']))
    if entry is None or c['seed']!=0 or c['spec']['mode']!='b3':return None
    r,path,line=entry;cfg=c['scenario_payload']['config']
    return dict(success=r['success'],outcome=r['outcome'],episode_length=r['length'],
        numerical_failure=int(r['outcome']=='unknown'),collision=int(r['outcome']=='collision'),timeout=int(r['outcome']=='timeout'),
        elapsed_seconds=r['seconds'],completion_time=r['length']*cfg['dt'] if r['success'] else None,
        truncated_time=r['length']*cfg['dt'] if r['success'] else cfg['dt']*cfg['max_steps'],
        horizon_time=cfg['dt']*cfg['max_steps'],local_progress_time=None,
        **{k:c[k]for k in ('scene','state_index','spec','seed','rollout_uid','state_uid','controller_uid','eta_uid')},
        controller=c['controller_payload']['nominal']['type_code'],source='historical_fresh_exact_semantics',
        source_path=path,source_line=line,trace=None)


def behavior_key(scene,state,controller,seed,sp):
    """Exact zero-column aliases; never equate different nominal/safety/state/seed."""
    keep=sp.get('keep',range(5));e=[x if k in keep else 0. for k,x in enumerate(sp['eta'])]
    mode=sp['mode'];role=sp.get('role')if e[4] else None
    recipe=sp.get('recipe')if e[3] else None
    if mode in ('b3','add','blend'):
        if not e[4]:mode='add'
        if not e[3] and not e[4]:mode='b3'
    release=sp.get('release_eta')
    if sp.get('schedule')and release:release=[x if k in keep else 0. for k,x in enumerate(release)]
    return digest(dict(scene=scene,state=state,controller=controller,seed=seed,eta=e,mode=mode,
        role=role,recipe=recipe,schedule=sp.get('schedule'),release=release,
        theta=sp.get('theta'),eta_b=sp.get('eta_b')))


def alias_row(row,c):
    return dict(row,**{k:c[k]for k in ('spec','rollout_uid','state_uid','controller_uid','eta_uid')},
        eta=c['eta_a'],source='exact_zero_coefficient_reuse',source_rollout_uid=row['rollout_uid'],wall_seconds=0.)


def execute(cases,stage,workers=18,reuse=True):
    cases=list({c['rollout_uid']:c for c in cases}.values())
    missing=register(cases,stage)
    path=OUT/'candidates.jsonl';done={}
    if path.exists():done={r['rollout_uid']:r for r in map(json.loads,path.open())}
    semantic={behavior_key(r['scene'],r['state_index'],r['controller'],r['seed'],r['spec']):r for r in done.values()if not r.get('numerical_failure')}
    todo=[];reused=0;pending={};aliases={}
    with path.open('a') as f:
        for c in cases:
            if c['rollout_uid'] in done:continue
            bk=behavior_key(c['scene'],c['state_index'],c['controller_payload']['nominal']['type_code'],c['seed'],c['spec'])
            old=old_row(c) if reuse else None
            if old is None and bk in semantic:old=alias_row(semantic[bk],c)
            if old:
                f.write(json.dumps(clean(old))+'\n');f.flush();done[c['rollout_uid']]=old;semantic[bk]=old;reused+=1
            else:
                if bk in pending:
                    aliases.setdefault(pending[bk],[]).append(c);continue
                key=tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                if key not in missing:
                    raise RuntimeError('Global exact cache hit without local result: export it before resuming '+c['rollout_uid'])
                todo.append(c);pending[bk]=c['rollout_uid']
        print(stage,'new',len(todo),'historical reused',reused,'local reused',len(cases)-len(todo)-reused,flush=True)
        t0=time.monotonic()
        with ProcessPoolExecutor(max_workers=min(workers,18),mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
            futures={pool.submit(work,c):c for c in todo}
            for k,fu in enumerate(as_completed(futures),1):
                r=fu.result();f.write(json.dumps(clean(r))+'\n');f.flush();done[r['rollout_uid']]=r
                for c in aliases.get(r['rollout_uid'],[]):
                    alias=alias_row(r,c);f.write(json.dumps(clean(alias))+'\n');f.flush();done[c['rollout_uid']]=alias
                if k%10==0 or k==len(todo): print(stage,k,'/',len(todo),round(time.monotonic()-t0,1),'s',flush=True)
    dump(OUT/f'{stage}_execution.json',dict(cases=len(cases),new=len(todo),historical_reused=reused,completed=True))
    return [done[c['rollout_uid']]for c in cases]


def dev_cases(specs=None):
    scenes,ctls,_,_=inputs()
    return [make_case(scenes[s],ctl,8,sp,0)for s in DEV_SCENES for ctl in ctls for sp in (specs or b3_pool())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','A','A_deep','B','C','D','E','range']);ap.add_argument('--workers',type=int,default=18);a=ap.parse_args()
    scenes,ctls,_,_=inputs()
    if a.stage=='smoke':
        execute(dev_cases([dict(name='smoke_b3',mode='b3',eta=[.875,0.,.375,0.,0.])])[:2],'smoke',2,False);return
    if a.stage=='A':
        rows=execute(dev_cases(),'A',a.workers)
        failures=[]
        for s in DEV_SCENES:
            for ci,ctl in enumerate(ctls):
                rr=[r for r in rows if r['scene']==s and r['controller']==ctl['type_code']]
                if not any(r['success']for r in rr):failures.append(dict(scene=s,controller_index=ci,state_index=8))
        dump(OUT/'A_failures.json',failures);return
    failures=json.load(open(OUT/'A_failures.json'))
    cases=[]
    if a.stage=='A_deep':
        points=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102703).random_base2(8),[.5,-.5,0.],[1.25,.5,.75])
        for f in failures:
            for k,e in enumerate(points):
                sp=dict(name='b3_deep_'+str(k),mode='b3',eta=e.tolist()+[0.,0.])
                cases.append(make_case(scenes[f['scene']],ctls[f['controller_index']],f['state_index'],sp,0))
    elif a.stage in ('B','C','range'):
        for f in failures:
            s=scenes[f['scene']];ctl=ctls[f['controller_index']];si=f['state_index']
            if a.stage=='range':
                specs=b3_pool(True)
                # Matched lower-bound ablation: same candidate coordinates except G.
                for recipe in ('goal','cluster','wall'):specs+=b5_pool(s,si,'add',recipe,True)
            else:
                mode='add' if a.stage=='B' else 'blend'
                # B3 candidate subset is evaluated/reused for each B5 family.
                specs=b3_pool()+[sp for recipe in ('goal','cluster','wall')for sp in b5_pool(s,si,mode,recipe)]
            cases.extend(make_case(s,ctl,si,sp,0)for sp in specs)
    elif a.stage=='D':
        for f in failures:
            s=scenes[f['scene']];ctl=ctls[f['controller_index']];si=f['state_index']
            for k,sp in enumerate(b3_pool()[:16]):
                sp=dict(sp,name='sched_'+sp['name'],theta=.5,eta_b=b3_pool()[(k+8)%16]['eta'][:3])
                cases.append(make_case(s,ctl,si,sp,0))
        rows=list(map(json.loads,(OUT/'candidates.jsonl').open()))
        for f in failures:
            for mode in ('add','blend'):
                for recipe in ('goal','cluster','wall'):
                    rr=[r for r in rows if r['scene']==f['scene'] and r['controller']==ctls[f['controller_index']]['type_code'] and r['spec']['mode']==mode and r['spec'].get('recipe')==recipe and r['source']=='new_rollout']
                    # Prespecified ranking of fixed candidates; retain the 4 smallest final distance.
                    rr=sorted(rr,key=lambda r:(-r['success'],sum(r.get('final_goal_distances',[1e9])),r['spec']['name']))[:4]
                    for r in rr:
                        sp=dict(r['spec'],name='release_'+r['spec']['name'],schedule='release_third',release_eta=[.875,0.,.375])
                        cases.append(make_case(scenes[f['scene']],ctls[f['controller_index']],f['state_index'],sp,0))
    elif a.stage=='E':
        rows=list(map(json.loads,(OUT/'candidates.jsonl').open()))
        for r in rows:
            if not r['success'] or r['spec']['mode']not in ('add','blend') or r['spec'].get('keep'):continue
            if r['scene'] not in DEV_SCENES:continue
            ci=next(i for i,c in enumerate(ctls)if c['type_code']==r['controller'])
            for label,keep in [('GRTY',[0,2,3,4]),('GPTY',[0,1,3,4]),('GTR',[0,2,3]),('GTY',[0,3,4])]:
                sp=dict(r['spec'],name=label+'_'+r['spec']['name'],keep=keep,ablation=label)
                cases.append(make_case(scenes[r['scene']],ctls[ci],r['state_index'],sp,r['seed']))
        # Same 96-candidate recipe pool in each reduced family, not only deletion
        # of successful witnesses. Goal recipe fixed by pilot B; other T recipes
        # remain separate results and are not silently selected per test state.
        for f in failures:
            for original in b5_pool(scenes[f['scene']],f['state_index'],'add','goal'):
                for label,keep in [('GRTY',[0,2,3,4]),('GPTY',[0,1,3,4]),('GTR',[0,2,3]),('GTY',[0,3,4])]:
                    sp=dict(original,name=label+'_'+original['name'],keep=keep,ablation=label)
                    cases.append(make_case(scenes[f['scene']],ctls[f['controller_index']],f['state_index'],sp,0))
    execute(cases,a.stage,a.workers)


if __name__=='__main__':main()
