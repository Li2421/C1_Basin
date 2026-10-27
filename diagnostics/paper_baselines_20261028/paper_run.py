"""Cache-registered actual closed-loop evaluation of paper controller ports."""
import sys,json,time,subprocess,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'basis_discovery_20261027'))
import runner as R
import numpy as np
from paper_adapters import METHODS
R.OUT=HERE/'paper';R.OUT.mkdir(exist_ok=True);R.EXP='exp_paper_baselines_20261028'

def make(c,method,seed,options=None):
    scenes,ctls,_,_=R.inputs()
    x=R.make_case(scenes[c['scene']],ctls[0],c['state_index'],dict(eta=[0.,0.,0.],coefficients={}),seed)
    pins={k:subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE/'vendor'/k,text=True).strip()for k in ('dac','mgr','impc_ob','qpswift')}
    source_files=['paper_run.py','paper_adapters.py','impc_adapter.py','mgr_geometry.py','formula_baselines.py']
    x['controller_payload'].update(nominal=dict(type_code=method),paper_method=method,pins=pins,paper_sources={p:R.file_sha(HERE/p)for p in source_files},dependencies=dict(ompl='1.7.0',cvxopt='1.3.3',qpsolvers='4.13.0',shapely='2.2.0',PyQt5='5.15.11'),correction_semantics='full_paper_mechanisms_common_SI_port_v1',safety_layer='paper_own_controller_safety')
    x['controller_payload']['paper_options']=options or {}
    x['controller_uid']=R.db.uid('ctl',dict(scenario=x['scenario_uid'],payload=x['controller_payload']))
    x['rollout_uid']=R.db.uid('roll',dict(state=x['state_uid'],eta=x['eta_uid'],controller=x['controller_uid'],seed=x['seed_key']))
    return x

def run(c):
    start=time.monotonic();env=R.RN.SwapEnv(c['state'],R.RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config
    positions=[env.positions.copy()];velocities=[env.velocities.copy()];actions=[];durations=[];clear=[];traces=[]
    error=None;termination='timeout';ctl=None;speed_clipped=0
    try:
        ctl=METHODS[c['controller_payload']['paper_method']](env,c['seed'],c['controller_payload'].get('paper_options'));setup=time.monotonic()-start
        for t in range(cfg.max_steps):
            before=time.monotonic();u=np.asarray(ctl.action(env,t),float)
            if u.shape!=env.action_shape or not np.isfinite(u).all():raise ValueError('Nonfinite or malformed paper action')
            # Common physical speed limit: explicitly measured, no production CBF appended.
            norms=np.linalg.norm(u,axis=1);speed_clipped+=int((norms>cfg.max_speed+1e-7).any());u*=np.minimum(1.,cfg.max_speed/np.maximum(norms,1e-30))[:,None]
            _,done,info=env.step(u);termination=info['termination']
            positions.append(env.positions.copy());velocities.append(env.velocities.copy());actions.append(u);clear.append(info['swept_clearance']);durations.append(time.monotonic()-before)
            if done:break
    except Exception as exc:
        setup=time.monotonic()-start if ctl is None else setup;termination='unknown';error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    stats=ctl.stats()if ctl else {};solver_failures=stats.get('solver',{}).get('failures',[])
    success=int(termination=='success'and not env.collided)
    result=dict(success=success,collision=int(env.collided),timeout=int(termination=='timeout'),numerical_failure=int(error is not None),solver_failure_count=len(solver_failures),outcome=termination,error=error,episode_length=len(actions),elapsed_seconds=time.monotonic()-start,setup_seconds=setup,controller_stats=stats,paper_options=c['controller_payload'].get('paper_options',{}),speed_clipped_steps=speed_clipped,completion_time=len(actions)*cfg.dt if success else None,truncated_time=len(actions)*cfg.dt if success else cfg.max_steps*cfg.dt,horizon_time=cfg.max_steps*cfg.dt,final_goal_distances=np.linalg.norm(env.positions-env.goals,axis=1),minimum_swept_clearance=min(clear)if clear else None,collision_types=env.collision_types)
    arrays=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),actions=np.asarray(actions).reshape(-1,env.num_agents,2),goals=env.goals,swept_clearance=np.asarray(clear),step_seconds=np.asarray(durations))
    return R.clean(result),arrays
R.run=run

def main():
    name=sys.argv[1];path=R.OUT/f'{name}_planned.json'
    if '--plan'in sys.argv:
        p=json.load(open(HERE.parent/'basis_discovery_20261027/PROTOCOL.json'))
        # Pilot only: distinct original development scenes, never test-driven repair.
        cases=[]
        for scene in sorted({c['scene']for c in p['cases']}):
            c=next(c for c in p['cases']if c['scene']==scene)
            for m in METHODS:
                if name.startswith('pilot_dac') and m!='dac_author_si':continue
                if name.startswith('pilot_impc') and m!='chen_impc_ob_si_tracking':continue
                if name.startswith('pilot_mgr') and m!='mgr_author_unicycle_tracking':continue
                if name.startswith('pilot_formula') and not(m.startswith('wang')or m.startswith('zhang')):continue
                cases.append(make(c,m,0))
        R.register(cases,name)
    else:
        cases=json.load(path.open())['exact_configs']
        # Each vendor has global module state, so one task per interpreter is required.
        from concurrent.futures import ProcessPoolExecutor,as_completed
        missing=R.register(cases,name);out=R.OUT/'candidates.jsonl';done={r['rollout_uid']for r in map(json.loads,out.open())}if out.exists()else set()
        todo=[c for c in cases if c['rollout_uid']not in done]
        assert all(tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))in missing for c in todo)
        with out.open('a')as f,ProcessPoolExecutor(max_workers=min(len(R.AVAILABLE_CPUS),len(todo)or 1),max_tasks_per_child=1,initializer=R.init_worker)as pool:
            for fu in as_completed([pool.submit(R.work,c)for c in todo]):
                r=fu.result();f.write(json.dumps(r)+'\n');f.flush();print(r['controller'],r['scene'],r['success'],r['outcome'],r['error'],flush=True)
        subprocess.run([sys.executable,'-m','shared_rollout_db.src.merger','--experiment',R.EXP],cwd=R.ROOT,check=True,stdout=(R.OUT/f'{name}_merge.log').open('w'))
        R.dump(R.OUT/f'{name}_execution.json',dict(complete=True,new=len(todo),requested=len(cases)))
if __name__=='__main__':main()
