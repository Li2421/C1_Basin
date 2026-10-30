"""Unmodified author narrow-passage entrypoint, with headless plotting only."""
import guard
from guard import HERE
from budget import reserve
from pathlib import Path
import os,sys,json,runpy,time,traceback,hashlib
root=HERE/'native_reproduction/dac_random';author=root/'author'
base=HERE/'inputs/diagnostics/paper_baselines_20261028'
sys.path[:0]=[str(author),str(base/'vendor/qpswift/python')]+[str(p)for p in base.glob('deps*')if p.is_dir()]
os.environ['MPLBACKEND']='Agg'
os.chdir(author)
freeze=json.loads((root/'SOURCE_FROZEN.json').read_text())
assert all(hashlib.sha256((author/n).read_bytes()).hexdigest()==h for n,h in freeze['files'].items())
reserve({'rollout_uid':'native_dac_author_random20_default_v1','purpose':'native_author_reproduction_no_shared_db'})
t=time.monotonic();result={'method':'dac_author_si','native_entrypoint':'sim-random_obs_navigation.py','author_algorithm_edits':0,'dependency_adjustments':['Copied qpSWIFT extension added to import path','MPLBACKEND=Agg for unattended display']}
try:
    values=runpy.run_path(str(author/'sim-random_obs_navigation.py'),run_name='__main__')
    import numpy as np
    positions=np.array([a.cur_pos for a in values['agent_set']]);goals=np.asarray(values['agent_tar_pos_set'])
    errors=np.linalg.norm(positions-goals,axis=1)
    result.update(run_completed=True,all_arrived=bool(np.all(errors<values['Cfg'].sim_stop_err)),goal_errors=errors.tolist(),author_goal_tolerance=values['Cfg'].sim_stop_err,author_collision_flag=bool(values.get('collision_flag',False)),steps=int(values['step']))
    np.savez_compressed(root/'native_trace.npz',positions=np.concatenate([np.asarray(values['agent_init_pos'])[:,None,:],np.asarray(values['pos_save'])],axis=1),goals=goals,obstacle_centers=np.asarray(values['obs_pos_set']),obstacle_radius=np.asarray(values['Cfg'].obs_radius),robot_radius=np.asarray(values['Cfg'].agent_r))
except Exception as exc:
    result.update(run_completed=False,error=repr(exc),traceback=traceback.format_exc())
result['elapsed_seconds']=time.monotonic()-t
result['author_sources_unchanged']=all(hashlib.sha256((author/n).read_bytes()).hexdigest()==h for n,h in freeze['files'].items())
(root/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
