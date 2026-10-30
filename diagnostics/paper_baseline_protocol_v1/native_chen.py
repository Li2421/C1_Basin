"""Original Chen entrypoint with dependency/path setup only."""
import guard
from guard import HERE
from budget import reserve
import os,sys,json,time,runpy,traceback,hashlib
root=HERE/'native_reproduction/chen';author=root/'author';base=HERE/'inputs/diagnostics/paper_baselines_20261028';src=author/'src'
sys.path[:0]=[str(HERE/'native_deps'),str(base/'deps_ompl17')]+[str(p)for p in base.glob('deps*')if p.is_dir()and p.name!='deps_ompl17']+[str(src)]
# The author bundled an ABI-incompatible OMPL; use the copied compatible dependency before resolving author imports.
import ompl
os.environ['MPLBACKEND']='Agg';os.chdir(src)
frozen=json.loads((root/'SOURCE_FROZEN.json').read_text())
assert all(hashlib.sha256((author/n).read_bytes()).hexdigest()==frozen['files'][n]for n in frozen['algorithm_files'])
reserve({'rollout_uid':'native_chen_author_H8_default_v1','purpose':'native_author_reproduction_no_shared_db'})
t=time.monotonic();result={'method':'chen_impc_ob_si_tracking','entrypoint':'src/test.py','author_algorithm_edits':0,'dependency_changes':['CPython3.9openGJKbinary rebuilt from official openGJK source for3.11','map/data output directories created','MPLBACKEND=Agg']}
try:
    runpy.run_path(str(src/'test.py'),run_name='__main__')
    import SET,numpy as np
    data=np.loadtxt(src/'data/data.csv');stride=SET.K+5;assert (len(data)-4)%stride==0
    starts=list(range(4,len(data),stride));positions=np.array([data[1].reshape(SET.Num,2)]+[data[k+4].reshape(SET.Num,2)for k in starts]);goals=np.asarray(SET.target)
    np.savez_compressed(root/'native_trace.npz',positions=positions,goals=goals,robot_radius=np.asarray(SET.ExtendWidth),obstacles=np.asarray([o.vertex_list for o in SET.ini_obstacle_list]))
    errors=np.linalg.norm(positions[-1]-goals,axis=1)
    result.update(run_completed=True,all_arrived=bool(np.all(errors<=.01)),goal_errors=errors.tolist(),steps=len(starts),native_goal_tolerance=.01,author_prediction_steps=SET.K,author_dt=SET.h,author_process_workers=SET.core_num)
except Exception as exc:result.update(run_completed=False,error=repr(exc),traceback=traceback.format_exc())
result['elapsed_seconds']=time.monotonic()-t;result['author_python_sources_unchanged']=all(hashlib.sha256((author/n).read_bytes()).hexdigest()==frozen['files'][n]for n in frozen['algorithm_files'])
(root/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
