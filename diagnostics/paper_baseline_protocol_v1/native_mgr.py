"""Read-only observation of the unchanged author GUI simulation."""
import guard
from guard import HERE
from budget import reserve
import os,sys,runpy,json,time,traceback,hashlib
root=HERE/'native_reproduction/mgr';author=root/'author';base=HERE/'inputs/diagnostics/paper_baselines_20261028'
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['XDG_RUNTIME_DIR']=str(HERE/'tmp/qt-runtime')
from pathlib import Path
Path(os.environ['XDG_RUNTIME_DIR']).mkdir(exist_ok=True,mode=0o700)
sys.path[:0]=[str(author),str(HERE/'native_deps')]+[str(p)for p in base.glob('deps*')if p.is_dir()]
os.chdir(author)
instance='./instances/rect_maps/RectEnv15/agents60/RectEnv_15_60_0.yaml'
sys.argv=['main.py','--instance',instance]
freeze=json.loads((root/'SOURCE_FROZEN.json').read_text());assert all(hashlib.sha256((author/n).read_bytes()).hexdigest()==h for n,h in freeze['files'].items())
from visualization import MainWindow
import numpy as np
frames=[];window=[None];code=MainWindow.update_simulation.__code__
def observe(frame,event,arg):
    if event=='return'and frame.f_code is code:
        w=frame.f_locals['self'];window[0]=w
        frames.append([[a.position.x(),a.position.y()]for a in w.simulator.agents])
# Observation only: no author functions or arguments replaced.
sys.setprofile(observe)
reserve({'rollout_uid':'native_mgr_author_rect60_default_v1','purpose':'native_author_reproduction_no_shared_db'})
t=time.monotonic();result={'method':'mgr_author_unicycle_tracking','instance':instance,'author_algorithm_edits':0,'recording':'Python profile return hook copies positions after author GUI step; does not write controller state'}
try:
    try:runpy.run_path(str(author/'main.py'),run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0,None):raise
    w=window[0];assert w is not None
    goals=np.array([[a.target.x(),a.target.y()]for a in w.simulator.agents]);radii=np.array([a.radius for a in w.simulator.agents])
    np.savez_compressed(root/'native_trace.npz',positions=np.asarray(frames),goals=goals,radii=radii)
    result.update(run_completed=True,all_arrived=bool(all(w.arrived_robots)),arrived=sum(w.arrived_robots),robots=len(w.arrived_robots),arrival_times=w.arrival_times,frames=len(frames),final_errors=np.linalg.norm(np.asarray(frames[-1])-goals,axis=1).tolist())
except Exception as exc:result.update(run_completed=False,error=repr(exc),traceback=traceback.format_exc())
finally:sys.setprofile(None)
result['elapsed_seconds']=time.monotonic()-t;result['author_sources_unchanged']=all(hashlib.sha256((author/n).read_bytes()).hexdigest()==h for n,h in freeze['files'].items())
(root/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
