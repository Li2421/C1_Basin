from audit_supplement import *
import re,subprocess
records=[]
for p in (HERE/'backend').glob('*.py'):
 lines=[dict(line=i,text=l) for i,l in enumerate(p.read_text().splitlines(),1) if any(t in l for t in ('HERE/','ROOT =','SOURCE =','write(','open(','mkdir','os.environ'))]
 records.append(dict(path=str(p),sha256=sha(p),path_lines=lines))
libs=[]
for p in [*HERE.glob('*.so'),*(HERE/'native_resume').glob('*.so')]:
 strings=subprocess.check_output(['strings',str(p)],text=True,errors='replace');absolute=[s for s in strings.splitlines() if s.startswith('/home/')]
 libs.append(dict(path=str(p),sha256=sha(p),absolute_path_strings=absolute))
write(HERE/'WRITE_PATH_AUDIT.json',dict(utc=datetime.now(timezone.utc).isoformat(),status='PRE_REPLAY_INVENTORY',modules=records,libraries=libs,
 write_roots=[str(HERE),str(ROOT/'shared_rollout_db/journals/exp_baseline_literature_scenes_supplement_eta_swap8_v1')],
 pure_cache_root=str(HERE/'runtime_cbs_query_cache'),backend_environment='TMPDIR/MPLCONFIGDIR/XDG_CACHE_HOME/CUDA_CACHE_PATH all supplement; bytecode disabled',
 readonly_paths='PoC ROOT, SOURCE and external import paths are inputs only; actors never instantiate PoC Runtime; compiled ASTAR/PA do not open disk files',
 new_geometry_adapter='Only finite wall-segment cost dispatch added to private copies; all old-scene branches unchanged. Replays verify the path-only backend.'))
print('Inventoried',len(records),'modules',len(libs),'libraries',flush=True)
