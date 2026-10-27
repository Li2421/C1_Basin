from audit import HERE,ROOT,sha,write
from pathlib import Path
import subprocess,re

def main():
    backend=HERE/'resume_backend';python=[]
    for p in sorted(backend.glob('*.py')):
        lines=p.read_text().splitlines()
        relevant=[dict(line=i+1,text=s.strip()) for i,s in enumerate(lines) if any(t in s for t in ('mkdir','write_text','write_bytes',".open(",'os.environ','getenv','CDLL','ROOT =','SOURCE ='))]
        python.append(dict(path=str(p),sha256=sha(p),path_statements=relevant))
    names=['astar_kernel.so','cbs_kernel.so','astar_kernel_tiled.so','cbs_kernel_tiled.so','astar_kernel_spatial_mask_candidate_20261022.so','pa_query_memo_facts_candidate_20261022.so','native_resume/cbs_kernel_resume.so']
    libraries=[]
    for name in names:
        p=HERE/name;strings=subprocess.check_output(['strings',str(p)],text=True)
        absolutes=sorted({s for s in strings.splitlines() if s.startswith('/')})
        assert not any('baseline_literature_scenes_distributed_v2/runtime_cbs' in s for s in absolutes),p
        libraries.append(dict(path=str(p),sha256=sha(p),absolute_path_strings=absolutes,
            writes='Private query cache only' if name.endswith('cbs_kernel_resume.so') else 'No file write primitives in corresponding A*/CBS/memory-cache source'))
    write(HERE/'WRITE_PATH_AUDIT.json',dict(status='STATIC_PATHS_VERIFIED_REPLAY_PENDING',python=python,libraries=libraries,
        allowed_output_roots=[str(HERE),str(ROOT/'shared_rollout_db/journals/exp_baseline_literature_scenes_stage2_collision_fix_v1')],
        private_disk_cache=str(HERE/'runtime_cbs_query_cache_resume'),
        python_env_output_roots={k:str(HERE/v) for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cuda_cache')]},
        read_only_constants='poc_field_controller ROOT/SOURCE are input roots only; its module has no file writer. No historical coordinators imported.',
        old_unsafe_library='Retained for historical provenance but not loaded by private backend',
        source_changes='Path-only experiment HERE and native cache root/library/namespace paths; controller algorithms unchanged'))
    print('Inventoried',len(python),'Python modules and',len(libraries),'loaded libraries')
if __name__=='__main__':main()
