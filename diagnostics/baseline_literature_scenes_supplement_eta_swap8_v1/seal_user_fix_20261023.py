import bootstrap
from experiment import *
import re

def main():
    assert read(HERE/'MOVING_USER_FIX_UNIT_GATE.json')['passed']
    assert read(HERE/'USER_FIX_UNCHANGED_FREE_AND_GEOMETRY.receipt.json')['passed']
    assert read(HERE/'USER_FIX_WRITE_PATH_PROBE.json')['passed']
    for p,h in read(HERE/'IMPLEMENTATION_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    persistent=[];handles=[];bad=[]
    for p in HERE.glob('STRACE_USER_FIX_WRITE_PATH.*'):
        for line in p.read_text().splitlines():
            if not (('open' in line and any(flag in line for flag in ('O_WRONLY','O_RDWR','O_CREAT'))) or line.startswith(('mkdir','rename'))):continue
            if ' = -1 ' in line:continue
            for target in re.findall(r'"([^\"]+)"',line):
                if not target.startswith('/'):continue
                row=dict(log=str(p),path=target,operation=line)
                if target.startswith(('/proc/','/dev/')):handles.append(row);continue
                persistent.append(row)
                if not Path(target).is_relative_to(HERE):bad.append(row)
    assert not bad,bad
    oldaudit=read(HERE/'WRITE_PATH_AUDIT.json')
    for lib in oldaudit['libraries']:assert sha(lib['path'])==lib['sha256']
    names=('swap_geometry_user_fix_20261023.py','swap_safety_user_fix_20261023.py','controllers_layout_user_fix_20261023.py','nominal_geometry_user_fix_20261023.py','swap_runtime_user_fix_20261023.py','case_factory_user_fix_20261023.py','worker_user_fix_20261023.py','LAYOUT_REVISION_DESIGN.json','MOVING_USER_FIX_UNIT_GATE.json')
    write(HERE/'WRITE_PATH_AUDIT_USER_FIX_20261023.json',dict(utc=utc(),passed=True,old_audit_sha256=sha(HERE/'WRITE_PATH_AUDIT.json'),persistent_writes=persistent,os_handles=handles,violations=[],libraries_unchanged=True,write_roots=[str(HERE),str(ROOT/'shared_rollout_db/journals/exp_baseline_literature_scenes_supplement_eta_swap8_v1')],new_python_modules={n:sha(HERE/n) for n in names if n.endswith('.py')},cache_key_geometry='PA sufficient_key includes walls.tobytes and margin; CBS prepare world includes wall bytes and margin; changed layout cannot reuse incompatible pure query',free_bitwise_replays=8))
    seal('USER_FIX_IMPLEMENTATION_FROZEN_20261023.receipt.json',[HERE/n for n in names],before_any_revised_research_rollout=True,only_authorized_scene_changes=True)
    seal('USER_FIX_READY_20261023.receipt.json',[HERE/n for n in ('USER_FIX_IMPLEMENTATION_FROZEN_20261023.receipt.json','WRITE_PATH_AUDIT_USER_FIX_20261023.json','USER_FIX_UNCHANGED_FREE_AND_GEOMETRY.receipt.json','MOVING_USER_FIX_UNIT_GATE.json','USER_FIX_DERIVATION_20261023.md','PARAMETERS_USER_FIX_20261023.md','PREREGISTRATION_USER_FIX_20261023.md','LAYOUT_REVISION_DESIGN.json')],passed=True,before_any_revised_research_rollout=True)
    print('User fix frozen and READY; original/free unchanged',flush=True)
if __name__=='__main__':main()
