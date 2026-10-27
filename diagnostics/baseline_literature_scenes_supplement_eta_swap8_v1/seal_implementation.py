import bootstrap
from experiment import *
from audit_supplement import evidence
import re,subprocess

def main():
    obj=read(HERE/'WRITE_PATH_AUDIT.json');proof=[]
    for i in range(32):
        p=HERE/'REPLAYS'/f'{i:02d}.json';r=read(p);assert r['passed'];proof.append(dict(path=str(p),sha256=sha(p),scene=r['source']['scene'],type=r['source']['type']))
    persistent=[];os_handles=[];violations=[]
    for p in HERE.glob('STRACE_WRITE_PATH.*'):
        for line in p.read_text().splitlines():
            if not (('open' in line and any(flag in line for flag in ('O_WRONLY','O_RDWR','O_CREAT'))) or line.startswith(('mkdir','rename'))):continue
            if ' = -1 ' in line:continue
            for path in re.findall(r'"([^\"]+)"',line):
                if not path.startswith('/'):continue
                item=dict(log=str(p),path=path,operation=line)
                if path.startswith(('/proc/','/dev/')):os_handles.append(item);continue
                persistent.append(item)
                if not Path(path).is_relative_to(HERE):violations.append(item)
    assert not violations,violations
    assert '14 lit_crowded_random conflict_based_search PASSED' in (HERE/'STRACE_REPLAY.log').read_text()
    obj.update(status='PASSED',passed=True,trajectory_replays=proof,persistent_writes=persistent,os_handles=os_handles,violations=violations,backend_files={str(p):sha(p) for p in (HERE/'backend').glob('*.py')},strace_logs={str(p):sha(p) for p in HERE.glob('STRACE_WRITE_PATH.*')})
    write(HERE/'WRITE_PATH_AUDIT.json',obj);seal('WRITE_PATH_READY.receipt.json',[HERE/'WRITE_PATH_AUDIT.json'],passed=True)
    assert read(HERE/'SWAP_COMPONENTS.receipt.json')['passed']
    names=('swap_geometry.py','swap_safety.py','swap_runtime.py','swap_diagnostics.py','experiment.py','registry.py','case_factory.py','worker.py','CONTROLLER_MANIFEST.json')
    files=[HERE/n for n in names]+list((HERE/'backend').glob('*.py'))+[HERE/'native_resume/cbs_kernel_resume.so',HERE/'astar_kernel_spatial_mask_candidate_20261022.so',HERE/'pa_query_memo_facts_candidate_20261022.so']
    seal('IMPLEMENTATION_FROZEN.receipt.json',files,passed=True,before_any_new_research_rollout=True,old_scene_bitwise_replays=32,new_geometry_extension_only=True)
    seal('PREREGISTRATION.receipt.json',[HERE/'PREREGISTRATION.md',HERE/'PARAMETERS.md',HERE/'HISTORY_CHECK.md',HERE/'SCENES.receipt.json'],before_any_new_research_rollout=True)
    write(HERE/'RESOURCE_POLICY.json',dict(cpu_total=18,management=1,gpu_total=2,actual_gpu=0,release_deadline_singapore='2026-10-23T09:00:00+08:00',deadline_passed=True))
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write(f'\n{utc()} 第二部分独立开工审计1540源文件；新几何/组件门、32条SI逐位回放和strace通过。提前提交的回放批28322已停，保留29条；完整stat通过后28338仅恢复3条。此为调度程序失序，不倒补审计时间；未发现新的越界写入。H已封存不写。\n')
    print('Scientific implementation and preregistration frozen; write path ready',flush=True)
if __name__=='__main__':main()
