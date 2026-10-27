from audit import HERE,sha,write
from pathlib import Path
import json,re

def main():
    r=json.loads((HERE/'WRITE_PATH_AUDIT.json').read_text());proof=[]
    for i in range(32):
        p=HERE/'RESUME_REPLAYS'/f'{i:02d}.json';x=json.loads(p.read_text());assert x['passed'],p
        proof.append(dict(path=str(p),sha256=sha(p),scene=x['source']['scene'],type=x['source']['type']))
    regular=[];os_handles=[];violations=[]
    for p in sorted(HERE.glob('STRACE_WRITE_PATH.*')):
        for line in p.read_text().splitlines():
            if not (('open' in line and any(f in line for f in ('O_WRONLY','O_RDWR','O_CREAT'))) or line.startswith(('mkdir','rename'))):continue
            if ' = -1 ' in line:continue
            paths=re.findall(r'"([^\"]+)"',line)
            for path in paths:
                if not path.startswith('/'):continue
                item=dict(log=str(p),path=path,operation=line)
                if path.startswith(('/proc/','/dev/')):os_handles.append(item);continue
                regular.append(item)
                if not Path(path).is_relative_to(HERE):violations.append(item)
    assert not violations,violations
    r.update(status='PASSED',passed=True,trajectory_replays=proof,trajectories_per_scene=8,
       persistent_write_paths=regular,os_pseudo_file_or_device_handles=os_handles,violations=violations,
       strace_logs={str(p):sha(p) for p in HERE.glob('STRACE_WRITE_PATH.*')},
       note='OS process-name pseudo-files and CUDA device handles are not persistent research output. JAX execution backend remains CPU. No historical cache was written.')
    write(HERE/'WRITE_PATH_AUDIT.json',r)
    write(HERE/'WRITE_PATH_READY.receipt.json',dict(files={str(HERE/'WRITE_PATH_AUDIT.json'):sha(HERE/'WRITE_PATH_AUDIT.json')},passed=True))
    print('Write path audit passed:',len(proof),'bitwise replays;',len(regular),'persistent writes all private',flush=True)
if __name__=='__main__':main()
