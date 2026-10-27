"""Fresh interpreter selecting only the corrected IMPC occupancy-domain adapter."""
from core import *
import sys,importlib.util,time,os
sys.path.insert(0,str(HERE/'ports'))
spec=importlib.util.spec_from_file_location('impc_adapter',HERE/'ports/impc_adapter_v2.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['impc_adapter']=module
import paper_episode
import runner,runtime

if __name__=='__main__':
    c=read(sys.argv[1]);dump(HERE/'attempts'/f"{c['rollout_uid']}_{time.time_ns()}.json",dict(rollout_uid=c['rollout_uid'],job=os.environ.get('SLURM_JOB_ID'),time=time.time()))
    if len(list((HERE/'attempts').glob('*.json')))>45000:raise RuntimeError('HARD_STOP rollout budget')
    runtime.run=paper_episode.run;r=runner.work(c);print('paper_impc_v2',r['type_id'],r['state_index'],r['outcome'],round(r['wall_seconds'],3),flush=True)
