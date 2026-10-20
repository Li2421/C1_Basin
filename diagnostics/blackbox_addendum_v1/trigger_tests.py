from common import *
from trigger import Trigger
from audit_utils import source_inventory
import ast
from types import SimpleNamespace

def main():
    src=D/'teacher_student_macflow_v1d/fair_rhr.py';old=src.read_text();node=next(n for n in ast.parse(old).body if isinstance(n,ast.ClassDef) and n.name=='Trigger');oldbody=ast.get_source_segment(old,node);new=(HERE/'trigger.py').read_text();assert oldbody in new
    checks=0
    # Independent sliding-history formula on actual historical physical trajectories.
    cs=read(R33/'main_CASES.json');paths=[]
    for n in ['G0','G3','DM1']:
        c=next(c for c in cs if c['base_controller_name']==n and c['state_index']==0 and c['seed']==0 and c['arm']=='safety_baseline');raw=read(R33/'main/raw'/f"{c['rollout_uid']}.json");paths.append(raw['trace_path'])
    for path in paths:
        z=np.load(path);pos=z['positions'];vel=z['velocities'];goals=z['goals'];dist=np.linalg.norm(goals[None]-pos,axis=-1);notgoal=dist>.09;low=(np.linalg.norm(vel,axis=-1)<.05)&notgoal
        for rule in read(D/'teacher_student_macflow_v1d/RHR_GRID.json'):
            trigger=Trigger(rule);w=rule['steps'];cfg=SimpleNamespace(goal_tolerance=.09)
            for t in range(len(pos)-1):
                expected=notgoal[t] if rule['mode']=='always' else (np.all(low[t-w+1:t+1],axis=0)&notgoal[t] if t>=w else np.zeros(4,bool)) if rule['mode']=='slow' else ((dist[t-w]-dist[t]<.05)&notgoal[t] if t>=w else np.zeros(4,bool));actual=trigger.update(SimpleNamespace(step_count=t,positions=pos[t],velocities=vel[t],goals=goals,config=cfg));assert np.array_equal(actual,expected);checks+=1
    write(HERE/'TRIGGER_TESTS.json',dict(utc=utc(),passed=True,source_Trigger_byte_identical=True,activation_history_checks=checks,reference_source_sha256=sha(src),new_physical_steps=0));source_inventory('imports_trigger_tests.json')
if __name__=='__main__':main()
