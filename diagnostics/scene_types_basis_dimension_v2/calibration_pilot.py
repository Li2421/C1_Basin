"""Combine independent geometry-only gates and first witness pilots in one batch."""
from dispatch import *
from calibration_gates import tasks
from witnesses import make

def main():
    assert read(HERE/'calibration_base_gates_execution.json')['complete']
    gates=read(HERE/'CALIBRATION_BASE_GATES.json');cases=[];manifest=[]
    for t in range(1,18):
        rr=[r for r in gates if r['type_id']==t]
        rate=sum(r['safety_success']for r in rr)/sum(r['safety_n']for r in rr)
        if rate>.6:
            scene=read(HERE/'scene_design'/f't{t:02d}_l1.json')['scene']
            cases.extend(tasks(scene));manifest.append(dict(type_id=t,action='harder_level_gate',level=1,base_safety_rate=rate))
        else:
            scene=read(HERE/'scene_design'/f't{t:02d}_l0.json')['scene']
            order='left_first'if t in(8,10,11,12)else'shortest'
            cases.append(make(scene,controllers()[0],0,'sequential_release',order))
            manifest.append(dict(type_id=t,action='sequential_witness_pilot',level=0,state=0,order=order))
    dump(HERE/'CALIBRATION_PILOT_SELECTION.json',dict(selection=manifest,rule='Eliminate levels already failing the safety-only hardness gate before spending witness compute. Only geometry/solo/safety evidence observed. No corrections or paper outcomes.'))
    execute(cases,'calibration_witness_pilots',worker_entry='worker_witness.py')
if __name__=='__main__':main()
