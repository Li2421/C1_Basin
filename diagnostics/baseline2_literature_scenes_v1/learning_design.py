from experiment import *
from registry import case,register
from states import config
from flow_scene import physical12
import numpy as np
import collections

def selected_states(scene,role):
    states=read(HERE/f'FLOW_STATES_{scene}.json')
    if scene in ('lit_narrow_passage','lit_doorway'):ids=set(range(1,33)) if role=='train' else {0,*range(33,64)};result=[s for s in states if s['index'] in ids]
    elif scene=='lit_circle_five':ids=set(range(32)) if role=='train' else set(range(32,64));result=[s for s in states if s['index'] in ids]
    else:ids=set(range(16)) if role=='train' else set(range(16,32));result=[s for s in states if s['N'] in (8,14) and s['index'] in ids]
    if len(result)!=32:raise ValueError(f'{scene} {role}: needs exact preregistered 32 valid states, only {len(result)} remain after geometry exclusion')
    return sorted(result,key=lambda s:(s['N'],s['index']))

def make_case(s,ctl,a,index,seed,method=None):
    hashes={n:sha(HERE/n) for n in ('flow_runtime.py','flow_scene.py','projection_adapter.py','controllers.py','planning.py','astar_kernel.cpp','cbs_kernel.cpp')}
    cfg=config(s['scene'],s['N']).to_dict()
    if s['scene']=='lit_crowded_random':cfg['max_steps']=int(50/.15)
    sc=dict(version='flow',config=cfg,walls_sha256=sha(HERE/'states.py'),environment_sha256=hashes['flow_scene.py'])
    payload=dict(nominal=ctl,execution_sources=hashes,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false')
    physical_method='safety' if index==0 else f'learning_action_{index:02d}'
    c=case(s,'flow',physical_method,sc,payload,seed,a['eta_a'],a['eta_b'],a['theta'])
    c['menu_index']=index
    if method:c['method']=method
    return c

def plan(scene):
    prereg();usability=read(HERE/f'BASELINE_USABILITY_{scene}.json')
    if not usability['informative']:
        write(HERE/f'LEARNING_GAP_{scene}.json',dict(reason=usability['reason'],required_by_protocol=False));return
    try:states=selected_states(scene,'train');selected_states(scene,'evaluation')
    except ValueError as e:write(HERE/f'LEARNING_GAP_{scene}.json',dict(reason=str(e),required_by_protocol=True));return
    rows=read(HERE/'CONTROLLER_MANIFEST.json');groups=collections.defaultdict(list)
    for r in rows:groups[r['type_code']].append(r)
    controllers=[r for t in sorted(groups) for r in groups[t][:2]];menu=read(HERE/'LEARNING_MENU.json')
    cases=[make_case(s,ctl,a,ai,0) for ctl in controllers for s in states for ai,a in enumerate(menu)]
    write(HERE/f'LEARNING_SPLIT_{scene}.json',dict(training=states,evaluation=selected_states(scene,'evaluation'),training_controllers=controllers,menu_sha256=sha(HERE/'LEARNING_MENU.json')))
    seal(f'LEARNING_SPLIT_{scene}.receipt.json',[HERE/f'LEARNING_SPLIT_{scene}.json',HERE/'learning_design.py'],before_training_labels=True)
    register('learning_labels_'+scene,cases)

def encode(a):
    early=np.asarray(a['eta_a']);late=np.asarray(a['eta_b']);progress=a['theta'] is not None
    return np.asarray([*early,*late,*(late-early),0. if progress else 500/700,float(np.all(early==0)),float(np.all(late==0)),float(progress),0.,float(a['theta'] or 0.),1.,0.,0.,0.],np.float32)

def find_raw(rid):
    for folder in ('learning_labels_', 'safety_', 'fixed_'):
        for p in HERE.glob(folder+'*'):
            q=p/'raw'/f'{rid}.json'
            if q.exists():return q
    raise FileNotFoundError(rid)

def assemble(scene):
    split=read(HERE/f'LEARNING_SPLIT_{scene}.json');states=split['training'];ctls=split['training_controllers'];cases=read(HERE/f'learning_labels_{scene}_CASES.json');menu=read(HERE/'LEARNING_MENU.json')
    labels=np.full((len(ctls),32,12),-1,np.int8);provenance=[];ci={c['controller']:i for i,c in enumerate(ctls)};si={(s['N'],s['index']):i for i,s in enumerate(states)}
    for c in cases:
        path=find_raw(c['rollout_uid']);r=read(path);a=c['menu_index'];i=ci[c['controller_payload']['nominal']['controller']];j=si[(c['state']['N'],c['state']['index'])]
        labels[i,j,a]=-1 if r['numerical_failure'] else r['success'];provenance.append(dict(rollout_uid=c['rollout_uid'],source=str(path),sha256=sha(path)))
    X=np.asarray([physical12(s) for s in states]);actions=np.asarray([encode(a) for a in menu]);assert X.shape==(32,12) and actions.shape==(12,19)
    path=HERE/f'LEARNING_BUNDLE_{scene}.npz';np.savez_compressed(path,labels=labels,physical12=X,action=actions,controllers=np.asarray([c['controller'] for c in ctls]),types=np.asarray([c['type_code'] for c in ctls]))
    write(HERE/f'LEARNING_BUNDLE_SOURCES_{scene}.json',provenance);seal(f'LEARNING_BUNDLE_{scene}.receipt.json',[path,HERE/f'LEARNING_BUNDLE_SOURCES_{scene}.json',HERE/f'LEARNING_SPLIT_{scene}.receipt.json'],before_training=True,unknown_labels=int((labels<0).sum()),observed_labels=int((labels>=0).sum()))
    print(scene,'bundle',labels.shape,'unknown',int((labels<0).sum()))
if __name__=='__main__':plan(sys.argv[2]) if sys.argv[1]=='plan' else assemble(sys.argv[2])
