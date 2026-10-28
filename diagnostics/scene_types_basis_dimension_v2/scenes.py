"""v2 builders and explicit comparisons with the frozen v1 physical tasks."""
from core import *
import importlib.util

spec=importlib.util.spec_from_file_location('private_scene_v2_impl',HERE/'generated/scenes_v2_impl.py')
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
TITLES,LADDERS,SEED=impl.TITLES,impl.LADDERS,impl.SEED
geometry=impl.geometry

def comparable(s):
    s=json.loads(json.dumps(s));s.pop('definition_sha256',None)
    s['config'].pop('max_steps',None)
    return s

def build(type_id,level=0,widen=0.,crossing=None):
    s=impl.build(type_id,level,widen,crossing)
    old=next((e['scene']for e in read(ROOT/'diagnostics/scene_types_basis_dimension_v1/SCENES_FROZEN.json')['entries']if e['type_id']==type_id and e.get('scene')),None)
    equal=old is not None and comparable(s)==comparable(old)
    # The horizon field is the only permitted difference for v1 reuse.
    original_without_time_sha=digest(comparable(old)) if old else None
    generated_without_time_sha=digest(comparable(s))
    if equal:assert original_without_time_sha==generated_without_time_sha
    if not equal:
        name=f'xsc_stb2_t{type_id:02d}_l{level}_w{round(widen*1000):03d}'
        s['name']=name;s['config']['scene']=name
        for state in s['states']:state['scene']=name
    length=s['longest_initial_route'];cfg=s['config']
    s['time_limits']={str(k):dict(max_steps=int(math.ceil(k*length/cfg['max_speed']/cfg['dt'])),seconds=int(math.ceil(k*length/cfg['max_speed']/cfg['dt']))*cfg['dt'])for k in (4,5,6)}
    s['v1_equivalent_except_horizon']=equal
    s['v1_definition_sha256']=old['definition_sha256']if equal else None
    s['definition_without_time_sha256']=generated_without_time_sha
    s['v1_definition_without_time_sha256']=original_without_time_sha
    s.pop('definition_sha256',None);s['definition_sha256']=digest(s)
    return s

def build_all():
    index=[]
    for t in range(1,18):
        for level in range(len(LADDERS[t])):
            path=HERE/'scene_design'/f't{t:02d}_l{level}.json'
            if path.exists():v=read(path)
            else:
                try:v=dict(status='valid',scene=build(t,level))
                except Exception as exc:v=dict(status='geometry_gap',type_id=t,level=level,error=str(exc))
                dump(path,v)
            index.append(dict(type_id=t,level=level,status=v['status'],path=str(path),error=v.get('error')))
            print(TITLES[t],level,v['status'],v.get('error',''),flush=True)
    dump(HERE/'SCENE_DESIGN_INDEX.json',index)
if __name__=='__main__':build_all()
