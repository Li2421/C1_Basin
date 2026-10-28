"""All prescribed ladder entries; never select a level using rollout results."""
from core import *
import importlib.util

OLD=ROOT/'diagnostics/scene_types_basis_dimension_v1'
spec=importlib.util.spec_from_file_location('private_scene_v3_impl',HERE/'generated/scenes_v3_impl.py')
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
TITLES,LADDERS,SEED=impl.TITLES,impl.LADDERS,impl.SEED
geometry=impl.geometry
MODIFIED={2,7,16,17}


def build(type_id,level=0,widen=0.,crossing=None):
    assert widen==0.,'No calibration or widening in v3'
    old_path=OLD/'scene_design'/f't{type_id:02d}_l{level}.json'
    if type_id not in MODIFIED:
        old=read(old_path);assert old['status']=='valid'
        s=json.loads(json.dumps(old['scene']))
        original_sha=s['definition_sha256']
        original_steps=s['config']['max_steps']
        old_states=json.loads(json.dumps(s['states']))
    else:
        s=impl.build(type_id,level,widen,crossing)
        name=f'xsc_stb3_t{type_id:02d}_l{level}_w000'
        s['name']=name;s['config']['scene']=name
        for state in s['states']:state['scene']=name
        original_sha=None;original_steps=None
    cfg=s['config'];length=s['longest_initial_route']
    s['time_limits']={str(k):dict(max_steps=int(math.ceil(k*length/cfg['max_speed']/cfg['dt'])),
                                  seconds=int(math.ceil(k*length/cfg['max_speed']/cfg['dt']))*cfg['dt'])for k in (4,6)}
    cfg['max_steps']=s['time_limits']['6']['max_steps']
    if type_id not in MODIFIED:
        assert s['states']==old_states
        assert original_steps==s['time_limits']['4']['max_steps']
        check=json.loads(json.dumps(s));check.pop('time_limits')
        check['config']['max_steps']=original_steps
        assert check==old['scene']
    s['v1_equivalent_except_horizon']=type_id not in MODIFIED
    s['v1_definition_sha256']=original_sha
    s['v1_scene_source']=str(old_path)if original_sha else None
    s.pop('definition_sha256',None);s['definition_sha256']=digest(s)
    return s


def build_all():
    p=HERE/'SCENES.json'
    if p.exists():return read(p)
    entries=[]
    for t in range(1,18):
        for level in range(len(LADDERS[t])):
            s=build(t,level)
            entry=dict(type_id=t,level=level,title=TITLES[t],scene=s,
                       source='v1_literal_scene'if t not in MODIFIED else'prescribed_geometry_repair',
                       no_difficulty_selection=True)
            entries.append(entry)
            dump(HERE/'scene_design'/f't{t:02d}_l{level}.json',dict(status='valid',scene=s))
            print('SCENE',t,level,'N',s['config']['num_agents'],'horizon',s['config']['max_steps'],flush=True)
    assert len(entries)==49
    result=dict(entries=entries,generation_seed=SEED,case_states=list(range(8)),
                physical_instances=49*8,nominal_cases=49*8*2,
                no_correction_seen=True,no_calibration=True,no_family_selection=True)
    dump(p,result);return result


if __name__=='__main__':build_all()
