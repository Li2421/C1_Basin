"""Replay copied geometry samplers, then extend only the accepted RNG stream.

No controller is instantiated. Frozen geometry, routes and horizons are retained.
The actual generation is queued behind delivery of the current study.
"""
import guard
from guard import HERE
import ast, dataclasses, datetime, functools, hashlib, json, math, shutil, sys, types
from pathlib import Path

BASE=HERE/'inputs/diagnostics/scene_types_oracle_coverage_v3'
V1=HERE/'inputs/diagnostics/scene_types_basis_dimension_v1'
BACKEND=BASE/'generated/backend'

def read(path): return json.loads(Path(path).read_text())
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(data): return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def definitions(path,names,ns,extend=False):
    """Execute selected original definitions without importing the old harness."""
    tree=ast.parse(path.read_text());nodes=[];replacements=0
    for node in tree.body:
        found=getattr(node,'name',None)
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):found=node.targets[0].id
        if found not in names:continue
        if extend and found=='build':
            for loop in ast.walk(node):
                if isinstance(loop,ast.For) and isinstance(loop.target,ast.Name) and loop.target.id=='si':
                    assert ast.dump(loop.iter)==ast.dump(ast.parse('range(12)',mode='eval').body)
                    loop.iter.args=[ast.Name(id='STATE_COUNT',ctx=ast.Load())];replacements+=1
        nodes.append(node)
    assert len(nodes)==len(names), (path,names,[getattr(n,'name',None)for n in nodes])
    assert not extend or replacements==1
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(module,str(path),'exec'),ns)

def namespace():
    import numpy as np
    sys.path.insert(0,str(HERE/'inputs/production'))
    from bottleneck_family.scenario import Config,Geometry
    from single_integrator.environment import point_segment_distance,segment_distance
    work=HERE/'state_generation_workspace';work.mkdir(exist_ok=True)
    for source,target in [(BASE/'SCENE2_RESOLUTION.json',work/'SCENE2_RESOLUTION.json'),(BASE/'generated/LAYOUT_REVISION_DESIGN.json',work/'LAYOUT_REVISION_DESIGN.json')]+[(p,work/'forest_layouts'/p.name)for p in(BASE/'forest_layouts').glob('*.json')]:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:shutil.copyfile(source,target)
    module=types.ModuleType('pbp1_private_scene_definitions');module.__file__=str(Path(__file__).resolve());sys.modules[module.__name__]=module
    ns=module.__dict__;ns.update(np=np,math=math,json=json,Path=Path,HERE=work,read=read,dump=dump,digest=digest,lru_cache=functools.lru_cache,dataclass=dataclasses.dataclass,asdict=dataclasses.asdict,point_segment_distance=point_segment_distance,segment_distance=segment_distance)
    definitions(BACKEND/'legacy_states.py',{'LiteratureConfig'},ns)
    definitions(BACKEND/'swap_geometry.py',{'Geometry'},ns);ns['OriginalGeometry']=ns['Geometry']
    definitions(BACKEND/'swap_geometry_user_fix_20261023.py',{'layout','static_walls','GeometryRevision'},ns)
    ns.update(Config=Config,Geometry=Geometry,RN=types.SimpleNamespace(LiteratureConfig=ns['LiteratureConfig']))
    definitions(BASE/'generated/scene_environment.py',{'bottleneck_geometry'},ns)
    definitions(BASE/'routes.py',{'SegmentRouteGeometry','route_geometry'},ns)
    definitions(BASE/'scene_helpers.py',{'forest_layout'},ns)
    helper=types.ModuleType('scene_helpers');helper.__file__=str(BASE/'scene_helpers.py');helper.forest_layout=ns['forest_layout'];sys.modules['scene_helpers']=helper
    return ns

def sampler(source,ns,count):
    ns['STATE_COUNT']=count
    definitions(source,{'SEED','TITLES','LADDERS','rect_complement','bcgeom','separate','sample','build'},ns,extend=True)
    return ns['build']

def compare_state(actual,expected):
    import numpy as np
    fields={name:np.asarray(actual[name],np.float64).tobytes()==np.asarray(expected[name],np.float64).tobytes()for name in ('positions','velocities','goals')}
    fields['metadata']=digest({k:v for k,v in actual.items()if k not in fields})==digest({k:v for k,v in expected.items()if k not in fields})
    return fields

def normalized(states,scene):
    # Literal rename performed by the archived v3 wrapper for these four types.
    if scene['type_id']in(2,7,16,17):
        for state in states:state['scene']=scene['states'][0]['scene']
    return states

def main():
    assert(HERE/'PRIOR_TASK_DELIVERED.json').exists(),'Numerical preparation remains queued until the previous study is delivered.'
    assert not(HERE/'PARAMETERS_FROZEN.json').exists(),'State splits must precede parameter selection.'
    original=read(BASE/'SCENES.json');representatives=read(BASE/'REPRESENTATIVE17_FROZEN.json')['selected'];selected={r['scene']for r in representatives}
    ns=namespace();checks=[];new={};error=None
    try:
        for entry in original['entries']:
            scene=entry['scene'];source=BASE/'generated/scenes_v3_impl.py'if scene['type_id']in(2,7,16,17)else V1/'scenes.py'
            regenerated=sampler(source,ns,12)(scene['type_id'],scene['level'],scene['widen'])
            old=normalized(regenerated['states'],scene)
            fields=[compare_state(a,b)for a,b in zip(old,scene['states'])]
            passed=len(old)==12 and all(all(f.values())for f in fields)
            checks.append(dict(type_id=scene['type_id'],level=scene['level'],source=str(source.relative_to(HERE)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),state_fields=fields,passed=passed))
            assert passed, ('Cannot reproduce frozen prefix',scene['type_id'],scene['level'],fields)
            if scene['name']in selected:
                extended=sampler(source,ns,20)(scene['type_id'],scene['level'],scene['widen'])
                values=normalized(extended['states'],scene)
                prefix=[compare_state(a,b)for a,b in zip(values[:12],scene['states'])]
                assert all(all(f.values())for f in prefix),('Extended RNG prefix changed',scene['name'])
                new[scene['name']]=values[12:20]
            print('Geometry sampler prefix verified',scene['type_id'],scene['level'],flush=True)
    except Exception as exc:
        error=repr(exc);new={}
    fresh=error is None and len(new)==17
    indices=list(range(12,20))if fresh else list(range(8,12))
    chosen=[]
    for entry in original['entries']:
        scene=entry['scene']
        if scene['name']not in selected:continue
        states=new[scene['name']]if fresh else[scene['states'][i]for i in indices]
        chosen.append(dict(type_id=scene['type_id'],level=scene['level'],scene=scene['name'],states=states))
    output=dict(time=datetime.datetime.now().astimezone().isoformat(),fresh=fresh,test_state_indices=indices,physical_states=17*len(indices),prefix_checks=checks,error=error,scenes=chosen,new_rollouts=0,
        generation_edit='Only range(12) becomes range(STATE_COUNT), first12 replayed before extending to20. Four modified types retain literal v3 scene aliases.',
        frozen_geometry_and_horizon_unchanged=True,
        limitation=None if fresh else'测试状态曾被固定参数评估使用过（未用于任何选择）')
    dump(HERE/'NEW_STATES.json',output)
    print('State preparation:', 'new12..19'if fresh else'protocol fallback8..11', 'physical states',output['physical_states'], 'error',error)

if __name__=='__main__':main()
