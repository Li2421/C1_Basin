"""Deterministic state proposals, then append-only sequential feasibility proofs."""
from common import *
from dataclasses import asdict
from geometry import make_geometry,task,GeneratedGeometry
from design_generated import identities
from registry import register
from bottleneck_family.scenario import Config

def sharing(paths, resources, family):
    memberships=[]
    for k,(x,y,w) in enumerate(resources):
        robots=[]
        horizontal=family=='F' and k>=2
        axis=1 if horizontal else 0;level=y if horizontal else x;center=x if horizontal else y
        for i,path in enumerate(paths):
            for a,b in zip(path[:-1],path[1:]):
                if (a[axis]-level)*(b[axis]-level)>0 or a[axis]==b[axis]:continue
                fraction=(level-a[axis])/(b[axis]-a[axis]);q=a+fraction*(b-a)
                if abs(q[1-axis]-center)<w/2:robots.append(i);break
        memberships.append(dict(resource=k,robots=robots))
    return any(len(r['robots'])>=2 for r in memberships),memberships

def proposal(family,role,index,geo_seed,task_seed):
    geo,resources=make_geometry(family,int(geo_seed))
    p,g,paths,H=task(geo,family,int(task_seed));shared,witness=sharing(paths,resources,family)
    if family=='G':
        # Random obstacles have no named doors; shared route intersections define junction resources.
        from single_integrator.environment import segment_distance
        shared=any(segment_distance(a[:-1,None],a[1:,None],b[None,:-1],b[None,1:]).min()<.32 for i,a in enumerate(paths) for b in paths[i+1:])
        witness=dict(shared_path_intersection=bool(shared),threshold=.32)
    if not shared:raise ValueError('no shared resource competition')
    return dict(family=family,role=role,index=index,geometry_seed=int(geo_seed),task_seed=int(task_seed),
        alias=f'round39_{family}_{role}_{index}_{geo_seed}_{task_seed}',config=asdict(geo.config),
        rectangles=geo.rectangles.tolist(),positions=p.tolist(),goals=g.tolist(),velocities=np.zeros((4,2)).tolist(),
        paths=[x.tolist() for x in paths],H=H,resource_witness=witness,
        geometry_valid=True,single_robot_paths_exist=True,joint_feasibility='pending')

def prepare():
    cfg=frozen_config();assert read(HERE/'GEOMETRY_EQUIVALENCE.json')['passed'];catalog={};rejections=[]
    # Content dedup uses every recoverable physical triple, without accessing source directories.
    from shared_rollout_db.src.rollout_db import connect,canonical
    signatures=set();unrecoverable=0
    with connect(True) as db:
        for row in db.execute('SELECT physical_state_json FROM state'):
            try:
                v=json.loads(row[0]);trip={k:v[k] for k in ['positions','goals','velocities']}
                signatures.add(canonical(trip))
            except Exception:unrecoverable+=1
    for ordinal,family in enumerate('ABCDEFG'):
        specs=[];geo_cache={}
        for role,count in [('train',16),('evaluation',32),('pilot',16),('probe',16)]:
            for index in range(count):
                map_index=index if role=='train' else index//4 if role=='evaluation' else (index//4 if role=='pilot' else index//4+4)
                key=('train' if role=='train' else 'evaluation',map_index)
                if key not in geo_cache:
                    base=cfg['geometry_seeds'][key[0]]+ordinal*cfg['family_seed_offset']+map_index*100
                    for retry in range(cfg['reject_next_seed_limit']):
                        gs=base+retry
                        try:proposal(family,role,index,gs,gs*100+index);geo_cache[key]=gs;break
                        except ValueError as e:rejections.append(dict(family=family,role=role,geometry_seed=gs,reason=str(e)))
                    else:raise RuntimeError('geometry proposal seed range exhausted '+family)
                gs=geo_cache[key];ts=cfg['geometry_seeds'][role]+ordinal*cfg['family_seed_offset']+index*100000
                for retry in range(cfg['reject_next_seed_limit']):
                    try:
                        state=proposal(family,role,index,gs,ts+retry)
                        signature=canonical({k:state[k] for k in ['positions','goals','velocities']})
                        if signature in signatures:raise ValueError('physical state already present')
                        signatures.add(signature);specs.append(state);break
                    except ValueError as e:rejections.append(dict(family=family,role=role,index=index,task_seed=ts+retry,reason=str(e)))
                else:raise RuntimeError('state proposal seed range exhausted '+family)
        catalog[family]=specs
    write(HERE/'CANDIDATE_CATALOG_V2.json',catalog);write(HERE/'GEOMETRY_REJECTIONS_V2.json',rejections)
    write(HERE/'STATE_DEDUP_AUDIT.json',dict(recoverable_signatures=len(signatures),unrecoverable=unrecoverable,
        method='All DB physical JSON triples, no source-directory access; generated candidates checked against history and one another'))
    rows=[];registered=[]
    for specs in catalog.values():
        for s in specs:
            c,r=identities(s,'sequential_expert','expert');rows.append(c);registered+=r
    register('joint_initial_v2',rows,registered)
    seal('GENERATOR_INPUTS_FROZEN_V2.receipt.json',[HERE/'CANDIDATE_CATALOG_V2.json',HERE/'catalog.py',HERE/'geometry.py',HERE/'sequential_expert.py',HERE/'SEQUENTIAL_EXPERT_PROVENANCE.json'],before_joint_proof=True)

def accept(stage):
    catalog=read(HERE/'VALIDATED_CATALOG.json') if (HERE/'VALIDATED_CATALOG.json').exists() else read(HERE/'CANDIDATE_CATALOG_V2.json');cases=read(HERE/f'{stage}_CASES.json');failed=[]
    for c in cases:
        raw=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");s=c['state'];row=next(v for v in catalog[s['family']] if v['alias']==s['alias'])
        row['joint_proof']=dict(raw_path=str(HERE/stage/'raw'/f"{c['rollout_uid']}.json"),trace_sha256=raw['trace_sha256'],success=bool(raw['success']),outcome=raw['outcome'])
        if raw['success']:row['joint_feasibility']='certified'
        else:failed.append(row)
    write(HERE/'VALIDATED_CATALOG.json',catalog)
    return failed

def retry(number,failed):
    cfg=frozen_config();catalog=read(HERE/'VALIDATED_CATALOG.json');rows=[];registered=[];exhausted=[]
    groups={(s['family'],s['geometry_seed']) for s in failed}
    for family_name,old_seed in sorted(groups):
        group=[s for s in catalog[family_name] if s['geometry_seed']==old_seed]
        old=group[0]
        attempts=int(old.get('joint_rejection_count',0))+1
        if attempts>=cfg['reject_next_seed_limit']:
            exhausted.append(dict(family=old['family'],role=old['role'],index=old['index'],reason='preregistered joint-validity seed limit exhausted'));continue
        gs=old_seed+1
        for _ in range(cfg['reject_next_seed_limit']):
            try:
                newgroup=[proposal(q['family'],q['role'],q['index'],gs,q['task_seed']+1) for q in group]
                break
            except ValueError:gs+=1
        else:
            exhausted.append(dict(family=old['family'],role=old['role'],index=old['index'],reason='preregistered next-state seed limit exhausted'));continue
        family=catalog[family_name]
        for old,new in zip(group,newgroup):
            new['joint_rejection_count']=attempts
            i=next(i for i,v in enumerate(family) if v['alias']==old['alias']);family[i]=new
            c,r=identities(new,'sequential_expert','expert');rows.append(c);registered+=r
    write(HERE/'VALIDATED_CATALOG.json',catalog)
    write(HERE/f'joint_{number}_EXHAUSTED.json',exhausted)
    if rows:register(f'joint_{number}',rows,registered)
    return rows

def finalize():
    catalog=read(HERE/'VALIDATED_CATALOG.json');eligible=[];gaps=[]
    for family,states in catalog.items():
        valid=all(s['joint_feasibility']=='certified' for s in states)
        if valid:eligible.append(family)
        else:gaps.append(dict(family=family,reason='Some registered states lack joint-solvability proof'))
    write(HERE/'CATALOG_FROZEN.json',dict(families=eligible,gaps=gaps,catalog=catalog))
    seal('CATALOG_FROZEN.receipt.json',[HERE/'CATALOG_FROZEN.json'],geometry_and_joint_proof=True)
    folder=HERE/'geometry_previews';folder.mkdir(exist_ok=True)
    for family,states in catalog.items():
        seen=set()
        for s in states:
            if s['geometry_seed'] in seen:continue
            seen.add(s['geometry_seed']);parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">','<rect width="800" height="400" fill="white" stroke="black"/>']
            for x0,y0,x1,y1 in s['rectangles']:
                parts.append(f'<rect x="{(x0+8)*50}" y="{(4-y1)*50}" width="{(x1-x0)*50}" height="{(y1-y0)*50}" fill="#555"/>')
            for i,(p,g,path) in enumerate(zip(s['positions'],s['goals'],s['paths'])):
                color=['#3465a4','#cc0000','#4e9a06','#75507b'][i]
                points=' '.join(f'{(x+8)*50},{(4-y)*50}' for x,y in path)
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" opacity="0.5"/>')
                parts.append(f'<circle cx="{(p[0]+8)*50}" cy="{(4-p[1])*50}" r="8" fill="{color}"/>')
                parts.append(f'<circle cx="{(g[0]+8)*50}" cy="{(4-g[1])*50}" r="8" fill="none" stroke="{color}"/>')
            parts.append('</svg>');(folder/f"{family}_{s['geometry_seed']}.svg").write_text('\n'.join(parts))

if __name__=='__main__':prepare()
