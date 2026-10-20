"""Single-robot navigation demonstrations on training geometry only."""
from common import *
from geometry import GeneratedGeometry,task
from bottleneck_family.scenario import Config
from design_generated import identities
from registry import register
import math

def prepare():
    cfg=frozen_config();catalog=read(HERE/'CATALOG_FROZEN.json');rows=[];registered=[];rejected=[]
    for family in catalog['families']:
        sources=[s for s in catalog['catalog'][family] if s['role']=='train']
        assert len(sources)==16
        for index,s in enumerate(sources):
            geo=GeneratedGeometry(Config(**s['config']),s['rectangles'])
            for split,count in [('train',32),('dev',8)]:
                for j in range(count):
                    base=s['geometry_seed']*10000+(0 if split=='train' else 5000)+j*100
                    for retry in range(100):
                        try:p,g,paths,_=task(geo,family,base+retry);break
                        except ValueError as e:rejected.append(dict(family=family,map=index,split=split,seed=base+retry,reason=str(e)))
                    else:continue
                    i=j%4;length=np.linalg.norm(np.diff(paths[i],axis=0),axis=1).sum();H=int(math.ceil(2.5*length/.5/.05/100)*100)
                    state=dict(s,role='demonstration_'+split,index=index*count+j,alias=f'round39_demo_{family}_{split}_{index}_{j}',
                        positions=p[i:i+1].tolist(),goals=g[i:i+1].tolist(),velocities=np.zeros((1,2)).tolist(),paths=[paths[i].tolist()])
                    state['config']=dict(s['config'],num_agents=1,max_steps=H)
                    state['H']=H;state['split']=split;state['source_training_geometry_seed']=s['geometry_seed']
                    c,r=identities(state,'navigation','navigation');rows.append(c);registered+=r
    write(HERE/'DEMONSTRATION_REJECTIONS.json',rejected)
    register('demonstrations',rows,registered)
    seal('DEMONSTRATION_IMPLEMENTATION.receipt.json',[HERE/'demonstrations.py',HERE/'observation.py',HERE/'single_agent.py',HERE/'GEOMETRY_EQUIVALENCE.json',HERE/'CATALOG_FROZEN.json'],single_robot_only=True,no_evaluation_geometry=True,no_sequential_expert_demonstrations=True)

def freeze():
    catalog=read(HERE/'CATALOG_FROZEN.json');cases=read(HERE/'demonstrations_CASES.json');families=[];summary={};folder=HERE/'datasets';folder.mkdir(exist_ok=True)
    for family in catalog['families']:
        arrays={k:[] for k in ['train_X','train_Y','dev_X','dev_Y']};provenance=[];counts={}
        for c in cases:
            s=c['state']
            if s['family']!=family:continue
            raw=read(HERE/'demonstrations/raw'/f"{c['rollout_uid']}.json");counts[raw['outcome']]=counts.get(raw['outcome'],0)+1
            provenance.append(dict(rollout_uid=c['rollout_uid'],raw_path=str(HERE/'demonstrations/raw'/f"{c['rollout_uid']}.json"),source_geometry_seed=s['geometry_seed'],split=s['split'],success=raw['success'],trace_sha256=raw['trace_sha256']))
            if not raw['success']:continue
            assert s['geometry_seed'] in {q['geometry_seed'] for q in catalog['catalog'][family] if q['role']=='train'}
            z=np.load(raw['trace_path']);assert sha(raw['trace_path'])==raw['trace_sha256'];observed=z['observations'].reshape(-1,35);actions=z['actions'].reshape(-1,2);assert len(observed)==len(actions)
            arrays[s['split']+'_X'].append(observed);arrays[s['split']+'_Y'].append(actions)
        good=all(arrays[k] for k in arrays)
        if good:
            arrays={k:np.concatenate(v).astype(np.float32) for k,v in arrays.items()};np.savez_compressed(folder/f'{family}.npz',**arrays);families.append(family)
        summary[family]=dict(outcomes=counts,usable=good,transitions={k:len(v) for k,v in arrays.items()} if good else {},source_geometry_count=16)
        write(folder/f'{family}_PROVENANCE.json',provenance)
    write(HERE/'DATASET_FROZEN.json',dict(families=families,summary=summary,selection='All successful single-robot navigation episodes; failed or unknown episodes retained in DB and provenance, never replaced',G_correction_labels_used=False,G_evaluation_geometry_used=False))
    seal('DATASET_FROZEN.receipt.json',[HERE/'DATASET_FROZEN.json']+list(folder.glob('*')),before_training=True)

if __name__=='__main__':globals()[sys.argv[1]]()
