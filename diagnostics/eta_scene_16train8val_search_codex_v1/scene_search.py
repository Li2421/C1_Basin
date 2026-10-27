"""Independent parameter candidates, discovery0-3 and locked confirmation8-23."""
from common import *
from design import controllers,menu,CONTROL_IDS
from scenes import construct,features,make_new,geometry_report
from legacy_states import LiteratureConfig
import numpy as np,math
SEED=2026102707
NS=24

def build():
 out=[]
 # Different from the parallel search: fixed revised V obstacles, radius/time/phase variants.
 static_params=[(1.28,.045,4.,-.035),(1.32,.055,4.,.035),(1.36,.045,3.6,-.025),(1.44,.055,4.4,.025),(1.48,.045,4.,-.045),(1.52,.055,3.6,.045),(1.38,.06,4.4,-.02),(1.46,.04,4.,.02),(1.3,.05,4.4,.06),(1.5,.05,4.4,-.06)]
 walls=read(HERE/'LAYOUT_REVISION_DESIGN.json')['walls']
 for k,(radius,r,f,phase)in enumerate(static_params):
  name=f'xsc_cdx_obst_{k:02d}';n=8;th=np.arange(n)*2*np.pi/n+phase;initial=radius*np.stack([np.cos(th),np.sin(th)],-1);goals=-initial;rng=np.random.default_rng(np.random.SeedSequence([SEED,1,k]));states=[]
  for i in range(NS):
   p=initial+rng.uniform(-.08,.08,(n,2));states.append(dict(scene='lit_swap8_obstacles',design_name=name,index=i,N=n,positions=p.tolist(),goals=goals.tolist(),velocities=np.zeros_like(p).tolist()))
  cfg=LiteratureConfig('lit_swap8_obstacles',n,r,.05,.5,int(math.ceil(f*(2*radius+math.sqrt(2)*.08)/.5/.05))).to_dict()
  s=dict(name=name,title=name,family='revised_static_V_exchange',group='obstacle',factor=f,config=cfg,walls=walls,states=states,sampler_seed=[SEED,1,k],free_area=None,min_opening_over_diameter=2.,serial_walls=0,diametric=True,moving=False,specification=dict(radius=radius,r=r,f=f,phase=phase,static_layout='static_straight_corridors_20261023'))
  s['definition_sha256']=uid('scene_definition',s);s['features']=features(s);out.append(s)
 # Parameterized exact old constructor in a private copy: only seed/count differ.
 for k,w in enumerate([.21,.24,.27,.31,.37,.43,.49,.58]):
  s=construct(dict(name=f'xsc_cdx_plus_{k:02d}',group='E',N=4,r=.05,f=4.,v=.5,w=w,seed=SEED,state_count=NS));s['group']='plus';s['family']='plus_crossing';out.append(s)
 for k,(n,r,f)in enumerate([(8,.14,2.7),(9,.145,2.6),(10,.125,2.65),(10,.135,2.85),(11,.12,2.7),(9,.155,2.75)]):
  s=construct(dict(name=f'xsc_cdx_box_{k:02d}',group='C',N=n,r=r,f=f,v=.5,seed=SEED,state_count=NS));s['group']='box';s['family']='square_random';out.append(s)
 for k,(n,r,f)in enumerate([(6,.055,4.),(8,.045,4.2),(10,.055,4.2),(12,.045,3.8)]):
  s=construct(dict(name=f'xsc_cdx_swap_{k:02d}',group='A',N=n,r=r,f=f,v=.5,seed=SEED,state_count=NS));s['group']='swap';s['family']='open_exchange';out.append(s)
 # Preserve the original circle/static case construction exactly when rerunning literature instances.
 old=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/main_CASES.json');templates={}
 for c in old:
  key=c['controller_payload']['nominal']['controller']
  if c['state']['scene']=='lit_swap8_obstacles'and key in CONTROL_IDS and key not in templates:templates[key]=c
 write(HERE/'STATIC_REQUEST_TEMPLATES.json',templates)
 checks=[]
 for ctl,c in templates.items():
  rebuilt=make_case(c['state'],c['version'],c['method'],c['scenario_payload'],{k:v for k,v in c['controller_payload'].items()if k not in ['method','eta_b','theta']},c['seed'],c['eta_a'],c['state_provenance'])
  for key in ['state_uid','scenario_uid','controller_uid','eta_uid','seed_key','rollout_uid']:assert rebuilt[key]==c[key],(ctl,key)
  checks.append(dict(controller=ctl,original_uid=c['rollout_uid'],all_identity_fields_match=True))
 write(HERE/'REQUEST_RECONSTRUCTION.json',dict(passed=True,checks=checks))
 write(HERE/'SCENE_LIBRARY.json',out);write(HERE/'SCENES.json',out)
 reports={}
 from single_integrator.environment import point_segment_distance
 for s in out:
  report=geometry_report(s)
  if s['group']=='obstacle':
   W=np.asarray(s['walls']);r=s['config']['agent_radius'];margin=r+.005
   # Each task follows its straight radial corridor in the revised V layout.
   from single_integrator.environment import segment_distance
   for st,row in zip(s['states'],report['states']):
    p=np.asarray(st['positions']);g=np.asarray(st['goals']);valid=bool(np.all(point_segment_distance(p[:,None],W[:,0],W[:,1])>margin)and np.all(point_segment_distance(g[:,None],W[:,0],W[:,1])>margin))
    direct=np.min(segment_distance(p[:,None],g[:,None],W[None,:,0],W[None,:,1]),axis=-1)>margin
    row['geometric_valid']=row['geometric_valid']and valid;row['single_robot_paths']=bool(direct.all());row['eligible']=bool(row['geometric_valid']and direct.all());row['path_witnesses']=[np.stack([a,b]).tolist()if ok else None for a,b,ok in zip(p,g,direct)]
  reports[s['name']]=report
 write(HERE/'GEOMETRY_REPORTS.json',reports)
 print('Prepared',len(out),'independent candidates; eligibility',[(s['name'],sum(r['eligible']for r in reports[s['name']]['states']))for s in out],flush=True)

def case(scene,ctl,state,action,seed=0,original_identity=False):
 if scene['group']!='obstacle':return make_new(scene,ctl,state,action,seed)
 template=read(HERE/'STATIC_REQUEST_TEMPLATES.json')[ctl['controller']];scenario=copy.deepcopy(template['scenario_payload']);scenario['config']=dict(scene['config'],num_agents=state['N']);scenario['search_definition_sha256']=scene['definition_sha256']
 payload=copy.deepcopy(template['controller_payload'])
 for k in ['method','eta_b','theta']:payload.pop(k,None)
 # Explicit new source identity; preserve every historical revision and geometry field.
 payload['execution_sources']={**payload['execution_sources'],**{str(HERE/p):sha(HERE/p)for p in ['backend/swap_safety.py','fast_projection.py','accel_mode.py','backend/swap_runtime_user_fix_20261023.py']}}
 c=make_case(state,template['version'],action['key'],scenario,payload,seed,action['eta_a'],dict(scene=scene['name'],index=state['index'],sampler_seed=scene['sampler_seed']))
 c['scene_revision']=template['scene_revision'];return c

if __name__=='__main__':build()
