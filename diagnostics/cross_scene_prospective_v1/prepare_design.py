"""Outcome-free prescribed scenes, exact historical learner, and hard freeze gates."""
from common import *
from scenes import features,geometry_report,FEATURE_NAMES
from models import select
from bottleneck_family.scenario import Config,Geometry,build_instance
from bottleneck_family.__main__ import preset
from legacy_states import LiteratureConfig
from dataclasses import replace
import numpy as np

OLD=ROOT/'diagnostics/cross_scene_correction_v1'
EXPECTED=['lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_swap8_obstacles','xsc_swap_n4',
    'xsc_swap_n12','xsc_swap_n8_dense','xsc_gap_n4','xsc_gap_n6','xsc_double_n4','xsc_offset_n4']
NAMES={'xsp_long_tunnel_n4':'长隧道','xsp_clutter_n4':'杂物','xsp_wall_side_door_n4':'贴墙门',
    'xsp_zigzag_three_n4':'三墙之字','xsp_chicane_n4':'小弯道','xsp_narrow_door_n4':'窄门',
    'xsp_low_room_n6':'矮房间','xsp_chamber_pillars_n6':'双墙夹柱','xsp_wide_wall_door_n6':'贴墙宽门',
    'xsp_unequal_doors_n6':'不等宽三门','xsp_offset_tunnel_n6':'偏心隧道','lit_crowded_random_n14':'拥挤随机14机'}

def stop(reason,details):
    write(HERE/'STOP.receipt.json',dict(utc=utc(),reason=reason,details=details,new_test_rollouts=0,
        frozen=False,required_by='PROTOCOL.md sections 0,3,4'))
    raise RuntimeError(reason)

def rank(v):
    order=np.argsort(v,kind='stable');r=np.empty(len(v));r[order]=np.arange(len(v))
    for val in np.unique(v):r[v==val]=r[v==val].mean()
    return r

def training():
    old=[s for s in read_evidence(OLD/'SCENES.json')if s['family']!='extrapolation']
    labels=read_evidence(OLD/'LABEL_TABLES.json');names=[s['name']for s in old]
    q=np.array([np.asarray(labels[n]['y'],float).mean((0,1))for n in names])
    for i,n in enumerate(names):assert np.array_equal(q[i],labels[n]['type_equal_candidate_success_lower'])
    f=np.array([s['features']for s in old]);rho=np.corrcoef(np.array([rank(v)for v in q]))
    regrets=q.max(1)[:,None]-q;J=np.array([[np.maximum(a,b).min()for b in regrets]for a in regrets])
    keep=set(range(len(names)));dropped=[]
    while True:
        pairs=[(rho[i,j],i,j)for i in keep for j in keep if i<j and rho[i,j]>=.9 and J[i,j]<=.02]
        if not pairs:break
        _,i,j=max(pairs);mi=np.mean([rho[i,k]for k in keep if k!=i]);mj=np.mean([rho[j,k]for k in keep if k!=j])
        d=i if mi>=mj else j;o=j if d==i else i;keep.remove(d)
        dropped.append(dict(dropped=names[d],paired_with=names[o],rho=float(rho[i,j]),shared_regret=float(J[i,j]),means=[float(mi),float(mj)]))
    idx=sorted(keep);kept=[names[i]for i in idx]
    write(HERE/'DEDUPLICATION.json',dict(names=names,kept=kept,dropped=dropped,rho=rho.tolist(),shared_regret=J.tolist(),q=q.tolist(),
        thresholds=dict(spearman=.9,shared_regret=.02),new_rollouts=0))
    if kept!=EXPECTED:stop('DEDUPLICATED_SCENES_MISMATCH',dict(expected=EXPECTED,actual=kept))
    return old,f,q,idx

def configs():
    specs=[('xsp_long_tunnel_n4',4,dict(barrier_thickness=3.)),
        ('xsp_clutter_n4',4,{}),('xsp_wall_side_door_n4',4,dict(openings=(((3.,.5),),))),
        ('xsp_zigzag_three_n4',4,dict(barrier_x=(-3.,0.,3.),openings=(((1.,.5),),((-1.,.5),),((1.,.5),)))),
        ('xsp_chicane_n4',4,dict(barrier_x=(-.6,.6),openings=(((-.5,.5),),((.5,.5),)))),
        ('xsp_narrow_door_n4',4,dict(openings=(((0.,.4),),))),
        ('xsp_low_room_n6',6,dict(half_height=2.)),
        ('xsp_chamber_pillars_n6',6,dict(barrier_x=(-1.5,1.5),openings=(((0.,.5),),((0.,.5),)),extra_rectangles=((-.4,.6,.4,1.4),(-.4,-1.4,.4,-.6)))),
        ('xsp_wide_wall_door_n6',6,dict(openings=(((2.8,1.2),),))),
        ('xsp_unequal_doors_n6',6,dict(openings=(((-2.,.4),(0.,.6),(2.,.9)),))),
        ('xsp_offset_tunnel_n6',6,dict(barrier_thickness=2.,openings=(((-2.,.5),),)))]
    for name,n,changes in specs:
        base=preset('clutter',n)if name=='xsp_clutter_n4'else Config(num_agents=n)
        yield name,replace(base,max_steps=max(2000,800*n),split='train',**changes)

def prepare_scenes(old):
    scenes=[];excluded=[]
    for name,base in configs():
        geom=Geometry(base);states=[];invalid=[]
        for k in range(16):
            seed=2026102600+k
            try:
                inst=build_instance(replace(base,seed=seed))
                states.append(dict(scene=name,index=k,N=base.num_agents,positions=inst.positions.tolist(),
                    goals=inst.goals.tolist(),velocities=np.zeros_like(inst.positions).tolist(),bottleneck_config=base.to_dict(),
                    generation_seed=seed,physical_source='bottleneck_family.scenario.build_instance'))
            except ValueError as e:invalid.append(dict(index=k,seed=seed,error=str(e),replacement=False))
        cfg=LiteratureConfig(name,base.num_agents,base.agent_radius,base.dt,base.max_speed,base.max_steps,
            base.goal_tolerance,base.wall_radius,base.wall_collision_margin,base.agent_collision_margin).to_dict()
        area=4*base.half_length*base.half_height-sum((x1-x0)*(y1-y0)for x0,y0,x1,y1 in geom.rectangles)
        scene=dict(name=name,full_name=NAMES[name],family='bottleneck',factor='geometry',config=cfg,walls=geom.walls.tolist(),states=states,
            bottleneck_config=base.to_dict(),sampler_seed=2026102600,free_area=float(area),
            min_opening_over_diameter=min(w for openings in base.openings for _,w in openings)/(2*base.agent_radius),
            serial_walls=len(base.barrier_x),diametric=False,moving=False)
        report=geometry_report(scene);report['invalid_generation']=invalid
        scene['eligible']=not invalid and all(r['eligible']for r in report['states'])
        if not scene['eligible']:excluded.append(name)
        report['joint_solvable']='not_certified';report['controller_success']='not_evaluated';report['deadlock_resolution']='not_established'
        write(HERE/'geometry'/f'{name}.json',report)
        scene['definition_sha256']=uid('scene_definition',scene);scene['features']=features(scene)
        scenes.append(scene);print('Geometry',name,'eligible',scene['eligible'],flush=True)
    source=V2/'LEARNING_SPLIT_REVISED_20261021_lit_crowded_random.json';split=read_evidence(source)
    states=[copy.deepcopy(s)for s in split['training']if s['N']==14][:16];assert len(states)==16
    template=next(s for s in old if s['name']=='lit_crowded_random')
    scene=dict(name='lit_crowded_random_n14',full_name=NAMES['lit_crowded_random_n14'],family='crowded',factor='density',
        config=dict(template['config'],num_agents=14),walls=copy.deepcopy(template['walls']),states=states,
        free_area=template['free_area'],min_opening_over_diameter=template['min_opening_over_diameter'],
        serial_walls=template['serial_walls'],diametric=False,moving=False,eligible=True,physical_source=str(source))
    assert scene['config']['max_steps']==333
    scene['definition_sha256']=uid('scene_definition',scene);scene['features']=features(scene);scenes.append(scene)
    write(HERE/'SCENES.json',scenes);write(HERE/'FEATURES.json',dict(names=FEATURE_NAMES,scenes={s['name']:s['features']for s in scenes}))
    write(HERE/'EXCLUDED_LAYOUTS.json',dict(layouts=excluded,replacement=False))
    lines=['# 新布局五层证据\n\n','| 场景 | 几何有效与单机器人有路 | 联合可解 | 控制器成功 | 死锁解决 |\n|---|---|---|---|---|\n']
    for s in scenes[:-1]:lines.append(f"| {s['full_name']}（{s['name']}） | {'16/16'if s['eligible']else '不合法，整体排除'} | 未认证 | 未评估 | 未建立 |\n")
    lines.append('\n几何仅提供保守单机器人路径见证。未缩小机器人或更改物理参数；timeout 不标为 deadlock。\n')
    (HERE/'FIVE_LAYER.md').write_text(''.join(lines))
    return scenes

def main():
    assert read(HERE/'REPLAY_CHECK.json')['passed']
    assert not(HERE/'MODEL_FROZEN.receipt.json').exists()
    old,f,q,idx=training();scenes=prepare_scenes(old);etas=np.array([a['eta_a']for a in actions()]);keys=[a['key']for a in actions()]
    choices={};details={};mismatches=[]
    for s in scenes:
        reduced=select(f[idx],q[idx],s['features'],etas);full=select(f,q,s['features'],etas)
        for label,res,k in [('dedup',reduced,2),('full',full,3)]:
            if res['k']!=k or res['penalty']!=100 or keys[res['global_fixed']]!='sE29':
                mismatches.append(dict(scene=s['name'],pool=label,parameters=dict(k=res['k'],penalty=res['penalty'],global_fixed=keys[res['global_fixed']])))
        choices[s['name']]={**{m+'_'+pool:keys[r[m]]for pool,r in [('dedup',reduced),('full',full)]for m in ('knn','ridge')},
            'global_fixed':'sE29','eta13':'eta13','safety':'safety'}
        details[s['name']]=dict(dedup=reduced,full=full)
        expected=dict(knn_dedup='sE01'if s['family']=='crowded'else 'sE05',
            knn_full='sE01'if s['family']=='crowded'else 'sE22'if s['name']=='xsp_offset_tunnel_n6'else 'sE05',
            ridge_dedup='sE19'if s['family']=='crowded'else 'sE13',ridge_full='sE19'if s['family']=='crowded'else 'sE13')
        for m,a in expected.items():
            if choices[s['name']][m]!=a:mismatches.append(dict(scene=s['name'],method=m,expected=a,actual=choices[s['name']][m]))
        print('Choices',s['name'],choices[s['name']],flush=True)
    write(HERE/'FROZEN_CHOICES.json',choices);write(HERE/'MODEL_DETAILS.json',details)
    write(HERE/'DESIGN_EXPECTATION_CHECK.json',dict(passed=not mismatches,mismatches=mismatches,new_test_rollouts=0))
    if mismatches:stop('FROZEN_SELECTION_EXPECTATION_MISMATCH',mismatches)
    paths=[HERE/n for n in ('PROTOCOL.md','prepare_design.py','models.py','scenes.py','analysis_stats.py','SCENES.json','FEATURES.json',
        'DEDUPLICATION.json','FROZEN_CHOICES.json','MODEL_DETAILS.json','DESIGN_EXPECTATION_CHECK.json')]
    seal('MODEL_FROZEN.receipt.json',paths,training_rollouts_added=0,test_rollouts_before_freeze=0,
        training_label_source=str(OLD/'LABEL_TABLES.json'),training_label_sha256=sha(OLD/'LABEL_TABLES.json'),
        methods=list(choices[scenes[0]['name']]),statistical_seed=2026102513,replicates=5000,selection_changes_after_freeze_forbidden=True)
    print('FROZEN',flush=True)

if __name__=='__main__':main()
