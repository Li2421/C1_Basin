"""Build a design handoff only. No scene constructors, simulation, training or DB writes."""
from pathlib import Path
import json,hashlib,datetime,math,copy
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
H=Path(__file__).resolve().parent;PRE=H.parent;R=PRE.parents[1];sources={}
def read(p):
 p=Path(p);b=p.read_bytes();sources[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def write(n,d):(H/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
old=read(PRE/'PROPOSED_16_TRAIN_8_VALIDATION.json');menu=read(R/'diagnostics/critic_pool_supplement_v1/ETA_MENU.json');all_profiles=read(PRE/'EXISTING_PATTERN_ANALYSIS.json');size=read(R/'diagnostics/eta_diverse_scene_test_v1/NEW_LABEL_TABLES.json')['xsc_div_size_r014'];pool=read(R/'diagnostics/critic_pool_supplement_v1/SCENES.json');basebox=next(s for s in pool if s['name']=='xsc_tr_box_n08_r015_f25')
F=[
 dict(id='F1',name='V形障碍直走廊',family='obstacle',priority=1,evidence='Existing49-menu revisedV anchor; local variants untested',need='Suppress lateral velocity while retaining contact response',anchor='lit_swap8_obstacles',parameter='arm_length',values=[.23,.25,.29,.31,.24,.30],sentinels=[1,2],risk='Wall routing must be identical in physical step, nominal costs and safety rows; scalar geometry feasibility does not establish joint solvability.'),
 dict(id='F2',name='开放九机交汇',family='open',priority=1,evidence='Existing49-menu9-agent anchor,8 states',need='Amplify lateral motion and maintain contact response',anchor='xsc_tr_swap_n09',parameter='initial_ring_radius',values=[1.,1.08,1.16,1.24,1.04,1.20],sentinels=[1,2],risk='Changing radius could flatten the response; do not assume every nearby geometry retains the anchor winner.'),
 dict(id='F3',name='等宽十字路口',family='plus',priority=1,evidence='Several existing49-menu widths; width.32 is a counterexample to universal separation',need='Weak goal correction with moderate lateral/contact response',anchor='xsc_tr_plus_w016',parameter='corridor_halfwidth',values=[.15,.19,.23,.27,.17,.255],sentinels=[0,2],risk='The former proposed validation width.25 already has labels; changed explicitly to.255. Mandatory full-config history check still applies.'),
 dict(id='F4',name='圆形五机密集随机',family='circle',priority=1,evidence='Existing49-menu circle anchor,16 states, low-contact-response region',need='Small contact coefficient with meaningful goal progress',anchor='lit_circle_five',parameter='agent_radius',values=[1.85,1.95,2.05,2.15,1.90,2.10],sentinels=[1,2],risk='Keep original flow runtime and11-radius wall model, never use xsc0.5-sensing safety for these meter-scale robots; longer episodes require measured cost.'),
 dict(id='F5',name='大半径开放六机',family='large_radius_six',priority=1,evidence='Strong25-menu historical result at r.14; missing24 candidates are unknown, not failure',need='Larger positive lateral and goal correction than the small-robot open anchor',anchor='xsc_div_size_r014',parameter='agent_radius',values=[.10,.12,.14,.16,.11,.15],sentinels=[1,2],risk='25-menu evidence cannot prove49-menu separation; preserve original geometry, time budget and exact historical state identities for reuse.'),
 dict(id='F6',name='方形八机适中时间余量',family='box_time',priority=2,evidence='Existing49-menu f2.5 vs f3.0: best.75 vs1.0,10 vs30 candidates within5pp of best',need='Probe goal-strength/contact tradeoff without inducing all-failure or near-universal success',anchor='xsc_tr_box_n08_r015_f25',parameter='time_factor',values=[2.20,2.35,2.50,2.65,2.425,2.575],sentinels=[1,2],risk='Floor/ceiling risk; do not shorten an already launched episode. Horizon is a frozen physical task parameter.'),
 dict(id='F7',name='异宽十字路口',family='anisotropic_plus',priority=3,evidence='Mechanistic extrapolation only; no formal labels for this anisotropy sweep',need='Two narrow approaches and two wider approaches share one crossing; test a different compromise across approach constraints',anchor='xsc_tr_plus_w016',parameter='vertical_corridor_halfwidth',values=[.22,.28,.36,.44,.25,.40],sentinels=[0,2],risk='May simply preserve the equal-width preferred eta. Count as a new response regime only after actual full49 curves show reciprocal transfer loss.'),
 dict(id='F8',name='分批到达的九机交汇',family='staggered_open',priority=3,evidence='Mechanistic hypothesis from contact gating; no historical outcome claim',need='Separate simultaneous central contact from sequential resource competition while retaining all robots in the shared crossing',anchor='xsc_tr_swap_n09',parameter='radial_stagger_delta',values=[.12,.24,.36,.48,.18,.42],sentinels=[0,2],risk='Large offsets may make every eta work; such candidates do not supply heterogeneity and should not be expanded.')]
rows=[]
for fi,f in enumerate(F):
 for i,value in enumerate(f['values']):
  split='train_candidate' if i<4 else 'sealed_validation_candidate';oldrow=next((r for r in old['scenes']if r['family']==f['family']and r['split']==('train'if i<4 else'validation')and r['value']==value),None)
  # Deliberate width change retains the corresponding sampler and geometry definition.
  if f['id']=='F3' and i==5:oldrow=next(r for r in old['scenes']if r['family']=='plus'and r['split']=='validation'and r['value']==.25)
  row=copy.deepcopy(oldrow)if oldrow else {}
  row.update(id=f"{f['id']}_{'T' if i<4 else 'V'}{i+1 if i<4 else i-3}",split=split,family_id=f['id'],family=f['family'],priority=f['priority'],anchor=f['anchor'],changed_parameter=f['parameter'],value=value,status='DESIGN_ONLY_NOT_SIMULATED',role='sentinel'if i in f['sentinels']else'expand_only_if_informative'if i<4 else'no_multiagent_labels_before_final_set_freeze',parameter_seed_components=[2026102723,fi,0 if i<4 else 1,i if i<4 else i-4],states_generated=False,labels_generated=False,evidence_level=f['evidence'],hypothesis=f['need'],main_risk=f['risk'])
  row.pop('sampler_seed_components',None)
  if f['id']=='F5':
   row['fixed']=dict(N=6,initial_ring_radius=.75,dt=.05,max_speed=.5,max_steps=260,goals='unperturbed antipodes',walls=[])
   row['sampler']='Exact source xsc_div_size_r014 physical state bank for indices0-11 wherever available; reuse UID/provenance for r.14, derive other radii without altering physical samples. New confirmation states from the original base six-agent sampler with a predeclared disjoint seed; keep260 steps.'
   row['execution_note']='cross_scene_si_v1 distributed safety with0.5 sensing; no solver changes.'
   row['old_labels_reusable']=dict(exact_configuration_match=(value==.14),existing_candidates=25 if value==.14 else 0,existing_states=12 if value==.14 else 0,full49_missing_candidates=24 if value==.14 else 49,full12_state_candidate_completion_upper=864 if value==.14 else None,official_preflight_required=True)
   row['sensing_margin_diagnostic']=.5-(2*value+2*.5*.05+.0001)
  elif f['id']=='F6':
   row['fixed']=dict(N=8,robot_radius=.15,dt=.05,max_speed=.5,bottleneck_config=basebox['bottleneck_config'])
   row['base_config']=basebox['config'];row['proposed_max_steps']=math.ceil(basebox['config']['max_steps']*value/2.5)
   row['sampler']='Reuse the exact source f2.5 state bank at indices0-7 as discovery inputs for all time factors; preserve the f2.5 source case unchanged for reuse. New state draws use the same fixed geometry/radius sampler, independent of f.'
   row['execution_note']='max_steps is fixed before sampling/scoring. For f2.5 retain archived config exactly; other variants change max_steps only.'
   row['old_labels_reusable']=dict(exact_configuration_match=(value==2.5),existing_candidates=49 if value==2.5 else 0,existing_states=8 if value==2.5 else 0,official_preflight_required=True)
  elif f['id']=='F7':
   row['fixed']=dict(N=4,robot_radius=.05,horizontal_corridor_halfwidth=.16,half_length=1.6,half_height=1.6,initial_radius=1.,dt=.05,max_speed=.5,time_factor=4.)
   wx=value;wy=.16;row['extra_rectangles']=[[wx,wy,1.6,1.6],[-1.6,wy,-wx,1.6],[-1.6,-1.6,-wx,-wy],[wx,-1.6,1.6,-wy]]
   row['geometry_definition']='Free corridor union |y|<.16 or |x|<value. Retain original plus outer boundary/dummy barrier contract; ensure no inconsistency between advertised geometry and actual walls.'
   row['sampler']='Original four opposing radial tasks at angles0,90,180,270 degrees; independent radial[-.1,.1], tangential[-.02,.02] perturbation; fixed antipodal goals.'
   row['horizon_rule']='Common upper bound ceil(4*(2+sqrt(.1**2+.02**2))/(.5*.05)); freeze before state generation.'
  elif f['id']=='F8':
   row['fixed']=dict(N=9,robot_radius=.05,base_ring_radius=1.125,dt=.05,max_speed=.5,time_factor=4.,walls=[])
   row['sampler']='theta_j=2*pi*j/9; start radius=1.125+value*((j%3)-1), then independent radial[-.08,.08], tangential[-.02,.02] perturbations. Goal_j=-1.125*(cos theta_j,sin theta_j), velocity0. All9 routes cross the same center; no tiled independent games.'
   row['horizon_rule']='ceil(4*(2*1.125+delta+sqrt(.08**2+.02**2))/(.5*.05)), frozen before draws.'
   row['proposed_max_steps']=math.ceil(4*(2*1.125+value+math.sqrt(.08**2+.02**2))/(.5*.05))
  if split=='sealed_validation_candidate':
   row['state_source_rule']='Generate an independent physical state bank from the validation sampler seed. Never reuse a training/discovery or historical physical state; canonical positions/velocities/goals hashes must be disjoint.'
   row['sampler']+=' Validation override: no old physical state bank; independent draws only, using the same declared family distribution.'
  else:
   row['state_source_rule']='Discovery indices0-7; exact archival state banks may be reused where declared. Final training-set confirmation uses16 independently generated, previously unscored states, distinct from all discovery and historical labeled physical states.'
  row['physics_success_rules']='Same frozen SI entry for the relevant family, three fixed controller instances, seed0, full49 exact eta menu; geometry validity and capability >=95% for every controller required.'
  spec={k:row[k]for k in ['family','changed_parameter','value','fixed']};row['design_spec_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  rows.append(row)
assert len(rows)==48 and len({r['id']for r in rows})==48
# Derive evidence only from stored tables; no new inference or simulation.
q=np.asarray(size['y']).mean((0,1));sq={k:float(q[i])for i,k in enumerate(size['eta_keys'])}
evidence=dict(size_r014=dict(candidates=25,states=12,best_key=size['eta_keys'][int(q.argmax())],best_rate=float(q.max()),rates=sq,reciprocal_with_V_obstacles=dict(size_using_V_eta_loss=float(q.max()-sq['sE22']),V_using_size_eta_loss=all_profiles['all_scene_profiles']['lit_swap8_obstacles']['rates']['sE22']-all_profiles['all_scene_profiles']['lit_swap8_obstacles']['rates']['sE01']),causal_component_effect_established=False),box_profiles={n:all_profiles['all_scene_profiles'][n]for n in ['xsc_tr_box_n08_r015_f25','xsc_tr_box_n08_r015_f30']},primary4_anchor_evidence=all_profiles['full_sample'],warning='All new configurations and new response-direction predictions remain hypotheses. Historical25 and49 candidate comparisons are not pooled as complete49 evidence.')
write('EVIDENCE.json',evidence)
write('CANDIDATES.json',dict(status='ANALYSIS_HANDOFF_ONLY_NO_RUN_AUTHORIZATION',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_count=48,train_candidates=32,sealed_validation_candidates=16,final_target=dict(train=16,validation=8),families=F,candidates=rows,full_eta_menu=menu,source_sha256=sources,changes_from_v1=['24additional reserve configurations in4new search directions','F3 validation width.25 replaced by.255 because .25 already has labels','Earlier pipeline.py is incomplete and disabled; these files are specifications, not a runnable rollout harness']))
print('48 candidates;32 training candidates;16 sealed validation candidates; all analysis-only.')
print('F5 reciprocal sample losses',evidence['size_r014']['reciprocal_with_V_obstacles'])
