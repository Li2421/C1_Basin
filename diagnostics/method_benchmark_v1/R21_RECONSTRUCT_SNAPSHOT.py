"""Read-only label identity reconstruction and historical-number gate only."""
import os,sys,json,hashlib,datetime
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics';DATA={};checks=[];assets={}
def read(p):
 p=Path(p);assets[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def load(p):
 p=Path(p);assets[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return dict(np.load(p,allow_pickle=False))
def save(tag,y,u,names,arms,states,scene='ring_exchange',available=None,folder=None):
 y=np.asarray(y,np.int8);u=np.asarray(u,np.int8);assert y.shape==u.shape and y.ndim==4;assert y.shape[0]==len(names) and y.shape[2]==len(arms);assert len(set(names))==len(names);assert len(set(states))==len(states)==y.shape[1];assert not np.any((y==1)&(u==1));av=np.ones((len(names),len(arms)),bool) if available is None else available
 np.savez_compressed(HERE/(tag+'.npz'),success=y,unknown=u,names=np.asarray(names),arms=np.asarray(arms),states=np.asarray(states),available=av);DATA[tag]=dict(y=y,u=u,names=list(names),arms=list(arms),states=list(states),available=av,scene=scene,folder=str(folder) if folder else None)
def check(name,actual,expected):checks.append(dict(name=name,actual=np.asarray(actual).tolist(),expected=np.asarray(expected).tolist(),passed=bool(np.allclose(actual,expected,rtol=0,atol=2e-15))))
def sids(p,count):
 cs=read(p/'main_CASES.json');x={c['state_index']:c['state_uid'] for c in cs};assert len(x)==count;return [x[i] for i in range(count)]
for tag,folder in [('R10','controller_universal_eta_schedule_v1'),('R12','controller_new_checkpoint_confirmation_v1/fp32_n8'),('R14A','ring_independent_flow_confirmation_v2')]:
 p=D/folder;z=load(p/'LABELS.npz');names=z.get('controller_names',z.get('controllers')).tolist();save(tag,z['success'].transpose(0,2,1,3),z['unknown'].transpose(0,2,1,3),names,z['arms'].tolist(),sids(p,64),available=z.get('available'),folder=p)
 if tag=='R10':
  rates=z['success'].mean((2,3));a=z['arms'].tolist();fixed=np.where(z['available'][:,:3],rates[:,:3],np.nan);value=float(np.mean(rates[:,a.index('S500')]-np.nanmax(fixed,axis=1)));report=read(p/'STATISTICAL_RESULTS.json');check('R10 S500 minus fixed oracle',value,report['group_metrics']['S500']['all']['mean_gain'])
 else:
  q=read(p/'SELECTOR_MODELS.json');choices={r['controller']:r['primary_action'] for r in q['choices']};rates=z['success'].mean((2,3));a=z['arms'].tolist();value=float(np.mean(rates.max(1)-[rates[i,a.index(choices[n])] for i,n in enumerate(names)]));report=read(p/'results.json');check(tag+' frozen G_E mean regret',value,report['methods']['primary']['mean_regret'])
p=D/'ring_controller_population_v1';rows=[json.loads(l) for l in (p/'dataset.jsonl').read_text().splitlines() if l];assets[str(p/'dataset.jsonl')]=hashlib.sha256((p/'dataset.jsonl').read_bytes()).hexdigest();names=list(dict.fromkeys(r['controller'] for r in rows));arms=['F10','F13','F2','S200','S300','S400','S500','S600'];y=np.zeros((56,32,8,4),np.int8);u=y.copy();seen=set();si={}
for r in rows:
 i=names.index(r['controller']);s=r['state_index'];a=arms.index(r['arm']);q=r['future_seed'];key=(i,s,a,q);assert key not in seen;seen.add(key);y[key]=r['success'];u[key]=r['unknown'];si[s]=r['state_uid']
assert len(seen)==57344;save('R15',y,u,names,arms,[si[i] for i in range(32)],folder=p)
for tag,folder,file in [('R17','ring_critic_new_recipe_confirmation_v1','LABELS.npz'),('R18TRAIN','ring_eta2_gap_v1','train_LABELS.npz'),('R18TEST','ring_eta2_gap_v1','test_LABELS.npz')]:
 p=D/folder;z=load(p/file);save(tag,z['Y']==1,z['Y']==-1,z['controller_names'].tolist(),z['arms'].tolist(),z['state_uids'].tolist(),folder=p)
 if tag=='R17':
  q=read(p/'FROZEN_CHOICES.json');names=z['controller_names'].tolist();a=z['arms'].tolist();indices=[i for i,n in enumerate(names) if n.endswith('__arch512')];rates=(z['Y']==1).sum((1,3))/(z['Y']!=-1).sum((1,3));ch={r['controller']:r['A3_fixed'] for r in q['controllers']};value=float(np.mean(rates[indices].max(1)-[rates[i,a.index(ch[names[i]])] for i in indices]));report=read(p/'results.json');check('R17 A3 N2 known-denominator historical regret',value*100,report['groups']['N2']['policies']['A3_fixed']['mean_regret_pp'])
 elif tag=='R18TEST':
  report=read(p/'results.json');rates=(z['Y']==1).sum((1,3))/(z['Y']!=-1).sum((1,3));check('R18 known-denominator rates',rates,[list(r['success_rate_by_arm'].values()) for r in report['controllers']])
p=D/'cross_scene_eta_schedule_v1';z=load(p/'LABELS.npz');report=read(p/'results.json');protocol=read(p/'protocol.json')
for i,sc in enumerate(z['scenes'].tolist()):
 states=[s['state_uid'] for s in sorted([s for s in protocol['states'] if s['scenario']==sc],key=lambda s:s['study_index'])];save('R13_'+sc,z['success'][i].transpose(0,2,1,3),z['unknown'][i].transpose(0,2,1,3),z['controllers'].tolist(),z['arms'].tolist(),states,scene=sc,folder=p)
check('R13 reconstructed arm table',z['success'].mean((3,4)),report['controller_rates_lower'])
rates=z['success'].mean((1,3,4));reg=[]
for menu in ([0,1,2],list(range(7))):
 rr=[]
 for i in range(4):
  source=rates[np.arange(4)!=i].mean(0);selected=menu[int(np.argmax(source[menu]))];rr.append(float(rates[i].max()-rates[i,selected]))
 reg.append(float(np.mean(rr)))
check('R13 LOSO fixed/full mean regrets',reg,[report['X2']['methods'][k]['mean_regret'] for k in ('fixed','expanded')])
p=D/'cross_scene_probe_selection_v1';active=read(p/'PROBE_SELECTION_FREEZE.json')['active_variants'];cases=read(p/'main_ACTIVE_CASES.json');report=read(p/'results.json');arms=['F10','F13','F15','S13_t1','S13_t2','S15_t1','S15_t2'];names=['flow_isotropic','flow_anisotropic','flow_coupled']
for variant in active:
 y=np.zeros((3,40,7,2),np.int8);u=y.copy();states={};seen=set();sc=None
 for c in [c for c in cases if c['variant']==variant]:
  r=json.loads((p/'main/raw'/(c['rollout_uid']+'.json')).read_text());key=(names.index(c['base_controller_name']),c['state_index'],arms.index(c['arm']),c['seed']);assert key not in seen;seen.add(key);y[key]=r['success'];u[key]=r['numerical_failure'];states[c['state_index']]=c['state_uid'];sc=c['scene']
 assert len(seen)==1680;save('R20_'+variant,y,u,names,arms,[states[i] for i in range(40)],scene=sc,folder=p);check('R20 '+variant+' arm rates',y.mean((0,1,3)),[report['rates'][variant][a] for a in arms])
p=D/'four_controller_axis_v2';report=read(p/'SCHEDULE_RESULTS.json');names=report['controller_order'];arms=report['all_arms'];y=np.zeros((25,32,8,2),np.int8);u=y.copy();states={};seen=set()
for stage,file in [('stage2','FIXED_CASES.json'),('stage3','SCHEDULE_CASES.json')]:
 for c in read(p/file):
  r=json.loads((p/stage/'raw'/(c['rollout_uid']+'.json')).read_text());key=(names.index(c['base_controller_name']),c['state_index'],arms.index(c['arm']),c['seed']);assert key not in seen;seen.add(key);y[key]=r['success'];u[key]=r['numerical_failure'];states[c['state_index']]=c['state_uid']
assert len(seen)==12800;save('R19b',y,u,names,arms,[states[i] for i in range(32)],scene='four_way_intersection',folder=p);a=arms.index('S15to10_786');check('R19b default successes/known', [int(y[:,:,a].sum()),int((1-u[:,:,a]).sum())],[1597,1597])
# R9 includes exact old fixed evidence plus both switching directions.
p=D/'controller_eta_switch_time_v1';protocol=read(p/'protocol.json');names=protocol['controller_order'] if 'controller_order' in protocol else ['base_u_macflow','base_u_v10_local_macflow','base_u_v9_local_macflow','base_u_plus_w_v4_wide_macflow'];names=list(dict.fromkeys(c['base_controller_name'] for c in read(p/'main_CASES.json')));arms=['F10','F13']+[f'S{a}to{b}_{t}' for a,b in [(10,13),(13,10)] for t in [100,200,300,400,500]];y=np.zeros((4,64,12,4),np.int8);u=y.copy();states={}
for item in read(p/'HISTORICAL_RAW_INDEX.json'):
 c=item['case']
 if c['controller_name'] in names and c['eta_index'] in (10,13) and c['seed']<4:
  r=json.loads(Path(item['path']).read_text());key=(names.index(c['controller_name']),c['state_index'],arms.index('F'+str(c['eta_index'])),c['seed']);y[key]=r['success'];u[key]=r['numerical_failure'];states[c['state_index']]=c['state_uid']
for c in read(p/'main_CASES.json'):
 r=json.loads((p/'main/raw'/(c['rollout_uid']+'.json')).read_text());key=(names.index(c['base_controller_name']),c['state_index'],arms.index(f"S{c['eta_a_index']}to{c['eta_b_index']}_{c['switch_step']}"),c['seed']);y[key]=r['success'];u[key]=r['numerical_failure'];states[c['state_index']]=c['state_uid']
save('R9',y,u,names,arms,[states[i] for i in range(64)],folder=p)
inventory={k:dict(shape=list(v['y'].shape),unknown=int(v['u'].sum()),available=v['available'].tolist(),names=v['names'],arms=v['arms'],states=v['states'],scene=v['scene'],folder=v['folder'],label_evidence='prospectively collected; current analysis DEVELOPMENT') for k,v in DATA.items()};(HERE/'DATA_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n');(HERE/'SOURCE_HASHES.json').write_text(json.dumps(assets,indent=2)+'\n');out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':all(c['passed'] for c in checks),'checks':checks,'schema':'controller,state,arm,seed; success and unknown separate','historical_denominator_note':'R17/R18/R19b historical checks use known-only where their reports did; new standardized lower/upper convention is not applied until preregistration'}
(HERE/'RECONSTRUCTION_GATE.json').write_text(json.dumps(out,indent=2)+'\n');print('GATE',out['passed'],len(checks));assert out['passed'],'STOP before new statistics'
