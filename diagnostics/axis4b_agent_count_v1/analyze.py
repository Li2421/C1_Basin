from common import *
from data import load_labels,ARMS
from statistics import summarize,evaluate,ci,source_choices

def expert_metrics():
 out={};rows=read(HERE/'expert_CASES.json') if (HERE/'expert_CASES.json').exists() else []
 for n in read(HERE/'DESIGN_CONFIG.json')['N']:
  rr=[read(HERE/'expert/raw'/f"{c['rollout_uid']}.json") for c in rows if c['N']==n]
  if not rr:out[str(n)]=dict(status='GAP',reason='Expert reproduction/reference unavailable');continue
  assert len(rr)==32;a=np.asarray([q['success'] for q in rr]);b=np.asarray([q['success']+q['numerical_failure'] for q in rr]);rng=np.random.default_rng(2026101536999+n);ix=rng.integers(32,size=(5000,32));out[str(n)]=dict(status='JOINT_WITNESSES_COMPLETE' if a.sum()==32 else 'PARTIAL_OR_UNESTABLISHED',successes=int(a.sum()),states=32,success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=ci(np.stack([a[ix].mean(1),b[ix].mean(1)],1)),events={k:sum(q[k] for q in rr) for k in ['collision','timeout','numerical_failure']},unresolved=sum(q['outcome']=='reference_unresolved' for q in rr),planning_seconds=sum(q['planning_seconds'] or 0 for q in rr),scope='Privileged sequential Q1 reference; a failed planner is not an infeasibility proof')
 return out

def pilot_metrics(n,kind):
 rows=read(HERE/'pilot_CASES.json') if (HERE/'pilot_CASES.json').exists() else [];out={};rng=np.random.default_rng(2026101536500+n);indices=rng.integers(16,size=(5000,16))
 for arm in ['safety_baseline','F10','S13t2']:
  rr=[read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json") for c in rows if c['N']==n and c['kind']==kind and c['arm']==arm]
  if len(rr)!=16:continue
  a=np.asarray([q['success'] for q in rr]);b=np.asarray([q['success']+q['numerical_failure'] for q in rr]);out[{'safety_baseline':'safety baseline','S13t2':'S500 class'}.get(arm,arm)]=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=ci(np.stack([a[indices].mean(1),b[indices].mean(1)],1)),events_pct={k:np.mean([q[k] for q in rr])*100 for k in ['collision','timeout','numerical_failure']},scope='Capability-only16states seed0; no main/menu/transfer estimate')
 return out

def main():
 assert (HERE/'ALL_LABELS_COMPLETE.receipt.json').exists();cfg=read(HERE/'DESIGN_CONFIG.json');cap=read(HERE/'CAPABILITY_GATE.json');frozen=read(HERE/'TRANSFER_CHOICES.json');n0=cap['source_N'];source=load_labels('source',n0) if n0 is not None else None
 if source:assert source_choices(source)==frozen['choices']
 seal('ENDPOINT_OPENING.receipt.json',[HERE/'ALL_LABELS_COMPLETE.receipt.json',HERE/'TRANSFER_CHOICES_FROZEN.receipt.json',HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json'],target_evaluation_labels_read_before_choice_freeze=False)
 experts=expert_metrics();out=dict(source_N=n0,units={},N_pooled={},experts=experts,endpoints={},scope='Fixed-map Gap1, native N-specific horizons; Flow checkpoints differ acrossN; onlyP2 is clean N-independent transfer');datasets={};p2targets=[];ac1={};ac2={};stepcounts={};unknowns=[]
 for n in cfg['N']:
  stage='source' if n==n0 else 'target';has=(HERE/f'{stage}_CASES.json').exists() and any(c['N']==n for c in read(HERE/f'{stage}_CASES.json'));data=load_labels(stage,n) if has else None
  if data and len(data['classes']):
   datasets[n]=data;np.savez_compressed(HERE/f'N{n}_LABELS.npz',lower=data['lower'],upper=data['upper'],classes=np.asarray(data['classes']),arms=np.asarray(ARMS),**data['events']);res=summarize(data,source,cfg['bootstrap_seed']+n*10);out['N_pooled'][str(n)]=res
   if n>n0:
    v=res['paired']['transfer_mean_minus_safety'];ac1[str(n)]=dict(status='SUPPORTED' if v['CI_pp'][0]>0 else 'NOT_DEMONSTRATED',criterion_passed=v['CI_pp'][0]>0,**v,controllers=data['classes'],choice='source mean, no target labels',Flow_checkpoint_changes_across_N='Flow' in data['classes'])
  for kind in ['Flow','P2','reciprocal']:
   label=f'N{n}_{kind}';cp=cap['controllers'][str(n)].get(kind);row=dict(N=n,controller=kind,expert=experts[str(n)])
   if data and kind in data['classes']:
    j=data['classes'].index(kind);sub=dict(N=n,classes=[kind],lower=data['lower'][j:j+1],upper=data['upper'][j:j+1],events={k:v[j:j+1] for k,v in data['events'].items()});r=summarize(sub,source,cfg['bootstrap_seed']+n*10+(0 if kind=='Flow' else 1));row.update(status='MAIN_COMPLETE',evidence='CONFIRMATORY',methods=r['methods'],analysis=r)
    if kind=='P2' and n>n0 and 'source per-controller transfer' in r['methods']:
     v=r['methods']['source per-controller transfer'];bounds=v['mean_regret_pp'];ac2[str(n)]=dict(status='SUPPORTED' if bounds[1]<=5 else 'REJECTED' if bounds[0]>5 else 'INCONCLUSIVE',mean_regret_pp=bounds,regret_CI_pp=r['paired']['transfer_controller_regret']['CI_pp']);p2targets.append(sub)
   else:row.update(status='GAP',evidence='CAPABILITY_ONLY_MAIN_GAP' if kind!='reciprocal' else 'GAP',reason='Historical reciprocal nominal strict action-byte gate unavailable and diagnostic termination differs' if kind=='reciprocal' else (cp['status'] if cp else 'No qualified asset'),pilot_methods=pilot_metrics(n,kind) if kind!='reciprocal' else {})
   out['units'][label]=row
 # Macro target-N P2 regret uses identical state draws acrossall nestedNs andsource.
 if p2targets:
  lows=[];highs=[];point=[out['units'][f"N{d['N']}_P2"]['analysis']['methods']['source per-controller transfer']['mean_regret_pp'] for d in p2targets];rng=np.random.default_rng(cfg['bootstrap_seed']+999)
  for _ in range(5000):
   ss=rng.integers(32,size=(1,32));v=[]
   for d in p2targets:
    m,_=evaluate(d['lower'][np.zeros((1,1),int),ss],d['upper'][np.zeros((1,1),int),ss],['P2'],source,ss);v.append([(m['full menu oracle'][0]-m['source per-controller transfer'][1]).mean(),(m['full menu oracle'][1]-m['source per-controller transfer'][0]).mean()])
   z=np.mean(v,axis=0);lows.append(z[0]);highs.append(z[1])
  bounds=np.mean(point,axis=0);pooled2=dict(status='SUPPORTED' if bounds[1]<=5 else 'REJECTED' if bounds[0]>5 else 'INCONCLUSIVE',mean_regret_pp=bounds,regret_95CI_pp=ci(np.stack([lows,highs],1)),target_N=[d['N'] for d in p2targets],controller='same P2 implementation/gains',individual_N=ac2)
 else:pooled2=dict(status='GAP',reason='No same N-independent controller qualified at source and any larger target',individual_N=ac2)
 out['endpoints']['AC1']=dict(status='GAP' if not ac1 else 'SUPPORTED' if all(v['criterion_passed'] for v in ac1.values()) else 'NOT_DEMONSTRATED',targets=ac1,source_N=n0,interpretation='Only capability-qualified targets; checkpoint changes inFlow preclude clean same-controller quantity claims');out['endpoints']['AC2']=pooled2
 # Performance and numerical evidence exclude deterministic copies fromphysical cost.
 for stage in ['gate','pilot','source','target','expert']:
  if not (HERE/f'{stage}_CASES.json').exists():continue
  for c in read(HERE/f'{stage}_CASES.json'):
   q=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");key=f"N{c['N']}_{c['kind']}";v=stepcounts.setdefault(key,dict(new_physical_steps=0,deterministic_reused_steps=0,episodes=0,numerical_unknown=0,individual_goal_fraction_sum=0.,physical_episode_elapsed_seconds=0.,physical_episode_count=0));v['episodes']+=1;v['new_physical_steps']+=q['new_physical_steps'];v['deterministic_reused_steps']+=q['deterministic_reused_steps'];v['numerical_unknown']+=q['numerical_failure'];v['individual_goal_fraction_sum']+=q['final_goal_fraction']
   if q['new_physical_steps']>0:v['physical_episode_elapsed_seconds']+=q['elapsed_seconds'];v['physical_episode_count']+=1
   if q['numerical_failure']:unknowns.append(dict(stage=stage,N=c['N'],kind=c['kind'],rollout_uid=c['rollout_uid'],error=q['error']))
 write(HERE/'PERFORMANCE_ACCOUNTING.json',stepcounts);write(HERE/'UNKNOWN_RECORDS.json',unknowns)
 levels=[];first_observed=None
 for n in cfg['N']:
  ss=read(HERE/'STATES.json')[str(n)]['eval'];e=experts[str(n)];controllers={k:cap['controllers'][str(n)][k] for k in ['Flow','P2']};row=dict(N=n,H=cfg['H'][str(n)],geometry_valid_states=sum(s['geometry_validation']['valid_initial_state'] and s['geometry_validation']['valid_goals'] for s in ss),single_robot_path_witness_states=sum(s['geometry_validation']['single_agent_path_witnesses'] for s in ss),joint_reference=e,controller_capability=controllers,controller_main_rates={k:out['units'][f'N{n}_{k}'].get('methods',{}) for k in ['Flow','P2']},correction_effect=ac1.get(str(n),dict(status='GAP or source-only')),deadlock_resolution='NOT_ESTABLISHED; no certified native deadlock event; timeout/stall not deadlock',shared_opposing_pairs=(n//2)*((n+1)//2));levels.append(row)
  if first_observed is None:
   if len(ss)!=32 or row['geometry_valid_states']!=32:first_observed=dict(N=n,layer=1,reason='Geometric validation incomplete')
   elif row['single_robot_path_witness_states']!=32:first_observed=dict(N=n,layer=2,reason='Single-robot path evidence incomplete')
   elif e.get('successes',0)!=32:first_observed=dict(N=n,layer=3,reason='Joint witnesses incomplete; expert failure does not prove impossibility')
   else:
    failed=[k for k,v in controllers.items() if v['status']!='PASS']
    if failed:first_observed=dict(N=n,layer=4,controllers=failed,reason='Prespecified three-arm controller capability gate not passed; not universal eta impossibility')
 out['five_layers']=levels;out['first_observed_failure_or_gap']=first_observed or dict(layer=5,N=2,reason='No certified deadlock-resolution evidence; preceding measured layers passed');out['earliest_asset_gap']=dict(N=2,layer=4,controller='reciprocal',reason='Not a measured performance failure; missing strict nominal trace interface');write(HERE/'results.json',out);write(HERE/'FIVE_LAYER_EVIDENCE.json',levels);print('Round36 analysis complete',out['endpoints'],flush=True)
if __name__=='__main__':main()
