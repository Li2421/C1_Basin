from common import *
from plan_E import PARAMS,ARMS
from audit_utils import source_inventory
import importlib.util
# An exact copy of the previous round's inference, with local core resolved safely.
from agent_statistics import summarize,source_choices

def capability():
 rows={r['rollout_uid']:r for r in read(HERE/'gap_capability_DB_ROWS.json')};cases=read(HERE/'gap_capability_CASES.json');budget=read(HERE/'MPC_TIMING_REPORT.json');out={}
 for n in [2,10,20]:
  out[str(n)]={}
  for name in PARAMS:
   arms={}
   if not budget[str(n)]['eligible']:out[str(n)][name]=dict(status='GAP',reason=budget[str(n)]['reason']);continue
   for arm in ['safety_baseline','F10','S13t2']:
    rr=[rows[c['rollout_uid']] for c in cases if c['N']==n and c['base_controller_name']==name and c['arm']==arm];assert len(rr)==16;s=sum(r['success'] and not r['numerical_failure'] for r in rr);u=sum(r['numerical_failure'] for r in rr);arms[arm]=[s/16,(s+u)/16]
   lower=max(a[0] for a in arms.values());upper=max(a[1] for a in arms.values());out[str(n)][name]=dict(status='PASS' if lower>=.2 else 'FAIL_CONTROLLER_CAPABILITY' if upper<.2 else 'INCONCLUSIVE',max_success_bounds=[lower,upper],arms=arms)
 write(HERE/'MPC_CAPABILITY_GATE.json',out);seal('MPC_CAPABILITY_GATE.receipt.json',[HERE/'MPC_CAPABILITY_GATE.json'],no_gate_rejudgment=True)

def load(n):
 gate=read(HERE/'MPC_CAPABILITY_GATE.json')[str(n)];names=[k for k in PARAMS if gate[k]['status']=='PASS'];shape=(len(names),32,9,2);data={k:np.zeros(shape) for k in ['lower','upper','collision','timeout','unknown']};seen=np.zeros(shape,bool);prefix=[]
 for phase in [f'gap_N{n}_fixed',f'gap_N{n}_schedules']:
  rows={r['rollout_uid']:r for r in read(HERE/f'{phase}_DB_ROWS.json')}
  for c in read(HERE/f'{phase}_CASES.json'):
   r=rows[c['rollout_uid']];i=(names.index(c['base_controller_name']),c['state_index'],ARMS.index(c['arm']),c['seed']);assert not seen[i];seen[i]=True;u=r['numerical_failure'];data['lower'][i]=r['success'] if not u else 0;data['upper'][i]=data['lower'][i]+u
   for k in ['collision','timeout','unknown']:data[k][i]=r['numerical_failure' if k=='unknown' else k]
   raw=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json")
   if raw.get('exact_prefix'):prefix.append(dict(rollout_uid=c['rollout_uid'],**raw['exact_prefix']))
 assert seen.all();np.savez_compressed(HERE/f'MPC_N{n}_LABELS.npz',**data,controller_names=names,arms=ARMS);write(HERE/f'MPC_N{n}_PREFIX_AUDIT.json',prefix);return dict(N=n,classes=names,lower=data['lower'],upper=data['upper'],events={k:data[k] for k in ['collision','timeout','unknown']})

def freeze(n):
 data=load(n);choices=source_choices(data);write(HERE/'MPC_SOURCE_CHOICES.json',dict(utc=utc(),source_N=n,choices=choices,controller_names=data['classes'],parameters={k:PARAMS[k] for k in data['classes']},source_label_sha256=sha(HERE/f'MPC_N{n}_LABELS.npz'),target_evaluation_labels_opened=False));seal('MPC_SOURCE_CHOICES_FROZEN.receipt.json',[HERE/'MPC_SOURCE_CHOICES.json',HERE/f'MPC_N{n}_LABELS.npz'],before_any_target_main_labels=True)

def finish():
 gate=read(HERE/'MPC_CAPABILITY_GATE.json');out={};sourceN=read(HERE/'MPC_SOURCE_CHOICES.json')['source_N'] if (HERE/'MPC_SOURCE_CHOICES.json').exists() else None;source=load(sourceN) if sourceN else None;layers=[];oldlayers={q['N']:q for q in read(R36/'FIVE_LAYER_EVIDENCE.json')}
 for j,n in enumerate([2,10,20]):
  qualified=[name for name in PARAMS if gate[str(n)][name]['status']=='PASS'];q=dict(N=n,geometry_valid_states=oldlayers[n]['geometry_valid_states'],single_robot_path_witness_states=oldlayers[n]['single_robot_path_witness_states'],joint_reference=oldlayers[n]['joint_reference'],joint_reference_source=str(R36/'FIVE_LAYER_EVIDENCE.json'),controller_capability=gate[str(n)],deadlock_resolution='NOT_ESTABLISHED: successful navigation and stagnation candidates do not establish deadlock resolution')
  if qualified:
   data=load(n);stats=summarize(data,source,202610163750+j);stats.update(evidence='CONFIRMATORY',original_ring_critics='N/A: no certified variable-N Gap interface; no padding or refitting',parameters={k:PARAMS[k] for k in qualified});common=[k for k in qualified if source and k in source['classes']]
   if common:
    ix=[qualified.index(k) for k in common];clean=dict(N=n,classes=common,lower=data['lower'][ix],upper=data['upper'][ix],events={k:v[ix] for k,v in data['events'].items()});cs=summarize(clean,source,202610163750+j);paired=cs['paired'];gain=paired['transfer_controller_minus_safety']['CI_pp'];reg=paired['transfer_controller_regret']['estimate_pp'];stats['clean_same_controller_transfer']=dict(instances=common,source_N=sourceN,gain_CI_pp=gain,mean_regret_pp=reg,correction_status='SUPPORTED' if gain[0]>0 else 'REJECTED' if gain[1]<=0 else 'INCONCLUSIVE',transfer_status='SUPPORTED' if reg[1]<=5 else 'REJECTED' if reg[0]>5 else 'INCONCLUSIVE',scope='Source cross-fit reference' if n==sourceN else 'Same fixed-parameter black-box controller across N');q['correction_and_transfer']=stats['clean_same_controller_transfer']
   else:stats['clean_same_controller_transfer']=dict(status='GAP',reason='No identical parameter instance passes both source and target capability')
   stats['missing_source_instances']=[k for k in qualified if not source or k not in source['classes']];out[str(n)]=stats
  else:out[str(n)]=dict(status='GAP',reason='No budget-eligible instance passed the frozen capability gate',capability=gate[str(n)])
  layers.append(q)
 first=[]
 for row in layers:
  for name,v in row['controller_capability'].items():
   if v['status']!='PASS':first.append(dict(N=row['N'],instance=name,layer=4 if v['status']=='FAIL_CONTROLLER_CAPABILITY' else None,status=v['status'],reason=v.get('reason','Capability below threshold or unknown interval crossing threshold')))
 write(HERE/'MPC_FIVE_LAYER_EVIDENCE.json',dict(layers=layers,first_observed_failures=first,unestablished_fifth_layer_all_N=True));write(HERE/'E_RESULTS.json',dict(utc=utc(),source_N=sourceN,units=out,first_observed_failure=first[0] if first else None,timing=read(HERE/'MPC_TIMING_REPORT.json')));seal('E_COMPLETE.receipt.json',[HERE/'E_RESULTS.json',HERE/'MPC_FIVE_LAYER_EVIDENCE.json',HERE/'MPC_CAPABILITY_GATE.json',HERE/'MPC_TIMING_REPORT.json']);source_inventory('imports_analyze_E.json')
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze(int(sys.argv[2]))
 else:globals()[sys.argv[1]]()
