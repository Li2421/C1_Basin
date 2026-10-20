from common import *
from planning import protocol,ring_state,ring_case,source
from registry import register,uid,canonical,eta_identity
import copy

def novel_state(p,old,validation=None):
    s=copy.deepcopy(old);s['physical']={k:s['physical'][k] for k in ['positions','velocities','goals']};s['transport_physical']=s['physical']
    if validation:
        s['content_hash']=uid('content',dict(physical=s['physical'],original_uid=s['uid'],validation=validation));s['state_uid']=uid('state',dict(scenario=s['scenario_uid'],content=s['content_hash']))
    if s['state_uid'] not in {q['state_uid'] for q in p['states']}:p['states'].append(s)
    return s

def novel_case(p,s,name,rule,seed=0,validation=None):
    parent=read(R33/'main_protocol.json');p['native_manifest']=parent['native_manifest'];v=copy.deepcopy(parent['profiles'][name]['payload']);v.update(scenario_uid=s['scenario_uid'],conditioning='Original black-box TT baseline plus continuing clockwise tangent and same safety projection',round37_engine='continuing_rhr',continuing_rhr=dict(rule=rule,validation=validation,implementation_sha256=sha(HERE/'continuing_rhr.py'),trigger_sha256=sha(HERE/'trigger.py')),execution_source_sha256={f:sha(HERE/f) for f in ['controllers.py','legacy_controllers.py','gaussian.py','continuing_rhr.py','trigger.py']});key=uid('profile',v);cid=uid('ctl',v);p['profiles'][key]=dict(controller_uid=cid,payload=v);p['profiles'][name]=p['profiles'][key];ea=[0.,0.,0.];eid=eta_identity(ea)[0];sk=canonical(dict(future_index=seed));return dict(engine='continuing_rhr',kind='G' if name.startswith('G') else 'DM',H=700,r=.9,profile_key=key,base_controller_name=name,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),arm='RHR',seed=seed,eta_a=ea,eta_b=ea,switch_step=700,state_index=s['study_index'],rhr_rule=rule,placement='TT')

def gate():
    assert (HERE/'B_COMPLETE.receipt.json').exists();p=protocol();cases=[];rule=dict(mode='progress',steps=1000000,speed=.1);oldp=read(R35/'baseline_protocol.json');rows=read(R35/'baseline_CASES.json');names=source()['controller_names'];spec=[(n,0,0) for n in names]+[(n,1,1) for n in ['base_u_macflow','base_u_v10_local_macflow']]
    for n,i,seed in spec:
        ref=next(c for c in rows if c['base_controller_name']==n and c['state_index']==i and c['seed']==seed and c['arm']=='safety_baseline');old=next(s for s in oldp['states'] if s['state_uid']==ref['state_uid']);s=ring_state(p,old,700,.9,'round37_inactive_flow');s['study_index']=i;c=ring_case(p,s,n,700,.9,'RHR',seed,'round37_inactive_flow',rule=rule);c['engine']='continuing_rhr';path=R35/'baseline/raw'/f"{ref['rollout_uid']}.json";c.update(reference_path=str(path),reference_sha256=sha(path),replay_group='Flow_inactive');cases.append(c)
    oldp=read(R33/'main_protocol.json');rows=read(R33/'main_CASES.json');names=[f'G{i}' for i in range(6)]+[f'DM{i}' for i in range(6)];spec=[(n,0) for n in names]+[(n,1) for n in ['G0','G1','DM0','DM1']]
    for n,seed in spec:
        ref=next(c for c in rows if c['base_controller_name']==n and c['state_index']==0 and c['seed']==seed and c['arm']=='safety_baseline');old=next(s for s in oldp['states'] if s['state_uid']==ref['state_uid']);s=novel_state(p,old,'round37_inactive_novel');c=novel_case(p,s,n,rule,seed,'round37_inactive_novel');path=R33/'main/raw'/f"{ref['rollout_uid']}.json";c.update(reference_path=str(path),reference_sha256=sha(path),replay_group='Novel_inactive');cases.append(c)
    register('replay_D',p,cases)

def physical_signature(v):
    try:
        return uid('physical',[(np.asarray(v[k],np.float64).shape,np.asarray(v[k],np.float64).tobytes().hex()) for k in ['positions','velocities','goals']])
    except (KeyError,TypeError,ValueError):return None

def tuning_states():
    from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
    from registry import connect
    with connect(True) as db:old=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
    sigs=set();unrecoverable=0
    for q in old:
        try:sig=physical_signature(json.loads(q['physical_state_json']))
        except (ValueError,TypeError):sig=None
        if sig:sigs.add(sig)
        else:unrecoverable+=1
    all_eval=read(R32/'EVAL_STATES.json')+read(R33/'EVAL_STATES.json')+read(R35/'EVAL_STATES_ORIGINAL.json')+read(HERE/'A1_STATES.json')
    for q in all_eval:sigs.add(physical_signature(q['physical']))
    native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');chosen=[];rejected=[]
    for seed in range(202610370000,202610371000):
        s=generate('round37_blackbox_tuning',1,seed,native)[0];sig=physical_signature(s['physical'])
        if sig in sigs:rejected.append(dict(seed=seed,reason='physical duplicate'));continue
        s['study_index']=len(chosen);chosen.append(s);sigs.add(sig)
        if len(chosen)==16:break
    assert len(chosen)==16;write(HERE/'TUNING_STATES.json',chosen);write(HERE/'TUNING_STATE_DEDUP.json',dict(utc=utc(),passed=True,DB_states=len(old),unrecoverable_old_physical=unrecoverable,all_explicit_evaluation_states=len(all_eval),chosen_seeds=[s['metadata']['draw_seed'] for s in chosen],rejected=rejected,outcomes_used=False));return chosen

def tune():
    assert read(HERE/'D_REPRODUCTION_GATE.json')['Flow_inactive_passed'] and read(HERE/'D_REPRODUCTION_GATE.json')['Novel_inactive_passed'];p=protocol();rows=[];states=tuning_states();grid=read(D/'teacher_student_macflow_v1d/RHR_GRID.json');names=source()['controller_names']
    for k,rule in enumerate(grid):
        for old in states:
            s=ring_state(p,old,700,.9)
            for n in names:
                c=ring_case(p,s,n,700,.9,'RHR',0,rule=rule);c.update(engine='continuing_rhr',grid_index=k,pool='Flow');rows.append(c)
            old=copy.deepcopy(old);old['scenario_uid']=read(R33/'main_protocol.json')['states'][0]['scenario_uid'];s=novel_state(p,old,'round37_novel_tuning')
            for n in [f'G{i}' for i in range(6)]+[f'DM{i}' for i in range(6)]:
                c=novel_case(p,s,n,rule);c.update(grid_index=k,pool='Gaussian' if n.startswith('G') else 'MPC');rows.append(c)
    register('right_hand_tune',p,rows)

def evaluation():
    assert (HERE/'RIGHT_HAND_CHOICES_FROZEN.receipt.json').exists();selected=read(HERE/'RIGHT_HAND_SELECTED.json');p=protocol();rows=[];names=source()['controller_names']
    for label,states,H,r,seeds in [('D1',read(R32/'EVAL_STATES.json')[:64],700,.9,[0,1]),('D2',read(R32/'EVAL_STATES.json')[64:],700,.9,[0,1]),('D3_H500',read(R35/'EVAL_STATES_ORIGINAL.json'),500,.9,[0,1]),('D3_H1000',read(R35/'EVAL_STATES_ORIGINAL.json'),1000,.9,[0,1])]+[(f'D4_r{radius}',read(HERE/'A2_ORIGINAL_STATES.json'),700,radius,[0]) for radius in [.8,.85,.95,1.]]:
        for j,old in enumerate(states):
            s=ring_state(p,old,H,r);s['study_index']=j
            for n in names:
                for seed in seeds:
                    c=ring_case(p,s,n,H,r,'RHR',seed,rule=selected['Flow']['rule']);c.update(engine='continuing_rhr',subunit=label,pool='Flow');rows.append(c)
    for old in read(R33/'EVAL_STATES.json'):
        s=novel_state(p,old)
        for n in [f'G{i}' for i in range(6)]+[f'DM{i}' for i in range(6)]:
            pool='Gaussian' if n.startswith('G') else 'MPC'
            for seed in [0,1]:
                c=novel_case(p,s,n,selected[pool]['rule'],seed);c.update(subunit='D5',pool=pool);rows.append(c)
    expected=6144;assert len(rows)==expected,(len(rows),expected);register('right_hand_evaluation',p,rows)

if __name__=='__main__':globals()[sys.argv[1]]()
