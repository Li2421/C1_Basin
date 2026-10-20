"""Actual zero-prefix and arbitrary-progress gates, with official identity preflight."""
from common import *
from actions import action,case,fixed_source_uid
from records import actual
from registry import register
def design_basic():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'];base=read(HERE/'BASE_PROTOCOL.json');objects=read(HERE/'A_OBJECTS.json')['records'];zero=[0.,0.,0.];rows=[];profiles={};states={};checks=[]
    flow=[n for n in base['controller_order'] if base['types'][n]=='Flow'];native='base_u_v10_local_macflow';adapter=next(n for n in flow if n!=native)
    with rollout_db.connect(True) as db:
        for name in (native,adapter,'transformer_bc0','conflict_based_search0'):
            obj=next(x for x in objects if x['controller']==name);s=obj['state'];target=action(zero,obj['proposals'][0],'fixed',200,'zero_prefix');c,cp=case(base,name,s,target,0);source=fixed_source_uid(base,name,s['state_uid'],zero,0);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(source,)).fetchone();assert row is not None
            evidence=actual(dict(row));assert evidence is not None;rows.append(c);profiles[cp['controller_uid']]=cp;states[s['state_uid']]=s;checks.append(dict(rollout_uid=c['rollout_uid'],role='zero_prefix',source=evidence,prefix_stop=200))
        for name in (adapter,'transformer_bc0','centralized_mppi0','distributed_cross_entropy0'):
            obj=next(x for x in objects if x['controller']==name);s=obj['state'];a=obj['selected_action'];target=action(a['eta_a'],a['eta_b'],'progress',700,'arbitrary_progress')
            if np.array_equal(target['eta_a'],target['eta_b']):target=action(obj['proposals'][0],obj['proposals'][1],'progress',700,'arbitrary_progress')
            c,cp=case(base,name,s,target,0);rows.append(c);profiles[cp['controller_uid']]=cp;states[s['state_uid']]=s;checks.append(dict(rollout_uid=c['rollout_uid'],role='online_offline'))
    assert len({c['rollout_uid'] for c in rows})==8
    register('equivalence_basic',dict(base,states=list(states.values()),phase='equivalence'),rows,profiles);write(HERE/'EQUIVALENCE_BASIC_CHECKS.json',checks)
    seal('equivalence_basic_EXECUTION_FROZEN.receipt.json',[HERE/'equivalence_basic_protocol.json',HERE/'equivalence_basic_CASES.json',HERE/'equivalence_basic_PREFLIGHT.receipt.json',HERE/'EQUIVALENCE_BASIC_CHECKS.json',HERE/'equivalence.py',HERE/'actions.py',*[HERE/n for n in ('new_execution.py','mixed_execution.py','mixed_runtime.py','progress.py','core.py','common.py')]],before_submission=True)
def run_basic(worker):
    from new_execution import execute
    rows=read(HERE/'equivalence_basic_MISSING_CASES.json');p=read(HERE/'equivalence_basic_protocol.json');cache={}
    for i in range(worker,len(rows),8):
        c=rows[i];raw=HERE/'equivalence_basic/raw'/f"{c['rollout_uid']}.json";assert not raw.exists();out=execute(c,p,cache,'equivalence_basic');write(raw,out);cache_writer.append_journal([out],EXP,f'equivalence_basic_{worker}')
def collect_basic():
    from progress import offline_switch
    rows=read(HERE/'equivalence_basic_CASES.json');checks=read(HERE/'EQUIVALENCE_BASIC_CHECKS.json');result=[]
    with rollout_db.connect(True) as db:
        for c,check in zip(rows,checks):
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None;evidence=actual(dict(row));assert evidence is not None;out=evidence['record']
            with np.load(evidence['trace_path'],allow_pickle=False) as z:
                positions=z['positions'];fields={}
                if check['role']=='zero_prefix':
                    with np.load(check['source']['trace_path'],allow_pickle=False) as old:
                        count=min(len(positions),len(old['positions']),check['prefix_stop']+1);fields['prefix_bytes']=positions[:count].tobytes()==old['positions'][:count].tobytes();fields['velocity_prefix_bytes']=z['velocities'][:count].tobytes()==old['velocities'][:count].tobytes()
                else:fields['online_offline_equal']=out['progress_switch_step']==offline_switch(positions,z['goals'],700,.9681620884830543)
            result.append(dict(controller=c['base_controller_name'],type=c['family'],role=check['role'],checks=fields,passed=all(fields.values())))
    assert all(r['passed'] for r in result),result
    write(HERE/'EXECUTION_EQUIVALENCE.json',dict(utc=utc(),passed=True,checks=result,zero_prefix_checks=4,arbitrary_progress_checks=4,progress_human_reproductions=8,all_zero_reproductions=8))
    seal('EXECUTION_EQUIVALENCE.receipt.json',[HERE/'EXECUTION_EQUIVALENCE.json',HERE/'equivalence.py',HERE/'equivalence_basic_COMPLETE.receipt.json',HERE/'EXECUTION_REPRODUCTION.receipt.json'],no_core_execution_change=True)
if __name__=='__main__':run_basic(int(sys.argv[2])) if sys.argv[1]=='run_basic' else globals()[sys.argv[1]]()
