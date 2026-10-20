"""Eight t0 means, eight actual prefix/recompute gates, exact preflight."""
from common import *
from actions_v3 import action,case,fixed_source_uid
from records import actual
from registry import register

def design():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'];base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');old=D/'crosstype4_no_repair_schedule_diagnosis_v1'
    with np.load(old/'STATE_INPUTS.npz',allow_pickle=False) as z:proposals=z['proposals']
    names=[n for n in base['controller_order'] if base['types'][n]=='Flow'][:6]+['transformer_bc0','centralized_mppi0'];rows=[];profiles={};checks=[]
    with rollout_db.connect(True) as db:
        for i,name in enumerate(names):
            found=None
            for si,s in enumerate(states):
                eta=proposals[si,0];rid=fixed_source_uid(base,name,s['state_uid'],eta,0);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                if row is None or row['episode_length']<601:continue
                source=actual(dict(row))
                if source is None:continue
                found=(si,s,eta,source);break
            assert found is not None,name
            si,s,eta,source=found;mode='fixed' if i%2==0 else 'progress';a=action(eta,proposals[si,0],mode,500,'requery_prefix','requery');c,cp=case(base,name,s,a,0,si);rows.append(c);profiles[cp['controller_uid']]=cp;checks.append(dict(case=c,source=source,t0_state=states[i],t0_expected_mean=proposals[i,0]))
    register('requery_gates_v3',dict(base,states=states,phase='equivalence'),rows,profiles);assert len(read(HERE/'requery_gates_v3_MISSING_CASES.json'))==8
    write(HERE/'REQUERY_GATE_CHECKS.json',checks);seal('requery_gates_v3_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('requery_gates_v3_protocol.json','requery_gates_v3_CASES.json','requery_gates_v3_PREFLIGHT.receipt.json','REQUERY_GATE_CHECKS.json','requery_gates_v3.py','actions.py','actions_v3.py','execution_v3.py','requery_client.py','requery_helper.py','snapshot_conditioning.py','given_actions.py','execution.py','requery_generator_v2.py','requery_execution_v3.py','legacy_conditioning.py','new_execution.py','mixed_runtime.py')],before_submission=True)
def run(worker):
    from execution_v3 import execute
    from requery_client import FrozenMean
    for p,h in read(HERE/'requery_gates_v3_EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    check=read(HERE/'REQUERY_GATE_CHECKS.json')[worker];fm=FrozenMean();s=check['t0_state'];out0=fm.query(s['state_uid'],s['physical']);equal=np.asarray(out0['eta_b'],np.float64).tobytes()==np.asarray(check['t0_expected_mean'],np.float64).tobytes();write(HERE/f'requery_gates_v3/t0_{worker}.json',dict(utc=utc(),state_uid=s['state_uid'],bytes_equal=equal,query=out0,expected=check['t0_expected_mean']));assert equal
    c=check['case'];path=HERE/'requery_gates_v3/raw'/f"{c['rollout_uid']}.json";assert not path.exists();out=execute(c,read(HERE/'requery_gates_v3_protocol.json'),{'_frozen_mean':fm},'requery_gates_v3');write(path,out);cache_writer.append_journal([out],EXP,f'requery_gates_v3_{worker}')
def collect():
    from requery_client import FrozenMean
    fm=FrozenMean();results=[]
    for i,check in enumerate(read(HERE/'REQUERY_GATE_CHECKS.json')):
        c=check['case'];r=read(HERE/'requery_gates_v3/raw'/f"{c['rollout_uid']}.json");source=check['source'];fields=dict(t0_mean=read(HERE/f'requery_gates_v3/t0_{i}.json')['bytes_equal']);q=r['late_eta_requery'];fields['requery_executed']=q is not None
        if q is not None:
            stop=q['switch_step'];recomputed=fm.query(c['state_uid'],q['physical']);fields['late_eta_recomputed_bytes']=np.asarray(recomputed['eta_b'],np.float64).tobytes()==np.asarray(q['eta_b'],np.float64).tobytes()
            with np.load(source['trace_path'],allow_pickle=False) as a,np.load(r['trace_path'],allow_pickle=False) as b:
                for key,end in [('positions',stop+1),('velocities',stop+1),('actions',stop),('nominal_actions',stop)]:fields[key+'_prefix']=a[key][:end].dtype==b[key][:end].dtype and a[key][:end].shape==b[key][:end].shape and a[key][:end].tobytes()==b[key][:end].tobytes()
                fields['trace_eta_stored']=b['requery_eta_b'].tobytes()==np.asarray(q['eta_b'],np.float64).tobytes()
        results.append(dict(controller=c['base_controller_name'],type=c['family'],checks=fields,passed=all(fields.values())))
    write(HERE/'REQUERY_EQUIVALENCE_v3.json',dict(utc=utc(),passed=all(r['passed'] for r in results),checks=results,t0_states=8,actual_prefixes=8));seal('REQUERY_EQUIVALENCE_v3.receipt.json',[HERE/n for n in ('REQUERY_EQUIVALENCE_v3.json','requery_gates_v3_COMPLETE.receipt.json','requery_gates_v3_EXECUTION_FROZEN.receipt.json')]);assert all(r['passed'] for r in results)
if __name__=='__main__':run(int(sys.argv[2])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
