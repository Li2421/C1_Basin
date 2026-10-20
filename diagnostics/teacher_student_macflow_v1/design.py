from core import *
from audit_utils import source_inventory
from registration import *
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
import jax
jax.config.update('jax_enable_x64',False)
def sig(p):
    if not p:return None
    try:
        vals=[]
        for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
            x=np.asarray(next(p[k] for k in keys if k in p),np.float64).copy();assert x.shape==(4,2);x[x==0]=0;vals.append(x.tobytes().hex())
        return uid('physical',dict(arrays=vals,timestep=float(p.get('timestep',0))))
    except (StopIteration,AssertionError,TypeError):return None

def initial():
    assert read(HERE/'NOISE_GATE.json')['status']=='PASS'
    with connect(True) as db:
        historical=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
    ids={r['state_uid'] for r in historical};content={r['content_hash'] for r in historical};signatures=set();unresolved=0
    for r in historical:
        key=sig(json.loads(r['physical_state_json']) if r['physical_state_json'] else None)
        if key is None:unresolved+=1
        else:signatures.add(key)
    refs=read(D/'decentralized_pipeline_v1/PILOT_RESERVED_STATES.json')+read(D/'decentralized_pipeline_v1/EVAL_STATES.json');signatures.update(sig(s['physical']) for s in refs)
    native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');pilot=[];rejected=[]
    for seed in range(202610300000,202610301000):
        s=generate('round30_pilot',1,seed,native)[0];key=sig(s['physical'])
        if s['state_uid'] in ids or s['content_hash'] in content or key in signatures:rejected.append(seed);continue
        s['study_index']=len(pilot);pilot.append(s);signatures.add(key)
        if len(pilot)==16:break
    assert len(pilot)==16
    write(HERE/'PILOT_STATES_ORIGINAL.json',pilot);pilot=[normalized_state(s) for s in pilot];evaluate=[normalized_state(s) for s in read(D/'decentralized_pipeline_v1/EVAL_STATES.json')];write(HERE/'PILOT_STATES.json',pilot);write(HERE/'EVAL_STATES.json',evaluate)
    write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=True,DB_rows=len(historical),unresolved_physical=unresolved,pilot_seeds=[s['metadata']['draw_seed'] for s in pilot],rejected=rejected,round28_reserved_seed_ranges=[[202610280000,202610280007],[202610281000,202610281999]],state_uid_representation='new exact metadata representation without redundant timestep=0, same original eval positions/velocities/goals and RNG namespace; original_state_uid preserves pairing',evaluation_states_generated=0))
    register('initial',[exp_row(phase) for phase in ['pilot','probe','main','teacher']]+[state_row(s) for s in pilot+evaluate]);source_inventory('imports_design.json')
if __name__=='__main__':globals()[sys.argv[1]]()
