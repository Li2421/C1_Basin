"""Same frozen episode and postprocessing as run.py; explicit future-case transfer."""
from core import *
from shared_rollout_db.src.cache_writer import append_journal
from audit_utils import source_inventory
from episode_frozen import episode
from student import Student
from controls import Control
import jax
jax.config.update('jax_enable_x64',False)
def worker(index,workers):
    for receipt in ['PREREGISTRATION.receipt.json','IMPLEMENTATION_FROZEN.receipt.json','C_PREFLIGHT.receipt.json','HANDOFF_PREFLIGHT.receipt.json']:
        for p,h in read(HERE/receipt)['files'].items():assert sha(p)==h,p
    p=read(HERE/'C_protocol.json');bases={};count=0
    for c in read(HERE/'HANDOFF_CASES.json')[index::workers]:
        out=HERE/'C/raw'/f"{c['rollout_uid']}.json"
        if out.is_symlink():assert out.resolve()==HERE/'handoff/RESERVED.json','Reservation must be this dispatch plan'
        else:assert not out.exists(),'No completed or unknown rollout may be run twice'
        n=c['base_controller_name'];assert c['rhr_rule'] is None
        if n not in bases:bases[n]=Student(c['student_index']) if c['kind']=='student' else Control(c['kind'])
        base=bases[n];assert sha(base.checkpoint)==p['base_profiles'][n]['payload']['flow_checkpoint_sha256'];r=episode(c,p,{n:base},'C')
        s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);r.update(experiment_uid=p['experiment_uid'],cache_identity_schema='registered_exact_uid_v1',scenario_uid=p['base_profiles'][n]['payload']['scenario_uid'],flow_checkpoint_sha256=p['base_profiles'][n]['payload']['flow_checkpoint_sha256'],eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],phase_protocol_sha256=sha(HERE/'C_protocol.json'),rhr_rule=None);write(out,r);journal=append_journal([r],p['experiment_uid'],f'C_handoff_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');count+=1
        if count%16==0:print('handoff',index,count,flush=True)
    write(HERE/'handoff'/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory(f'imports_handoff_{index}.json')
if __name__=='__main__':worker(int(sys.argv[1]),int(sys.argv[2]))
