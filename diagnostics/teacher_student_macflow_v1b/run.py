from core import *
from shared_rollout_db.src.cache_writer import append_journal
from audit_utils import source_inventory
from episode_extended import episode,CountedRuntime
from data import teacher
from student import Student
import jax
jax.config.update('jax_enable_x64',False)
def worker(phase,index,workers):
    import shared_rollout_db.src.cache_writer as official_writer
    assert Path(official_writer.__file__).resolve()==ROOT/'shared_rollout_db/src/cache_writer.py'
    assert official_writer.ROOT==ROOT/'shared_rollout_db'
    assert read(HERE/'ROUND28_REPLAY_GATE.json')['passed'];assert read(HERE/'COMPONENT_TESTS.json')['status']=='PASS';assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists()
    p=read(HERE/f'{phase}_protocol.json');writepath=HERE/'main_protocol.json'
    # episode's exact frozen bookkeeping reads main_protocol.json for all non-gate phases.
    assert writepath.exists()
    cache={};count=0
    for c in read(HERE/f'{phase}_MISSING_CASES.json')[index::workers]:
        out=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
        if out.exists():continue
        n=c['base_controller_name']
        if n not in cache:cache[n]=CountedRuntime(teacher(c['student_index']) if c['teacher'] else Student(c['student_index']))
        r=episode(c,p,cache,phase);s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);r.update(experiment_uid=p['experiment_uid'],cache_identity_schema='registered_exact_uid_v1',scenario_uid=p['base_profiles'][n]['payload']['scenario_uid'],flow_checkpoint_sha256=p['base_profiles'][n]['payload']['flow_checkpoint_sha256'],eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],phase_protocol_sha256=sha(HERE/f'{phase}_protocol.json'))
        write(out,r);journal=append_journal([r],p['experiment_uid'],f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');count+=1
        if count%8==0:print(phase,index,count,c['base_controller_name'],c['arm'],flush=True)
    write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count));source_inventory(f'imports_{phase}_{index}.json')
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
