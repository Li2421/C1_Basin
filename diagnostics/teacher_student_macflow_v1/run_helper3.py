"""Extra shards for untouched tail cases; identical frozen execution, no rollout retries."""
from core import *
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src import cache_writer
from audit_utils import source_inventory
from episode_extended import episode,CountedRuntime
from student import Student
import jax
jax.config.update('jax_enable_x64',False)
def worker(index):
    assert Path(cache_writer.__file__).resolve()==ROOT/'shared_rollout_db/src/cache_writer.py'
    frozen=read(HERE/'MAIN_IMPLEMENTATION_FROZEN.receipt.json')
    for path,h in frozen['files'].items():assert sha(path)==h
    p=read(HERE/'main_protocol.json');plan=read(HERE/'HELPER3_CASES.json');guard=[HERE/'main/raw'/f"{c['rollout_uid']}.json" for c in read(HERE/'HELPER3_GUARDS.json')];cache={};count=0;stopped=False
    for c in plan[index::8]:
        # End between episodes, two controller blocks before the original workers
        # can reach these tail cases; never interrupt or repeat an episode.
        if any(f.exists() for f in guard):stopped=True;break
        out=HERE/'main/raw'/f"{c['rollout_uid']}.json"
        if out.exists():continue
        n=c['base_controller_name']
        if n not in cache:cache[n]=CountedRuntime(Student(c['student_index']))
        r=episode(c,p,cache,'main');s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);r.update(experiment_uid=p['experiment_uid'],cache_identity_schema='registered_exact_uid_v1',scenario_uid=p['base_profiles'][n]['payload']['scenario_uid'],flow_checkpoint_sha256=p['base_profiles'][n]['payload']['flow_checkpoint_sha256'],eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],phase_protocol_sha256=sha(HERE/'main_protocol.json'),execution_driver_sha256=sha(__file__),producer='helper_tail_final')
        write(out,r);append_journal([r],p['experiment_uid'],f'helper_tail_final_{index}');count+=1
        if count%8==0:print('helper',index,count,n,c['arm'],flush=True)
    write(HERE/'main'/f'helper3_worker_{index}.json',dict(utc=utc(),new=count,finished=True,stopped_between_episodes=stopped));source_inventory(f'imports_main_helper3_{index}.json')
if __name__=='__main__':worker(int(sys.argv[1]))
