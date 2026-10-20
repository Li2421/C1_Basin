from core import *
from shared_rollout_db.src.cache_writer import append_journal
from audit_utils import source_inventory
from episode_extended import episode,CountedRuntime
from tuned import TunedRuntime
from student import Student
import jax
jax.config.update('jax_enable_x64',False)
def worker(phase,index,workers):
    from shared_rollout_db.src import cache_writer
    assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db');assert cache_writer.ROOT/'journals'==ROOT/'shared_rollout_db/journals'
    assert (HERE/'REGATE_FROZEN.receipt.json').exists() and read(HERE/'COMPONENT_TESTS.json')['status']=='PASS';assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists()
    if phase=='main':assert (HERE/'CHOICES_FROZEN.receipt.json').exists() and (HERE/'RHR_FROZEN.receipt.json').exists()
    p=read(HERE/f'{phase}_protocol.json');cache={};bases={};count=0
    for c in read(HERE/f'{phase}_MISSING_CASES.json')[index::workers]:
        out=HERE/phase/'raw'/f"{c['rollout_uid']}.json";assert not out.exists(),'No retry or overwrite of completed/unknown trial'
        n=c['base_controller_name'];i=c['student_index']
        if n not in bases:
            bases[n]=Student(i);assert sha(bases[n].checkpoint)==read(HERE/'STUDENTS.json')[str(i)]['checkpoint_sha256']
        speed=c.get('rhr_speed');trigger=c.get('rhr_trigger');tuned=speed is not None and (speed,trigger)!=(.15,40);cache[n]=TunedRuntime(bases[n],speed,trigger) if tuned else CountedRuntime(bases[n]);r=episode(c,p,cache,phase)
        if tuned:
            path=Path(r['trace_path'])
            with np.load(path) as z:values={k:z[k] for k in z.files}
            positions=values['positions'];velocities=values['velocities'];goals=values['goals'];counts=np.zeros(4,int);active=[]
            for t in range(len(positions)-1):
                active.append((counts>=trigger)&(np.linalg.norm(goals-positions[t],axis=1)>bases[n].config.goal_tolerance));mask=(np.linalg.norm(velocities[t+1],axis=1)<.01)&(np.linalg.norm(goals-positions[t+1],axis=1)>bases[n].config.goal_tolerance);counts=np.where(mask,counts+1,0)
            values['rhr_active']=np.asarray(active,bool);meta=json.loads(str(values['metadata_json']));meta['rhr_params']=dict(speed=speed,trigger_steps=trigger);values['metadata_json']=np.array(json.dumps(meta));np.savez_compressed(path,**values);r['trace_sha256']=sha(path)
        s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);r.update(experiment_uid=p['experiment_uid'],cache_identity_schema='registered_exact_uid_v1',scenario_uid=p['base_profiles'][n]['payload']['scenario_uid'],flow_checkpoint_sha256=p['base_profiles'][n]['payload']['flow_checkpoint_sha256'],eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],phase_protocol_sha256=sha(HERE/f'{phase}_protocol.json'),rhr_params=dict(speed=speed,trigger_steps=trigger) if speed is not None else None);write(out,r);journal=append_journal([r],p['experiment_uid'],f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');count+=1
        if count%16==0:print(phase,index,count,flush=True)
    write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count));source_inventory(f'imports_{phase}_{index}.json')
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
