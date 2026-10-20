from core import *
from shared_rollout_db.src.cache_writer import append_journal
from audit_utils import source_inventory
from episode_frozen import episode
from student import Student
from controls import Control
from fair_rhr import FairRuntime
import jax
jax.config.update('jax_enable_x64',False)
def worker(phase,index,workers):
    assert read(HERE/'COMPONENT_TESTS.json')['status']=='PASS'
    for receipt in ['PREREGISTRATION.receipt.json','IMPLEMENTATION_FROZEN.receipt.json',f'{phase}_PREFLIGHT.receipt.json']+(['RHR_FROZEN.receipt.json'] if phase=='fair' else []):
        for p,h in read(HERE/receipt)['files'].items():assert sha(p)==h,p
    p=read(HERE/f'{phase}_protocol.json');bases={};count=0
    for c in read(HERE/f'{phase}_MISSING_CASES.json')[index::workers]:
        out=HERE/phase/'raw'/f"{c['rollout_uid']}.json";assert not out.exists(),'Never retry or overwrite unknown/completed evidence';n=c['base_controller_name']
        if n not in bases:bases[n]=Student(c['student_index']) if c['kind']=='student' else Control(c['kind'])
        base=bases[n];assert sha(base.checkpoint)==p['base_profiles'][n]['payload']['flow_checkpoint_sha256'];rt=FairRuntime(base,c['rhr_rule']) if c['rhr_rule'] else base;cache={n:rt};r=episode(c,p,cache,phase)
        if c['rhr_rule']:
            path=Path(r['trace_path'])
            with np.load(path) as z:values={k:z[k] for k in z.files}
            values['rhr_active']=np.asarray(rt.active_log[:r['physical_steps']],bool).reshape(-1,4);values['neighbor_counts']=np.asarray(rt.neighbor_log[:r['physical_steps']],int).reshape(-1,4);meta=json.loads(str(values['metadata_json']));meta['rhr_rule']=c['rhr_rule'];meta['display_pipeline']='D-local-RHR';values['metadata_json']=np.array(json.dumps(meta));np.savez_compressed(path,**values);r['trace_sha256']=sha(path)
        s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);r.update(experiment_uid=p['experiment_uid'],cache_identity_schema='registered_exact_uid_v1',scenario_uid=p['base_profiles'][n]['payload']['scenario_uid'],flow_checkpoint_sha256=p['base_profiles'][n]['payload']['flow_checkpoint_sha256'],eta=c['eta_a'],future_index=c['seed'],initial_positions=s['physical']['positions'],initial_velocities=s['physical']['velocities'],goals=s['physical']['goals'],phase_protocol_sha256=sha(HERE/f'{phase}_protocol.json'),rhr_rule=c['rhr_rule']);write(out,r);journal=append_journal([r],p['experiment_uid'],f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');count+=1
        if count%16==0:print(phase,index,count,flush=True)
    write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory(f'imports_{phase}_{index}.json')
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
