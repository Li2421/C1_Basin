"""Bind immutable round10 numerical loop to audited checkpoint-agnostic adapter."""
from core import *
import importlib.util,types
from shared_rollout_db.src import cache_writer
_append=cache_writer.append_journal
cache_writer.append_journal=lambda rows,*a,**k:_append(clean(rows),*a,**k)
spec=importlib.util.spec_from_file_location('round12_frozen_schedule_runner',R9/'run.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)

def make_runtime(name,p,cache):
    if name in cache:return cache[name]
    import jax
    jax.config.update('jax_enable_x64',False)
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    v=p['base_profiles'][name]['payload'];manifest=v.get('adapter_manifest_path',str(FIELD/'assets/diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json'));frame=v.get('observation_frame','local');rt=RingFlowAdapter(v['flow_checkpoint_path'],read(manifest),frame);assert sha(rt.checkpoint)==v['flow_checkpoint_sha256'];assert not jax.config.x64_enabled;original=rt.safety
    def observed(self,env):self.last_safety_tracker=frozen.SafetyTracker(original(env));return self.last_safety_tracker
    rt.safety=types.MethodType(observed,rt);rt.last_safety_tracker=None
    original_env=rt.make_env
    def make_env(self,state):
        env=original_env(state);initial=clean(env.augmented_state());path=HERE/'actual_initial_states'/f"{state['state_uid']}_{v['flow_checkpoint_sha256'][:12]}_{os.getpid()}.json"
        if path.exists():assert read(path)==initial
        else:write(path,initial)
        return env
    rt.make_env=types.MethodType(make_env,rt);cache[name]=rt;return rt
frozen.make_runtime=make_runtime
if __name__=='__main__':
    phase=sys.argv[1];index=int(sys.argv[2]);workers=int(sys.argv[3]);prereg();labels_allowed();p=read(HERE/'protocol.json');assert sha(__file__)==p['io_binding_sha256'];rows=read(HERE/f'{phase}_MISSING_CASES.json');perm=np.random.default_rng(202610131221).permutation(len(rows))
    write(HERE/phase/f'START_AUTHORIZATION_{index}.json',dict(utc=utc(),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),probe_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),selector_receipt_sha256=sha(HERE/'SELECTOR_FROZEN.receipt.json'),phase=phase,worker=index,before_any_worker_rollout=True))
    frozen.execute([rows[int(i)] for i in perm[index::workers]],phase,index)
