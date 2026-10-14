"""IO/runtime composition around unchanged round10's frozen round9 loop."""
from core import *
import importlib.util,types
from shared_rollout_db.src import cache_writer
_append=cache_writer.append_journal
cache_writer.append_journal=lambda records,*a,**k:_append(clean(records),*a,**k)
spec=importlib.util.spec_from_file_location('round11_frozen_schedule_runner',R9/'run.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
original_make=frozen.make_runtime

def make_runtime(name,p,cache):
    if 'blend' not in p['base_profiles'][name]['payload']:return original_make(name,p,cache)
    if name in cache:return cache[name]
    sys.path.insert(0,str(R5))
    from blend_runtime import make
    rt=make(name,dict(controllers={'ring_exchange':p['base_profiles']}),cache);original=rt.safety
    def observed(self,env):self.last_safety_tracker=frozen.SafetyTracker(original(env));return self.last_safety_tracker
    rt.safety=types.MethodType(observed,rt);rt.last_safety_tracker=None;return rt
frozen.make_runtime=make_runtime
if __name__=='__main__':
    phase=sys.argv[1];index=int(sys.argv[2]);workers=int(sys.argv[3]);prereg();labels_allowed();assert (HERE/'SELECTOR_FROZEN.receipt.json').exists();p=read(HERE/'protocol.json');assert sha(__file__)==p['io_binding_sha256'];rows=read(HERE/f'{phase}_MISSING_CASES.json');perm=np.random.default_rng(202610131111).permutation(len(rows));frozen.execute([rows[int(j)] for j in perm[index::workers]],phase,index)
