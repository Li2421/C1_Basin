"""Audit actual research imports; do not traverse any unrelated study directory."""
from common import *
from controllers import make
from source_training import flow_execute
from validation_v2.adapters import FourRingRuntime
from learned import LearnedNominal
from fast_astar import FastPrioritizedAStar
from expanded import old
# Instantiate all actual nominal classes/checkpoints, without physical steps or labels.
import jax
for filename in ['source_training_protocol.json','main_protocol.json']:
 if not (HERE/filename).exists():continue
 protocol=read(HERE/filename);cache={}
 for name,profile in protocol['profiles'].items():
  payload=profile['payload']
  if name=='base_u_v10_local_macflow':
   rt=FourRingRuntime('ring_exchange');jax.config.update('jax_enable_x64',False)
  elif payload.get('controller_class')=='ASTAR':rt=FastPrioritizedAStar(payload['nominal_parameters'],protocol['native_manifest'])
  else:rt=make(name,protocol,cache)
  env=rt.make_env(protocol['states'][0]);rt.observation(env) if payload.get('controller_class','Flow') in ['Flow','G','deterministic','mixture','diffusion'] else None
  assert env.step_count==0
roots=[ROOT,FIELD,Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')];paths=set()
for m in list(sys.modules.values()):
 value=getattr(m,'__file__',None)
 if not value:continue
 path=Path(value).resolve()
 if path.suffix=='.py' and path.exists() and any(path.is_relative_to(root) for root in roots) and not path.is_relative_to(HERE):paths.add(path)
begin=read(HERE/'INTEGRITY_BEGIN.json');historical=read(D/'axis2_controller_v1/INTEGRITY_BEGIN.json');base={**begin['source_py'],**begin['poc_py']};prior_imports=read(D/'blackbox_addendum_v1/imports_right_hand_tune_0.json');hist={**historical.get('poc_importable_py',{}),**historical.get('MACFlow_py',{}),**prior_imports['sources']};rows=[]
for path in sorted(paths):
 current=sha(path);pre=base.get(str(path));prior=hist.get(str(path));rows.append(dict(path=str(path),current_sha256=current,round_begin_sha256=pre,historical_prior_sha256=prior,baseline='current round initial snapshot' if pre else 'earlier immutable study source snapshot' if prior else 'newly enumerated import',matches_baseline=current==(pre or prior) if pre or prior else None))
write(HERE/'IMPORTED_SOURCES.json',dict(utc=utc(),sources=rows,historical_baseline_source=str(D/'axis2_controller_v1/INTEGRITY_BEGIN.json'),historical_baseline_sha256=sha(D/'axis2_controller_v1/INTEGRITY_BEGIN.json'),uncovered=[x['path'] for x in rows if x['matches_baseline'] is None],mismatches=[x for x in rows if x['matches_baseline'] is False],historical_baseline_scope_explicit=True,additional_historical_baseline_source=str(D/'blackbox_addendum_v1/imports_right_hand_tune_0.json'),additional_historical_baseline_sha256=sha(D/'blackbox_addendum_v1/imports_right_hand_tune_0.json'),all_nominal_types_and_checkpoints_instantiated_without_physical_steps=True))
print('research imports',len(rows),'historical supplemental',sum(x['baseline'].startswith('earlier') for x in rows),'uncovered',sum(x['matches_baseline'] is None for x in rows))
