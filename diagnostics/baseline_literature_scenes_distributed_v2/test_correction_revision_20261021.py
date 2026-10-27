"""No new research rollouts: algebra checks and32 full saved-episode replays."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from correction_revision_20261021 import *
from distributed_safety import local_basis
from shared_control.basis_families import get_basis_family
from states import config
import numpy as np,time

def same(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});os.nice(10)
 verify_receipt('CORRECTION_REVISION_DEFINITION_20261021.receipt.json')
 started=time.monotonic();rng=np.random.default_rng(2026102109);basis=get_basis_family('orthoflow3');tests={};max_permutation_error=0.
 for n in (2,3,5,14):
  for repeat in range(8):
   p=rng.normal(size=(n,2));g=rng.normal(size=(n,2));v=rng.normal(size=(n,2));speed=.37;eta=rng.normal(size=3)
   for i in range(n):assert same(distributed_fields(p,g,v,i,speed,.08,.4,True),local_basis(p,g,v,i,np.inf,speed))
   original=basis.compute(p,g,v,speed);revised=central_fields(p,g,v,speed,.08,.4,basis,True)
   assert all(same(a,b) for a,b in zip(original.values,revised.values));assert same(original.correction(eta),revised.correction(eta))
   assert not contact_mask(p,p,.08,.4).any()
   assert not any(np.any(distributed_fields(p,p,v,i,speed,.08,.4)[2]) for i in range(n))
   assert not np.any(central_fields(p,p,v,speed,.08,.4,basis).values[2])
   permutation=rng.permutation(n);inverse=np.argsort(permutation)
   assert np.array_equal(contact_mask(p,g,.08,.4),contact_mask(p[permutation],g[permutation],.08,.4)[inverse])
   a=np.asarray([distributed_fields(p,g,v,i,speed,.08,.4) for i in range(n)])
   b=np.asarray([distributed_fields(p[permutation],g[permutation],v[permutation],i,speed,.08,.4) for i in range(n)])[inverse]
   err=float(np.max(np.abs(a-b)));max_permutation_error=max(max_permutation_error,err);assert np.allclose(a,b,rtol=0,atol=2e-15)
   a=central_fields(p,g,v,speed,.08,.4,basis);b=central_fields(p[permutation],g[permutation],v[permutation],speed,.08,.4,basis)
   for aa,bb in zip(a.values,b.values):assert np.allclose(aa,bb[inverse],rtol=0,atol=2e-15)
 tests.update(force_all_open_original_fields_and_corrections_bitwise=True,all_arrived_relative_field_zero=True,permutation_invariant_with_original_float_reduction_roundoff=True,maximum_local_permutation_error=max_permutation_error)
 p=np.asarray([[.25,0.],[0.,.25],[0.,-.25]]);g=p.copy();g[0]=[-.25,0.];v=np.zeros_like(p);cfg=config('lit_narrow_passage');mask=contact_mask(p,g,cfg.goal_tolerance,contact_radius(cfg));assert mask.all()
 for i in range(3):assert same(distributed_fields(p,g,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg)),local_basis(p,g,v,i,np.inf,cfg.max_speed))
 assert all(same(a,b) for a,b in zip(central_fields(p,g,v,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg),basis).values,basis.compute(p,g,v,cfg.max_speed).values))
 tests['narrow_paper_three_agents_contact_and_original_basis_bitwise']=True
 R=.375125;p=np.asarray([[0.,0.],[R,0.]]);g=p+np.asarray([1.,0.]);assert contact_mask(p,g,.08,R).all();p[1,0]=R+1e-12;assert not contact_mask(p,g,.08,R).any()
 tests['inclusive_contact_boundary_and_both_arrived_exception']=True
 write(HERE/'CORRECTION_REVISION_ALGEBRA_TESTS_20261021.json',dict(utc=utc(),passed=True,tests=tests,new_rollouts=0))
 from runtime_correction_revised_20261021 import run as local_or_crowd
 from central_correction_revised_20261021 import run as joint
 kinds=('centralized_sampling','centralized_mppi','gradient_mpc','cross_entropy')
 cases=[]
 for scene in NAMES:
  for layer in ('central','distributed'):
   stage=f'{layer}_baseline_{scene}';originals=read(HERE/f'{stage}_CASES.json')
   source={row['rollout_uid']:row for entry in read(HERE/('analysis_'+stage)/'GROUP_INDEX.json') for row in read(entry['file'])}
   for j,kind in enumerate(kinds):
    requested_n=(2,6,10,14)[j] if scene=='lit_crowded_random' else originals[0]['state']['N']
    c=next(c for c in originals if c['controller_payload']['nominal']['type_code']==kind and c['seed']==0 and c['state']['N']==requested_n)
    row=source[c['rollout_uid']];cases.append(dict(case=c,source=row,scene=scene,layer=layer))
 assert len(cases)==32 and len({x['case']['rollout_uid'] for x in cases})==32
 write(HERE/'CORRECTION_REVISION_REPLAY_PLAN_20261021.json',dict(utc=utc(),cases=cases,scope='8 saved safety-only episodes per scene;4 central+4 distributed; full frozen nominal actors, projections, plant and terminal events; no journals or research data generated',source_runtime_diff='CORRECTION_REVISION_SOURCE_DIFF_20261021.json'))
 results=[]
 for item in cases:
  c=item['case'];row=item['source'];path=Path(row['path']);assert sha(path)==row['sha256'];raw=read(path);assert not raw['numerical_failure'];assert sha(raw['trace_path'])==raw['trace_sha256']
  assert not np.any(c['eta_a']) and not np.any(c['eta_b']) and c['theta'] is None
  run=joint if item['layer']=='central' and item['scene']!='lit_crowded_random' else local_or_crowd
  result,arrays=run(c)
  with np.load(raw['trace_path']) as z:
   checks={k:same(z[k],arrays[k]) for k in z.files if k!='step_seconds'}
  terminal={k:result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length','switch_step')}
  proof=dict(scene=item['scene'],layer=item['layer'],controller=c['controller_payload']['nominal']['controller'],source=str(path),source_sha256=row['sha256'],trace_sha256=raw['trace_sha256'],arrays=checks,terminal=terminal,nominal_actor_recomputed=True,steps=result['episode_length'],seconds=result['elapsed_seconds'],passed=all(checks.values()) and all(terminal.values()))
  results.append(proof);write(HERE/'CORRECTION_REVISION_REPLAY_PROGRESS_20261021.json',dict(utc=utc(),completed=len(results),planned=32,results=results))
  assert proof['passed'],proof
 write(HERE/'CORRECTION_REVISION_UNIT_GATES_20261021.json',dict(utc=utc(),passed=True,tests=tests,replays=results,all32_saved_full_actor_replays_bitwise=True,new_research_rollouts=0,new_training=0,journals_written=0,seconds=time.monotonic()-started))
 print('PASS: algebra and32 full saved-episode replays; no research rollout or journal',flush=True)
if __name__=='__main__':main()
