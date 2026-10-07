"""Diagnostic invariants and a matched complete P0 pipeline check."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np
from shared_control.diagnostic_corrector import DiagnosticCorrector
from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import correction,raw_ortho_flow
from diagnostics.double_bottleneck_eta_basis_redesign.tools.run_rollouts import run_one,DATASETS,CHECKPOINT
from diagnostics.double_bottleneck_eta3_full_sobol.tools.run_long_v2 import run_one as reference_run
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.flowbc_4a_agent import load_checkpoint

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'diagnostics/double_bottleneck_eta_basis_redesign'
rng=np.random.default_rng(927)
scale=json.loads((OUT/'P1_SCALE.json').read_text())['scale']
for _ in range(100):
    p,g,u=rng.normal(size=(3,4,2)); eta=rng.normal(size=3)
    assert np.array_equal(correction('P0-3D',eta,p,g,u,.5,scale),DiagnosticCorrector(eta)(p,g,u,.5)[0])
    c=correction('P1-OrthoFlow3',eta,p,g,u,.5,scale)
    permutation=rng.permutation(4)
    assert np.allclose(c[permutation],correction('P1-OrthoFlow3',eta,p[permutation],g[permutation],u[permutation],.5,scale),atol=1e-12)
    rot=np.array([[0.,-1.],[1.,0.]])
    assert np.allclose(c@rot,correction('P1-OrthoFlow3',eta,p@rot,g@rot,u@rot,.5,scale),atol=1e-12)
assert np.array_equal(raw_ortho_flow(np.zeros((4,2)),np.ones((4,2)),.5),np.ones((4,2)))
goal=np.tile([.5,0.],(4,1));u=goal.copy()
assert np.linalg.norm(raw_ortho_flow(goal,u,.5))<1e-10
e=json.loads((OUT/'pilot_catalog.json').read_text())['episodes'][0]
point=json.loads((ROOT/'diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json').read_text())['points'][0]
job={**e,**point,'representation':'P0-3D','stage':'equivalence','job_id':'p0_pipeline_equivalence','eta_index':0}
d=FlowBC4ADataset(DATASETS[e['set']],'val',seed=45 if e['set'].startswith('existing') else 46)
policy,_=load_checkpoint(CHECKPOINT,d.environment_fingerprint);episode=d.by_family[e['family_id']][0]
a=run_one(policy,d,episode,job,scale);b=reference_run(policy,d,episode,job)
fields=('success','outcome','episode_steps','wall_collision','agent_collision','minimum_wall_clearance','minimum_agent_clearance','final_goal_errors')
assert all(a[k]==b[k] for k in fields)
(OUT/'implementation_checks.json').write_text(json.dumps({'random_invariant_trials':100,'canonical_p0_exact':True,
    'permutation_equivariant':True,'rotation_equivariant':True,'zero_goal_fallback_finite':True,
    'aligned_flow_has_no_invented_lateral_vector':True,'complete_p0_rollout_exact_match':True,
    'episode_id':e['episode_id'],'eta_index':0,'matched_fields':list(fields),'auxiliary_rollouts':2},indent=2)+'\n')
print('All basis invariants and full P0 pipeline equivalence passed')
