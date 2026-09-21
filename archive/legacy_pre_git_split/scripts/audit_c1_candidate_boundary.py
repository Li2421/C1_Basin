"""Controlled risk-function boundary probe, not a physical rollout."""
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.risk.soft_activity import soft_risk_diagnostics, SoftRiskConfig


def main():
    jax.config.update('jax_enable_x64', True)
    target = ROOT/'results/c1_candidate_boundary_probe.json'
    if target.exists():
        raise FileExistsError(target)
    cfg = SoftRiskConfig()
    force = jnp.array([[-.1,-.1],[0.,0.]])
    blocks = jnp.array([[[1.,0.],[0.,0.]],[[0.,1.],[0.,0.]]])
    rows=[]
    cutoff=cfg.rho+cfg.candidate_sigmas*cfg.tau_h
    for eps in (1e-4,1e-6,1e-8,1e-10):
        pair=[]
        for sign in (-1,1):
            d=soft_risk_diagnostics(force,blocks,jnp.array([0.,cutoff+sign*eps]),jnp.zeros(2),cfg)
            pair.append(dict(h2=cutoff+sign*eps,risk=float(d['risk']),
                agent0_margin=float(d['margins'][0]),cone_type=int(d['cone_types'][0]),
                activity=float(d['local_activity'][0])))
        rows.append(dict(epsilon=eps,endpoints=pair,jump=pair[0]['risk']-pair[1]['risk']))
    report=dict(scope='synthetic independent risk-function inputs; not evidence of physical frequency',
                force=force.tolist(),blocks=blocks.tolist(),h1=0.,slack=[0.,0.],risk=cfg.__dict__,rows=rows)
    assert all(x['jump']>.49 for x in rows)
    assert all(x['endpoints'][0]['activity']==x['endpoints'][1]['activity'] for x in rows)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
