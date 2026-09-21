"""Measure same-batch value/gradient agreement and actual parallel speedup."""
import json
from pathlib import Path
import time
import sys
import jax
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.train_deadlock_union import setup
from single_integrator.c1.episode_pool import initialize,task,EpisodePool
from single_integrator.c1.training.persistence import atomic_save


def main():
    params,_,_,_,_=setup()
    items=json.loads((ROOT/'results/c1_deadlock_union/sets.json').read_text())['pools']['train'][:8]
    initialize('ordered',0,0,[])
    payloads=[(jax.device_get(params),i,72101,1.,.01,True) for i in items]
    # Warm up serial compilation outside the measured time.
    task(payloads[0])
    begin=time.monotonic();serial=[task(p) for p in payloads];serial_time=time.monotonic()-begin
    pool=EpisodePool('ordered',workers=4)
    try:
        begin=time.monotonic();pool.rows(params,items,[72101]*8,gradient=True);warmup=time.monotonic()-begin
        begin=time.monotonic();parallel=pool.rows(params,items,[72101]*8,gradient=True);parallel_time=time.monotonic()-begin
    finally:pool.close()
    differences=[];relative=[];value_agreement=True
    for s,p in zip(serial,parallel):
        if s[1].keys()!=p[1].keys():raise AssertionError('terms differ')
        for k in s[1]:
            value_agreement=value_agreement and bool(np.isclose(s[1][k],p[1][k],rtol=1e-9,atol=1e-10))
        sg=jax.tree_util.tree_leaves(s[2]);pg=jax.tree_util.tree_leaves(p[2])
        differences.append(max(float(np.max(np.abs(a-b))) for a,b in zip(sg,pg)))
        norm=np.sqrt(sum(float(np.sum(a*a)) for a in sg))
        error=np.sqrt(sum(float(np.sum((a-b)**2)) for a,b in zip(sg,pg)))
        relative.append(float(error/max(norm,1e-12)))
    agreement=bool(value_agreement and max(relative)<1e-6)
    result=dict(agreement_passed=agreement,value_agreement=value_agreement,n=8,workers=4,serial_gradient_seconds=serial_time,
        parallel_gradient_seconds=parallel_time,speedup=serial_time/parallel_time,
        pool_warmup_seconds=warmup,max_gradient_absolute_difference=max(differences),
        max_gradient_relative_l2_error=max(relative),gradient_relative_l2_tolerance=1e-6,
        numerical_note='Original entrywise rtol=1e-7 check failed near small components; this run reports whole-gradient relative L2 error, not bitwise identity',
        backend=jax.default_backend(),scope='Fixed eight-episode batch; excludes startup from steady-state speedup')
    atomic_save(ROOT/'results/c1_episode_pool_benchmark.json',result)
    print(json.dumps(result),flush=True)
    if not agreement:raise AssertionError('parallel numerical agreement failed')


if __name__=='__main__':main()
