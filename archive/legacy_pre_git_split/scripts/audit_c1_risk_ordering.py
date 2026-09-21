"""Score saved development traces and test all cross-outcome orderings."""
import argparse
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.risk.exact_margin import trajectory
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.environment import Config, GiveWayEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--candidate',choices=('exact','ordered'),default='exact')
    args=p.parse_args()
    risk=trajectory
    if args.candidate=='ordered':
        from single_integrator.c1.risk.ordered_guidance import trajectory as risk
    jax.config.update('jax_enable_x64',True)
    config=Config(**json.loads((ROOT/'results/c1_deadlock_union/sets.json').read_text())['environment'])
    goals=jnp.asarray(GiveWayEnv(config).goals)
    fn=jax.jit(lambda b,a,u,m,to: risk(b,a,u,goals,m,terminal_timeout=to,
        dt=config.dt,max_speed=config.max_speed,goal_tolerance=config.goal_tolerance,
        hold_seconds=config.deadlock_hold_seconds,progress_window_seconds=config.progress_window_seconds,
        progress_epsilon=config.progress_epsilon,speed_epsilon_fraction=config.speed_epsilon_fraction)['J_live'])
    records=json.loads((args.folder/'records.json').read_text())
    values=[]
    for r in records:
        with np.load(args.folder/f"{r['rid']}_{r['noise_seed']}_{r['variant']}.npz") as z:
            n=len(z['applied'])
            pad=lambda a: np.concatenate([a,np.repeat(a[-1:],config.max_steps-n,axis=0)])
            score=float(fn(pad(z['positions_before']),pad(z['positions_after']),
                np.pad(z['applied'],((0,config.max_steps-n),(0,0))),np.arange(config.max_steps)<n,r['timeout']))
        values.append(dict(rid=r['rid'],variant=r['variant'],deadlock=bool(r['any_deadlock'] or r['label']=='stalled_deadlock'),
                           label=r['label'],**{args.candidate:score},original=r['risk_original'],bounded=r['risk_bounded']))
    result=dict(scope='Retrospective development ordering; does not prove forecast calibration',n=len(values),scores={})
    for key in (args.candidate,'original','bounded'):
        d=np.array([r[key] for r in values if r['deadlock']])
        nd=np.array([r[key] for r in values if not r['deadlock']])
        diff=d[:,None]-nd[None,:]
        result['scores'][key]=dict(deadlock_n=len(d),non_deadlock_n=len(nd),min_deadlock=float(d.min()),
            max_non_deadlock=float(nd.max()),strict_ordering=bool(d.min()>nd.max()),
            reversed_pairs=int((diff<0).sum()),tied_pairs=int((diff==0).sum()),total_pairs=diff.size,
            deadlock_below_one=int((d<1).sum()),non_deadlock_above_one=int((nd>1).sum()))
    atomic_save(args.folder/('ordering_audit.json' if args.candidate=='exact' else 'ordered_ordering_audit.json'),result)
    atomic_save(args.folder/(args.candidate+'_scores.json'),values)
    print(json.dumps(result))


if __name__=='__main__':main()
