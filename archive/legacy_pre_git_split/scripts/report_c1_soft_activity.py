"""Plot full tasks and decompose matched-prefix soft-risk changes."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/c1_soft_activity_edge'))
    args=parser.parse_args();out=args.out
    config=json.loads((out/'config.json').read_text());risk=config['risk'];dt=config['environment']['dt']
    with np.load(out/'baseline_trajectory.npz') as f:base=dict(f)
    with np.load(out/'after_trajectory.npz') as f:after=dict(f)
    length=min(len(base['risk']),len(after['risk']))
    def aggregate(r):
        return float(risk['alpha']*np.mean(r**risk['p'])**(1/risk['p'])+(1-risk['alpha'])*r.mean())
    def mixed(use_after):
        h=(after if 0 in use_after else base)['h'][:length]
        s=(after if 1 in use_after else base)['residual'][:length]
        c=(after if 2 in use_after else base)['cone_risk'][:length]
        a=1/(1+np.exp(-(risk['rho']-h)/risk['tau_h']))*np.exp(-(s/risk['tau_s'])**2)
        candidate=h<=risk['rho']+risk['candidate_sigmas']*risk['tau_h']
        owners=np.zeros((17,2),bool);owners[0]=True;owners[1:9,0]=True;owners[9:,1]=True
        activity=np.max(np.where(candidate[:,:,None]&owners[None],a[:,:,None],0.),axis=1)
        return np.max(activity*c,axis=1)
    np.testing.assert_allclose(mixed(set()),base['risk'][:length],atol=1e-10)
    np.testing.assert_allclose(mixed({0,1,2}),after['risk'][:length],atol=1e-10)
    values={mask:aggregate(mixed(set(mask))) for k in range(4) for mask in itertools.combinations(range(3),k)}
    effects={}
    for i,name in enumerate(['h_and_local_candidates','CBF_slack','signed_margin_and_cone_geometry']):
        effects[name]=sum(math.factorial(len(s))*math.factorial(2-len(s))/6*
            (values[tuple(sorted((*s,i)))]-value) for s,value in values.items() if i not in s)
    delta=values[(0,1,2)]-values[()]
    np.testing.assert_allclose(sum(effects.values()),delta,atol=1e-12)
    report=dict(common_prefix_steps=length,seconds=length*dt,baseline_R=values[()],after_R=values[(0,1,2)],
        delta_R=delta,shapley_factor_contributions=effects,
        interpretation='Exact algebraic decomposition on matched time indices; not a causal intervention experiment.',
        baseline_goal_error_at_common_end=base['goal_errors'][length-1].tolist(),
        after_goal_error_at_common_end=after['goal_errors'][length-1].tolist(),
        baseline_mean_pair_h=float(base['h'][:length,0].mean()),after_mean_pair_h=float(after['h'][:length,0].mean()),
        baseline_mean_pair_slack=float(base['residual'][:length,0].mean()),after_mean_pair_slack=float(after['residual'][:length,0].mean()))
    (out/'risk_attribution.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/c1_soft_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from single_integrator.environment import Config,GiveWayEnv
    env=GiveWayEnv(Config(**config['environment']))
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for a,b in env.walls:axes[0,0].plot([a[0],b[0]],[a[1],b[1]],color='black',lw=1)
    for label,trace,color in [('Baseline',base,'tab:blue'),('After short training',after,'tab:orange')]:
        t=(np.arange(len(trace['risk']))+1)*dt
        for agent,style in [(0,'-'),(1,'--')]:
            xy=np.concatenate((trace['initial_positions'][None],trace['positions']))
            axes[0,0].plot(xy[:,agent,0],xy[:,agent,1],style,color=color,label=f'{label} agent {agent}')
        axes[0,1].plot(t,trace['risk'],label=label,color=color)
        axes[1,0].plot(t,trace['min_swept_agent_distance']+2*env.config.agent_radius,label=label,color=color)
        axes[1,1].plot(t,trace['goal_errors'].sum(axis=1),label=label,color=color)
        axes[2,0].plot(t,trace['residual'][:,0],label=label,color=color)
        axes[2,1].plot(t,np.max(trace['local_activity'],axis=1),label=label+' soft A',color=color)
        axes[2,1].plot(t,np.any(trace['active_mask'],axis=1).astype(float),':',alpha=.4,color=color,label=label+' hard active')
    axes[0,0].set_aspect('equal');axes[0,0].set_title('Complete trajectories')
    axes[0,1].set_title('Continuous instantaneous risk')
    axes[1,0].set_title('Minimum swept inter-agent center distance (m)')
    axes[1,0].axhline(.3201,color='red',ls=':',label='CBF separation')
    axes[1,1].set_title('Sum of goal errors (m)')
    axes[2,0].set_title('Pairwise post-projection CBF slack');axes[2,0].set_yscale('symlog',linthresh=1e-7)
    axes[2,1].set_title('Soft activity and hard-active diagnostics')
    for ax in axes.ravel():ax.legend(fontsize=7);ax.grid(alpha=.2)
    for ax in [axes[0,1],*axes[1],*axes[2]]:ax.set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
