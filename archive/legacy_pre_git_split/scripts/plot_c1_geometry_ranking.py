"""Plot the frozen matched-branch audit; no simulation or optimization."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'results/c1_geometry_ranking_analysis'
fig, axes = plt.subplots(7, 2, figsize=(14, 19), sharex=True)
for col, seed in enumerate((20260916, 20260915)):
    for action, color in ((1, 'tab:orange'), (2, 'tab:blue')):
        z = np.load(OUT / f'0192_{seed}_{action}.npz')
        t = (np.arange(850) + 1) * .05
        n = len(z['applied']); live = t[:n]
        label = f'action {action}'
        for row, key in enumerate(('G', 'S', 'Hgeom', 'P')):
            axes[row, col].plot(t, z[key], color=color, alpha=.7, lw=.8, label=label)
            axes[row, col].set_ylabel(key)
        axes[4, col].plot(t, z['goal_distance'].max(axis=1), color=color, label=label)
        axes[4, col].set_ylabel('max goal distance')
        axes[5, col].plot(live, z['applied_speed'][:n], color=color, label=label+' joint speed')
        axes[5, col].plot(live, z['relative_speed'][:n], color=color, ls=':', alpha=.7)
        axes[5, col].set_ylabel('projected speed / relative (:)')
        axes[6, col].plot(live, z['pair_h'][:n], color=color, label=label)
        axes[6, col].set_ylabel('pair h')
    axes[0, col].set_title('Planning noise' if col == 0 else 'Execution noise')
    for ax in axes[:, col]:
        ax.axvspan(5, 7, color='grey', alpha=.15)
        ax.grid(alpha=.2)
    axes[0, col].legend()
    axes[-1, col].set_xlabel('time (s); pulse at 5–7 s')
fig.tight_layout()
fig.savefig(OUT / 'id192_dense_comparison.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for col, seed in enumerate((20260916, 20260915)):
    for action, color in ((1, 'tab:orange'), (2, 'tab:blue')):
        z = np.load(OUT / f'0192_{seed}_{action}.npz')
        x = z['positions_after']; u = z['applied']; t=(np.arange(len(u))+1)*.05
        for agent, style in ((0, '-'), (1, '--')):
            axes[0, col].plot(x[:, agent, 0], x[:, agent, 1], color=color, ls=style, label=f'a{action}, agent{agent}')
        axes[1, col].plot(t, u[:, 0], color=color, label=f'a{action}, u0x')
        axes[1, col].plot(t, u[:, 1], color=color, ls='--', label=f'a{action}, u0y')
    axes[0, col].set_title('Planning' if col == 0 else 'Execution')
    axes[0, col].set(xlabel='x', ylabel='y')
    axes[1, col].set(xlabel='time (s)', ylabel='actual projected agent0 controls')
    axes[1, col].axvspan(5,7,color='grey',alpha=.15)
    for ax in axes[:, col]: ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT / 'id192_paths_controls.png', dpi=150)
