"""Plot saved training/validation evidence without loading or selecting policies."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    history = json.loads((args.run_dir/'history.json').read_text())
    validation = json.loads((args.run_dir/'validation.json').read_text())
    config = json.loads((args.run_dir/'config.json').read_text())
    fig, axes = plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    steps = np.array([r['update']+1 for r in history])
    axes[0].plot(steps,[r['loss'] for r in history],'o--',label='Before step')
    axes[0].plot(steps,[r['post_step']['loss'] for r in history],'o-',label='After step')
    axes[0].set(title='Same-batch objective at fixed dual',xlabel='Update',ylabel='Primal objective')
    axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[0].legend()
    axes[1].step(steps,[r['step_control']['scale'] for r in history],where='mid')
    axes[1].set(title='Accepted fraction of Adam proposal',xlabel='Update',ylabel='Scale',ylim=(-.03,1.03))
    vx=[0 if r['update']=='before_training' else r['update'] for r in validation]
    axes[2].plot(vx,[r['J_live'] for r in validation],'o-',label='Independent validation')
    axes[2].axhline(config['training']['epsilon'],color='tab:red',linestyle='--',label='Fixed epsilon')
    axes[2].set(title='Constraint feasibility',xlabel='Completed updates',ylabel='Mean risk')
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(args.run_dir/'training_evidence.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
