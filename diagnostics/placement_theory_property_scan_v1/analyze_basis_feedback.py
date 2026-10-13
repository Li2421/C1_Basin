"""Finite-eta closed-loop effect of time-indexed nominal B versus live B."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    p=json.loads((HERE/'protocol.json').read_text());rows=[]
    for uid in p['state_probe_uids']:
        scene='ring_exchange' if 'ring' in uid else 'toy_give_way'
        for chain in ('TT','FF'):
            for i in (4,8):
                base=f'{uid}_{chain}_c{i}_'
                files={mode:HERE/'rollout_results'/(base+mode) for mode in ('live_B_full','frozen_B')}
                meta={mode:json.loads(Path(str(path)+'.json').read_text()) for mode,path in files.items()}
                with np.load(Path(str(files['live_B_full'])+'.npz')) as z:
                    lp=z['positions'].copy();la=z['actions'].copy();lk=z['noise_keys'].copy()
                with np.load(Path(str(files['frozen_B'])+'.npz')) as z:
                    fp=z['positions'].copy();fa=z['actions'].copy();fk=z['noise_keys'].copy()
                n=min(len(la),len(fa));dp=np.linalg.norm((lp[:n+1]-fp[:n+1]).reshape(n+1,-1),axis=1)
                du=np.linalg.norm((la[:n]-fa[:n]).reshape(n,-1),axis=1)
                keys_same=bool(np.array_equal(lk[:n],fk[:n]))
                rho_pair=[meta['live_B_full']['rho'],meta['frozen_B']['rho']]
                rr=dict(uid=uid,scene=scene,chain=chain,candidate_index=i,
                    live_termination=meta['live_B_full']['termination'],
                    frozen_termination=meta['frozen_B']['termination'],
                    live_success=meta['live_B_full']['success'],
                    frozen_success=meta['frozen_B']['success'],
                    valid_full_pair=meta['live_B_full']['success'] is not None and meta['frozen_B']['success'] is not None,
                    common_steps=n,noise_keys_paired=keys_same,
                    first_action_difference=float(du[0]) if n else None,
                    mean_action_difference=float(np.mean(du)) if n else None,
                    max_action_difference=float(np.max(du)) if n else None,
                    common_prefix_final_position_difference=float(dp[-1]),
                    maximum_position_difference=float(np.max(dp)),
                    live_rho=rho_pair[0],frozen_rho=rho_pair[1],
                    rho_difference=(rho_pair[0]-rho_pair[1]) if None not in rho_pair else None,
                    frozen_basis_delta_gap_mean=meta['frozen_B']['basis_delta_gap_mean'],
                    live_basis_delta_gap_mean_first_reference_steps=meta['live_B_full']['basis_delta_gap_mean'],
                    nominal_reference_steps=meta['frozen_B']['nominal_reference_steps'])
                rows.append(rr)
    valid=[r for r in rows if r['valid_full_pair']]
    stats=dict(pairs=len(rows),all_key_pairs_match=all(r['noise_keys_paired'] for r in rows),
        full_label_pairs=len(valid),censored_pairs=len(rows)-len(valid),
        full_label_flips=sum(r['live_success']!=r['frozen_success'] for r in valid),
        median_common_prefix_final_position_difference=float(np.median([r['common_prefix_final_position_difference'] for r in rows])),
        median_mean_action_difference=float(np.median([r['mean_action_difference'] for r in rows])),
        median_frozen_basis_delta_gap=float(np.median([r['frozen_basis_delta_gap_mean'] for r in rows])),
        first_action_max_difference=max(r['first_action_difference'] for r in rows),
        note='Frozen B_t^0 is undefined after the eta=0 nominal trajectory terminates. Its rollout is censored at that step. Pilot live_B also censored erroneously; it is excluded and superseded by live_B_full.')
    (HERE/'basis_feedback_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'basis_feedback_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(8.7,3.3))
    for scene,color in (('ring_exchange','#1f77b4'),('toy_give_way','#d95f02')):
        rr=[r for r in rows if r['scene']==scene]
        axs[0].scatter([r['frozen_basis_delta_gap_mean'] for r in rr],
            [r['common_prefix_final_position_difference'] for r in rr],color=color,label=scene)
        axs[1].scatter([r['mean_action_difference'] for r in rr],
            [abs(r['rho_difference']) for r in rr],color=color,label=scene)
    axs[0].set_xlabel('mean $||(B_{live}-B_t^0)\\eta||$');axs[0].set_ylabel('common-prefix final $||\\Delta p||$')
    axs[1].set_xlabel('mean action difference');axs[1].set_ylabel('$|\\Delta\\rho|$ (censored included)')
    for a in axs:a.legend()
    fig.tight_layout();fig.savefig(HERE/'figures/basis_feedback.pdf');fig.savefig(HERE/'figures/basis_feedback.png',dpi=180);plt.close(fig)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
