"""P0 development analysis with simultaneous max-over-K16 fixed reference."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diagnostics.ring_state_selection_transfer_v1.study import data

HERE = Path(__file__).resolve().parent


def main():
    _, _, _, _, _, source = data()
    assert (HERE / "P0_TRAINING.json").exists()
    split = json.loads((HERE / "P0_SPLITS.json").read_text())
    pred = np.load(HERE / "P0_PREDICTIONS.npz")
    outcome = source["outcomes"]
    Y = outcome[:, :, 8:16, 0].mean(-1)
    U = outcome[:, :, 8:16, 1].mean(-1)
    discover = outcome[:, :, :8, 0].mean((0, 2))
    fixed_discovery = int(discover.argmax())
    fixed_hindsight = int(Y.mean(0).argmax())
    nstates = len(Y)
    draws = np.random.default_rng(2026101204).integers(nstates, size=(5000, nstates))
    best_fixed_boot = Y[draws].mean(1).max(1)
    rows = []
    for n, k in [(128, 16), (256, 8), (512, 4)]:
        for rule in ("nll", "selected_success"):
            values = []
            unknown = []
            mse = []
            picks = []
            for sub in range(3):
                key = f"n{n}_k{k}_sub{sub}_{rule}"
                logits = pred[key]
                pick = logits.argmax(-1)
                chosen = np.take_along_axis(Y[None, :, :], pick[:, :, None], 2)[:, :, 0]
                un = np.take_along_axis(U[None, :, :], pick[:, :, None], 2)[:, :, 0]
                values.extend(list(chosen))
                unknown.extend(list(un))
                picks.extend(list(pick))
                trueq = np.divide(Y, 1-U, out=np.full_like(Y, np.nan), where=(U < 1))
                mse.extend([np.nanmean((expit(logits[i]) - trueq)**2) for i in range(3)])
            values = np.asarray(values)
            unknown = np.asarray(unknown)
            per_state = values.mean(0)
            per_state_unknown = unknown.mean(0)
            boot = per_state[draws].mean(1) - best_fixed_boot
            lower = np.quantile(boot, [.025, .975]).tolist()
            row = dict(n=n, eta_per_state=k, rule=rule, n_subsets=3, n_fits=9,
                       selected_success=float(per_state.mean()), selected_unknown=float(per_state_unknown.mean()),
                       best_fixed_discovery_eta=fixed_discovery, best_fixed_discovery_success=float(Y[:,fixed_discovery].mean()),
                       hindsight_best_fixed_eta=fixed_hindsight, hindsight_best_fixed_success=float(Y[:,fixed_hindsight].mean()),
                       gain_vs_all16_fixed=float(per_state.mean()-Y.mean(0).max()),
                       simultaneous_ci95=lower, simultaneous_worst_unknown_ci95=np.quantile(boot-per_state_unknown[draws].mean(1), [.025,.975]).tolist(),
                       success_by_subset=[float(values[i*3:(i+1)*3].mean()) for i in range(3)],
                       success_by_fit=values.mean(1).tolist(),
                       finite_oracle_regret=float((Y.max(1)-per_state).mean()),
                       q_mse=float(np.mean(mse)),
                       eta_choices=np.asarray(picks).reshape(3,3,nstates).tolist())
            rows.append(row)
    valid = [r for r in rows if r["rule"]=="nll" and r["simultaneous_ci95"][0]>0]
    if valid:
        bestmax=max(r["selected_success"] for r in valid)
        close=[r for r in valid if r["selected_success"] >= bestmax-.01]
        allocation=max(close,key=lambda r:r["n"])
    else:
        allocation=max([r for r in rows if r["rule"]=="nll"],key=lambda r:r["selected_success"])
    n=allocation["n"]
    nll=next(r for r in rows if r["n"]==n and r["rule"]=="nll")
    succ=next(r for r in rows if r["n"]==n and r["rule"]=="selected_success")
    paired=np.asarray(succ["eta_choices"]) # only used to recover per-state differences directly below
    diff=[]
    for sub in range(3):
        a=pred[f"n{n}_k{allocation['eta_per_state']}_sub{sub}_selected_success"].argmax(-1)
        b=pred[f"n{n}_k{allocation['eta_per_state']}_sub{sub}_nll"].argmax(-1)
        diff.extend(list(np.take_along_axis(Y[None],a[:,:,None],2)[:,:,0]-np.take_along_axis(Y[None],b[:,:,None],2)[:,:,0]))
    delta=np.asarray(diff).mean(0)
    rule_ci=np.quantile(delta[draws].mean(1), [.025,.975]).tolist()
    checkpoint_rule="selected_success" if (float(delta.mean()) >= .01 and rule_ci[0]>0) else "nll"
    selected=dict(allocation=dict(n=n,eta_per_state=allocation['eta_per_state']), checkpoint_rule=checkpoint_rule,
                  selected_success_minus_nll=float(delta.mean()), paired_ci95=rule_ci,
                  allocation_rule='positive simultaneous CI, then highest success within1pp prefer more states; fallback highest NLL-rule success',
                  checkpoint_rule_policy='switch only if >=1pp and paired state CI lower>0',
                  development_only=True)
    result=dict(status='COMPLETED DEVELOPMENT',rows=[{k:v for k,v in r.items() if k!='eta_choices'} for r in rows], decision=selected,
                reused_source_rollouts=512*16*4+32*16*16+256*16*16,new_rollouts=0,
                note='All new model comparisons on previously opened 256 states are development, not independent confirmation; max fixed re-estimated in each bootstrap sample.')
    (HERE/'P0_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'decision':selected,'summary':[{k:r[k] for k in ['n','eta_per_state','rule','selected_success','gain_vs_all16_fixed','simultaneous_ci95']} for r in rows]},indent=2))


if __name__=='__main__':main()
