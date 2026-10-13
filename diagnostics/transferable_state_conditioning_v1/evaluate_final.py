"""State-unit Q16/B15 and independent-half mechanism audit.

Run only after the model-gated final rollout jobs have completed and merged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE as GLOBAL_HERE, SCENES, read, sha
from diagnostics.transferable_state_conditioning_v1.final_worker import gate
from shared_rollout_db.src.rollout_db import connect

FLAGS = ("success", "deadlock", "timeout", "collision", "numerical_failure")
HERE = Path(os.environ.get("TRANSFER_Q_FINAL_ROOT", str(GLOBAL_HERE)))
SEED = 20261008
BOOT = 20000


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def paired_bootstrap(a, seed=SEED):
    x = np.asarray(a, float)
    assert x.ndim == 1 and len(x)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(x), size=(BOOT, len(x)))
    estimates = x[sample].mean(1)
    return {"states": len(x), "difference": float(x.mean()),
            "ci95": np.quantile(estimates, [.025,.975]).tolist(),
            "positive_states": int(np.sum(x>0)),
            "negative_states": int(np.sum(x<0)),
            "ties": int(np.sum(x==0))}


def collect(scene, bank):
    gate()
    assert scene in SCENES and bank in ("proposal", "anchor")
    panel = HERE / "final" / bank / scene
    cases = read(panel / "cases.json")
    split = read(HERE / "frozen_splits.json")
    ids = (split["final_confirmation"][scene]["state_uids"] if bank == "proposal"
           else split["anchor_subset"][scene]["state_uids"])
    n = len(ids); k = 17 if bank == "proposal" else 16
    assert len(cases) == n*k*16
    values = np.full((n,k,16,5), -1, np.int8)
    eta = np.zeros((n,k,3), np.float64)
    lookup = {uid:i for i,uid in enumerate(ids)}
    safety_issues = []
    with connect(True) as db:
        for c in cases:
            i = lookup[c["state_uid"]]; j = c["eta_index"]; t = c["future_index"]
            row = db.execute("SELECT * FROM rollout WHERE rollout_uid=? AND conflict_quarantined=0",
                             (c["rollout_uid"],)).fetchone()
            assert row is not None, c["rollout_uid"]
            assert row["state_uid"] == c["state_uid"] and row["eta_uid"] == c["eta_uid"]
            assert row["controller_uid"] == c["controller_uid"] and row["seed_key"] == c["seed_key"]
            assert row["compatibility_quality"] == "EXACT_REUSE"
            assert values[i,j,t,0] == -1
            values[i,j,t] = [row[f] for f in FLAGS]
            eta[i,j] = c["eta"]
            if row["collision"] or row["numerical_failure"]:
                safety_issues.append({"state_uid":c["state_uid"],"eta_index":j,
                                      "eta":c["eta"],"seed_key":c["seed_key"],
                                      "rollout_uid":c["rollout_uid"],
                                      "collision":int(row["collision"]),
                                      "numerical_failure":int(row["numerical_failure"]),
                                      "original_source_file":row["original_source_file"]})
    assert np.all(values>=0)
    outcome = panel / "outcomes.npz"
    np.savez_compressed(outcome, values=values, eta=eta, state_uids=np.asarray(ids))
    audit = {"scene":scene,"bank":bank,"independent_states":n,"eta_candidates":k,
             "seed_outcomes":int(np.prod(values.shape[:-1])),
             "collisions":int(values[...,3].sum()),
             "numerical_failures":int(values[...,4].sum()),
             "outcomes_sha256":sha(outcome)}
    write(panel / "collection_audit.json", audit)
    write(panel / "safety_issues.json", safety_issues)
    print(json.dumps(audit), flush=True)


def safe_corr(x, y):
    a = np.asarray(x, float).reshape(-1)
    b = np.asarray(y, float).reshape(-1)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return None
    return float(np.corrcoef(a,b)[0,1])


def binary_entropy(indices, k=16):
    p = np.bincount(np.asarray(indices,int), minlength=k) / len(indices)
    return float(-(p[p>0]*np.log2(p[p>0])).sum())


def method_values(values, choice):
    n = values.shape[0]
    c = np.asarray(choice, int)
    assert c.shape == (n,)
    chosen = values[np.arange(n), c]
    successes = chosen[:, :, 0].sum(1)
    unknown = chosen[:, :, 4].sum(1)
    return {"q_lower": successes / 16., "q_upper": (successes+unknown) / 16.,
            "B15_confirmed": successes >= 15,
            "B15_possible": successes+unknown >= 15,
            "collision_count": chosen[:,:,3].sum(1),
            "numerical_count": unknown,
            "deadlock_count": chosen[:,:,1].sum(1),
            "timeout_count": chosen[:,:,2].sum(1)}


def summarize_method(metric, oracle_q=None):
    result = {"states":len(metric["q_lower"]),
              "B15_confirmed":int(metric["B15_confirmed"].sum()),
              "B15_possible":int(metric["B15_possible"].sum()),
              "B15_rate_lower":float(metric["B15_confirmed"].mean()),
              "mean_success_lower":float(metric["q_lower"].mean()),
              "mean_success_upper":float(metric["q_upper"].mean()),
              "collisions":int(metric["collision_count"].sum()),
              "numerical_failures":int(metric["numerical_count"].sum()),
              "deadlocks":int(metric["deadlock_count"].sum()),
              "timeouts":int(metric["timeout_count"].sum())}
    if oracle_q is not None:
        result["mean_oracle_regret_lower"] = float(np.mean(oracle_q-metric["q_lower"]))
    return result


def proposal(panel):
    gate()
    scene = panel.removeprefix("within_").removeprefix("loso_")
    d = np.load(HERE / "final" / "proposal" / scene / "outcomes.npz")
    p = np.load(HERE / "predictions" / panel / "proposal.npz")
    assert np.array_equal(d["state_uids"], p["state_uids"])
    np.testing.assert_allclose(d["eta"][:,:16],p["eta"],rtol=0,atol=1e-7)
    v = d["values"]
    n = len(v)
    row = np.arange(n)
    q = v[:,:16,:,0].sum(2)/16.
    oracle_choice = q.argmax(1)
    oracle_q = q.max(1)
    state = method_values(v,p["state_choice"])
    eta = method_values(v,p["eta_choice"])
    safety = method_values(v,np.full(n,16,int))
    oracle = method_values(v,oracle_choice)
    mismatches = p["mismatch_choice"]
    assert mismatches.shape == (100,n)
    mismatch_q = q[np.arange(n)[None,:],mismatches]
    mismatch_b15 = (v[:,:16,:,0].sum(2)>=15)[np.arange(n)[None,:],mismatches]
    result = {"panel":panel,"bank":"frozen stochastic K16","states":n,
              "methods":{"Safety":summarize_method(safety,oracle_q),
                         "Q_eta":summarize_method(eta,oracle_q),
                         "Q_state":summarize_method(state,oracle_q),
                         "Oracle":summarize_method(oracle,oracle_q),
                         "state_mismatched_100":{"mean_B15":float(mismatch_b15.mean()),
                           "mean_success":float(mismatch_q.mean()),
                           "mean_selection_changed":float(np.mean(mismatches != p["state_choice"][None,:]))}},
              "paired":{"state_minus_eta_B15":paired_bootstrap(state["B15_confirmed"].astype(float)-eta["B15_confirmed"].astype(float)),
                        "state_minus_eta_mean_success":paired_bootstrap(state["q_lower"]-eta["q_lower"]),
                        "state_minus_Safety_B15":paired_bootstrap(state["B15_confirmed"].astype(float)-safety["B15_confirmed"].astype(float)),
                        "state_minus_Safety_mean_success":paired_bootstrap(state["q_lower"]-safety["q_lower"]),
                        "state_minus_mismatch_B15":paired_bootstrap(state["B15_confirmed"].astype(float)-mismatch_b15.mean(0)),
                        "state_minus_mismatch_mean_success":paired_bootstrap(state["q_lower"]-mismatch_q.mean(0))},
              "selected_candidate_index_counts":np.bincount(p["state_choice"],minlength=16).tolist(),
              "selected_candidate_index_entropy_bits":binary_entropy(p["state_choice"]),
              "selected_eta_mean":np.asarray(p["eta"])[row,p["state_choice"]].mean(0).tolist(),
              "selected_eta_std":np.asarray(p["eta"])[row,p["state_choice"]].std(0).tolist(),
              "top1_matches_Q16_oracle":{"Q_eta":float(np.mean(p["eta_choice"]==oracle_choice)),
                                          "Q_state":float(np.mean(p["state_choice"]==oracle_choice))},
              "within_state_pairwise_accuracy":{"Q_eta":pairwise(p["eta_logits"],q,.25),
                                                 "Q_state":pairwise(p["state_logits"],q,.25)},
              "failure_cases":[]}
    disadvantage = state["q_lower"]-eta["q_lower"]
    for i in np.argsort(disadvantage)[:min(20,n)]:
        if disadvantage[i] >= 0: break
        result["failure_cases"].append({"state_uid":str(d["state_uids"][i]),
                                        "state_eta_index":int(p["state_choice"][i]),
                                        "eta_only_index":int(p["eta_choice"][i]),
                                        "state_success_count":int(state["q_lower"][i]*16),
                                        "eta_only_success_count":int(eta["q_lower"][i]*16),
                                        "safety_success_count":int(safety["q_lower"][i]*16),
                                        "oracle_success_count":int(oracle_q[i]*16)})
    output = HERE / "final" / "metrics" / panel / "proposal.json"
    write(output,result)
    print(panel,"proposal",result["methods"]["Q_state"]["B15_confirmed"],
          result["paired"]["state_minus_mismatch_B15"],flush=True)


def pairwise(z, truth, threshold):
    n,k = truth.shape
    a,b = np.triu_indices(k,1)
    t = truth[:,a]-truth[:,b]
    valid = np.abs(t) >= threshold
    pred = np.sign(z[:,a]-z[:,b])
    good = pred == np.sign(t)
    by_state = np.array([np.mean(good[i,valid[i]]) if np.any(valid[i]) else np.nan for i in range(n)])
    return {"cases":int(valid.sum()),"states_with_cases":int(np.isfinite(by_state).sum()),
            "accuracy":float(np.mean(good[valid])) if valid.any() else None,
            "state_bootstrap":paired_bootstrap(by_state[np.isfinite(by_state)]) if np.isfinite(by_state).any() else None}


def reversal_audit(z_state,z_eta,discovery,evaluation,mismatch_q,state_choice,eta_choice):
    n,k = discovery.shape
    a,b = np.triu_indices(k,1)
    diff = discovery[:,a]-discovery[:,b]
    strong = np.abs(diff) >= 3/8
    reversal = np.array([np.any(diff[:,i]>=3/8) and np.any(diff[:,i]<=-3/8) for i in range(len(a))])
    valid = strong & reversal[None,:]
    eval_diff = evaluation[:,a]-evaluation[:,b]
    non_tie = valid & (eval_diff!=0)
    pred_state = np.sign(z_state[:,a]-z_state[:,b])
    pred_eta = np.sign(z_eta[:,a]-z_eta[:,b])
    correct_state = pred_state == np.sign(eval_diff)
    correct_eta = pred_eta == np.sign(eval_diff)
    persist = np.sign(eval_diff) == np.sign(diff)
    per_state = []
    per_eta = []
    for i in range(n):
        if np.any(non_tie[i]):
            per_state.append(float(np.mean(correct_state[i,non_tie[i]])))
            per_eta.append(float(np.mean(correct_eta[i,non_tie[i]])))
    active_states = np.flatnonzero(valid.any(1))
    gain = evaluation[np.arange(n),state_choice]-evaluation[np.arange(n),eta_choice]
    mismatch_gain = evaluation[np.arange(n),state_choice]-mismatch_q.mean(0)
    groups={}
    for label,mask in (("A",valid & (diff>0)),("B",valid & (diff<0))):
        observed=mask & (eval_diff!=0)
        group_states=np.flatnonzero(mask.any(1))
        groups[label]={"pair_state_cases":int(mask.sum()),
                       "independent_eval_non_tie_cases":int(observed.sum()),
                       "states":int(len(group_states)),
                       "switch_accuracy_state":float(correct_state[observed].mean()) if observed.any() else None,
                       "switch_accuracy_eta_only":float(correct_eta[observed].mean()) if observed.any() else None,
                       "selected_eta_counts":np.bincount(state_choice[group_states],minlength=k).tolist(),
                       "state_minus_eta_evalQ":paired_bootstrap(gain[group_states]) if len(group_states) else None,
                       "state_minus_mismatch_evalQ":paired_bootstrap(mismatch_gain[group_states]) if len(group_states) else None}
    predicted_reversals=int(sum(np.any(pred_state[valid[:,i],i]>0) and
                                np.any(pred_state[valid[:,i],i]<0)
                                for i in np.flatnonzero(reversal)))
    return {"strong_discovery_threshold":"3/8", "reversal_eta_pairs":int(reversal.sum()),
            "model_predicted_reversal_eta_pairs":predicted_reversals,
            "strong_pair_state_cases":int(valid.sum()),
            "strong_reversal_states":int(len(active_states)),
            "independent_eval_non_tie_cases":int(non_tie.sum()),
            "discovery_preference_persistence":float(persist[non_tie].mean()) if non_tie.any() else None,
            "switch_accuracy_state":float(correct_state[non_tie].mean()) if non_tie.any() else None,
            "switch_accuracy_eta_only":float(correct_eta[non_tie].mean()) if non_tie.any() else None,
            "switch_accuracy_state_bootstrap":paired_bootstrap(np.asarray(per_state)) if per_state else None,
            "switch_accuracy_state_minus_eta_bootstrap":paired_bootstrap(np.asarray(per_state)-np.asarray(per_eta)) if per_state else None,
            "strong_states_state_minus_eta_evalQ":paired_bootstrap(gain[active_states]) if len(active_states) else None,
            "strong_states_state_minus_mismatch_evalQ":paired_bootstrap(mismatch_gain[active_states]) if len(active_states) else None,
            "preference_groups":groups}


def anchor(panel):
    gate()
    scene = panel.removeprefix("within_").removeprefix("loso_")
    d = np.load(HERE / "final" / "anchor" / scene / "outcomes.npz")
    p = np.load(HERE / "predictions" / panel / "anchor.npz")
    assert np.array_equal(d["state_uids"],p["state_uids"])
    np.testing.assert_allclose(d["eta"],p["eta"],rtol=0,atol=1e-7)
    v=d["values"];n=len(v);row=np.arange(n)
    assert v.shape==(n,16,16,5)
    discovery=v[:,:,:8,0].sum(2)/8.
    evaluation=v[:,:,8:,0].sum(2)/8.
    full=v[:,:,:,0].sum(2)/16.
    state_choice=p["state_choice"].astype(int)
    eta_choice=p["eta_choice"].astype(int)
    mismatch_choice=p["mismatch_choice"].astype(int)
    assert mismatch_choice.shape==(100,n)
    mismatched_eval=evaluation[row[None,:],mismatch_choice]
    selected_eval=evaluation[row,state_choice]
    eta_eval=evaluation[row,eta_choice]
    oracle_eval=evaluation.max(1)
    fixed_disc=discovery.mean(0).argmax()
    fixed_posthoc=evaluation.mean(0).argmax()
    adaptive_demand=np.flatnonzero(discovery.max(1)-discovery[:,fixed_disc]>=3/8)
    preferred_disc=discovery.argmax(1)
    preferred_eval=evaluation.argmax(1)
    state_diff_vs_posthoc=selected_eval-evaluation[:,fixed_posthoc]
    rng=np.random.default_rng(SEED)
    sample=rng.integers(n,size=(BOOT,n))
    state_boot=selected_eval[sample].mean(1)
    fixed_boot=evaluation[sample].mean(1).max(1)
    fixed_reselect=state_boot-fixed_boot
    true_effect=full-full.mean(0,keepdims=True)
    predicted=jax_sigmoid(p["state_logits"])
    pred_effect=predicted-predicted.mean(0,keepdims=True)
    result={"panel":panel,"bank":"common fixed Sobol 16 eta","states":n,
            "discovery_seed_indices":list(range(8)),"evaluation_seed_indices":list(range(8,16)),
            "mean_eval_success":{"Q_eta":float(eta_eval.mean()),"Q_state":float(selected_eval.mean()),
                "mismatched_state_100":float(mismatched_eval.mean()),
                "fixed_discovery_best":float(evaluation[:,fixed_disc].mean()),
                "fixed_evaluation_posthoc_best":float(evaluation[:,fixed_posthoc].mean()),
                "Oracle":float(oracle_eval.mean())},
            "B15_full16":{"Q_eta":int(np.sum(full[row,eta_choice]>=15/16)),
                "Q_state":int(np.sum(full[row,state_choice]>=15/16)),
                "fixed_discovery_best":int(np.sum(full[:,fixed_disc]>=15/16)),
                "fixed_evaluation_posthoc_best":int(np.sum(full[:,fixed_posthoc]>=15/16)),
                "Oracle":int(np.sum(full.max(1)>=15/16))},
            "fixed_policies_eval_mean":evaluation.mean(0).tolist(),
            "evaluation_half_top1_matches_oracle":{"Q_eta":float(np.mean(eta_choice==evaluation.argmax(1))),
                                                     "Q_state":float(np.mean(state_choice==evaluation.argmax(1))),
                                                     "fixed_discovery_best":float(np.mean(fixed_disc==evaluation.argmax(1)))},
            "evaluation_half_mean_oracle_regret":{"Q_eta":float(np.mean(oracle_eval-eta_eval)),
                                                   "Q_state":float(np.mean(oracle_eval-selected_eval)),
                                                   "fixed_discovery_best":float(np.mean(oracle_eval-evaluation[:,fixed_disc]))},
            "fixed_discovery_best_index":int(fixed_disc),
            "fixed_evaluation_posthoc_best_index":int(fixed_posthoc),
            "ground_truth_preference_distribution":{
                "discovery_preferred_eta_counts":np.bincount(preferred_disc,minlength=16).tolist(),
                "evaluation_preferred_eta_counts":np.bincount(preferred_eval,minlength=16).tolist(),
                "discovery_preferred_differs_from_global_fixed":int(np.sum(preferred_disc!=fixed_disc)),
                "distinct_discovery_preferred_eta":int(len(set(preferred_disc.tolist()))),
                "discovery_top1_tied_states":int(np.sum(np.sum(discovery==discovery.max(1,keepdims=True),axis=1)>1)),
                "deterministic_tie_break":"lowest eta index"},
            "adaptive_demand":{"criterion":"discovery oracle minus discovery best fixed >=3/8",
                "states":int(len(adaptive_demand)),
                "state_minus_eta_evalQ":paired_bootstrap((selected_eval-eta_eval)[adaptive_demand]) if len(adaptive_demand) else None,
                "state_minus_fixed_evalQ":paired_bootstrap((selected_eval-evaluation[:,fixed_disc])[adaptive_demand]) if len(adaptive_demand) else None,
                "state_minus_mismatch_evalQ":paired_bootstrap((selected_eval-mismatched_eval.mean(0))[adaptive_demand]) if len(adaptive_demand) else None},
            "paired":{"state_minus_eta_evalQ":paired_bootstrap(selected_eval-eta_eval),
                      "state_minus_mismatch_evalQ":paired_bootstrap(selected_eval-mismatched_eval.mean(0)),
                      "state_minus_discovery_fixed_evalQ":paired_bootstrap(selected_eval-evaluation[:,fixed_disc]),
                      "state_minus_posthoc_fixed_evalQ":paired_bootstrap(state_diff_vs_posthoc),
                      "state_minus_bootstrap_reselected_fixed_evalQ":{
                          "states":n,"difference":float(selected_eval.mean()-evaluation.mean(0).max()),
                          "ci95":np.quantile(fixed_reselect,[.025,.975]).tolist()}},
            "state_effect_correlation":safe_corr(pred_effect,true_effect),
            "pairwise_reversal_accuracy":{"Q_eta":pairwise(p["eta_logits"],full,.25),
                                         "Q_state":pairwise(p["state_logits"],full,.25)},
            "switching":reversal_audit(p["state_logits"],p["eta_logits"],discovery,evaluation,
                                       mismatched_eval,state_choice,eta_choice),
            "selection_counts":np.bincount(state_choice,minlength=16).tolist(),
            "selection_entropy_bits":binary_entropy(state_choice),
            "collisions":int(v[...,3].sum()),"numerical_failures":int(v[...,4].sum()),
            "failure_cases":[]}
    for i in np.argsort(selected_eval-eta_eval)[:min(20,n)]:
        if selected_eval[i]>=eta_eval[i]:break
        result["failure_cases"].append({"state_uid":str(d["state_uids"][i]),
                                        "state_eta_index":int(state_choice[i]),
                                        "eta_only_index":int(eta_choice[i]),
                                        "state_eval_success_count":int(selected_eval[i]*8),
                                        "eta_eval_success_count":int(eta_eval[i]*8),
                                        "oracle_eval_success_count":int(oracle_eval[i]*8)})
    output=HERE/"final"/"metrics"/panel/"anchor.json"
    write(output,result)
    print(panel,"anchor",result["paired"]["state_minus_bootstrap_reselected_fixed_evalQ"],
          result["switching"],flush=True)


def jax_sigmoid(x):
    x=np.asarray(x,float)
    return 1/(1+np.exp(-np.clip(x,-50,50)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("collect","proposal","anchor"))
    p.add_argument("--scene",choices=SCENES)
    p.add_argument("--panel",choices=tuple(f"{prefix}_{s}" for prefix in ("within","loso") for s in SCENES))
    p.add_argument("--bank",choices=("proposal","anchor"))
    a=p.parse_args()
    if a.action=="collect":
        assert a.scene and a.bank
        collect(a.scene,a.bank)
    else:
        assert a.panel
        (proposal if a.action=="proposal" else anchor)(a.panel)
