"""Require all eight panels and all falsifications before issuing a verdict."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE as GLOBAL_HERE, SCENES, read, sha
from diagnostics.transferable_state_conditioning_v1.evaluate_final import BOOT, SEED, paired_bootstrap
from diagnostics.transferable_state_conditioning_v1.final_worker import gate

HERE = Path(os.environ.get("TRANSFER_Q_FINAL_ROOT", str(GLOBAL_HERE)))


def pct(value):
    return f"{100*value:.1f}%"


def ci_pp(pair):
    return f"{100*pair['difference']:+.1f} pp [{100*pair['ci95'][0]:+.1f}, {100*pair['ci95'][1]:+.1f}]"


def per_state_gap(scene, kind):
    d=np.load(HERE/"final"/"proposal"/scene/"outcomes.npz")
    p=np.load(HERE/"predictions"/f"loso_{scene}"/"proposal.npz")
    q=d["values"][:,:16,:,0].sum(2)/16.
    row=np.arange(len(q))
    matched=q[row,p["state_choice"]]
    if kind=="mismatch":
        control=q[row[None,:],p["mismatch_choice"]].mean(0)
    elif kind=="eta":
        control=q[row,p["eta_choice"]]
    elif kind=="safety":
        control=d["values"][:,16,:,0].sum(1)/16.
    else:raise ValueError(kind)
    return matched-control


def equal_scene_bootstrap(kind):
    gaps=[per_state_gap(scene,kind) for scene in SCENES]
    rng=np.random.default_rng(SEED)
    reps=np.zeros(BOOT)
    for gap in gaps:
        sample=rng.integers(len(gap),size=(BOOT,len(gap)))
        reps+=gap[sample].mean(1)/len(gaps)
    return {"scenes":len(gaps),"states":sum(len(x) for x in gaps),
            "equal_scene_weighted_difference":float(np.mean([x.mean() for x in gaps])),
            "ci95":np.quantile(reps,[.025,.975]).tolist()}


def causal_scene(proposal,anchor):
    p=proposal["paired"]["state_minus_mismatch_mean_success"]
    a=anchor["paired"]["state_minus_mismatch_evalQ"]
    f=anchor["paired"]["state_minus_bootstrap_reselected_fixed_evalQ"]
    sw=anchor["switching"]
    switched=(sw["reversal_eta_pairs"]>0 and
              sw["switch_accuracy_state"] is not None and
              sw["switch_accuracy_eta_only"] is not None and
              sw["switch_accuracy_state"]>sw["switch_accuracy_eta_only"])
    collision_ok=(proposal["methods"]["Q_state"]["collisions"]<=
                  proposal["methods"]["Q_eta"]["collisions"])
    return bool(p["ci95"][0]>0 and a["ci95"][0]>0 and f["ci95"][0]>0 and switched and collision_ok)


def main():
    gate()
    split=read(HERE/"frozen_splits.json")
    panel_names=[f"{prefix}_{scene}" for prefix in ("within","loso") for scene in SCENES]
    panels={}
    for name in panel_names:
        path=HERE/"final"/"metrics"/name
        panels[name]={bank:read(path/f"{bank}.json") for bank in ("proposal","anchor")}
        scene=name.removeprefix("within_").removeprefix("loso_")
        assert panels[name]["proposal"]["states"]==split["final_confirmation"][scene]["count"]
        assert panels[name]["anchor"]["states"]==split["anchor_subset"][scene]["count"]
    aggregate={"matched_minus_eta":equal_scene_bootstrap("eta"),
               "matched_minus_Safety":equal_scene_bootstrap("safety"),
               "matched_minus_same_scene_mismatch":equal_scene_bootstrap("mismatch")}
    passes={scene:causal_scene(panels[f"loso_{scene}"]["proposal"],
                               panels[f"loso_{scene}"]["anchor"]) for scene in SCENES}
    within={scene:panels[f"within_{scene}"]["proposal"]["paired"]["state_minus_mismatch_mean_success"]["ci95"][0]>0
            for scene in SCENES}
    collision_total=sum(panels[f"loso_{scene}"]["proposal"]["methods"]["Q_state"]["collisions"] for scene in SCENES)
    collision_eta=sum(panels[f"loso_{scene}"]["proposal"]["methods"]["Q_eta"]["collisions"] for scene in SCENES)
    if (sum(passes.values())>=2 and
        aggregate["matched_minus_same_scene_mismatch"]["ci95"][0]>0 and
        aggregate["matched_minus_Safety"]["ci95"][0]>0 and
        sum(within.values())>=2 and collision_total<=collision_eta):
        verdict="STATE_CONDITIONED_GENERALIZATION_CONFIRMED"
    elif any(passes.values()):
        verdict="STATE_CONDITIONING_LEARNED_BUT_TRANSFER_LIMITED"
    else:
        verdict="CROSS_SCENE_STATE_CONDITIONING_NOT_ESTABLISHED"
    result={"schema":"transferable_q_final_consolidation_v1",
            "final_protocol_sha256":sha(HERE/"final_protocol.json"),
            "model_freeze_sha256":sha(HERE/"final_models_frozen.json"),
            "verdict":verdict,"prespecified_loso_causal_scene_pass":passes,
            "within_scene_matched_mismatch_positive_CI":within,
            "aggregate_loso":aggregate,
            "collision_total_state":collision_total,"collision_total_eta_only":collision_eta,
            "panels":panels}
    target=HERE/"final"/"consolidated_results.json"
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    title="phase-1 confirmation" if "phase1" in split else "final confirmation"
    lines=[f"# Transferable physical-only Q(s,η): {title}", "",
           f"**Verdict: `{verdict}`.**", "",
           "All numbers below use frozen, naturally sampled physical states. The stochastic K16 and common fixed-η bank are distinct candidate panels."]
    if "phase1" in split:
        lines += ["", "This is the prospectively frozen first-32-state tranche per scene. A positive or borderline result requires expansion to the remaining untouched TEST states before a full positive claim."]
    frozen_models=read(HERE/"final_models_frozen.json")
    lines += ["", "## Source-only model choice", "",
              "All critic inputs use the masked physical-entity encoding: physical positions, velocities, goals, fixed map geometry and η only. Flow, response/context, policy and controller channels are zeroed. Both critics used identical source rollout labels; the eta-only head has the same downstream width as the state critic.", "",
              "| Panel | Chosen state objective | Source-val concat regret | Source-val residual regret | Source-val eta-only regret |",
              "|---|---|---:|---:|---:|"]
    for name in panel_names:
        choice=frozen_models["panels"][name]
        lines.append(f"| {name} | {choice['state_kind']} | {choice['source_validation_concat']['selection_regret']:.4f} | {choice['source_validation_residual']['selection_regret']:.4f} | {choice['source_validation_eta']['selection_regret']:.4f} |")
    lines += ["", "## Within-scene frozen stochastic K16: B15 / physical states", "",
              "| Scene | Safety | Q(η) | matched Q(s,η) | mismatched state mean | Oracle | state−eta B15 95% CI | state−mismatch B15 95% CI |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels={"toy_give_way":"Toy", "double_bottleneck":"DB", "four_way_intersection":"Four-Way", "ring_exchange":"Ring"}
    for scene in SCENES:
        r=panels[f"within_{scene}"]["proposal"];m=r["methods"];n=r["states"]
        lines.append(f"| {labels[scene]} ({n}) | {m['Safety']['B15_confirmed']}/{n} | {m['Q_eta']['B15_confirmed']}/{n} | {m['Q_state']['B15_confirmed']}/{n} | {m['state_mismatched_100']['mean_B15']*n:.1f}/{n} | {m['Oracle']['B15_confirmed']}/{n} | {ci_pp(r['paired']['state_minus_eta_B15'])} | {ci_pp(r['paired']['state_minus_mismatch_B15'])} |")
    lines += ["", "## Critic-only LOSO, frozen stochastic K16: B15 / physical states", "",
              "| Held-out scene | Safety | Q(η) | matched Q(s,η) | mismatched state mean | Oracle | state−eta B15 95% CI | state−mismatch B15 95% CI |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scene in SCENES:
        r=panels[f"loso_{scene}"]["proposal"];m=r["methods"];n=r["states"]
        lines.append(f"| {labels[scene]} ({n}) | {m['Safety']['B15_confirmed']}/{n} | {m['Q_eta']['B15_confirmed']}/{n} | {m['Q_state']['B15_confirmed']}/{n} | {m['state_mismatched_100']['mean_B15']*n:.1f}/{n} | {m['Oracle']['B15_confirmed']}/{n} | {ci_pp(r['paired']['state_minus_eta_B15'])} | {ci_pp(r['paired']['state_minus_mismatch_B15'])} |")
    lines += ["", "## Common fixed-η bank: independent evaluation half (8 seeds)", "",
              "| Panel | Q(η) | matched Q(s,η) | mismatched | best fixed, discovery | best fixed, posthoc evaluation | Oracle | state−bootstrap-reselected best fixed 95% CI |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in panel_names:
        r=panels[name]["anchor"];m=r["mean_eval_success"];b=r["paired"]["state_minus_bootstrap_reselected_fixed_evalQ"]
        lines.append(f"| {name} ({r['states']}) | {pct(m['Q_eta'])} | {pct(m['Q_state'])} | {pct(m['mismatched_state_100'])} | {pct(m['fixed_discovery_best'])} | {pct(m['fixed_evaluation_posthoc_best'])} | {pct(m['Oracle'])} | {100*b['difference']:+.1f} pp [{100*b['ci95'][0]:+.1f},{100*b['ci95'][1]:+.1f}] |")
    lines += ["", "## State-conditioning mechanism on the common bank", "",
              "| Panel | state-effect corr. | strong reversal pairs | switch accuracy Q(η) | switch accuracy Q(s,η) | selection entropy (bits) | matched−mismatch eval Q 95% CI |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in panel_names:
        r=panels[name]["anchor"];sw=r["switching"];pa=r["paired"]["state_minus_mismatch_evalQ"]
        c="n/a" if r["state_effect_correlation"] is None else f"{r['state_effect_correlation']:.3f}"
        ae="n/a" if sw["switch_accuracy_eta_only"] is None else pct(sw["switch_accuracy_eta_only"])
        ast="n/a" if sw["switch_accuracy_state"] is None else pct(sw["switch_accuracy_state"])
        lines.append(f"| {name} | {c} | {sw['reversal_eta_pairs']} | {ae} | {ast} | {r['selection_entropy_bits']:.2f} | {ci_pp(pa)} |")
    agg=aggregate["matched_minus_same_scene_mismatch"]
    safety_agg=aggregate["matched_minus_Safety"]
    lines += ["", f"Equal-scene-weighted LOSO matched−mismatch mean success: {100*agg['equal_scene_weighted_difference']:+.1f} pp, 95% state-bootstrap CI [{100*agg['ci95'][0]:+.1f}, {100*agg['ci95'][1]:+.1f}], 4 scenes / {agg['states']} physical states.", "",
              f"Equal-scene-weighted LOSO Q(s,η)−Safety mean success: {100*safety_agg['equal_scene_weighted_difference']:+.1f} pp, 95% state-bootstrap CI [{100*safety_agg['ci95'][0]:+.1f}, {100*safety_agg['ci95'][1]:+.1f}].", "",
              "The LOSO statement is limited to **critic** transfer conditional on frozen historical proposal generators. Those generators previously used scene-specific training data; this is not end-to-end zero-shot transfer of the whole pipeline. The common fixed bank removes generator response context from candidate selection, while the frozen FF execution pipeline still uses Flow to execute actions.", "",
              "Every panel's JSON records mean success, oracle regret, B15 bounds, collisions, numerical failures, deadlocks, timeouts, all 16 fixed-policy outcomes, adaptive-demand and reversal groups, state-level paired bootstrap intervals, and individual failure cases. Exact per-seed evidence is in the shared rollout DB and the frozen raw/NPZ artifacts.", ""]
    report=HERE/"final"/"FINAL_REPORT.md"
    report.write_text("\n".join(lines))
    print(json.dumps({"verdict":verdict,"report":str(report),"consolidated":str(target),
                      "loso_causal_scenes":passes,"aggregate_loso_mismatch":aggregate["matched_minus_same_scene_mismatch"]},indent=2))


if __name__=="__main__":main()
