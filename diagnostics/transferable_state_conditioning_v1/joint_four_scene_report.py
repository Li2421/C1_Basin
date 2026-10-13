"""Consolidate fixed-model evidence; no fitting or rollout generation here."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from diagnostics.transferable_state_conditioning_v1.joint_four_scene import HERE, ROOT, SCENES, read, sha

OLD = HERE / "stage_v3_coverage/final_phase1_32"
FRESH = ROOT / "fresh_confirmation"
BOOT = 20000
SEED = 20261008
SHORT = dict(zip(SCENES, ("Toy", "DB", "Four-Way", "Ring")))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def ci(samples, point):
    return {"difference": float(point), "ci95": np.quantile(samples, [.025,.975]).tolist(),
            "bootstrap_unit": "physical state", "replicates": BOOT}


def fixed_and_switching():
    """Extra falsification on already opened full common-bank DEVELOPMENT."""
    report = {}
    for scene in SCENES:
        outcome = np.load(OLD / "final/anchor" / scene / "outcomes.npz")
        v = outcome["values"]
        disc = v[:,:,:8,0].mean(2)
        ev = v[:,:,8:,0].mean(2)
        unknown = v[:,:,8:,4].mean(2)
        n,k = ev.shape
        row = np.arange(n)
        rng = np.random.default_rng(SEED)
        draws = rng.integers(n, size=(BOOT,n), dtype=np.int16)
        counts = np.zeros((BOOT,n), np.int16)
        np.add.at(counts, (np.arange(BOOT)[:,None], draws), 1)
        disc_fixed = disc.mean(0).argmax()
        eval_fixed = ev.mean(0).argmax()
        means_fixed = ev[draws].mean(1)
        a,b = np.triu_indices(k,1)
        d = disc[:,a]-disc[:,b]
        positive,negative = d >= 3/8, d <= -3/8
        eligible = positive.any(0) & negative.any(0)
        for kind, path in (
            ("within", OLD / "predictions" / ("within_"+scene) / "anchor.npz"),
            ("loso", OLD / "predictions" / ("loso_"+scene) / "anchor.npz"),
            ("joint", ROOT / "development/predictions" / ("within_"+scene) / "anchor.npz")):
            p = np.load(path)
            assert np.array_equal(p["state_uids"], outcome["state_uids"])
            chosen = ev[row,p["state_choice"]]
            eta = ev[row,p["eta_choice"]]
            mismatch = ev[row[None,:],p["mismatch_choice"]].mean(0)
            samples = {"state_indices": draws}
            stats = {}
            for label, other in (("eta_only",eta),("mismatch",mismatch),
                                 ("discovery_best_fixed",ev[:,disc_fixed]),
                                 ("evaluation_posthoc_fixed",ev[:,eval_fixed])):
                diff = chosen-other
                samples[label] = diff[draws].mean(1)
                stats[label] = ci(samples[label], diff.mean())
            samples["fixed_reselected_each_bootstrap"] = chosen[draws].mean(1)-means_fixed.max(1)
            stats["fixed_reselected_each_bootstrap"] = ci(samples["fixed_reselected_each_bootstrap"], chosen.mean()-ev.mean(0).max())
            # A constant ranking cannot pass both sides. Evaluation signs must
            # independently confirm the discovery direction; ties score zero.
            two_side = None
            if eligible.any():
                pos,neg = positive[:,eligible],negative[:,eligible]
                e = (ev[:,a]-ev[:,b])[:,eligible]
                z = (p["state_logits"][:,a]-p["state_logits"][:,b])[:,eligible]
                right_pos, right_neg = pos & (e>0) & (z>0), neg & (e<0) & (z<0)
                point_by_pair = (right_pos.sum(0)/pos.sum(0))*(right_neg.sum(0)/neg.sum(0))
                den_pos,den_neg = counts@pos.astype(float),counts@neg.astype(float)
                valid = (den_pos>0)&(den_neg>0)
                scores = (counts@right_pos.astype(float)/np.maximum(den_pos,1))*(counts@right_neg.astype(float)/np.maximum(den_neg,1))
                usable = valid.any(1)
                samples["both_direction_switch"] = (scores*valid).sum(1)[usable]/valid.sum(1)[usable]
                two_side = {"mean_pair_score":float(point_by_pair.mean()),
                            "ci95":np.quantile(samples["both_direction_switch"],[.025,.975]).tolist(),
                            "pair_scores":point_by_pair.tolist(),
                            "pair_indices":np.column_stack((a[eligible],b[eligible])).tolist(),
                            "bootstrap_valid_replicates":int(usable.sum()),
                            "definition":"For each discovery-reversal eta pair, P(correct on A)*P(correct on B), then equal-pair mean; evaluation signs verify both sides; constant ranking scores zero."}
            dest = ROOT / "mechanism_bootstrap" / f"{kind}_{scene}.npz"
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest, **samples)
            freq = np.bincount(p["state_choice"],minlength=k)
            prob = freq[freq>0]/n
            report[kind+"/"+scene] = {
                "scope":"DEVELOPMENT mechanism audit; no new target-based model selection",
                "physical_states":n,"independent_seed_split":"0..7 discovery; 8..15 evaluation",
                "state_eval_success":float(chosen.mean()),"all_16_fixed_eval_success":ev.mean(0).tolist(),
                "discovery_best_fixed_index":int(disc_fixed),"evaluation_posthoc_fixed_index":int(eval_fixed),
                "evaluation_numerical_unknown_seed_count":int(v[:,:,8:,4].sum()),
                "state_minus_best_fixed_worst_case_unknown":float(chosen.mean()-(ev+unknown).mean(0).max()),
                "selection_counts":freq.tolist(),"selection_entropy_bits":float(-np.sum(prob*np.log2(prob))),
                "paired":stats,"both_direction_switch":two_side,
                "bootstrap_artifact":str(dest),"bootstrap_sha256":sha(dest)}
    save(ROOT / "mechanism_audit.json", report)
    return report


def fresh_artifacts():
    results = read(FRESH / "results.json")
    rows, failures = [], []
    for scene in SCENES:
        p = np.load(FRESH / "predictions" / ("within_"+scene) / "proposal.npz")
        v = np.load(FRESH / "final/proposal" / scene / "selected_outcomes.npz")["values"]
        n=len(v); idx=np.arange(n)
        choices={"Safety":np.full(n,16),"joint_Q_eta":p["eta_choice"],"joint_Q_state":p["state_choice"],
                 "frozen_LOSO_Q_state":p["historical_loso_choice"],"frozen_within_Q_state":p["historical_within_choice"]}
        q={k:v[idx,c,:,0].mean(1) for k,c in choices.items()}
        b={k:v[idx,c,:,0].sum(1)>=15 for k,c in choices.items()}
        miss=v[idx[None,:],p["mismatch_choice"],:,0]
        q["mismatch"]=miss.mean(2).mean(0)
        b["mismatch"]=(miss.sum(2)>=15).mean(0)
        draws=np.random.default_rng(SEED).integers(n,size=(BOOT,n))
        boot={"state_indices":draws.astype(np.int16),"state_uids":p["state_uids"]}
        for metric, vals in (("mean_success",q),("B15",b)):
            for name,other in vals.items():
                if name=="joint_Q_state":continue
                diff=vals["joint_Q_state"].astype(float)-other.astype(float)
                boot[metric+"_minus_"+name]=diff[draws].mean(1)
                boot[metric+"_state_differences_"+name]=diff
                expected=results[scene]["paired"]["joint_state_minus_"+name][metric]
                np.testing.assert_allclose(np.quantile(boot[metric+"_minus_"+name],[.025,.975]),expected["ci95"],atol=1e-12)
        np.savez_compressed(FRESH / "final/proposal" / scene / "paired_bootstrap.npz",**boot)
        for i,uid in enumerate(p["state_uids"]):
            row={"scene":scene,"state_uid":str(uid)}
            for name in q:
                row[name+"_mean_success"]=float(q[name][i]);row[name+"_B15"]=float(b[name][i])
            for name,c in choices.items():row[name+"_eta_index"]=int(c[i])
            rows.append(row)
            if not b["joint_Q_state"][i] or q["joint_Q_state"][i]<q["joint_Q_eta"][i]:
                failures.append({**row,"matched_eta":p["eta"][i,p["state_choice"][i]].astype(float).tolist(),
                                 "eta_baseline_eta":p["eta"][i,p["eta_choice"][i]].astype(float).tolist()})
    with (FRESH / "per_state_methods.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(FRESH / "failure_cases.json",failures)
    return results


def pp(result):
    lo,hi=result["ci95"]
    return f"{100*result['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"


def main():
    mechanism=fixed_and_switching()
    fresh=fresh_artifacts()
    old=read(OLD / "final/consolidated_results.json")
    dev=read(ROOT / "development/results.json")
    prior=read(HERE / "posthoc_cause_audit/prior_intervention.json")
    info=read(HERE / "posthoc_cause_audit/controller_information.json")
    protocol=read(ROOT / "protocol.json")
    gate=read(FRESH / "final_protocol.json")
    lines=["# Physical-only state conditioning: joint-four test and causal diagnosis", "",
        "**Verdict for source-only four-scene LOSO: `CROSS_SCENE_STATE_CONDITIONING_NOT_ESTABLISHED`.**",
        "Within-scene state use is established in Toy and previously in Ring under its separately validated 512-state execution profile. The current four-scene experiment has not established transferable within-target-scene adaptation. The joint-four control is generalization to new states of seen scenes, not LOSO.","",
        "## Frozen protocol and cache reuse", "",
        "One joint physical-entity + eta critic and a fair eta-only critic were fitted to exactly the existing TRAIN labels: Toy 256, DB 48, Four-Way 48, Ring 112 independent states (464 total), plus 32/16/16/16 independent source validation states (80 total). All source labels are Q4 on K16. Equal-scene sampling and optimization budgets were shared. Concat and prior-plus-ranking-residual were selected using source validation only; concat was selected. No scene ID, Flow, response, controller fingerprint or rollout preview enters the model.",
        "No new TRAIN rollout was needed. Existing full K16 and common-bank outcomes on 32 states per scene were reused only for DEVELOPMENT diagnosis. A different 32 states per scene (128 total) were frozen from the original unfiltered TEST sampler order before outcomes; all four scenes are represented in TRAIN, none of these physical states is. Model, predictions and 100 within-scene derangements were frozen before fresh Q16 generation.",
        f"Fresh confirmation requested {sum(x['requested_seeds'] for x in gate['panels'].values()):,} selected-policy seed outcomes; exact preflight reused {sum(x['reused_seeds'] for x in gate['panels'].values()):,}, leaving {sum(x['missing_seeds'] for x in gate['panels'].values()):,} genuinely missing. Simulate the union of joint, fair eta-only, frozen LOSO, frozen within, 100 mismatches and Safety selections, once per exact state/eta/controller/seed. This avoids regenerating a 34,816-seed full pool. Unmeasured entries are explicitly -1; **there is no fresh full-pool oracle or best-fixed certificate**. Complete all-fixed falsification is reported separately on the existing common-bank development panel.",
        "B15 means at least 15 successes in 16 predeclared seeds; numerical failures remain unresolved upper/lower bounds. All paired intervals use 20,000 physical-state bootstrap resamples. No video is required for this pure Q audit. CPU rollout concurrency is capped at 12 (at least 5 of 24 CPUs remain outside this task).", "",
        "## New test: train on four scenes, test on fresh states of all four", "",
        "Main values are B15 states /32. Mismatch is the mean count over 100 derangements, not another independent sample.","",
        "| Scene | Safety | Joint Q(eta) | Joint Q(s,eta) | Mismatched state | Frozen LOSO state | Frozen within state |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s in SCENES:
        m=fresh[s]["methods"]
        vals=[f"{m[k]['B15_confirmed']}/32" for k in ("Safety","joint_Q_eta","joint_Q_state")]
        vals += [f"{m['mismatched_state_100']['mean_B15_count']:.2f}/32"]
        vals += [f"{m[k]['B15_confirmed']}/32" for k in ("frozen_LOSO_Q_state","frozen_within_Q_state")]
        lines.append("| "+SHORT[s]+" | "+" | ".join(vals)+" |")
    lines += ["", "Four-Way has numerical unresolved bounds: joint state is 29 confirmed /32, up to 32 possible; joint eta-only is 30 confirmed, up to 32 possible. All non-numerical selected Four-Way trials succeed. Its small negative differences below are differences in certified-success lower estimates, not established physical-outcome degradation. JSON also reports paired worst/best-case numerical sensitivity; do not claim state hurts from this row.", "",
              "Mean selected rollout success (%), followed by paired differences in percentage points and 95% CI:","",
              "| Scene | Safety | Joint eta | Joint state | Mismatch | State − eta | State − mismatch | State − Safety |",
              "|---|---:|---:|---:|---:|---|---|---|"]
    for s in SCENES:
        m=fresh[s]["methods"];p=fresh[s]["paired"]
        rates=[f"{100*m[k]['mean_success_lower']:.2f}" for k in ("Safety","joint_Q_eta","joint_Q_state")]
        rates += [f"{100*m['mismatched_state_100']['mean_success']:.2f}"]
        comparisons=[pp(p['joint_state_minus_'+k]['mean_success']) for k in ('joint_Q_eta','mismatch','Safety')]
        lines.append("| "+SHORT[s]+" | "+" | ".join(rates+comparisons)+" |")
    lines += ["", "| Scene | B15 state − eta (pp, CI) | B15 state − mismatch (pp, CI) | B15 state − Safety (pp, CI) |", "|---|---|---|---|"]
    for s in SCENES:
        p=fresh[s]['paired'];lines.append("| "+SHORT[s]+" | "+" | ".join(pp(p['joint_state_minus_'+k]['B15']) for k in ('joint_Q_eta','mismatch','Safety'))+" |")
    lines += ["", "## Original completed confirmation: within-scene versus LOSO", "",
              "These are the earlier, separate 32 states per scene; do not pool them with the new test after diagnostic use. Oracle and all K16 candidates were actually measured here.","",
              "| Regime | Scene | Safety | Eta-only | State | Same-scene mismatch mean | Oracle |", "|---|---|---:|---:|---:|---:|---:|"]
    for regime in ('within','loso'):
        for s in SCENES:
            m=old['panels'][regime+'_'+s]['proposal']['methods']
            vals=[str(m[k]['B15_confirmed'])+'/32' for k in ('Safety','Q_eta','Q_state')]
            vals += [f"{32*m['state_mismatched_100']['mean_B15']:.2f}/32",str(m['Oracle']['B15_confirmed'])+'/32']
            lines.append('| '+regime+' | '+SHORT[s]+' | '+' | '.join(vals)+' |')
    lines += ["", "## State mechanism and all-fixed falsification", "",
              "Common fixed K16 bank, all eligible pairs; discovery threshold 3/8, independent evaluation half. `Both-side switch` requires correct positive ranking in Group A AND correct negative ranking in Group B. Unlike unbalanced pair-state accuracy, a fixed ranking necessarily scores zero. Equal-pair score is bootstrapped by resampling physical states, never pair cases.","",
              "| Regime | Scene | State effect corr. | True reversal pairs | Both-side switch | Selection entropy (bits) | State eval success | Best fixed eval success | State − bootstrap-reselected fixed (pp, CI) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for kind in ('within','loso','joint'):
        for s in SCENES:
            m=mechanism[kind+'/'+s]
            if kind=='joint':
                d=dev[s+'/anchor'];corr=d['state_effect_correlation'];rev=d['switching']['reversal_eta_pairs']
            else:
                d=old['panels'][kind+'_'+s]['anchor']
                corr=d.get('state_effect_correlation')
                if isinstance(corr,dict):corr=corr.get('Q_state')
                rev=len(m['both_direction_switch']['pair_indices']) if m['both_direction_switch'] else 0
            sw=m['both_direction_switch'];st='—' if sw is None else f"{100*sw['mean_pair_score']:.1f}%"
            cs='—' if corr is None else f'{corr:.3f}'
            lines.append('| '+kind+' | '+SHORT[s]+f" | {cs} | {rev} | {st} | {m['selection_entropy_bits']:.3f} | {100*m['state_eval_success']:.2f}% | {100*max(m['all_16_fixed_eval_success']):.2f}% | "+pp(m['paired']['fixed_reselected_each_bootstrap'])+' |')
    lines += ["", "All LOSO models choose a single fixed eta on the common bank. Their decision value cannot exceed the best of those 16 fixed policies and their correct-state value is exactly zero there. In joint training, Ring changes to the appropriate scene-level fixed eta, but still does not switch across Ring states. Toy joint training does use the within-scene state signal, while its common-bank success does not reliably exceed the posthoc best fixed policy with this sample.","",
        "## Direct diagnosis: two separable failures", "",
        "1. **Wrong eta prior on an unseen execution regime.** A diagnostic intercept adjustment uses target discovery-half mean Q for each eta, without fitting any state dependence. On independent evaluation seeds it changes LOSO selected success Toy 44.53→90.23%, DB 100→100%, Four-Way 92.97→100%, Ring 0.39→82.42%. All four then equal their best fixed eta and have zero matched-versus-mismatched gain. This is a privileged causal diagnostic, **not a source-only LOSO repair**: it uses target labels. It shows the catastrophic errors can be removed by correcting the eta main effect alone, without recovering state adaptation.",
        "2. **Missing transferred state×eta interaction.** Correcting the eta prior still leaves no decision-level use of target state. Training on all four scenes restores Ring success largely by learning the Ring eta prior; same-scene mismatch and common-bank fixed-policy checks expose this. Toy provides a positive control showing the same code and physical input can learn a real within-scene interaction.",
        "The supported training limitation is coverage of mechanisms and interactions, not merely the number of rollout rows. DB/Four source proposal labels are close to ceiling (mean Q approximately .936/.974), so many states provide little selection contrast. Ring has only 112 current-profile source states, and only 16 source states use the broad fixed bank. Source validation may reward a good scene prior even when no within-target-state interaction transfers. A ranking residual and source-state expansion were already tested; neither established LOSO matched-state value. Q4 seed-half preference signs persist at roughly 98% in informative Toy/Ring cases, so label noise alone is not a supported explanation.",
        "The source-zero Ring curvature channels really are untrained, but setting those channels to source-zero changes **none** of the current Ring selections. Therefore their arbitrary learned weights are not the direct explanation for these failed decisions. Ring target eta support is close to source support; candidate coverage alone also cannot explain Ring. Toy has a separate eta-support shift (60.4% of candidate vectors have some coordinate outside source bounds); this is a contributing confound, not a universal cause.","",
        "## Is the same Q(s,eta) input theoretically valid within-scene and in LOSO?", "",
        "**The same input signature is valid; it does not guarantee the same target function.** With execution contract c fixed, the target is Q^c(s,eta)=P(success | s,eta,c). A within-scene critic can learn that fixed controller, dynamics and evaluator implicitly in its weights. To identify a controller-independent target from physical state alone across changing contracts would require Q^{c1}(s,eta)=Q^{c2}(s,eta) on overlapping physical inputs, or another justified invariant mechanism. Otherwise Q(s,eta) is at best a conditional average over the training distribution of c; changing that distribution at deployment changes the target.",
        "The current encoding already includes relative agent/goal/obstacle geometry, radii and masks. It does not omit the map. It deliberately removes Flow, controller response, policy-frame information and global execution constants. The four pipelines have different frozen Flow hashes, safety fingerprints and horizons (Toy/DB 850, Four 1100, Ring 700). Thus current LOSO changes both geometry and execution mechanism. Joint-four training can infer seen regimes from physical geometry without an explicit scene ID; that is compatible with strong scene-prior shortcuts and does not establish state-level transfer.",
        f"An exact archived controller intervention was re-audited in the shared database, reusing {info['reused_seed_records']:,} seed records and no new simulation: the same physical state, eta, current Flow reference, step-0 action and matched seeds are kept; only the future frozen Flow from step 1 changes. {info['16of16_to_0of16_pairs']} of 64 state–eta pairs change from 16/16 to 0/16; {info['holm_significant_pairs']} pairs pass Holm correction. Any deterministic physical-only encoding is identical in each such pair. This proves information insufficiency **across that allowed controller family**. It does not prove equal-input contradictory labels within the present four fixed-scene supports, nor quantify what fraction of this LOSO failure is due to hidden controller conditions.",
        "Archived short-response context ablations already attempted to expose controller behavior and still failed LOSO (selected Ring 1/60 versus eta-only 27/60 under that historical protocol). These are diagnosis-only controls from a different execution/data cohort, not current baselines. Adding arbitrary context is therefore neither a demonstrated repair nor part of the final physical-only method.","",
        "## What the evidence supports, and the corresponding method", "",
        "- For new states of known frozen scene/controller regimes, keep Q^c(s,eta), reuse compatible labels, and demand matched-state versus within-scene mismatch evidence. Joint-four training addresses this deployment scope; it must not be described as LOSO.",
        "- For unseen scene AND unseen frozen controller, additional physical states of the same few source mechanisms do not establish identifiability. A physical-only transfer claim needs a justified shared execution mechanism or source diversity that demonstrates invariant state×eta effects, followed by untouched target confirmation. With the pipeline fixed as requested, the present source-only model fails that empirical test.",
        "- The current failure is not established as generic memorization or a broken state input. It is observed loss of state×eta transfer plus a wrong domain eta prior. Source training and source validation can improve while this domain relation fails. No test here proves that all future physical-only architectures must fail, and no controller-conditioned deployment is being proposed as an already validated solution.","",
        "## Safety, failures and reproducibility", "",
        "All fresh per-state methods, exact seed records, numerical bounds, failure states and source manifests are retained. Collision counts are checked independently of success. Numerical failures are not silently recoded as scientific failures; B15 is reported conservatively as confirmed success, with possible counts retained in JSON.","",
        "Shared working-tree MACFlow source had changed for an unrelated experiment since protocol freeze. Exact controller-identity guards rejected 68 first-seed attempts (zero accepted labels). The original source with matching SHA was restored in an isolated module snapshot; all original controller payload fields and UIDs were verified before continuing. Shared edits and all completed outcomes were preserved. These 68 rejected attempts are additional execution overhead, not part of the 8,832 accepted seed labels. See `frozen_execution_restore.json` and `rejected_identity_attempts.json`.","",
        "| Scene | All executed collisions | All executed numerical failures | Joint selected collisions | Joint selected numerical failures |", "|---|---:|---:|---:|---:|"]
    for s in SCENES:
        d=fresh[s];m=d['methods']['joint_Q_state']
        lines.append(f"| {SHORT[s]} | {d['all_observed_collisions']} | {d['all_observed_numerical_failures']} | {m['collisions']} | {m['numerical_failures']} |")
    lines += ["", "Artifacts:","",
        "- `protocol.json`, `source_data/joint_four_scenes/manifest.json`, `final_models_frozen.json`: source-only model and split provenance.",
        "- `fresh_confirmation/final_protocol.json`, `frozen_splits.json`, `predictions/`: prospectively frozen 128 fresh states, candidate choices and 100 mismatch controls.",
        "- `fresh_confirmation/final/proposal/*/{cases.json,cache_preflight.json,selected_outcomes.npz,paired_bootstrap.npz,safety_issues.json}`: exact state/eta/seed cases, raw labels and bootstrap draws/outputs.",
        "- `fresh_confirmation/results.json`, `per_state_methods.csv`, `failure_cases.json`: final paired results and every state-level outcome.",
        "- `development/results.json`, `mechanism_audit.json`, `mechanism_bootstrap/`: all-16-fixed and both-direction switching audits, with independent discovery/evaluation halves and bootstrap re-selection of the fixed comparator.",
        "- `../posthoc_cause_audit/{results.json,prior_intervention.json,controller_information.json}`: direct interventions and evidence boundaries.",
        "- Earlier complete within/LOSO results: `../stage_v3_coverage/final_phase1_32/final/INTERPRETATION.md` and `consolidated_results.json`.","",
        "Paper claim: physical state has demonstrated within-scene selection value, but the tested physical-only critics have not established transferable within-held-out-scene eta adaptation. Joint training on all four scenes is a different, easier generalization question and must be reported separately."]
    (ROOT / "FINAL_SYNTHESIS.md").write_text("\n".join(lines)+"\n")
    save(ROOT / "report_manifest.json",{"report_sha256":sha(ROOT/'FINAL_SYNTHESIS.md'),
        "fresh_results_sha256":sha(FRESH/'results.json'),"mechanism_audit_sha256":sha(ROOT/'mechanism_audit.json'),
        "report_code_sha256":sha(Path(__file__)),"verdict":"CROSS_SCENE_STATE_CONDITIONING_NOT_ESTABLISHED"})
    print("JOINT_REPORT_AND_BOOTSTRAPS_COMPLETE",flush=True)


if __name__=='__main__':main()
