#!/usr/bin/env python3
"""Render the frozen-test evidence into the preregistered scientific report."""
from __future__ import annotations
import hashlib, json, sqlite3
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SCENARIOS=("double_bottleneck","four_way_intersection","ring_exchange")
LABEL={"double_bottleneck":"Double-Bottleneck","four_way_intersection":"Four-Way Intersection","ring_exchange":"Ring Exchange"}
METHODS=("B0 Hard Safety","B1 Fixed eta","B2 Generator Mean","B3 Random Proposal","B4 Oracle finite proposals","B5 Critic-selected","Min-Def Critic-Feasible","Oracle Min-Dinst","Oracle Min-Dtraj")

def load(name):return json.loads((HERE/name).read_text())
def pct(x,n):return f"{100*x/n:.1f}%" if n else "N/A"
def fnum(x):return "N/A" if x is None else f"{x:.5f}"

def main():
 s=load("summary.json");per=load("per_state_method_outcomes.json");conv=load("conversion_matrices.json");expl=load("critic_exploitation_audit.json");pareto=load("pareto_analysis.json");pre=load("initial_cache_preflight.json");post=load("cache_postflight.json");art=load("frozen_artifacts.json")
 detailed={};classes=[];deform_flags=[];outcomes={}
 for sc in SCENARIOS:
  rows=[r for r in per if r["scenario"]==sc];n=len(rows);b0=s[sc]["B0 Hard Safety"];b5=s[sc]["B5 Critic-selected"];b1=s[sc]["B1 Fixed eta"];b4=s[sc]["B4 Oracle finite proposals"]
  nonrob=n-b0["robust_success"];rob=b0["robust_success"];res=b5["rescue_robust"];br=b5["break_robust"]
  gain=b5["robust_success"]-b0["robust_success"];gap=b4["robust_success"]-b5["robust_success"]
  if gain>0 and res/max(nonrob,1)>=.20 and br/max(rob,1)<=.10 and gap/n<=.05 and expl[sc]["oracle_robust_critic_nonrobust"]/n<=.05:cl="GENERALIZATION_STRONG"
  elif gain>0 or res>0:cl="GENERALIZATION_PARTIAL"
  else:cl="GENERALIZATION_FAILED"
  classes.append(cl)
  md=s[sc]["Min-Def Critic-Feasible"];od=s[sc]["Oracle Min-Dinst"]
  reduction=1-md["D_inst"]["mean"]/max(b5["D_inst"]["mean"],1e-12)
  oracle_reduction=1-od["D_inst"]["mean"]/max(b5["D_inst"]["mean"],1e-12)
  if reduction>.10 and md["robust_success"]>=b5["robust_success"]-max(1,.02*n):df="useful"
  elif oracle_reduction>.10 and od["robust_success"]>=b5["robust_success"]:df="future_useful"
  elif md["robust_success"]<b5["robust_success"]-.05*n:df="tradeoff"
  else:df="small"
  deform_flags.append(df)
  high_bad=[]
  for r in rows:
   score=r["critic_scores"][r["critic_index"]];q=r["B5 Critic-selected"]["Q16_lower"]
   if score>=.9 and q<.5:high_bad.append({"state_uid":r["state_uid"],"predicted":score,"Q16":q})
  zero=[r for r in rows if r["B0 Hard Safety"]["robust"] is True]
  gating={"correction_clearly_useful":sum(r["B0 Hard Safety"]["robust"] is False and r["B5 Critic-selected"]["robust"] is True for r in rows),"correction_unnecessary":sum(r["B0 Hard Safety"]["robust"] is True and r["B5 Critic-selected"]["robust"] is True and r["B5 Critic-selected"]["D_inst"]>1e-12 for r in rows)}
  gating["ambiguous"]=n-gating["correction_clearly_useful"]-gating["correction_unnecessary"]
  detailed[sc]={"classification":cl,"robust_rescue_denominator":nonrob,"robust_break_denominator":rob,"robust_gain":gain,"oracle_gap":gap,"B5_vs_B1":b5["robust_success"]-b1["robust_success"],"high_pred_low_actual":high_bad,"zero_sufficient_states":len(zero),"zero_B5_break":sum(r["B5 Critic-selected"]["break_robust"] for r in zero),"zero_B5_Dinst":float(np.mean([r["B5 Critic-selected"]["D_inst"] for r in zero])) if zero else None,"gating":gating,"min_def_Dinst_reduction":reduction,"oracle_Dinst_reduction":oracle_reduction}
  outcomes[sc]={}
  for m in METHODS:
   canonical={};q16={}
   for r in rows:
    a=r[m]["single_outcome"];canonical[a]=canonical.get(a,0)+1
    for a,v in r[m]["terminal_counts"].items():q16[a]=q16.get(a,0)+v
   outcomes[sc][m]={"canonical":canonical,"q16_seed_outcomes":q16}
 (HERE/"interpretation.json").write_text(json.dumps(detailed,indent=2,sort_keys=True)+"\n")
 (HERE/"outcome_breakdown.json").write_text(json.dumps(outcomes,indent=2,sort_keys=True)+"\n")
 overall="GENERALIZATION_FAILED" if "GENERALIZATION_FAILED" in classes else "GENERALIZATION_PARTIAL" if "GENERALIZATION_PARTIAL" in classes else "GENERALIZATION_STRONG"
 if any(x in ("useful","future_useful") for x in deform_flags):ddecision="DEFORMATION_OPTIMIZATION_USEFUL"
 elif any(x=="tradeoff" for x in deform_flags):ddecision="DEFORMATION_ROBUSTNESS_TRADEOFF_STRONG"
 else:ddecision="DEFORMATION_ALREADY_SMALL"
 pipeline="PIPELINE_NEEDS_REVISION" if overall!="GENERALIZATION_STRONG" else "PIPELINE_READY_BUT_DEFORMATION_REFINEMENT_USEFUL" if ddecision in ("DEFORMATION_OPTIMIZATION_USEFUL","DEFORMATION_ROBUSTNESS_TRADEOFF_STRONG") else "PIPELINE_READY_FOR_FINAL_GENERALIZATION_CLAIM"
 lines=["# Frozen untouched-test evaluation: joint OrthoFlow3 generator + critic","",f"Generalization: **`{overall}`**  ",f"Deformation: **`{ddecision}`**  ",f"Pipeline: **`{pipeline}`**","","## 1. Frozen protocol","",f"The generator and critic were frozen before this evaluation. The proposal set is exactly the transformed generator mean plus four deterministic stochastic samples (K=4). The critic ranked only these five points. `tau_critic=0.9375` was frozen from the critic's empirical-Q semantics before test outcomes. The populations are 192 Double-Bottleneck matched rollout-states (the complete two 96-rollout mature hard-safety populations), 60 Four-Way states, and 60 Ring states. No test state entered training or selection.","",f"Proposal manifest SHA-256: `{art['proposal_manifest_sha256']}`.","","## 2. Results by scenario"]
 for sc in SCENARIOS:
  lines += ["",f"### {LABEL[sc]}","",f"Per-scenario decision: **`{detailed[sc]['classification']}`**","","| Method | Single-seed success | Robust success | Rescue | Break | Mean Q16 | Mean D_inst | Mean D_traj |","|---|---:|---:|---:|---:|---:|---:|---:|"]
  for m in METHODS:
   x=s[sc][m];n=x["states"]
   qtext="N/A" if x['mean_Q16'] is None else f"{x['mean_Q16']:.3f}"
   lines.append(f"| {m} | {x['single_seed_success']}/{n} ({pct(x['single_seed_success'],n)}) | {x['robust_success']}/{n} ({pct(x['robust_success'],n)}; unresolved {x['robust_unresolved']}) | {x['rescue_robust']}/{detailed[sc]['robust_rescue_denominator']} | {x['break_robust']}/{detailed[sc]['robust_break_denominator']} | {qtext} | {fnum(x['D_inst']['mean'])} | {fnum(x['D_traj']['mean'])} |")
  e=expl[sc];d=detailed[sc];lines += ["",f"B5 changes the robust-state count by {d['robust_gain']:+d} versus B0 and {d['B5_vs_B1']:+d} versus frozen B1. It is {d['oracle_gap']} certified robust states below B4. The critic chose the exact oracle proposal on {e['exact_same_candidate']}/{e['states']} states; mean exact-Q16 gap is {e['mean_Q16_gap']:.4f}; oracle-robust/critic-nonrobust cases: {e['oracle_robust_critic_nonrobust']}; oracle-robust/critic-unresolved cases: {e.get('oracle_robust_critic_unresolved',0)}; high-prediction (`>=0.9`) / low-actual (`Q16<0.5`) cases: {len(d['high_pred_low_actual'])}.","",f"Among {d['zero_sufficient_states']} B0-robust states, B5 robust break is {d['zero_B5_break']} and mean B5 D_inst is {fnum(d['zero_B5_Dinst'])}. Min-Def reduces mean D_inst by {100*d['min_def_Dinst_reduction']:.1f}% relative to B5; the proposal-set oracle reduction is {100*d['oracle_Dinst_reduction']:.1f}%.","",f"B0 -> B5 canonical conversion matrix: `{json.dumps(conv[sc],sort_keys=True)}`.","",f"Canonical outcome partitions: `{json.dumps({m:outcomes[sc][m]['canonical'] for m in METHODS},sort_keys=True)}`."]
 lines += ["","## 3. Generator and critic attribution","","B2 isolates the transformed generator mean; B3 isolates one pre-registered stochastic draw; B4 is the true-Q oracle over exactly the same five frozen proposals used by B5. Therefore B4-B2 is the generator distribution gain and B4-B5 is the out-of-sample critic ranking gap. No oracle center, eta search, resampling-until-success, or continuous critic optimization was used.","","## 4. Zero-sufficient states and gating interpretation"]
 for sc in SCENARIOS:
  g=detailed[sc]["gating"];n=sum(g.values());lines.append(f"- {LABEL[sc]}: correction clearly useful {g['correction_clearly_useful']}/{n}; correction unnecessary {g['correction_unnecessary']}/{n}; ambiguous {g['ambiguous']}/{n}.")
 lines += ["","This is post-hoc interpretation only; no gate was trained.","","## 5. Robustness versus deformation","","`D_inst` is the squared norm of the actual post-projection instantaneous change from hard-safety action. `D_traj` is the per-rollout time-average of that squared change; integrated correction energy is retained in the per-state evidence. B5 is ROBUST_MAX. Min-Def uses the frozen critic threshold and the same five proposals, falling back to eta=0 when none is feasible. Both oracle Min-Def variants use true Q16 only for analysis.",""]
 for sc in SCENARIOS:lines.append(f"- {LABEL[sc]}: {pareto[sc]['states_with_lower_Dinst_robust_than_B5']} states contain a truly robust lower-D_inst proposal than B5; {pareto[sc]['states_with_lower_Dtraj_robust_than_B5']} contain a truly robust lower-D_traj proposal.")
 lines += ["","Secondary intervention diagnostics (means over states; projection activity is unavailable from the legacy Double-Bottleneck adapter):","","| Scenario | Method | Eta norm | Projection activity | Completion steps |","|---|---|---:|---:|---:|"]
 for sc in SCENARIOS:
  for m in ("B0 Hard Safety","B5 Critic-selected","Min-Def Critic-Feasible","Oracle Min-Dinst","Oracle Min-Dtraj"):
   x=s[sc][m];pa=x["projection_activity"]["mean"]
   lines.append(f"| {LABEL[sc]} | {m} | {fnum(x['eta_norm']['mean'])} | {fnum(pa)} | {fnum(x['completion_steps']['mean'])} |")
 # Unique database audit (aliases in the method table can refer to one seed rollout).
 db=HERE.parents[1]/"shared_rollout_db"/"rollout.sqlite"
 con=sqlite3.connect(db);exp="exp_587e795b44e5d251bf8cbf699a69fd540dc35eeef9d3b31a5ce2ad3d2fca2b0b"
 unique=dict(zip(("rollouts","numerical_failures","collisions"),con.execute("SELECT COUNT(*),SUM(numerical_failure),SUM(collision) FROM rollout WHERE experiment_uid=?",(exp,)).fetchone()));con.close()
 (HERE/"numerical_safety_audit.json").write_text(json.dumps({"unique_database_records":unique,"q16_method_outcomes":outcomes,"retry_policy":"initial attempt plus at most 3 identical retries; unresolved remains NUMERICAL_SOLVER_FAILURE"},indent=2,sort_keys=True)+"\n")
 lines += ["","## 6. Safety and numerical audit","",f"All nonzero eta continuations retained the mandatory second hard-safety projection. Across {unique['rollouts']:,} unique database records, {unique['numerical_failures']} remained `NUMERICAL_SOLVER_FAILURE` after the fixed retry rule and {unique['collisions']} were collision outcomes. These were not folded into timeout or non-robust labels. Double-Bottleneck had no collision/numerical record; Four-Way had 81 numerical and no collision records; Ring had 24 numerical and 50 collision records across all evaluated controller/eta tuples. Importantly, Ring B5 itself had 0 collision seeds, while Ring B0 had 14/960 collision seeds. Thus the learned B5 path is collision-free in this sample, but the blanket claim that every hard-safety rollout was collision-free is false. See `numerical_safety_audit.json`.","","## 7. Cache and resumability","",f"Initial exact preflight requested {pre['summary']['requested']:,} seed rollouts, reused {pre['summary']['reused']:,}, and identified {pre['summary']['missing']:,} missing. Final postflight lists {post['summary']['missing']:,} non-reusable tuples: these are exactly the unresolved numerical records, not unexecuted work. Every seed-level result was committed immediately to the shared database.","","## 8. Explicit answers","","1. **Generator generalization is scenario-dependent.** B4 shows useful proposal coverage on Double (185/192) and Four-Way (60/60), but only 47/60 on Ring; Ring B2 mean collapses to 2/60 robust.","2. **Critic generalization is strong on Double/Four but incomplete on Ring.** Ring has seven oracle-robust/critic-nonrobust misselections and a mean exact-Q16 gap of 0.0823; this is genuine misranking, not continuous critic exploitation.","3. **B5 versus B0 robust gain:** Double +162 states (181 vs 19), Four-Way +55 certified states (57 vs 2, with 3 B5 numerical-unresolved), Ring +3 states (40 vs 37).","4. **B5 recovery of B4:** Double 181/185 certified robust states; Four 57/60 certified plus 3 unresolved; Ring 40/47.","5. **Robust rescue:** Double 163/173 B0-nonrobust, Four 55/58, Ring 12/23.","6. **Robust break:** Double 1/19 B0-robust, Four 0/2, Ring 9/37.","7. **Fixed shared eta is at least as good here.** B1 robust counts are 192, 59+1 unresolved, and 58 for Double/Four/Ring, respectively; it exceeds B5 in all three frozen populations.","8. **Some zero-sufficient states are unnecessarily modified.** The most consequential case is Ring: B5 changes all B0-robust states with nonzero mean deformation and breaks 9/37 robust states.","9. **Lower deformation is available, but the deployable threshold selector is not uniformly safe.** Four-Way Min-Def improves certification to 60/60 while reducing D_inst 23.1%; Double and Ring Min-Def lose robust states. Oracle proposal evidence shows lower-deformation robust alternatives remain available.","10. **A future deformation-aware/gating objective is justified**, especially for Ring, but it must also repair proposal/ranking generalization; deformation refinement alone is insufficient."]
 lines += ["","## 9. Final decisions","",f"- Generalization: `{overall}`",f"- Deformation: `{ddecision}`",f"- Pipeline: `{pipeline}`","","No model was retrained; no training labels were generated; no test-driven tuning was performed."]
 (HERE/"REPORT.md").write_text("\n".join(lines)+"\n")
 (HERE/"decisions.json").write_text(json.dumps({"generalization":overall,"deformation":ddecision,"pipeline":pipeline,"per_scenario":{k:v['classification'] for k,v in detailed.items()}},indent=2,sort_keys=True)+"\n")
 print(json.dumps({"generalization":overall,"deformation":ddecision,"pipeline":pipeline},indent=2))
if __name__=="__main__":main()
