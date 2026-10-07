"""Render qualification figures in the lightweight plotting environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
RAW = OUT / "raw/stage1"
FIG = OUT / "figures"
PHIS = ("zero", "goal025", "damp035", "relative025")
K_VALUES = (20, 100)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    p_result = json.loads((OUT / "property_p.json").read_text())
    proj = json.loads((OUT / "property_proj.json").read_text())
    risks = json.loads((RAW / "risk_records.json").read_text())
    initial20 = [x for x in risks if x["checkpoint"] == "initial" and x["K"] == 20]
    rmean = [np.mean([x["value"] for x in initial20 if x["phi_name"] == p]) for p in PHIS]
    qrate = [np.mean([x["eventual_deadlock"] for x in initial20 if x["phi_name"] == p]) for p in PHIS]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    fig, ax = plt.subplots(figsize=(6, 4))
    label_offsets = {
        "zero": (5, 7), "goal025": (5, 7),
        "damp035": (5, -15), "relative025": (-72, 7),
    }
    for name, x, y, color in zip(PHIS, rmean, qrate, colors):
        ax.scatter(x, y, s=70, color=color)
        ax.annotate(name, (x, y), xytext=label_offsets[name], textcoords="offset points")
    ax.set(xlabel="mean frozen R_20", ylabel="full-horizon strict-deadlock rate", title="Independent Stage-1 R vs eventual outcome")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG / "r_vs_full_q.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(PHIS)); width=.38
    ax.bar(x-width/2,rmean,width,label="mean R_20"); ax.bar(x+width/2,qrate,width,label="deadlock rate")
    ax.set_xticks(x,PHIS,rotation=15); ax.set_ylim(0,1.05); ax.set_title("Closed-loop policy ranking (Stage 1 mixture)")
    ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FIG/"closed_loop_policy_ranking.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,3.5));ax.axis("off")
    ax.text(.5,.6,"G NOT TESTABLE",ha="center",va="center",fontsize=18,weight="bold")
    ax.text(.5,.38,"Frozen B is defined only for four discrete phi IDs;\ninterpolation/extrapolation is forbidden.",ha="center",va="center")
    fig.tight_layout();fig.savefig(FIG/"gradient_direction_vs_qd.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,4))
    data=[[x["executed_action_rms_difference"] for x in proj["records"] if x["K"]==K] for K in K_VALUES]
    ax.boxplot(data,tick_labels=[f"K={K}" for K in K_VALUES],showfliers=False);ax.axhline(1e-6,color="red",linestyle="--",linewidth=1,label="collapse threshold")
    ax.set_yscale("log");ax.set_ylabel("executed-action RMS separation");ax.set_title("Hard-projected trajectory separation (Stage 1)")
    ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FIG/"executable_trajectory_separation.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    for label,deadlock in [("eventual deadlock",True),("other outcome",False)]:
        xs=[];ys=[]
        for offset in (300,150,40):
            rows=[x for x in risks if x["K"]==20 and x["checkpoint"]==f"terminal_minus_{offset}" and x["eventual_deadlock"]==deadlock and x["steps_to_terminal"]>20]
            if rows: xs.append((offset-20)*.05);ys.append(np.mean([x["value"] for x in rows]))
        ax.plot(xs,ys,marker="o",label=label)
    ax.set(xlabel="lead after observed K=20 prefix (s)",ylabel="mean R_20",title="R_K versus time remaining to terminal event")
    ax.invert_xaxis();ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/"risk_and_q_vs_time_to_deadlock.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4));labels=[];vals=[];lows=[];highs=[]
    for item in p_result["strata"]:
        labels.append(f"K{item['K']}\n{item['checkpoint'].replace('terminal_minus_','-')}");vals.append(item["auroc"]);lows.append(item["auroc_ci95"][0]);highs.append(item["auroc_ci95"][1])
    x=np.arange(len(vals));ax.errorbar(x,vals,yerr=[np.array(vals)-np.array(lows),np.array(highs)-np.array(vals)],fmt="o",capsize=3)
    ax.axhline(.5,color="black",linestyle="--");ax.set_xticks(x,labels,rotation=30,ha="right");ax.set_ylim(0,1.03);ax.set_ylabel("AUROC");ax.set_title("Frozen-K early-prediction ablation")
    ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FIG/"k_ablation.png",dpi=150);plt.close(fig)

    dead=[x for x in risks if x["eventual_deadlock"] and isinstance(x["K"],int)]
    fig,ax=plt.subplots(figsize=(7,4))
    for K in K_VALUES:
        rows=[x for x in dead if x["K"]==K and x["checkpoint"]!="initial" and x["steps_to_terminal"]>K]
        ax.scatter([(x["steps_to_terminal"]-K)*.05 for x in rows],[x["value"] for x in rows],label=f"K={K}",alpha=.75)
    ax.set(xlabel="lead after prefix (s)",ylabel="R_K",title="Strict-deadlock examples: certificate rise near event")
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/"delayed_deadlock_examples.png",dpi=150);plt.close(fig)

    manifest_path=OUT/"manifest.json";manifest=json.loads(manifest_path.read_text())
    for path in sorted(FIG.glob("*.png")):
        relative=str(path.relative_to(OUT));manifest["deliverables"][relative]=digest(path)
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"figures: {len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
