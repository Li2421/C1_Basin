"""Frozen Round16 analysis. External outcome arrays are opened only in main()."""
import hashlib
import json
import os
import html
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from scipy.stats import pearsonr, spearmanr

import fit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = fit.ARMS
SEEDS = fit.SEEDS
MODELS = ("A0", "A1", "A2", "A3")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def gate():
    receipt = json.loads((HERE / "MODEL_FREEZE.receipt.json").read_text())
    assert receipt["status"] == "MODEL_RULES_HYPERPARAMETERS_FROZEN_BEFORE_EXTERNAL_OUTCOMES"
    amendment_path=HERE/("MODEL_FREEZE_AMENDMENT_V2.receipt.json" if (HERE/"MODEL_FREEZE_AMENDMENT_V2.receipt.json").exists() else "MODEL_FREEZE_AMENDMENT.receipt.json")
    amendment=json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    for p,digest in receipt["files"].items():
        current=sha(HERE/p)
        if p=="evaluate.py" and amendment is not None:
            assert amendment["original_analysis_code_sha256"]==digest
            assert amendment["corrected_analysis_code_sha256"]==current
        else:
            assert current==digest,p
    audit = json.loads((HERE / "DATA_AUDIT.json").read_text())
    for rel, digest in audit["input_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    return receipt


def preds(tag, model):
    arrays = []
    names = None
    for seed in SEEDS:
        with np.load(HERE / "predictions" / f"{tag}_{model}_seed{seed}.npz", allow_pickle=False) as z:
            this_names = list(z["controller_names"])
            assert list(z["arms"]) == ARMS
            if names is None: names = this_names
            assert names == this_names
            arrays.append(z["probability"])
    return names, np.stack(arrays)


def source_effect(data, source_idx):
    rates = data["y"][source_idx].sum(1) / (32 * 4)
    return rates[:, ARMS.index("S500")] - rates[:, ARMS.index("F10")]


def cv_ge(data, source_idx, held_idx, k):
    with np.load(HERE / "fits" / f"fold{k}_transform.npz") as z:
        tr = {key: z[key] for key in z.files}
    x = fit.features(data["physical"], data["r5"], data["r6"], tr, "A1")[:, 0, 0, 23:25]
    src = x[source_idx]; tar = x[held_idx]; y = source_effect(data, source_idx)
    xm = src.mean(0); ym = y.mean()
    beta = np.linalg.solve((src-xm).T @ (src-xm) + np.eye(2), (src-xm).T @ (y-ym))
    eh = (tar-xm) @ beta + ym
    return np.where(eh < .05, "F10", "S500").tolist()


def select_predictions(fit_preds, menu, label_arms):
    """Return external-arm-index choices for each model selection rule."""
    mean = fit_preds.mean(0)
    pessimistic = mean - fit_preds.std(0)
    nc, ns, _ = mean.shape
    result = {}
    for rule, scores in (("state", mean), ("pessimistic", pessimistic)):
        choices = np.empty((nc, ns), np.int32)
        for c, allowed in enumerate(menu):
            ix = [ARMS.index(a) for a in allowed]
            local = np.argmax(scores[c][:, ix], axis=1)
            choices[c] = [label_arms.index(allowed[i]) for i in local]
        result[rule] = choices
    choices = np.empty((nc, ns), np.int32)
    for c, allowed in enumerate(menu):
        ix = [ARMS.index(a) for a in allowed]
        chosen = allowed[int(np.argmax(mean[c].mean(0)[ix]))]
        choices[c] = label_arms.index(chosen)
    result["controller_fixed"] = choices
    return result


def score(choice, lower, upper, oracle, oracle_upper, label_arms, f10_index):
    realized = np.take_along_axis(lower, choice[..., None], axis=2)[..., 0]
    realized_hi = np.take_along_axis(upper, choice[..., None], axis=2)[..., 0]
    value = realized.mean(1)
    f10 = lower[:, :, f10_index].mean(1)
    regret = oracle - value
    regret_upper = oracle_upper - realized_hi.mean(1)
    counts = [{a: int(np.sum(choice[c] == i)) for i, a in enumerate(label_arms) if np.any(choice[c] == i)} for c in range(len(choice))]
    return dict(mean_regret_pp=float(regret.mean() * 100), max_regret_pp=float(regret.max() * 100),
                per_controller_regret_pp=(regret * 100).tolist(), per_controller_success=value.tolist(),
                mean_success=float(value.mean()), upper_success=float(realized_hi.mean()),
                unknown_upper_mean_regret_pp=float(regret_upper.mean()*100),
                unknown_upper_max_regret_pp=float(regret_upper.max()*100),
                catastrophes_vs_F10=int(np.sum(value <= f10 - .10)), choice_counts=counts,
                choices=choice.tolist())


def q_scores(fit_preds, success, unknown, label_arms, menu):
    mean = fit_preds.mean(0)
    p = []; y = []
    for c in range(len(menu)):
        for arm in menu[c]:
            j = label_arms.index(arm); a = ARMS.index(arm)
            outcome = success[c, :, j, :].reshape(-1)
            mask = ~unknown[c, :, j, :].reshape(-1)
            p.append(np.repeat(mean[c, :, a], 4)[mask]); y.append(outcome[mask])
    p = np.clip(np.concatenate(p), 1e-6, 1-1e-6); y = np.concatenate(y)
    bins = []
    for i in range(10):
        mask = (p >= i/10) & (p < (i+1)/10 if i < 9 else p <= 1)
        bins.append(dict(left=i/10, right=(i+1)/10, count=int(mask.sum()),
                         predicted=float(p[mask].mean()) if mask.any() else None,
                         observed=float(y[mask].mean()) if mask.any() else None))
    return dict(NLL=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))), Brier=float(np.mean((y-p)**2)), calibration=bins, known_count=len(y))


def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10: return dict(pearson=None, spearman=None)
    p=float(pearsonr(a,b).statistic);s=float(spearmanr(a,b).statistic)
    return dict(pearson=p if np.isfinite(p) else None, spearman=s if np.isfinite(s) else None)


def inferred_effect(fit_preds, lower, label_arms):
    mean = fit_preds.mean(0).mean(1)
    observed = lower.mean(1)
    result = {}
    for label, plus, minus in (("E", "S500", "F10"), ("L", "S500", "F13")):
        if plus in label_arms and minus in label_arms:
            a = mean[:, ARMS.index(plus)] - mean[:, ARMS.index(minus)]
            b = observed[:, label_arms.index(plus)] - observed[:, label_arms.index(minus)]
            result[label] = dict(correlation=corr(a, b), predicted=a.tolist(), observed=b.tolist())
    return result


def evaluate_cohort(tag, names, label_arms, success, unknown, available, ge_choices, source_mean, model_predictions):
    # label arrays here use controller,state,arm,seed order.
    nc, ns, na, nq = success.shape
    assert nq == 4 and len(names) == nc and len(label_arms) == na
    lower = success.mean(-1); upper = (success + unknown).mean(-1)
    menu = []
    oracle = []
    oracle_upper=[]
    for c in range(nc):
        acts = [a for j,a in enumerate(label_arms) if available[c,j]]
        chosen = [a for a in ARMS if a in acts]
        assert chosen and "F10" in chosen and "S500" in chosen
        menu.append(chosen)
        oracle.append(max(lower[c,:,label_arms.index(a)].mean() for a in acts))
        oracle_upper.append(max(upper[c,:,label_arms.index(a)].mean() for a in acts))
    oracle = np.array(oracle)
    oracle_upper=np.array(oracle_upper)
    choices = {}
    for model, (pred_names, p) in model_predictions.items():
        assert pred_names == names and p.shape == (3,nc,ns,8)
        for rule, choice in select_predictions(p, menu, label_arms).items():
            choices[f"{model}_{rule}"] = choice
    for name, chosen in (("G_E", ge_choices), ("source_mean", [source_mean] * nc), ("F10", ["F10"] * nc)):
        arr = np.empty((nc,ns), np.int32)
        for c, arm in enumerate(chosen):
            assert arm in menu[c]
            arr[c] = label_arms.index(arm)
        choices[name] = arr
    policy = {name: score(choice, lower, upper, oracle, oracle_upper, label_arms, label_arms.index("F10")) for name,choice in choices.items()}
    quality = {m: q_scores(p, success, unknown, label_arms, menu) for m,(_,p) in model_predictions.items()}
    effects = {m: inferred_effect(p, lower, label_arms) for m,(_,p) in model_predictions.items()}
    return dict(controller_names=names, arms=label_arms, available=available.tolist(), unknown_count=int(unknown.sum()),
                oracle_success=oracle.tolist(), policies=policy, quality=quality, effects=effects)


def bootstrap_difference(cohort, left, right, seed=1616):
    # Regret(left)-regret(right) = success(right)-success(left); fixed oracle cancels.
    lower = np.array(cohort["_lower"])
    a = np.array(cohort["policies"][left]["choices"])
    b = np.array(cohort["policies"][right]["choices"])
    va = np.take_along_axis(lower, a[...,None], axis=2)[...,0]
    vb = np.take_along_axis(lower, b[...,None], axis=2)[...,0]
    delta = (vb-va).mean(0)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta), size=(10000,len(delta)))
    boot = delta[draw].mean(1)*100
    return dict(point_pp=float(delta.mean()*100), CI95_pp=np.quantile(boot,[.025,.975]).tolist(), state_clusters=len(delta))


def plot_external(results):
    tags=("T1","T2","T3")
    def keys_for(tag):
        models=("A0","A1","A2","A3") if tag=="T2" else ("A0","A1","A3")
        return [f"{m}_{r}" for m in models for r in ("state","pessimistic","controller_fixed")]+["G_E","source_mean","F10"]
    palette=["#386cb0","#fdb462","#7fc97f","#ef3b2c","#984ea3","#a6cee3","#b15928","#d95f02"]
    def svg_start(w,h):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    def text(x,y,value,size=11,rotate=None):
        transform=f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}"{transform}>{html.escape(str(value))}</text>'
    # Three small panels, one bar per model/rule.
    w=1500;h=520;parts=svg_start(w,h)
    ymax=max(c["policies"][key]["mean_regret_pp"] for tag,c in results["external"].items() for key in keys_for(tag))*1.1+1
    for t,tag in enumerate(tags):
        x0=60+t*480;y0=400;parts.append(text(x0,28,tag,18))
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+430}" y2="{y0}" stroke="black"/>')
        for j,key in enumerate(keys_for(tag)):
            v=results["external"][tag]["policies"][key]["mean_regret_pp"]
            bh=max(0,v/ymax*340)
            x=x0+8+j*27
            parts.append(f'<rect x="{x}" y="{y0-bh:.2f}" width="20" height="{bh:.2f}" fill="{palette[j%len(palette)]}"/>')
            parts.append(text(x+2,y0+10,key,8,70))
    parts.append(text(5,20,"Mean controller regret (pp)",12));parts.append('</svg>')
    (HERE/"REGRET_BY_TEST.svg").write_text("\n".join(parts))
    # Per-controller regrets and selected action, retaining every policy.
    w=1600;h=1050;parts=svg_start(w,h)
    for t,tag in enumerate(tags):
        c=results["external"][tag];y0=50+t*330
        parts.append(text(10,y0,tag+" controller regret (pp)",15))
        vmax=max(max(c["policies"][key]["per_controller_regret_pp"]) for key in ("A1_state","A1_controller_fixed","G_E"))+1
        for i,n in enumerate(c["controller_names"]):
            x=260+i*90
            for j,key in enumerate(("A1_state","A1_controller_fixed","G_E")):
                v=c["policies"][key]["per_controller_regret_pp"][i]
                bh=max(0,v/vmax*210)
                parts.append(f'<rect x="{x+j*25}" y="{y0+235-bh:.2f}" width="22" height="{bh:.2f}" fill="{palette[j]}"/>')
            parts.append(text(x,y0+250,n.replace("base_u_","").split("__")[0][:11],8,55))
    parts.append(text(5,1030,"Blue=A1 state; orange=A1 fixed; green=G_E",11));parts.append('</svg>')
    (HERE/"CONTROLLER_CHOICES.svg").write_text("\n".join(parts))
    w=2100;h=1350;parts=svg_start(w,h)
    for t,tag in enumerate(tags):
        c=results["external"][tag];y0=30+t*440
        parts.append(text(10,y0,tag+" most frequent selected arm (state ties: first arm)",15))
        for i,n in enumerate(c["controller_names"]):
            y=y0+28+i*25;parts.append(text(5,y+15,n.replace("base_u_","").split("__")[0][:26],9))
            for j,key in enumerate(keys_for(tag)):
                choice=np.asarray(c["policies"][key]["choices"][i])
                frequencies=np.array([(choice==z).mean() for z in range(len(c["arms"]))])
                ix=int(np.argmax(frequencies));x=270+j*120
                parts.append(f'<rect x="{x}" y="{y}" width="115" height="21" fill="{palette[ix]}"/>')
                parts.append(text(x+3,y+15,c["arms"][ix]+f" {frequencies[ix]:.2f}",9))
        for j,key in enumerate(keys_for(tag)):parts.append(text(270+j*120,y0+420,key,8,55))
    parts.append('</svg>');(HERE/"CHOICES_BY_CONTROLLER.svg").write_text("\n".join(parts))


def main():
    receipt=gate()  # Explicitly before opening external success/unknown arrays.
    data=fit.source_data(); recipes=sorted(set(data["recipes"])); external={};cv=[]
    source_rates=data["y"].sum((0,1))/(len(data["names"])*32*4)
    source_action=ARMS[int(np.argmax(source_rates))]
    ge=json.loads((HERE/"predictions/G_E_FROZEN_CHOICES.json").read_text())["external"]
    for k in range(7):
        held=[recipes[k],recipes[k+7]]
        hi=np.flatnonzero(np.isin(data["recipes"],held));si=np.flatnonzero(~np.isin(data["recipes"],held))
        names=[data["names"][i] for i in hi]
        model_pred={m:preds(f"CV_fold{k}",m) for m in MODELS}
        lab=data["labels"][hi]
        success=(lab==1).astype(float);unknown=(lab==-1)
        rates=data["y"][si].sum((0,1))/(len(si)*32*4)
        fold_source=ARMS[int(np.argmax(rates))]
        result=evaluate_cohort(f"CV_fold{k}",names,ARMS,success,unknown,np.ones((len(hi),8),bool),cv_ge(data,si,hi,k),fold_source,model_pred)
        result["held_recipes"]=held
        cv.append(result)
    for tag,(folder,_,_) in fit.EXT.items():
        # This is the first access to the external success/unknown values.
        with np.load(folder/"LABELS.npz",allow_pickle=False) as z:
            names=list(z["controller_names"] if "controller_names" in z.files else z["controllers"])
            arms=list(z["arms"])
            success=np.transpose(z["success"],(0,2,1,3))
            unknown=np.transpose(z["unknown"],(0,2,1,3)).astype(bool)
            available=z["available"].astype(bool) if "available" in z.files else np.ones((len(names),len(arms)),bool)
        model_pred={m:preds(tag,m) for m in (("A0","A1","A2","A3") if tag=="T2" else ("A0","A1","A3"))}
        gmap={x["controller"]:x["action"] for x in ge[tag]}
        result=evaluate_cohort(tag,names,arms,success,unknown,available,[gmap[n] for n in names],source_action,model_pred)
        result["_lower"]=(success.mean(-1)).tolist()
        external[tag]=result
    t1=external["T1"]
    q1=bootstrap_difference(t1,"A1_state","A0_state")
    q3=bootstrap_difference(t1,"A1_state","A1_controller_fixed")
    q2_a=t1["policies"]["A1_state"]["mean_regret_pp"]
    q2_g=t1["policies"]["G_E"]["mean_regret_pp"]
    q2_m=t1["policies"]["A1_state"]["max_regret_pp"]
    hypotheses={
      "Q1":{"status":"SUPPORTED" if q1["CI95_pp"][1]<0 else "REJECTED","comparison":q1},
      "Q2":{"status":"SUPPORTED" if q2_a<=q2_g+.5 and q2_m<=5 else "REJECTED","A1_mean_regret_pp":q2_a,"G_E_mean_regret_pp":q2_g,"A1_max_regret_pp":q2_m,"noninferiority_margin_pp":.5,"maximum_regret_limit_pp":5},
      "Q3":{"status":"SUPPORTED" if q3["CI95_pp"][1]<0 else "REJECTED","comparison":q3},
      "Q4":{"status":"INCONCLUSIVE","note":"Descriptive comparison, no prespecified pass/fail threshold"},
    }
    for c in external.values():c.pop("_lower")
    cv_summary={}
    for name in cv[0]["policies"]:
        all_regret=np.concatenate([np.asarray(f["policies"][name]["per_controller_regret_pp"]) for f in cv])
        cv_summary[name]=dict(mean_regret_pp=float(all_regret.mean()),max_regret_pp=float(all_regret.max()))
    cv_quality={}
    for model in MODELS:
        counts=np.array([f["quality"][model]["known_count"] for f in cv])
        cv_quality[model]=dict(NLL=float(np.average([f["quality"][model]["NLL"] for f in cv],weights=counts)),
                               Brier=float(np.average([f["quality"][model]["Brier"] for f in cv],weights=counts)),known_count=int(counts.sum()))
    result=dict(evidence_level="DEVELOPMENT",model_freeze_receipt_sha256=sha(HERE/"MODEL_FREEZE.receipt.json"),
                prereg_sha256=sha(HERE/"PREREGISTRATION.md"),source_unknown_count=int((data["labels"]==-1).sum()),
                source_mean_action=source_action,source_mean_rates={a:float(v) for a,v in zip(ARMS,source_rates)},
                CV=cv,CV_summary=cv_summary,CV_quality=cv_quality,external=external,hypotheses=hypotheses)
    (HERE/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    plot_external(result)
    a=t1["policies"]
    lead=(f"通用 critic 在独立训练的新 controller 上：Q1 {hypotheses['Q1']['status']}，A1 相对 A0 的遗憾差 "
          f"{q1['point_pp']:.2f}pp（95% CI {q1['CI95_pp'][0]:.2f} 至 {q1['CI95_pp'][1]:.2f}）；"
          f"Q2 {hypotheses['Q2']['status']}，A1 遗憾 {q2_a:.2f}pp、G_E {q2_g:.2f}pp、A1 最大遗憾 {q2_m:.2f}pp；"
          f"逐 state 价值 Q3 {hypotheses['Q3']['status']}，遗憾差 {q3['point_pp']:.2f}pp（95% CI {q3['CI95_pp'][0]:.2f} 至 {q3['CI95_pp'][1]:.2f}）。")
    lines=["# Round16 controller critic report","",lead,"","全部结论为 DEVELOPMENT。T1 是冻结模型的外部零样本测试；T1 标签此前已由 round14A 打开，不是全新前瞻确认。","","## 事实","", "| 测试集 | A0 state 遗憾 | A1 state 遗憾 | A1 fixed 遗憾 | A3 state 遗憾 | G_E 遗憾 | A1 最大遗憾 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for tag,c in external.items():
        p=c["policies"]
        lines.append("| "+tag+" | "+" | ".join(f"{p[k]['mean_regret_pp']:.2f}pp" for k in ("A0_state","A1_state","A1_controller_fixed","A3_state","G_E"))+f" | {p['A1_state']['max_regret_pp']:.2f}pp |")
    lines += ["","Q4（描述性）：T1 上 A3 state 与 A1 state 的遗憾分别为 " + f"{a['A3_state']['mean_regret_pp']:.2f}pp / {a['A1_state']['mean_regret_pp']:.2f}pp；" + f"灾难 controller 数分别为 {a['A3_state']['catastrophes_vs_F10']} / {a['A1_state']['catastrophes_vs_F10']}。", f"T2 的 A2 state 遗憾 {external['T2']['policies']['A2_state']['mean_regret_pp']:.2f}pp（冻结后发现兼容 R6 特征，描述性补充）。", "", "## 源数据配方交叉验证","", "| 模型 | state 遗憾 | 最大遗憾 | NLL | Brier |", "|---|---:|---:|---:|---:|"]
    for m in MODELS:
        p=cv_summary[f"{m}_state"];q=cv_quality[m]
        lines.append(f"| {m} | {p['mean_regret_pp']:.2f}pp | {p['max_regret_pp']:.2f}pp | {q['NLL']:.4f} | {q['Brier']:.4f} |")
    lines += ["","Q 预测质量、校准、E/L 相关、unknown 上界敏感性、每 controller 选择和全部 CV 折见 `results.json`；图为 `REGRET_BY_TEST.svg`、`CONTROLLER_CHOICES.svg` 和 `CHOICES_BY_CONTROLLER.svg`。", "", "## 解释与范围","", "Q1 只检验 controller 表示对选择遗憾是否有增益；NLL/Brier 不作为其替代证据。Q2 按预注册的 0.5pp 与 5pp 双门槛判定。Q3 用配对 state 簇 bootstrap。T2/T3 为补充外部数据；T2 的 A2 预测使用事后发现的 Round11 冻结 R6 数组，仅作描述性补充，见 `T2_A2_EXTENSION_PLAN.receipt.json`。T1/T3 仍无兼容 R6。","", "训练和评估完全复用历史 rollout 标签及冻结探测特征；未新增 rollout、探测或共享 DB 写入。模型与规则冻结收据见 `MODEL_FREEZE.receipt.json`。"]
    (HERE/"REPORT.md").write_text("\n".join(lines)+"\n")
    hand=["# Round16 handoff","",lead,"", "## Reproduction","", "Python: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。从仓库根目录运行 `python -B diagnostics/ring_controller_critic_v1/evaluate.py`；它首先校验模型冻结收据和历史输入哈希，再读取外部标签。训练代码是 `fit.py`，Slurm 启动配置为 `train.sbatch`。", "", "## Artifacts","", "`PREREGISTRATION.md` 和收据固定设计；`DATA_AUDIT.json` 只含预冻结身份和形状；`POSTFREEZE_LABEL_AUDIT.json` 记录 unknown 和可用动作；`fits/` 含每模型每 fit checkpoint、变换、CV NLL 和选步；`predictions/` 含每模型每 fit 的 CV/训练/可用外部预测和冻结 G_E 选择；`MODEL_FREEZE.receipt.json` 封存原始模型与分析代码；`MODEL_FREEZE_AMENDMENT_V2.receipt.json` 记录冻结后非模型修正；`results.json` 是完整数值结果。", "", "## Limits","", "全部结论为 DEVELOPMENT；T1 标签已在历史 round14A 打开，不能声称全新前瞻确认。T2 A2 使用原有 Round11 R6，仅作为事后可用性纠正的描述性补充；T1/T3 无兼容 R6。未补跑探测。纯 Q 实验依据 AGENTS.md 无视频交付。"]
    (HERE/"HANDOFF.md").write_text("\n".join(hand)+"\n")
    print(lead)


if __name__=="__main__":main()
