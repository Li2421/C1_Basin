"""Retrospective target-label-free scene regime prediction and few-shot calibration."""
from pathlib import Path
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "controller_state_geometry_transfer_v1"
DATA = PARENT / "data/factors.npz"
CONTEXT = PARENT / "data/factor_contexts.json"
METHODS = ["source_prior", "raw_zero", "mechanism_zero", "prior_k1", "prior_k2", "prior_k4",
           "raw_calibrated_k1", "raw_calibrated_k2", "raw_calibrated_k4",
           "mechanism_calibrated_k1", "mechanism_calibrated_k2", "mechanism_calibrated_k4"]


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def fit_predict(x, y, train, test):
    a = x[train]
    center = a.mean(0)
    scale = a.std(0)
    scale[scale < 1e-12] = 1.0
    z = (a-center)/scale
    zz = (x[test]-center)/scale
    z = np.column_stack([np.ones(len(z)), z])
    zz = np.column_stack([np.ones(len(zz)), zz])
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0
    beta = np.linalg.solve(z.T@z+penalty, z.T@y[train])
    return np.clip(zz@beta, 0, 1)


def evaluate(y, xraw, xmech, contexts, mode):
    rng = np.random.default_rng(20261011)
    perms = [rng.permutation(8) for _ in range(20)]
    geometries = sorted({c["physical_geometry_index"] for c in contexts})
    out = []
    for g in geometries:
        target = [i for i,c in enumerate(contexts) if c["physical_geometry_index"] == g]
        assert len(target) == 2
        if mode == "leave_physical_geometry_out":
            train = [i for i in range(len(contexts)) if i not in target]
        else:
            top = contexts[target[0]]["topology"]
            train = [i for i,c in enumerate(contexts) if c["topology"] != top]
        # No target outcome contributes to these three zero-shot predictions.
        source_q = np.nanmean(y[train], axis=(0,1,4)) # controller × eta
        prior = np.broadcast_to(source_q, (2,)+source_q.shape).copy()
        source_config_q = np.nanmean(y, axis=(1,4)).reshape(24, 8)
        raw = fit_predict(xraw, source_config_q, train, target).reshape(2,2,4)
        mech = fit_predict(xmech, source_config_q, train, target).reshape(2,2,4)
        test_y = y[target, 8:16]
        assert test_y.shape == (2,8,2,4,2)
        for ti, ci in enumerate(target):
            for cc in range(2):
                qtest_low = np.nan_to_num(test_y[ti,:,cc], nan=0.0).mean(axis=(0,2))
                qtest_high = np.nan_to_num(test_y[ti,:,cc], nan=1.0).mean(axis=(0,2))
                empirical_best = int(np.argmax(qtest_low))
                methods = {"source_prior": prior[ti,cc], "raw_zero":raw[ti,cc],
                           "mechanism_zero":mech[ti,cc]}
                for k in [1,2,4]:
                    for draw, perm in enumerate(perms):
                        cal = y[ci,perm[:k],cc]
                        with np.errstate(invalid="ignore"):
                            calq = np.nanmean(cal,axis=(0,2))
                        calq = np.where(np.isfinite(calq),calq,source_q[cc])
                        w = (2*k)/(2*k+4)
                        methods[f"prior_k{k}_draw{draw}"] = np.clip((1-w)*prior[ti,cc]+w*calq,0,1)
                        methods[f"raw_calibrated_k{k}_draw{draw}"] = np.clip(raw[ti,cc]+w*(calq-raw[ti,cc]),0,1)
                        methods[f"mechanism_calibrated_k{k}_draw{draw}"] = np.clip(mech[ti,cc]+w*(calq-mech[ti,cc]),0,1)
                pred = {}
                for name in METHODS:
                    keys = [name] if name in methods else [f"{name}_draw{d}" for d in range(20)]
                    choices = [int(np.argmax(methods[key])) for key in keys]
                    pred[name] = {"choices":choices,
                                  "success_lower":float(np.mean(qtest_low[choices])),
                                  "success_upper":float(np.mean(qtest_high[choices])),
                                  "regret_vs_test_hindsight_fixed_lower":float(qtest_low[empirical_best]-np.mean(qtest_low[choices]))}
                out.append({"physical_geometry_index":g,"target_context_index":ci,
                            "topology":contexts[ci]["topology"],"N":contexts[ci]["N"],
                            "width":contexts[ci]["width"],"horizon":contexts[ci]["target_horizon"],
                            "controller_index":cc,"train_contexts":train,
                            "test_state_indices":list(range(8,16)),
                            "calibration_state_pool":list(range(8)),
                            "unknown_test_labels":int((~np.isfinite(test_y[ti,:,cc])).sum()),
                            "test_hindsight_fixed_eta_index":empirical_best,
                            "test_hindsight_fixed_success_lower":float(qtest_low[empirical_best]),
                            "methods":pred})
    arr=np.asarray([[r["methods"][m]["success_lower"] for m in METHODS] for r in out]).reshape(12,4,len(METHODS))
    top=np.asarray([r["topology"] for r in out]).reshape(12,4)[:,0]
    gen=np.random.default_rng(20261011)
    draws=gen.integers(0,12,(2000,12))
    means=arr[draws].mean(axis=(1,2))
    overall={m:{"success_lower":float(arr[:,:,j].mean()),
                "geometry_bootstrap_ci95":np.quantile(means[:,j],[.025,.975]).tolist(),
                "vs_source_prior_pp":float(100*(arr[:,:,j]-arr[:,:,0]).mean()),
                "vs_source_prior_ci95_pp":(100*np.quantile(means[:,j]-means[:,0],[.025,.975])).tolist()}
             for j,m in enumerate(METHODS)}
    for j,m in enumerate(METHODS):
        overall[m]["by_topology"]={t:float(arr[top==t,:,j].mean()) for t in sorted(set(top))}
    return {"mode":mode,"n_physical_geometry":12,"n_target_contexts":24,
            "n_target_controller_tasks":len(out),"method_summary":overall,
            "per_target":out,
            "unknown_total_in_source_data":int((~np.isfinite(y)).sum()),
            "calibration_rollouts_per_target_context_for_k":{"0":0,"1":16,"2":32,"4":64},
            "actual_new_rollouts":0,
            "CI_unit":"12 independent physical geometry clusters; paired bootstrap; horizon/control tasks nested"}


def main():
    z=np.load(DATA,allow_pickle=False)
    y=z["Y"]
    assert y.shape==(24,16,2,4,2)
    # Features use only the eight unlabeled calibration-pool initial states.
    # The eight final test initial states never enter the scene-level prior.
    xraw=z["raw"].reshape(24,16,5)[:,:8].mean(1)
    xmech=z["mech"].reshape(24,16,8)[:,:8].mean(1)
    contexts=json.loads(CONTEXT.read_text())
    assert len(contexts)==24
    out={"data_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
         "contexts_sha256":hashlib.sha256(CONTEXT.read_bytes()).hexdigest(),
         "regression":"source-fold standardized ridge, L2=1, unpenalized intercept; no target labels",
         "zero_shot_feature_states":"unlabeled initial states 0..7 only; final test states 8..15 excluded from scene-level features",
         "factor_coordinate_source":str(PARENT/"factor_data.py"),
         "leave_physical_geometry_out":evaluate(y,xraw,xmech,contexts,"leave_physical_geometry_out"),
         "leave_topology_out":evaluate(y,xraw,xmech,contexts,"leave_topology_out")}
    write(HERE/"E4_REGIME_RESULTS.json",out)
    for fold in ["leave_physical_geometry_out","leave_topology_out"]:
        print(fold,{k:round(v["success_lower"],3) for k,v in out[fold]["method_summary"].items()},flush=True)

if __name__=="__main__":main()
