"""One controlled B15 interaction branch: scene-only, joint, and few-shot.

All labels use the audited v2 safety semantics. Spatial eta and source-family
split is frozen in spatial_source_split.json. No rollout is performed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import flax.linen as nn
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq

from offline_baselines import ROOT, OUT, dump_json


SCENARIOS = ("four_way_intersection", "ring_exchange")
SPLIT = json.loads((OUT / "spatial_source_split.json").read_text())


def parse2(value):
    z = json.loads(value)
    return json.loads(z) if isinstance(z, str) else z


class InteractionCritic(nn.Module):
    def setup(self):
        self.four_adapter = nn.Dense(64)
        self.ring_adapter = nn.Dense(64)
        self.eta_encoder = nn.Dense(32)
        self.shared1 = nn.Dense(128)
        self.shared2 = nn.Dense(64)
        self.out = nn.Dense(1)

    def branch(self, h, eta, adapter):
        z = nn.silu(adapter(h))
        e = nn.silu(self.eta_encoder(eta))
        x = nn.silu(self.shared1(jnp.concatenate([z, e], axis=-1)))
        x = nn.silu(self.shared2(x))
        return self.out(x)[..., 0]

    def four(self, h, eta): return self.branch(h, eta, self.four_adapter)
    def ring(self, h, eta): return self.branch(h, eta, self.ring_adapter)
    def __call__(self, hf, ef, hr, er): return self.four(hf, ef), self.ring(hr, er)


def prepare():
    states = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/states.parquet",
                           columns=["scenario", "state_uid", "conditioning"]).to_pylist()
    labels = pq.read_table(ROOT / "datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet",
                           columns=["scenario", "state_uid", "eta_uid", "eta_raw", "robust_15of16", "label_semantics_version"]).to_pylist()
    for r in labels:
        if r["scenario"] == "ring_exchange":
            assert r["label_semantics_version"] == "ring_current_safety_v2"
    center = np.asarray(SPLIT["eta_coord_center"], np.float32)
    radius = np.asarray(SPLIT["eta_coord_radius"], np.float32)
    data = {}
    for sc in SCENARIOS:
        ss = {r["state_uid"]: np.asarray(parse2(r["conditioning"])["flat"], np.float32)
              for r in states if r["scenario"] == sc}
        train_ids = SPLIT["state_uids"][sc]["train"]
        mean = np.mean(np.stack([ss[s] for s in train_ids]), axis=0)
        std = np.std(np.stack([ss[s] for s in train_ids]), axis=0)
        std[std < 1e-6] = 1.0
        eta_uid_set = set().union(*[set(v) for v in SPLIT["eta_uids"].values()])
        rows = [r for r in labels if r["scenario"] == sc and r["eta_uid"] in eta_uid_set
                and r["robust_15of16"] is not None]
        keyed = {(r["state_uid"], r["eta_uid"]): r for r in rows}
        partitions = {}
        for stpart, etap in (("train", "train"), ("dev", "train"), ("test", "test"),
                            ("spatial_dev", "dev")):
            source = "dev" if stpart == "spatial_dev" else stpart
            hh, ee, yy, sid, eid = [], [], [], [], []
            for state in SPLIT["state_uids"][sc][source]:
                for eta_uid in SPLIT["eta_uids"][etap]:
                    r = keyed.get((state, eta_uid))
                    if r is None: continue
                    hh.append((ss[state] - mean) / std)
                    ee.append((np.asarray(json.loads(r["eta_raw"]), np.float32) - center) / radius)
                    yy.append(float(r["robust_15of16"]))
                    sid.append(state);eid.append(eta_uid)
            partitions[stpart] = {"h": np.asarray(hh, np.float32), "eta": np.asarray(ee, np.float32),
                                  "y": np.asarray(yy, np.float32), "state_uids": sid, "eta_uids": eid}
        data[sc] = {"parts": partitions, "h_mean": mean, "h_std": std}
    return data


def sample(part: dict, rng: np.random.Generator, batch: int):
    ix = rng.integers(0, len(part["y"]), size=batch)
    return jnp.asarray(part["h"][ix]), jnp.asarray(part["eta"][ix]), jnp.asarray(part["y"][ix])


def evaluate(model, params, data: dict, scenario: str, part: str):
    d = data[scenario]["parts"][part]
    method = model.four if scenario == "four_way_intersection" else model.ring
    logits = np.asarray(model.apply(params, jnp.asarray(d["h"]), jnp.asarray(d["eta"]), method=method))
    p = np.clip(1 / (1 + np.exp(-logits)), 1e-6, 1 - 1e-6)
    y = d["y"]
    nll = float(np.mean(-y * np.log(p) - (1-y) * np.log(1-p)))
    brier = float(np.mean((p-y)**2))
    # Finite-panel candidate ranking over held-out eta, on each held-out state.
    selected = [];oracle=[]
    by = defaultdict(list)
    for i, sid in enumerate(d["state_uids"]):by[sid].append(i)
    for indices in by.values():
        j = indices[int(np.argmax(p[indices]))]
        selected.append(float(y[j]))
        oracle.append(float(np.max(y[indices])))
    return {"pairs": len(y), "states": len(by), "nll": nll, "brier": brier,
            "b15_selected_states": int(sum(selected)), "b15_oracle_states": int(sum(oracle)),
            "selected_fraction_coverable": float(sum(selected)/max(1,sum(oracle)))}


def train(seed: int, run: str, data: dict, init_bytes: bytes | None = None):
    model = InteractionCritic()
    dims = {sc: len(data[sc]["h_mean"]) for sc in SCENARIOS}
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,dims[SCENARIOS[0]])), jnp.zeros((1,3)),
                        jnp.zeros((1,dims[SCENARIOS[1]])), jnp.zeros((1,3)))
    if init_bytes is not None:
        params = serialization.from_bytes(params, init_bytes)
    opt = optax.adamw(1e-3, weight_decay=1e-4)
    state = opt.init(params)
    rng = np.random.default_rng(seed + 100003)
    if run in ("four_full", "four_pretrain"):
        scenes = ("four_way_intersection",)
    elif run in ("ring_full", "ring_25_scratch", "ring_25_transfer", "ring_25_transfer_fixed", "ring_25_finetune"):
        scenes = ("ring_exchange",)
    elif run == "joint_full":
        scenes = SCENARIOS
    else:
        raise ValueError(run)
    subset = {}
    for sc in scenes:
        part = data[sc]["parts"]["train"]
        if run in ("ring_25_scratch", "ring_25_transfer", "ring_25_transfer_fixed", "ring_25_finetune"):
            allowed = set(SPLIT["state_uids"][sc]["train"][:12])
            mask = np.asarray([s in allowed for s in part["state_uids"]])
            subset[sc] = {key: part[key][mask] if key in ("h","eta","y") else [v for v,m in zip(part[key],mask) if m]
                          for key in part}
        else:
            subset[sc] = part
    max_steps = 1800 if run in ("four_full", "ring_full", "joint_full", "four_pretrain") else 900
    patience = 6

    @jax.jit
    def step(p, st, hf, ef, yf, hr, er, yr):
        def loss_fn(pp):
            loss = 0.0
            if "four_way_intersection" in scenes:
                z = model.apply(pp, hf, ef, method=model.four)
                loss = loss + jnp.mean(optax.sigmoid_binary_cross_entropy(z, yf)) / len(scenes)
            if "ring_exchange" in scenes:
                z = model.apply(pp, hr, er, method=model.ring)
                loss = loss + jnp.mean(optax.sigmoid_binary_cross_entropy(z, yr)) / len(scenes)
            return loss
        loss, grad = jax.value_and_grad(loss_fn)(p)
        if run in ("ring_25_transfer", "ring_25_transfer_fixed"):
            grad = {"params": {k: (v if k == "ring_adapter" else jax.tree_util.tree_map(jnp.zeros_like, v))
                               for k, v in grad["params"].items()}}
        updates, st = opt.update(grad, st, p)
        if run == "ring_25_transfer_fixed":
            # AdamW applies weight decay even when gradients are zero. Zero
            # non-adapter *updates*, not just gradients, to truly freeze the
            # shared trunk, eta encoder and head during target adaptation.
            updates = {"params": {k: (v if k == "ring_adapter" else jax.tree_util.tree_map(jnp.zeros_like, v))
                                  for k, v in updates["params"].items()}}
        return optax.apply_updates(p, updates), st, loss

    history=[];best=(float("inf"),None,0);stale=0
    empty_four = (jnp.zeros((1,dims[SCENARIOS[0]])),jnp.zeros((1,3)),jnp.zeros((1,)))
    empty_ring = (jnp.zeros((1,dims[SCENARIOS[1]])),jnp.zeros((1,3)),jnp.zeros((1,)))
    for iteration in range(1, max_steps+1):
        bfour = sample(subset["four_way_intersection"],rng,128) if "four_way_intersection" in scenes else empty_four
        bring = sample(subset["ring_exchange"],rng,128) if "ring_exchange" in scenes else empty_ring
        params,state,loss=step(params,state,*bfour,*bring)
        if iteration % 100: continue
        dev = [evaluate(model,params,data,sc,"dev")["nll"] for sc in scenes]
        criterion = float(np.mean(dev))
        history.append({"step":iteration,"train_loss":float(loss),"dev_nll":criterion})
        if criterion < best[0]-1e-5:
            best=(criterion,serialization.to_bytes(params),iteration);stale=0
        else: stale+=1
        if stale>=patience and iteration>=600:break
    params=serialization.from_bytes(params,best[1])
    if run == "ring_25_transfer_fixed":
        frozen_before=serialization.from_bytes(params,init_bytes)
        for key in params["params"]:
            if key == "ring_adapter": continue
            before=jax.tree_util.tree_leaves(frozen_before["params"][key])
            after=jax.tree_util.tree_leaves(params["params"][key])
            assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(after,before)), key
    metrics={sc:{part:evaluate(model,params,data,sc,part) for part in ("dev","spatial_dev","test")}
             for sc in scenes}
    subdir=OUT/"interaction_models"/f"seed{seed}"/run
    subdir.mkdir(parents=True,exist_ok=True)
    (subdir/"checkpoint.msgpack").write_bytes(best[1])
    (subdir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    result={"seed":seed,"run":run,"best_step":best[2],"dev_nll":best[0],
            "training_scenes":scenes,"train_source_families":{s:len(set(subset[s]["state_uids"])) for s in scenes},
            "metrics":metrics,"checkpoint":str(subdir/"checkpoint.msgpack"),"new_rollout":0}
    (subdir/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result,best[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True,choices=(17,23,41));parser.add_argument("--transfer-only",action="store_true");parser.add_argument("--finetune-only",action="store_true");args=parser.parse_args()
    data=prepare()
    if args.transfer_only or args.finetune_only:
        path=OUT/"interaction_models"/f"seed{args.seed}"/"four_pretrain"/"checkpoint.msgpack"
        assert path.exists()
        run_name="ring_25_finetune" if args.finetune_only else "ring_25_transfer_fixed"
        result,_=train(args.seed,run_name,data,init_bytes=path.read_bytes())
        print(json.dumps(result,indent=2));return
    all_results=[]
    four,b=train(args.seed,"four_pretrain",data);all_results.append(four)
    for run in ("ring_25_transfer_fixed","ring_25_scratch","ring_full","joint_full"):
        r,_=train(args.seed,run,data,init_bytes=b if run=="ring_25_transfer_fixed" else None)
        all_results.append(r)
    dump_json(f"interaction_seed{args.seed}_summary.json",all_results)
    print(json.dumps({"seed":args.seed,"runs":[{"run":r["run"],"metrics":r["metrics"]} for r in all_results]},indent=2))


if __name__ == "__main__":
    main()
