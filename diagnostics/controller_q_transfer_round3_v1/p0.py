"""Equal-label-budget Ring P0; source rollouts and frozen development only."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import data


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import model

    assert any(d.platform == "gpu" for d in jax.devices()), "P0 fit needs Slurm GPU shard"
    p, c, a, d, b, y = data()
    assert not jax.config.x64_enabled, "P0 must match frozen float32 training"
    x = np.concatenate((a["physical23"], b["state23"])).astype(np.float32)
    X = jnp.asarray(x)
    E = jnp.asarray(a["eta"])
    S, F = jnp.asarray(d["success"].astype(np.float32)), jnp.asarray(d["failure"].astype(np.float32))
    val = np.arange(512, 544)
    valid_s, valid_f = d["success"][val], d["failure"][val]
    val_rate = valid_s / np.maximum(valid_s + valid_f, 1)
    fixed_val = int(val_rate.mean(0).argmax())
    tie_priority = np.argsort(np.lexsort((np.arange(16), -val_rate.mean(0))))
    net = model("local_state_16heads")
    opt = optax.chain(optax.clip_by_global_norm(5), optax.adamw(0.0008, weight_decay=0.0001))
    schedule = [50, 100] + list(range(200, 12001, 200))

    @jax.jit
    def predict(par, xx):
        return net.apply(par, xx, jnp.zeros((len(xx), 3)), jnp.zeros((len(xx), 76)))

    @jax.jit
    def block(par, state, ii, jj, mean_count):
        def step(carry, ids):
            pa, op = carry
            i, j = ids

            def loss(z):
                pred = predict(z, X[i])
                chosen = jnp.take_along_axis(pred, j[:, None], 1)[:, 0]
                return jnp.mean(S[i, j] * jax.nn.softplus(-chosen) + F[i, j] * jax.nn.softplus(chosen)) / mean_count

            _, grad = jax.value_and_grad(loss)(pa)
            update, op = opt.update(grad, op, pa)
            return (optax.apply_updates(pa, update), op), None

        return jax.lax.scan(step, (par, state), (ii, jj))[0]

    layout = []
    for n, k in [(128, 16), (256, 8), (512, 4)]:
        for sub, root_seed in enumerate((2026101201, 2026101202, 2026101203)):
            rng = np.random.default_rng(root_seed + n * 10000)
            states = rng.permutation(512)[:n]
            pairs = []
            for i in states:
                chosen = rng.permutation(16)[:k]
                for j in chosen:
                    if d["success"][i, j] + d["failure"][i, j]:
                        pairs.append((int(i), int(j)))
            layout.append(dict(key=f"n{n}_k{k}_sub{sub}", n=n, k=k, states=states.tolist(), pairs=pairs, root_seed=root_seed))
    dump("P0_SPLITS.json", dict(layout=layout, validation=val.tolist(), development=list(range(608, 864)), train_seed_count=4, fixed_val=fixed_val, source_sha=sha(ROOT / "diagnostics/state_nuisance_rootcause_v1/real_state_expansion/source_dataset.npz")))

    train_records = []
    all_predictions = {}
    start = time.time()
    for task in layout:
        folder = HERE / "p0_fits" / task["key"]
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            rec = json.loads((folder / "complete.json").read_text())
            assert rec["pairs_sha256"] == hashlib.sha256(np.asarray(task["pairs"], dtype=np.int32).tobytes()).hexdigest()
            train_records.append(rec)
            old = np.load(folder / "predictions.npz")
            for key in old.files:
                all_predictions[task["key"] + "_" + key] = old[key]
            continue
        pairs = np.asarray(task["pairs"], dtype=np.int32)
        count = d["success"][pairs[:, 0], pairs[:, 1]] + d["failure"][pairs[:, 0], pairs[:, 1]]
        nmean = float(np.mean(count))
        history = []
        saved = []
        for seed in (17, 23, 41):
            rng = np.random.default_rng(seed)
            par = net.init(jax.random.PRNGKey(seed), X[:1], jnp.zeros((1, 3)), jnp.zeros((1, 76)))
            op = opt.init(par)
            prev = 0
            hh, snapshots = [], []
            for step in schedule:
                draw = rng.integers(len(pairs), size=(step - prev, 128))
                ii, jj = pairs[draw].transpose(2, 0, 1)
                par, op = block(par, op, jnp.asarray(ii), jnp.asarray(jj), nmean)
                z = np.asarray(predict(par, X[val]))
                nll = float((valid_s * np.logaddexp(0, -z) + valid_f * np.logaddexp(0, z)).sum() / (valid_s + valid_f).sum())
                choices = (z + (16 - tie_priority)[None, :] * 1e-10).argmax(1)
                selected_success = float(val_rate[np.arange(len(val)), choices].mean())
                hh.append(dict(step=step, nll=nll, selected_success=selected_success))
                snapshots.append(serialization.to_bytes(par))
                prev = step
            history.append(hh)
            saved.append(snapshots)
        mean_nll = np.mean([[z["nll"] for z in hist] for hist in history], axis=0)
        mean_success = np.mean([[z["selected_success"] for z in hist] for hist in history], axis=0)
        choices = {"nll": int(np.argmin(mean_nll)), "selected_success": int(np.lexsort((np.arange(len(schedule)), mean_nll, -mean_success))[0])}
        rec = dict(key=task["key"], n=task["n"], k=task["k"], sub=int(task["key"].split("sub")[-1]),
                   pairs=len(pairs), pairs_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),
                   seed_fits=[17, 23, 41], schedule=schedule, history=history,
                   selected_steps={key: schedule[pos] for key, pos in choices.items()},
                   elapsed_seconds=time.time() - start)
        predictions = {}
        for rule, pos in choices.items():
            rows = []
            for seed, fit_snapshots in zip((17, 23, 41), saved):
                cp = folder / f"{rule}_seed{seed}_step{schedule[pos]}.msgpack"
                cp.write_bytes(fit_snapshots[pos])
                fitted = serialization.from_bytes(par, fit_snapshots[pos])
                rows.append(np.asarray(predict(fitted, X[608:864])))
            predictions[rule] = np.stack(rows)
            all_predictions[task["key"] + "_" + rule] = predictions[rule]
        np.savez_compressed(folder / "predictions.npz", **predictions)
        dump_rel = json.dumps(rec, indent=2) + "\n"
        (folder / "complete.json").write_text(dump_rel)
        train_records.append(rec)
        print(task["key"], rec["selected_steps"], flush=True)
    np.savez_compressed(HERE / "P0_PREDICTIONS.npz", **all_predictions)
    dump("P0_TRAINING.json", dict(records=train_records, elapsed_seconds=time.time()-start, devices=[str(z) for z in jax.devices()], x64=bool(jax.config.x64_enabled)))


if __name__ == "__main__":
    run()
