"""No task rollout: objective, split and input-path invariants."""
import unittest
import json
import subprocess
import sys
import copy
import jax
import jax.numpy as jnp
import numpy as np

from .data import OUT, read, CONTROLLERS, DBROOT, EXPERIMENT, REVISION, connect, validate_record
from .train import Additive, controller_balanced_nll, metrics


class RepairTests(unittest.TestCase):
    def test_state_normalization_uses_only_train_and_no_slot_identity(self):
        from .state_normalization import normalize_entities
        x = dict(np.load(OUT / "entities.npz"))
        xx, norm = normalize_entities(x, np.arange(24))
        altered = {k: v.copy() for k, v in x.items()}
        altered["agents"][24:] += 100.
        _, other = normalize_entities(altered, np.arange(24))
        self.assertEqual(norm, other)
        perm = np.arange(x["agents"].shape[1])[::-1]
        rotated = {k: v.copy() for k, v in x.items()}
        for k in ("agents", "obstacles", "agent_mask"):
            rotated[k] = rotated[k][:, perm]
        rotated["pairs"] = rotated["pairs"][:, perm][:, :, perm]
        yy, _ = normalize_entities(rotated, np.arange(24))
        np.testing.assert_allclose(yy["agents"], xx["agents"][:, perm], atol=1e-6)
        np.testing.assert_allclose(yy["pairs"], xx["pairs"][:, perm][:, :, perm], atol=1e-6)
        np.testing.assert_array_equal(xx["agent_mask"], x["agent_mask"])

    def test_ingestion_rejects_semantic_mismatch(self):
        protocol = read(OUT / "protocol.json")
        states = {s["uid"]: s for s in read(OUT / "states.json")}
        pairs = {(p["state_uid"], p["eta_uid"]): p for p in read(OUT / "pairs.json")}
        path = next((DBROOT / "journals" / EXPERIMENT).glob(f"{REVISION}_base*.jsonl"))
        record = json.loads(path.read_text().splitlines()[0])["record"]
        with connect(True) as db:
            payloads = {c["controller_uid"]: json.loads(db.execute(
                "SELECT config_json FROM controller_config WHERE controller_uid=?",
                (c["controller_uid"],)).fetchone()[0]) for c in protocol["profiles"]}
        validate_record(record, protocol, states, pairs, payloads)
        for field, value in (("jax_enable_x64", True), ("flow_sampler_precision", "float64"),
                             ("future_index", 16), ("flow_checkpoint_sha256", "wrong"),
                             ("rng_namespace", -1), ("source_split", "test")):
            invalid = copy.deepcopy(record)
            invalid[field] = value
            with self.assertRaises(AssertionError, msg=field):
                validate_record(invalid, protocol, states, pairs, payloads)

    def test_imports_do_not_change_native_sampler_precision(self):
        script = """
import jax
jax.config.update('jax_enable_x64', False)
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features
assert not jax.config.x64_enabled
from diagnostics.orthoflow3_controller_training_repair_v1 import data
assert not jax.config.x64_enabled
assert str(jax.random.normal(jax.random.PRNGKey(0), (1,)).dtype) == 'float32'
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)

    def test_cross_matrix_and_splits(self):
        states, pairs = read(OUT / "states.json"), read(OUT / "pairs.json")
        tr = {s["source_group"] for s in states if s["split"] == "train"}
        va = {s["source_group"] for s in states if s["split"] == "validation"}
        self.assertEqual((len(tr), len(va), len(tr & va)), (24, 6, 0))
        expected = {p["eta_uid"] for p in pairs}
        for s in states:
            pp = [p for p in pairs if p["state_uid"] == s["uid"]]
            self.assertEqual({p["eta_uid"] for p in pp}, expected)
            self.assertEqual(len(pp), 16)
            self.assertEqual(s["physical"]["timestep"], 0)
        profiles = read(OUT / "protocol.json")["profiles"]
        self.assertEqual(len({p["sha256"] for p in profiles}), 3)

    def test_controller_equal_weight_not_equal_fake_labels(self):
        z = jnp.array([[1., -1.], [1., -1.], [1., -1.]])
        s = jnp.array([[4., 1.], [8., 2.], [16., 4.]])
        f = jnp.array([[0., 3.], [0., 6.], [0., 12.]])
        value = float(controller_balanced_nll(z, s, f))
        expected = float(jnp.sum(s[0]*jax.nn.softplus(-z[0])+f[0]*jax.nn.softplus(z[0]))/8.)
        self.assertAlmostEqual(value, expected, places=6)
        grad = np.asarray(jax.grad(controller_balanced_nll)(z, s, f))
        np.testing.assert_allclose(grad[0], grad[1], atol=1e-7)
        np.testing.assert_allclose(grad[0], grad[2], atol=1e-7)
        # An unobserved trial contributes zero derivative, never a fake failure.
        missing = np.asarray(jax.grad(controller_balanced_nll)(z, jnp.zeros_like(s), jnp.zeros_like(f)))
        np.testing.assert_array_equal(missing, 0.)

    def test_additive_cannot_change_eta_preference(self):
        entities = dict(np.load(OUT / "entities.npz"))
        x = {k: jnp.asarray(v[[0, 0, 1, 1]]) for k, v in entities.items()}
        eta = jnp.array([[.1, .2, .3], [.9, -.1, .2], [.1, .2, .3], [.9, -.1, .2]])
        c = np.zeros((4, 24), np.float32)
        c[2:, :10] = 1.
        model = Additive()
        params = model.init(jax.random.PRNGKey(17), x, eta, jnp.asarray(c), jnp.zeros((4, 3)))
        z = np.asarray(model.apply(params, x, eta, jnp.asarray(c), jnp.zeros((4, 3))))
        self.assertAlmostEqual(float(z[0]-z[1]), float(z[2]-z[3]), places=6)

    def test_reversal_metrics_and_standard_seed_certification(self):
        ss = np.array([[16, 0, 0, 16], [0, 16, 16, 0], [16, 0, 0, 16]], np.float32)
        ff = 16-ss
        d = {"success": ss, "failure": ff, "standard_success": ss.copy(), "standard_failure": ff.copy(),
             "rows": [{"state_uid": "a" if i < 2 else "b", "eta_uid": str(i % 2)} for i in range(4)]}
        z = (ss-8)/2
        m = metrics(z, d, np.arange(4))
        self.assertEqual(m["selected_B15"], 6)
        self.assertEqual(m["controller_reversals"]["accuracy"], 1.)
        self.assertEqual(m["state_reversals"]["accuracy"], 1.)
        json.dumps(m, allow_nan=False)
        # Only 14 successes + one failure + one unresolved numerical trial:
        # not a fake 14/16 failure, and not a B15 certificate.
        d["standard_success"][0, 0] = 14
        d["standard_failure"][0, 0] = 1
        m = metrics(z, d, np.arange(4))
        self.assertEqual(m["eligible"], 5)
        constant = np.tile(np.array([1., -1., 1., -1.]), (3, 1))
        m = metrics(constant, d, np.arange(4))
        self.assertEqual(m["controller_reversals"]["correct"], 0)
        self.assertEqual(m["state_reversals"]["correct"], 0)


if __name__ == "__main__":
    unittest.main()
