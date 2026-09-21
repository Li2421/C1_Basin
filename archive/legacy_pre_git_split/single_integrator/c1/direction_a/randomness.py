"""Index-addressed, replayable random sources for Direction A."""
import hashlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp


SOURCES = {
    "flow_bc": 0x464C4F57,
    "gate": 0x47415445,
    "gaussian_residual": 0x47415553,
    "environment": 0x454E5652,
    "evaluation_resampling": 0x4556414C,
}


def _stable_u32(label):
    if isinstance(label, int):
        if label < 0:
            raise ValueError("random indices must be nonnegative")
        return label & 0xFFFFFFFF
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class IndexedRandomTape:
    """Random values indexed by split/scenario/continuation/time/source.

    No source owns a mutable generator.  A closed gate therefore cannot shift
    Flow-BC, environment, or later residual draws.
    """

    seed: int

    def key(self, split, scenario, continuation, time, source):
        if source not in SOURCES:
            raise ValueError(f"unknown random source {source!r}")
        if min(int(scenario), int(continuation), int(time)) < 0:
            raise ValueError("scenario, continuation and time must be nonnegative")
        key = jax.random.PRNGKey(self.seed)
        for value in (_stable_u32(split), int(scenario), int(continuation),
                      int(time), SOURCES[source]):
            key = jax.random.fold_in(key, jnp.uint32(value))
        return key

    def flow_noise(self, split, scenario, continuation, time):
        return jax.random.normal(
            self.key(split, scenario, continuation, time, "flow_bc"),
            (4,), dtype=jnp.float32)

    def gate_uniform(self, split, scenario, continuation, time, *, dtype=jnp.float64):
        return jax.random.uniform(
            self.key(split, scenario, continuation, time, "gate"),
            (), dtype=dtype)

    def gaussian_residual(self, split, scenario, continuation, time, *, dtype=jnp.float64):
        return jax.random.normal(
            self.key(split, scenario, continuation, time, "gaussian_residual"),
            (4,), dtype=dtype)

    def environment_key(self, split, scenario, continuation, time):
        return self.key(split, scenario, continuation, time, "environment")

    def resampling_key(self, split, scenario, continuation, time=0):
        return self.key(split, scenario, continuation, time,
                        "evaluation_resampling")
