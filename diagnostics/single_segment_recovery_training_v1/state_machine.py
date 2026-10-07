"""Authoritative one-segment recovery state machine for Toy Give-Way.

This module implements only the deployment transition semantics shared by the
entry/exit training and evaluation jobs.  It deliberately does not implement
head training, branch-label construction, online oracle calls, or controller
selection.

The only legal mode path is::

    SAFETY_BEFORE -> RECOVERY -> SAFETY_AFTER

The first recovery action is executed on the entry step.  The exit head is not
queried until at least one recovery transition has been executed.  An exit
decision executes Safety on that same physical step and permanently disables
re-entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

import numpy as np


FEATURE_DIMENSION = 214
ETA_DIMENSION = 3


class Mode(str, Enum):
    SAFETY_BEFORE = "SAFETY_BEFORE"
    RECOVERY = "RECOVERY"
    SAFETY_AFTER = "SAFETY_AFTER"


class RecoverySystem(str, Enum):
    DIRECT_G = "G"
    STRUCTURED_ETA = "ETA"


class TerminalStateError(RuntimeError):
    """Raised when control is requested after a frozen terminal event."""


class DecisionHead(Protocol):
    def __call__(self, features: np.ndarray) -> bool:
        """Return the already-thresholded state-dependent decision."""


class FiniteHistoryView:
    """Expose only genuine saved history to the network feature adapter.

    ``restore_full`` represents unavailable older history by NaNs.  Removing
    those placeholders here is not monitor mutation: the real environment and
    monitor history are untouched.  The startup-aware feature builder performs
    its own causal left padding on this read-only view.
    """

    def __init__(self, env: Any):
        self._env = env
        history = np.asarray(env.distance_history, dtype=np.float64)
        if history.ndim != 2 or history.shape[1:] != (2,):
            raise ValueError(("invalid goal-error history", history.shape))
        self.distance_history = history[np.isfinite(history).all(axis=1)]
        if len(self.distance_history) == 0:
            raise ValueError("environment contains no genuine finite history")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


@dataclass
class ControllerMemory:
    mode: Mode = Mode.SAFETY_BEFORE
    eta_latched: np.ndarray | None = None
    recovery_used: bool = False
    entry_step: int | None = None
    exit_step: int | None = None
    recovery_transitions: int = 0

    def validate(self) -> None:
        if self.recovery_transitions < 0:
            raise AssertionError("negative recovery transition count")
        if self.mode is Mode.SAFETY_BEFORE:
            if self.recovery_used or self.entry_step is not None or self.exit_step is not None:
                raise AssertionError("SAFETY_BEFORE cannot contain recovery history")
            if self.recovery_transitions != 0 or self.eta_latched is not None:
                raise AssertionError("SAFETY_BEFORE has invalid recovery memory")
        elif self.mode is Mode.RECOVERY:
            if not self.recovery_used or self.entry_step is None or self.exit_step is not None:
                raise AssertionError("RECOVERY memory is inconsistent")
            if self.recovery_transitions < 1:
                raise AssertionError("RECOVERY must contain at least one executed transition")
        elif self.mode is Mode.SAFETY_AFTER:
            if not self.recovery_used or self.entry_step is None or self.exit_step is None:
                raise AssertionError("SAFETY_AFTER memory is inconsistent")
            if self.recovery_transitions < 1 or self.exit_step <= self.entry_step:
                raise AssertionError("zero-duration or invalid recovery segment")
        else:  # pragma: no cover - Enum prevents this in normal use.
            raise AssertionError(("unknown mode", self.mode))
        if self.eta_latched is not None:
            eta = np.asarray(self.eta_latched, dtype=np.float64)
            if eta.shape != (ETA_DIMENSION,) or not np.isfinite(eta).all():
                raise AssertionError(("invalid latched eta", eta))


@dataclass(frozen=True)
class PreparedStep:
    """Common current-step data computed once before either decision branch."""

    step: int
    observation: np.ndarray
    flow_key: np.ndarray
    u_flow: np.ndarray
    u_safe: np.ndarray
    feature: np.ndarray | None
    constraint_matrix: np.ndarray
    constraint_lower: np.ndarray
    first_projection_retry: bool


@dataclass(frozen=True)
class StepRecord:
    step: int
    mode_before: Mode
    mode_after_decision: Mode
    decision: str
    entry_queried: bool
    exit_queried: bool
    flow_sample_count: int
    direct_query_count: int
    eta_query_count: int
    recovery_action: bool
    flow_key: np.ndarray
    feature: np.ndarray | None
    u_flow: np.ndarray
    u_safe: np.ndarray
    correction: np.ndarray
    u_exec: np.ndarray
    first_projection_retry: bool
    second_projection_retry: bool
    entry_step: int | None
    exit_step: int | None
    recovery_transitions: int
    eta_latched: np.ndarray | None
    event: str | None
    done: bool
    monitor_info: Mapping[str, Any]


class AuthoritativeStepKernel:
    """Compute one Flow/Safety context and apply exactly one physical action.

    The caller supplies the already-loaded frozen Flow sampler and episode key.
    ``prepare`` is the sole location at which that sampler is invoked.  The
    resulting context is shared by the head decision, recovery model, and both
    projections, which rules out an accidental second Flow draw on entry/exit.
    """

    def __init__(
        self,
        *,
        env: Any,
        config: Any,
        cbf: Any,
        episode_key: Any,
        sample_action: Callable[[np.ndarray, Any], np.ndarray],
        feature_builder: Any,
        project: Callable[..., tuple],
        key_for_step: Callable[[Any, int], Any] | None = None,
        bounded_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
        constraints: Callable[[Any, Any], tuple] | None = None,
    ) -> None:
        if key_for_step is None:
            import jax

            key_for_step = jax.random.fold_in
        if bounded_action is None or constraints is None:
            from single_integrator.cbf import barrier_constraints
            from single_integrator.environment import bounded_nominal

            bounded_action = bounded_nominal if bounded_action is None else bounded_action
            constraints = barrier_constraints if constraints is None else constraints
        self.env = env
        self.config = config
        self.cbf = cbf
        self.episode_key = episode_key
        self.sample_action = sample_action
        self.feature_builder = feature_builder
        self.project = project
        self.key_for_step = key_for_step
        self.bounded_action = bounded_action
        self.constraints = constraints
        self.flow_sample_count = 0
        self.physical_transition_count = 0

    def _validate_projected(self, action: np.ndarray, matrix: np.ndarray, lower: np.ndarray,
                            *, label: str) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2, 2) or not np.isfinite(action).all():
            raise RuntimeError((label, "invalid/nonfinite projected action", action.shape))
        linear_min = float(np.min(np.asarray(matrix) @ action.reshape(4) - np.asarray(lower)))
        speed_excess = float(np.max(np.linalg.norm(action, axis=-1) - self.config.max_speed))
        if linear_min < -float(self.cbf.feasibility_tol) or speed_excess > float(self.cbf.speed_tol):
            raise RuntimeError((label, "projected action violates frozen constraints", linear_min, speed_excess))
        return action

    def prepare(self, *, need_feature: bool) -> PreparedStep:
        if bool(self.env.done):
            raise TerminalStateError("terminal event is absorbing; control decision refused")
        step = int(self.env.step_count)
        if step >= int(self.config.max_steps):
            raise TerminalStateError("official physical horizon is exhausted")
        observation = np.asarray(self.env.observation(), dtype=np.float32)
        key = self.key_for_step(self.episode_key, step)
        raw_flow = np.asarray(self.sample_action(observation, key), dtype=np.float64)
        self.flow_sample_count += 1
        u_flow = np.asarray(self.bounded_action(raw_flow, self.config.max_speed), dtype=np.float64)
        matrix, lower, _ = self.constraints(self.env.snapshot(), self.cbf)
        u_safe, _, retry, _ = self.project(
            u_flow, matrix, lower, self.config.max_speed, self.cbf
        )
        u_safe = self._validate_projected(u_safe, matrix, lower, label="first projection")
        feature = None
        if need_feature:
            feature, _ = self.feature_builder.build(
                FiniteHistoryView(self.env), {"u_flow": u_flow, "u_safe": u_safe},
                self.config, self.cbf,
            )
            feature = np.asarray(feature, dtype=np.float64)
            if feature.shape != (FEATURE_DIMENSION,) or not np.isfinite(feature).all():
                raise RuntimeError(("invalid decision feature", feature.shape))
        return PreparedStep(
            step=step,
            observation=observation,
            flow_key=np.asarray(key),
            u_flow=u_flow,
            u_safe=u_safe,
            feature=feature,
            constraint_matrix=np.asarray(matrix, dtype=np.float64),
            constraint_lower=np.asarray(lower, dtype=np.float64),
            first_projection_retry=bool(retry),
        )

    def second_projection(self, context: PreparedStep, correction: np.ndarray) -> tuple[np.ndarray, bool]:
        correction = np.asarray(correction, dtype=np.float64)
        if correction.shape != (2, 2) or not np.isfinite(correction).all():
            raise RuntimeError(("invalid recovery correction", correction.shape))
        action, _, retry, _ = self.project(
            context.u_safe + correction,
            context.constraint_matrix,
            context.constraint_lower,
            self.config.max_speed,
            self.cbf,
        )
        action = self._validate_projected(
            action, context.constraint_matrix, context.constraint_lower,
            label="second projection",
        )
        return action, bool(retry)

    def execute(self, action: np.ndarray) -> tuple[bool, Mapping[str, Any]]:
        if bool(self.env.done):
            raise TerminalStateError("terminal event is absorbing; physical step refused")
        _, _, done, info = self.env.step(np.asarray(action, dtype=np.float64))
        self.physical_transition_count += 1
        return bool(done), info


def _strict_boolean(value: Any, *, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != ():
        raise TypeError((name, "decision head must return a scalar", array.shape))
    if array.dtype.kind == "b":
        return bool(array)
    if array.dtype.kind in "iu" and int(array) in (0, 1):
        return bool(array)
    raise TypeError((name, "decision head must return bool or integer 0/1", value))


class SingleSegmentRecoveryMachine:
    """Execute an immutable recovery controller behind learned entry/exit heads."""

    def __init__(
        self,
        *,
        kernel: AuthoritativeStepKernel,
        system: RecoverySystem,
        entry_head: DecisionHead,
        exit_head: DecisionHead,
        direct_model: Callable[[np.ndarray], np.ndarray] | None = None,
        eta_model: Any | None = None,
        eta_corrector_factory: Callable[[np.ndarray], Callable[[np.ndarray, np.ndarray, float], np.ndarray]] | None = None,
    ) -> None:
        self.kernel = kernel
        self.system = RecoverySystem(system)
        self.entry_head = entry_head
        self.exit_head = exit_head
        self.direct_model = direct_model
        self.eta_model = eta_model
        if eta_corrector_factory is None and self.system is RecoverySystem.STRUCTURED_ETA:
            from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi

            eta_corrector_factory = lambda eta: DiagnosticCorrector(DiagnosticPhi(*eta.tolist()))
        self.eta_corrector_factory = eta_corrector_factory
        self.memory = ControllerMemory()
        self._eta_corrector: Callable | None = None
        self.direct_query_count = 0
        self.eta_query_count = 0
        self.entry_query_count = 0
        self.exit_query_count = 0
        self.records: list[StepRecord] = []
        if self.system is RecoverySystem.DIRECT_G and direct_model is None:
            raise ValueError("System G requires a frozen direct-g model")
        if self.system is RecoverySystem.STRUCTURED_ETA and eta_model is None:
            raise ValueError("System ETA requires a frozen eta model")

    def restore_controller_memory(self, memory: ControllerMemory) -> None:
        """Restore controller memory paired with an exact augmented plant state.

        This is intended for offline branch continuations.  It must be called
        before any step and refuses impossible/mismatched modes.  In
        particular, an ETA recovery state reconstructs its correction object
        from the saved latch without querying the eta network again.
        """

        if self.records or self.kernel.physical_transition_count:
            raise RuntimeError("controller memory must be restored before stepping")
        memory.validate()
        if memory.mode is Mode.SAFETY_BEFORE and memory.eta_latched is not None:
            raise ValueError("SAFETY_BEFORE cannot restore an eta latch")
        if self.system is RecoverySystem.STRUCTURED_ETA:
            if memory.mode in (Mode.RECOVERY, Mode.SAFETY_AFTER) and memory.eta_latched is None:
                raise ValueError("ETA recovery history requires a saved eta latch")
        elif memory.eta_latched is not None:
            raise ValueError("System G cannot restore eta controller memory")
        self.memory = ControllerMemory(
            mode=memory.mode,
            eta_latched=(
                None if memory.eta_latched is None
                else np.asarray(memory.eta_latched, dtype=np.float64).copy()
            ),
            recovery_used=bool(memory.recovery_used),
            entry_step=memory.entry_step,
            exit_step=memory.exit_step,
            recovery_transitions=int(memory.recovery_transitions),
        )
        if self.memory.eta_latched is not None:
            self.memory.eta_latched.flags.writeable = False
            if self.memory.mode is Mode.RECOVERY:
                self._eta_corrector = self.eta_corrector_factory(self.memory.eta_latched)
        self.memory.validate()

    @staticmethod
    def _feature(context: PreparedStep) -> np.ndarray:
        if context.feature is None:
            raise AssertionError("decision/recovery feature was not prepared")
        return context.feature

    def _latch_eta(self, feature: np.ndarray) -> None:
        if self.system is not RecoverySystem.STRUCTURED_ETA:
            return
        if self.memory.eta_latched is not None or self._eta_corrector is not None:
            raise AssertionError("eta may be predicted and latched only once")
        predicted, _, _ = self.eta_model.predict(feature[None])
        eta = np.asarray(predicted, dtype=np.float64)
        if eta.shape != (1, ETA_DIMENSION) or not np.isfinite(eta).all():
            raise RuntimeError(("invalid eta prediction", eta.shape))
        latched = eta[0].copy()
        latched.flags.writeable = False
        self.memory.eta_latched = latched
        self._eta_corrector = self.eta_corrector_factory(latched)
        self.eta_query_count += 1

    def _recovery_correction(self, context: PreparedStep) -> np.ndarray:
        if self.system is RecoverySystem.DIRECT_G:
            values = np.asarray(self.direct_model(self._feature(context)[None]), dtype=np.float64)
            if values.shape != (1, 4):
                raise RuntimeError(("invalid direct-g prediction", values.shape))
            self.direct_query_count += 1
            return values[0].reshape(2, 2)
        if self.memory.eta_latched is None or self._eta_corrector is None:
            raise AssertionError("structured eta must be latched before recovery action")
        before = self.memory.eta_latched.copy()
        correction = self._eta_corrector(
            np.asarray(context.observation, dtype=np.float64), context.u_safe,
            self.kernel.config.max_speed,
        )
        if not np.array_equal(before, self.memory.eta_latched):
            raise AssertionError("latched eta changed while recovery was active")
        return np.asarray(correction, dtype=np.float64)

    def step(self) -> StepRecord:
        """Perform exactly one nonterminal physical transition."""

        self.memory.validate()
        if bool(self.kernel.env.done):
            raise TerminalStateError("terminal event is absorbing; no state-machine step allowed")
        mode_before = self.memory.mode
        need_feature = mode_before is not Mode.SAFETY_AFTER
        flow_before = self.kernel.flow_sample_count
        direct_before = self.direct_query_count
        eta_before = self.eta_query_count
        context = self.kernel.prepare(need_feature=need_feature)
        entry_queried = exit_queried = recovery_action = False
        decision = "SAFETY"
        correction = np.zeros((2, 2), dtype=np.float64)
        second_retry = False

        if mode_before is Mode.SAFETY_BEFORE:
            entry_queried = True
            self.entry_query_count += 1
            enter = _strict_boolean(self.entry_head(self._feature(context)), name="entry")
            if enter:
                self.memory.mode = Mode.RECOVERY
                self.memory.recovery_used = True
                self.memory.entry_step = context.step
                self._latch_eta(self._feature(context))
                correction = self._recovery_correction(context)
                u_exec, second_retry = self.kernel.second_projection(context, correction)
                self.memory.recovery_transitions += 1
                recovery_action = True
                decision = "ENTER"
            else:
                u_exec = context.u_safe
                decision = "WAIT"
        elif mode_before is Mode.RECOVERY:
            if self.memory.recovery_transitions < 1:
                raise AssertionError("exit queried before first recovery transition")
            exit_queried = True
            self.exit_query_count += 1
            exit_features = self._feature(context)
            if self.system is RecoverySystem.STRUCTURED_ETA:
                if self.memory.eta_latched is None:
                    raise AssertionError("ETA exit input requires latched eta")
                exit_features = np.concatenate((exit_features, self.memory.eta_latched))
                if exit_features.shape != (FEATURE_DIMENSION + ETA_DIMENSION,):
                    raise AssertionError(exit_features.shape)
            should_exit = _strict_boolean(self.exit_head(exit_features), name="exit")
            if should_exit:
                self.memory.mode = Mode.SAFETY_AFTER
                self.memory.exit_step = context.step
                u_exec = context.u_safe
                decision = "EXIT"
            else:
                correction = self._recovery_correction(context)
                u_exec, second_retry = self.kernel.second_projection(context, correction)
                self.memory.recovery_transitions += 1
                recovery_action = True
                decision = "CONTINUE"
        elif mode_before is Mode.SAFETY_AFTER:
            u_exec = context.u_safe
            decision = "SAFETY_AFTER"
        else:  # pragma: no cover
            raise AssertionError(mode_before)

        # The frozen environment applies collision/success/deadlock/timeout
        # priority during this sole physical transition.  No decision is made
        # after a terminal event on the same step.
        done, info = self.kernel.execute(u_exec)
        self.memory.validate()
        if self.kernel.flow_sample_count - flow_before != 1:
            raise AssertionError("each physical step must consume exactly one Flow sample")
        record = StepRecord(
            step=context.step,
            mode_before=mode_before,
            mode_after_decision=self.memory.mode,
            decision=decision,
            entry_queried=entry_queried,
            exit_queried=exit_queried,
            flow_sample_count=self.kernel.flow_sample_count - flow_before,
            direct_query_count=self.direct_query_count - direct_before,
            eta_query_count=self.eta_query_count - eta_before,
            recovery_action=recovery_action,
            flow_key=context.flow_key.copy(),
            feature=(None if context.feature is None else context.feature.copy()),
            u_flow=context.u_flow.copy(),
            u_safe=context.u_safe.copy(),
            correction=correction.copy(),
            u_exec=np.asarray(u_exec, dtype=np.float64).copy(),
            first_projection_retry=context.first_projection_retry,
            second_projection_retry=second_retry,
            entry_step=self.memory.entry_step,
            exit_step=self.memory.exit_step,
            recovery_transitions=self.memory.recovery_transitions,
            eta_latched=(None if self.memory.eta_latched is None else self.memory.eta_latched.copy()),
            event=info.get("termination"),
            done=done,
            monitor_info=dict(info),
        )
        self.records.append(record)
        return record

    def run_to_terminal(self) -> tuple[StepRecord, ...]:
        while not bool(self.kernel.env.done):
            self.step()
        return tuple(self.records)


__all__ = [
    "AuthoritativeStepKernel",
    "ControllerMemory",
    "DecisionHead",
    "ETA_DIMENSION",
    "FEATURE_DIMENSION",
    "FiniteHistoryView",
    "Mode",
    "PreparedStep",
    "RecoverySystem",
    "SingleSegmentRecoveryMachine",
    "StepRecord",
    "TerminalStateError",
]
