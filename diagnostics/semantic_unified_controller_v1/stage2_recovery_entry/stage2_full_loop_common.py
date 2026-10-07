"""Frozen Stage-2 full-episode NORMAL(S/L/R) semantics.

There is no periodic clock.  In NORMAL, a 214-D learned mode head chooses
Safety, one local Direct-g correction, or one-way entry into structured-eta
recovery.  Recovery predicts eta once, holds it immutable, and runs densely to
terminal; Stage 2 deliberately has no exit decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ModeStepRecord:
    step: int
    mode_before: str
    decision: str
    feature: np.ndarray | None
    u_safe: np.ndarray
    correction: np.ndarray
    u_exec: np.ndarray
    first_projection_retry: bool
    second_projection_retry: bool
    monitor_info: dict[str, Any]


class Stage2ModeMachine:
    """One-way NORMAL -> RECOVERY controller used only for Stage-2 evaluation."""

    def __init__(
        self,
        *,
        kernel: Any,
        mode_head: Callable[[np.ndarray], int],
        direct_model: Any,
        eta_model: Any,
        eta_corrector_factory: Callable[[np.ndarray], Callable] | None = None,
    ) -> None:
        self.kernel = kernel
        self.mode_head = mode_head
        self.direct_model = direct_model
        self.eta_model = eta_model
        if eta_corrector_factory is None:
            from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi

            eta_corrector_factory = lambda eta: DiagnosticCorrector(DiagnosticPhi(*eta.tolist()))
        self.eta_corrector_factory = eta_corrector_factory
        self.in_recovery = False
        self.entry_step: int | None = None
        self.eta_latched: np.ndarray | None = None
        self._eta_corrector: Callable | None = None
        self.mode_query_count = 0
        self.direct_query_count = 0
        self.eta_query_count = 0
        self.local_action_count = 0
        self.recovery_transition_count = 0
        self.records: list[ModeStepRecord] = []

    def _latch_eta(self, feature: np.ndarray) -> None:
        if self.eta_latched is not None or self._eta_corrector is not None:
            raise AssertionError("eta may be predicted only once")
        eta, _, _ = self.eta_model.predict(np.asarray(feature)[None])
        eta = np.asarray(eta, dtype=np.float64)
        if eta.shape != (1, 3) or not np.isfinite(eta).all():
            raise RuntimeError(("invalid structured-eta prediction", eta.shape))
        self.eta_latched = eta[0].copy()
        self.eta_latched.flags.writeable = False
        self._eta_corrector = self.eta_corrector_factory(self.eta_latched)
        self.eta_query_count += 1

    def _eta_correction(self, context: Any) -> np.ndarray:
        if self.eta_latched is None or self._eta_corrector is None:
            raise AssertionError("recovery requires a latched eta")
        before = self.eta_latched.copy()
        correction = np.asarray(
            self._eta_corrector(
                np.asarray(context.observation, dtype=np.float64),
                np.asarray(context.u_safe, dtype=np.float64),
                self.kernel.config.max_speed,
            ),
            dtype=np.float64,
        )
        if correction.shape != (2, 2) or not np.isfinite(correction).all():
            raise RuntimeError(("invalid structured-eta correction", correction.shape))
        if not np.array_equal(before, self.eta_latched):
            raise AssertionError("latched eta changed during recovery")
        return correction

    def step(self) -> ModeStepRecord:
        if bool(self.kernel.env.done):
            raise RuntimeError("terminal event is absorbing")
        mode_before = "RECOVERY" if self.in_recovery else "NORMAL"
        context = self.kernel.prepare(need_feature=not self.in_recovery)
        correction = np.zeros((2, 2), dtype=np.float64)
        second_retry = False

        if self.in_recovery:
            correction = self._eta_correction(context)
            u_exec, second_retry = self.kernel.second_projection(context, correction)
            self.recovery_transition_count += 1
            decision = "CONTINUE_RECOVERY"
        else:
            feature = np.asarray(context.feature, dtype=np.float64)
            if feature.shape != (214,) or not np.isfinite(feature).all():
                raise RuntimeError(("invalid mode-head feature", feature.shape))
            action = int(self.mode_head(feature))
            self.mode_query_count += 1
            if action == 0:
                u_exec = context.u_safe
                decision = "SAFETY"
            elif action == 1:
                predicted = np.asarray(self.direct_model(feature[None]), dtype=np.float64)
                if predicted.shape != (1, 4) or not np.isfinite(predicted).all():
                    raise RuntimeError(("invalid Direct-g prediction", predicted.shape))
                correction = predicted[0].reshape(2, 2)
                u_exec, second_retry = self.kernel.second_projection(context, correction)
                self.direct_query_count += 1
                self.local_action_count += 1
                decision = "LOCAL"
            elif action == 2:
                self._latch_eta(feature)
                self.in_recovery = True
                self.entry_step = int(context.step)
                correction = self._eta_correction(context)
                u_exec, second_retry = self.kernel.second_projection(context, correction)
                self.recovery_transition_count += 1
                decision = "ENTER_RECOVERY"
            else:
                raise RuntimeError(("mode head returned invalid action", action))

        _, info = self.kernel.execute(u_exec)
        row = ModeStepRecord(
            step=int(context.step), mode_before=mode_before, decision=decision,
            feature=None if context.feature is None else np.asarray(context.feature).copy(),
            u_safe=np.asarray(context.u_safe).copy(), correction=correction.copy(),
            u_exec=np.asarray(u_exec).copy(),
            first_projection_retry=bool(context.first_projection_retry),
            second_projection_retry=bool(second_retry), monitor_info=dict(info),
        )
        self.records.append(row)
        return row

    def run_to_terminal(self) -> tuple[ModeStepRecord, ...]:
        while not bool(self.kernel.env.done):
            self.step()
        return tuple(self.records)


__all__ = ["ModeStepRecord", "Stage2ModeMachine"]
