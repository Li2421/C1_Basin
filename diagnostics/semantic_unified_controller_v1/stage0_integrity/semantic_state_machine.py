"""Pure transition skeleton for the semantic Give-Way controller.

This module deliberately contains no policy inference, simulation, cadence, or
training.  It makes the allowed state transitions executable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


class Mode(str, Enum):
    NORMAL = "NORMAL"
    RECOVERY = "RECOVERY"
    SAFETY_AFTER = "SAFETY_AFTER"


class NormalAction(str, Enum):
    SAFETY = "S"
    LOCAL = "L"
    ENTER_RECOVERY = "R"


class RecoveryAction(str, Enum):
    CONTINUE = "C"
    EXIT = "X"


class ExecutedPrimitive(str, Enum):
    SAFETY = "SAFETY"
    LOCAL_DIRECT_G = "LOCAL_DIRECT_G"
    STRUCTURED_ETA = "STRUCTURED_ETA"


class TerminalStateError(RuntimeError):
    """Raised when a control decision is attempted after a terminal event."""


@dataclass(frozen=True)
class Transition:
    step: int
    mode_before: Mode
    mode_after: Mode
    decision: str
    executed: ExecutedPrimitive
    requires_second_projection: bool
    eta_predicted: bool
    eta_latched: tuple[float, float, float] | None


@dataclass
class SemanticMemory:
    mode: Mode = Mode.NORMAL
    eta_latched: tuple[float, float, float] | None = None
    recovery_used: bool = False
    entry_step: int | None = None
    exit_step: int | None = None
    recovery_steps: int = 0
    local_events: int = 0
    physical_step: int = 0

    def validate(self) -> None:
        if min(self.recovery_steps, self.local_events, self.physical_step) < 0:
            raise AssertionError("negative counter")
        if self.eta_latched is not None:
            if len(self.eta_latched) != 3 or not all(math.isfinite(v) for v in self.eta_latched):
                raise AssertionError("eta_latched must contain three finite coordinates")
        if self.mode is Mode.NORMAL:
            if self.recovery_used or self.entry_step is not None or self.exit_step is not None:
                raise AssertionError("NORMAL cannot contain recovery history")
            if self.eta_latched is not None or self.recovery_steps != 0:
                raise AssertionError("NORMAL cannot contain recovery state")
        elif self.mode is Mode.RECOVERY:
            if not self.recovery_used or self.entry_step is None or self.exit_step is not None:
                raise AssertionError("inconsistent RECOVERY memory")
            if self.eta_latched is None or self.recovery_steps < 1:
                raise AssertionError("RECOVERY requires latched eta and an executed entry step")
        elif self.mode is Mode.SAFETY_AFTER:
            if not self.recovery_used or self.entry_step is None or self.exit_step is None:
                raise AssertionError("inconsistent SAFETY_AFTER memory")
            if self.eta_latched is None or self.recovery_steps < 1:
                raise AssertionError("SAFETY_AFTER requires prior recovery")
            if self.exit_step <= self.entry_step:
                raise AssertionError("zero-duration recovery is forbidden")
        else:  # pragma: no cover
            raise AssertionError(f"unknown mode {self.mode!r}")


class SemanticStateMachine:
    """Advance the semantic controller by one physical decision step."""

    def __init__(self, memory: SemanticMemory | None = None) -> None:
        self.memory = memory or SemanticMemory()
        self.memory.validate()

    @staticmethod
    def _eta(value: Iterable[float] | None) -> tuple[float, float, float]:
        if value is None:
            raise ValueError("ENTER_RECOVERY requires the already-predicted eta")
        result = tuple(float(x) for x in value)
        if len(result) != 3 or not all(math.isfinite(x) for x in result):
            raise ValueError("eta must contain three finite coordinates")
        return result  # type: ignore[return-value]

    def step(
        self,
        *,
        normal_action: NormalAction | None = None,
        recovery_action: RecoveryAction | None = None,
        predicted_eta: Iterable[float] | None = None,
        terminal: bool = False,
    ) -> Transition:
        self.memory.validate()
        if terminal:
            raise TerminalStateError("terminal event has priority over controller decisions")

        before = self.memory.mode
        step = self.memory.physical_step
        eta_predicted = False

        if before is Mode.NORMAL:
            if normal_action is None or recovery_action is not None:
                raise ValueError("NORMAL requires exactly one NormalAction")
            if normal_action is NormalAction.SAFETY:
                executed = ExecutedPrimitive.SAFETY
            elif normal_action is NormalAction.LOCAL:
                self.memory.local_events += 1
                executed = ExecutedPrimitive.LOCAL_DIRECT_G
            elif normal_action is NormalAction.ENTER_RECOVERY:
                eta = self._eta(predicted_eta)
                self.memory.mode = Mode.RECOVERY
                self.memory.eta_latched = eta
                self.memory.recovery_used = True
                self.memory.entry_step = step
                self.memory.recovery_steps = 1
                eta_predicted = True
                executed = ExecutedPrimitive.STRUCTURED_ETA
            else:  # pragma: no cover
                raise AssertionError(normal_action)
        elif before is Mode.RECOVERY:
            if recovery_action is None or normal_action is not None or predicted_eta is not None:
                raise ValueError("RECOVERY requires exactly one RecoveryAction and cannot re-predict eta")
            if recovery_action is RecoveryAction.CONTINUE:
                self.memory.recovery_steps += 1
                executed = ExecutedPrimitive.STRUCTURED_ETA
            elif recovery_action is RecoveryAction.EXIT:
                self.memory.mode = Mode.SAFETY_AFTER
                self.memory.exit_step = step
                executed = ExecutedPrimitive.SAFETY
            else:  # pragma: no cover
                raise AssertionError(recovery_action)
        elif before is Mode.SAFETY_AFTER:
            if normal_action is not None or recovery_action is not None or predicted_eta is not None:
                raise ValueError("SAFETY_AFTER accepts no learned decision or eta")
            executed = ExecutedPrimitive.SAFETY
        else:  # pragma: no cover
            raise AssertionError(before)

        decision = (
            normal_action.value if normal_action is not None
            else recovery_action.value if recovery_action is not None
            else "SAFETY_AFTER"
        )
        transition = Transition(
            step=step,
            mode_before=before,
            mode_after=self.memory.mode,
            decision=decision,
            executed=executed,
            requires_second_projection=executed is not ExecutedPrimitive.SAFETY,
            eta_predicted=eta_predicted,
            eta_latched=self.memory.eta_latched,
        )
        self.memory.physical_step += 1
        self.memory.validate()
        return transition
