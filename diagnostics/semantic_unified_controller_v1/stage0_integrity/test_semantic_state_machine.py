from __future__ import annotations

from dataclasses import fields
import unittest

from semantic_state_machine import (
    ExecutedPrimitive,
    Mode,
    NormalAction,
    RecoveryAction,
    SemanticMemory,
    SemanticStateMachine,
    TerminalStateError,
)


class SemanticStateMachineTests(unittest.TestCase):
    def test_safety_and_local_are_state_driven_and_stay_normal(self):
        machine = SemanticStateMachine()
        safety = machine.step(normal_action=NormalAction.SAFETY)
        local = machine.step(normal_action=NormalAction.LOCAL)
        self.assertEqual(safety.executed, ExecutedPrimitive.SAFETY)
        self.assertFalse(safety.requires_second_projection)
        self.assertEqual(local.executed, ExecutedPrimitive.LOCAL_DIRECT_G)
        self.assertTrue(local.requires_second_projection)
        self.assertEqual(machine.memory.mode, Mode.NORMAL)
        self.assertEqual(machine.memory.local_events, 1)

    def test_entry_predicts_eta_once_and_executes_recovery_same_step(self):
        machine = SemanticStateMachine()
        entered = machine.step(
            normal_action=NormalAction.ENTER_RECOVERY,
            predicted_eta=(0.25, -0.5, 0.75),
        )
        self.assertEqual(entered.mode_after, Mode.RECOVERY)
        self.assertEqual(entered.executed, ExecutedPrimitive.STRUCTURED_ETA)
        self.assertTrue(entered.eta_predicted)
        self.assertEqual(machine.memory.entry_step, 0)
        self.assertEqual(machine.memory.recovery_steps, 1)

        continued = machine.step(recovery_action=RecoveryAction.CONTINUE)
        self.assertFalse(continued.eta_predicted)
        self.assertEqual(continued.eta_latched, entered.eta_latched)
        self.assertEqual(machine.memory.recovery_steps, 2)

    def test_exit_executes_safety_same_step_and_reentry_is_impossible(self):
        machine = SemanticStateMachine()
        machine.step(
            normal_action=NormalAction.ENTER_RECOVERY,
            predicted_eta=(0.25, -0.5, 0.75),
        )
        exited = machine.step(recovery_action=RecoveryAction.EXIT)
        after = machine.step()
        self.assertEqual(exited.executed, ExecutedPrimitive.SAFETY)
        self.assertEqual(exited.mode_after, Mode.SAFETY_AFTER)
        self.assertEqual(after.executed, ExecutedPrimitive.SAFETY)
        with self.assertRaises(ValueError):
            machine.step(normal_action=NormalAction.ENTER_RECOVERY, predicted_eta=(0, 0, 0))

    def test_zero_duration_recovery_is_impossible(self):
        memory = SemanticMemory(
            mode=Mode.SAFETY_AFTER,
            eta_latched=(0.0, 0.0, 0.0),
            recovery_used=True,
            entry_step=3,
            exit_step=3,
            recovery_steps=1,
            physical_step=4,
        )
        with self.assertRaises(AssertionError):
            SemanticStateMachine(memory)

    def test_terminal_event_preempts_decision_without_mutation(self):
        machine = SemanticStateMachine()
        with self.assertRaises(TerminalStateError):
            machine.step(normal_action=NormalAction.LOCAL, terminal=True)
        self.assertEqual(machine.memory.physical_step, 0)
        self.assertEqual(machine.memory.local_events, 0)

    def test_no_clock_or_burst_fields_exist(self):
        names = {item.name.lower() for item in fields(SemanticMemory)}
        forbidden = {"h", "l", "phase", "cadence", "burst", "cooldown", "period"}
        self.assertTrue(names.isdisjoint(forbidden), names & forbidden)

    def test_invalid_eta_and_reprediction_are_rejected(self):
        machine = SemanticStateMachine()
        with self.assertRaises(ValueError):
            machine.step(normal_action=NormalAction.ENTER_RECOVERY, predicted_eta=(1.0, 2.0))
        machine.step(
            normal_action=NormalAction.ENTER_RECOVERY,
            predicted_eta=(0.25, -0.5, 0.75),
        )
        with self.assertRaises(ValueError):
            machine.step(recovery_action=RecoveryAction.CONTINUE, predicted_eta=(0, 0, 0))


if __name__ == "__main__":
    unittest.main()
