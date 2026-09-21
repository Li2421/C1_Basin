"""CPU-only checks of causal intervention routing, not physical efficacy."""
import unittest

from single_integrator.c1.early_intervention import PrefixInterventionField


class RecordingField:
    def __init__(self):
        self.calls = []

    def prepare(self, params, observation, noise, projection=None):
        self.calls.append((params, observation, noise, projection))
        return self.calls[-1]


class PrefixTests(unittest.TestCase):
    def test_prefix_restores_reference_on_current_inputs(self):
        for prefix in [0, 1, 3, 5]:
            with self.subTest(prefix=prefix):
                field = RecordingField()
                reference, intervention, projection = object(), object(), object()
                wrapped = PrefixInterventionField(field, reference, intervention, prefix)
                for step in range(5):
                    observation, noise = object(), object()
                    result = wrapped.prepare(None, observation, noise, projection=projection)
                    self.assertEqual(result, (intervention if step < prefix else reference,
                                             observation, noise, projection))
                self.assertEqual(wrapped.calls, 5)


    def test_invalid_prefix(self):
        for invalid in [-1, 1.5, True, None]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PrefixInterventionField(RecordingField(), None, None, invalid)


    def test_separate_replays_do_not_share_schedule(self):
        field = RecordingField()
        first = PrefixInterventionField(field, 'ref', 'intervention', 1)
        second = PrefixInterventionField(field, 'ref', 'intervention', 1)
        self.assertEqual(first.prepare(None, None, None)[0], 'intervention')
        self.assertEqual(first.prepare(None, None, None)[0], 'ref')
        self.assertEqual(second.prepare(None, None, None)[0], 'intervention')
