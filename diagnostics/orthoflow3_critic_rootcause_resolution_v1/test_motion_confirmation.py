"""Contract tests; never train, simulate, or open the rollout database."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from . import motion_confirmation as m


class ConfirmationContracts(unittest.TestCase):
    def test_targets_are_predeclared_and_configure_is_metadata_only(self):
        with patch.object(m.parent, 'configure') as configure:
            for i, seed in enumerate((88132, 88133)):
                out = m.configure(i)
                self.assertEqual(m.parent.TARGET_SEED, seed)
                self.assertEqual(m.parent.FAMILY_SEED, 940091000+i*10000)
                self.assertTrue(str(out).endswith(str(seed)))
            self.assertEqual(configure.call_count, 2)
        with self.assertRaises(AssertionError):
            m.configure(-1)

    def test_setup_does_not_execute_continuations(self):
        with TemporaryDirectory(prefix='c1-confirmation-contract-') as folder, \
             patch.object(m.parent, 'OUT', Path(folder)), \
             patch.object(m, 'configure'), patch.object(m, 'bridge'), \
             patch.object(m.parent, 'prepare') as prepare, \
             patch.object(m.parent, 'physical') as physical, \
             patch.object(m.parent, 'features') as features, \
             patch.object(m, 'response_inputs'), patch.object(m, 'validate_inputs'), \
             patch.object(m, 'preflight') as preflight, \
             patch.object(m.parent.engine, 'run') as run:
            for action in ('prepare', 'physical', 'inputs', 'preflight'):
                m.main(action, 0, 0)
            prepare.assert_called_once()
            physical.assert_called_once()
            self.assertEqual(features.call_count, 2)
            preflight.assert_called_once()
            run.assert_not_called()

    def test_bridge_precedes_new_flow_and_state_construction(self):
        events = []
        with patch.object(m, 'configure'), \
             patch.object(m, 'bridge', side_effect=lambda: events.append('bridge')), \
             patch.object(m.parent, 'flow', side_effect=lambda: events.append('flow')), \
             patch.object(m.parent, 'prepare', side_effect=lambda: events.append('prepare')):
            m.main('flow', 0, 0)
            m.main('prepare', 0, 0)
        self.assertEqual(events, ['bridge', 'flow', 'bridge', 'prepare'])


if __name__ == '__main__':
    unittest.main()
