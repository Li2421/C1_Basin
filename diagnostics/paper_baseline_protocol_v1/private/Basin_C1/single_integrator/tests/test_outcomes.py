import itertools
import unittest
from single_integrator.outcomes import annotate, aggregate_outcomes


class OutcomeTests(unittest.TestCase):
    def test_all_event_combinations(self):
        rows = []
        for success, wall, agent, deadlock in itertools.product([False, True], repeat=4):
            row = annotate(dict(success=success, wall_collision=wall,
                                agent_collision=agent, deadlock=deadlock))
            expected = ('collision_failure' if wall or agent else 'success' if success
                        else 'safe_deadlock' if deadlock else 'other_timeout')
            self.assertEqual(row['outcome'], expected)
            self.assertEqual(row['success'], success and not (wall or agent))
            self.assertEqual(annotate(row), row)
            rows.append(row)
        stats = aggregate_outcomes(rows)
        self.assertEqual(sum(stats['outcome_counts'].values()), 16)
        self.assertEqual(sum(stats[k + '_rate'] for k in stats['outcome_counts']), 1.)
