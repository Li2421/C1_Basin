"""Delivery checks use explicitly synthetic fixtures, never research evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from new_benchmark_common.batch_videos import SCENARIOS, ROLES, load_trace, validate_plan, render_video, main


class VideoContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = {'batch_id': 'synthetic_test_only', 'scenarios': {}}
        for scenario in SCENARIOS:
            rows = []
            for index, role in enumerate(ROLES):
                trace = self.write_trace(scenario, index)
                row = {'role': role, 'trace': trace.name}
                if role.startswith('deadlock'):
                    row['baseline'] = self.write_trace(scenario, index, baseline=True).name
                rows.append(row)
            self.plan['scenarios'][scenario] = rows

    def write_trace(self, scenario, index, baseline=False):
        n = 2 if scenario == 'toy_giveway' else 4
        p = np.zeros((4, n, 2))
        p[:, :, 0] = np.arange(n) + index / 10
        p[:, :, 1] = np.arange(4)[:, None] / 10
        meta = dict(batch_id='synthetic_test_only', scenario=scenario,
                    rollout_id=f'{scenario}_{index}_{baseline}', controller='synthetic_fixture',
                    checkpoint='synthetic_fixture', seed=index, source_record='synthetic_fixture',
                    initial_state_sha256=f'synthetic_initial_state_{index}',
                    config=dict(dt=.05, agent_radius=.1, outer_radius=5., obstacle_radius=.5),
                    start_step=0, episode_steps=3, complete=True,
                    termination='deadlock' if baseline else 'success', collision=False,
                    safety_enabled=True, deadlock_detected=baseline, deadlock_criterion='test_only')
        path = self.root / f'{meta["rollout_id"]}.npz'
        np.savez_compressed(path, positions=p, steps=np.arange(4), goals=p[-1],
                            walls=np.array([[[-1., -1.], [5., -1.]]]),
                            swept_clearance=np.ones(3), metadata_json=np.array(json.dumps(meta)))
        return path

    def mutate(self, path, *, metadata=None, **arrays):
        with np.load(path) as z:
            content = dict(z)
        if metadata:
            meta = json.loads(str(content['metadata_json'].item()))
            meta.update(metadata)
            content['metadata_json'] = np.array(json.dumps(meta))
        content.update(arrays)
        np.savez_compressed(path, **content)

    def test_four_scene_plan(self):
        self.assertEqual(len(validate_plan(self.plan, self.root)), 12)

    def test_missing_scene_rejected(self):
        del self.plan['scenarios']['ring_exchange']
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.root)

    def test_duplicate_clip_rejected(self):
        rows = self.plan['scenarios']['toy_giveway']
        rows[1]['trace'] = rows[0]['trace']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_plan(self.plan, self.root)

    def test_partial_and_skipped_frames_rejected(self):
        path = self.root / self.plan['scenarios']['toy_giveway'][0]['trace']
        self.mutate(path, steps=np.array([0, 1, 3, 4]))
        with self.assertRaisesRegex(ValueError, 'full episode'):
            load_trace(path)

    def test_timeout_not_deadlock(self):
        path = self.root / self.plan['scenarios']['ring_exchange'][0]['baseline']
        self.mutate(path, metadata={'termination': 'timeout'})
        with self.assertRaisesRegex(ValueError, 'deadlock baseline'):
            validate_plan(self.plan, self.root)

    def test_swept_collision_not_success(self):
        path = self.root / self.plan['scenarios']['toy_giveway'][2]['trace']
        self.mutate(path, swept_clearance=np.array([.1, -.01, .1]))
        with self.assertRaisesRegex(ValueError, 'collision-free'):
            validate_plan(self.plan, self.root)

    def test_wrong_initial_state_rejected(self):
        path = self.root / self.plan['scenarios']['toy_giveway'][0]['baseline']
        self.mutate(path, metadata={'initial_state_sha256': 'different_velocity'})
        with self.assertRaisesRegex(ValueError, 'initial state'):
            validate_plan(self.plan, self.root)

    def test_wrong_batch_rejected(self):
        path = self.root / self.plan['scenarios']['toy_giveway'][2]['trace']
        self.mutate(path, metadata={'batch_id': 'older_batch'})
        with self.assertRaisesRegex(ValueError, 'belong to this batch'):
            validate_plan(self.plan, self.root)

    def test_failed_batch_removes_previous_delivery_marker(self):
        output = self.root / 'delivery'
        output.mkdir()
        marker = output / 'manifest.json'
        marker.write_text('{"complete": true}')
        result = subprocess.run(['bash', 'scripts/run_batch_with_videos.sh',
                                 str(self.root / 'plan.json'), str(output), '--', 'false'])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_cli_delivers_twelve_videos_and_invalidates_failed_rerun(self):
        plan_path = self.root / 'plan.json'
        output = self.root / 'videos'
        plan_path.write_text(json.dumps(self.plan))
        main(['--plan', str(plan_path), '--output', str(output)])
        manifest = output / 'manifest.json'
        result = json.loads(manifest.read_text())
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['videos']), 12)
        self.assertTrue(all((output / row['file']).is_file() for row in result['videos']))
        del self.plan['scenarios']['ring_exchange']
        plan_path.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError):
            main(['--plan', str(plan_path), '--output', str(output)])
        self.assertFalse(manifest.exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_render_all_scenarios_preserves_full_frames(self):
        for scenario, role, path, meta, data, baseline in validate_plan(self.plan, self.root):
            if role != 'deadlock_show':
                continue
            output = self.root / f'{scenario}.mp4'
            self.assertEqual(render_video(output, meta, data, baseline), 4)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
