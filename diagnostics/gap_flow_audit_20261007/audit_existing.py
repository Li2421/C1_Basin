"""Read-only audit of the frozen Gap1 test traces; never relabel a collision as deadlock."""
import json
from pathlib import Path

import numpy as np

from single_integrator.environment import point_segment_distance, segment_distance


ROOT = Path('diagnostics/gap_flow_v1')
OUTPUT = Path('diagnostics/gap_flow_audit_20261007')
SPECS = ((2, 'n2_recovery_wide'), (10, 'n10_wide_recovery'),
         (50, 'n50_wide_recovery'))


def audit(trace):
    with np.load(trace, allow_pickle=False) as archive:
        p, goals, walls = (archive[key] for key in ('positions', 'goals', 'walls'))
        swept = archive['swept_clearance']
        meta = json.loads(str(archive['metadata_json'].item()))
    cfg = meta['config']; dt = cfg['dt']; n = len(goals)
    distance = np.linalg.norm(p - goals[None], axis=2)
    speed = np.linalg.norm(np.diff(p, axis=0), axis=2) / dt
    progress = -(np.diff(distance, axis=0)) / dt
    near_gate = (np.abs(p[:, :, 0]) < 1.5) & (np.abs(p[:, :, 1]) < 1.0)
    first_gate = np.flatnonzero(near_gate.any(axis=1))
    # An opposing encounter requires proximity to the same gate at the same time.
    left = np.arange(0, n, 2); right = np.arange(1, n, 2)
    separation = np.linalg.norm(p[:, left, None, :] - p[:, None, right, :], axis=-1)
    at_gate = ((np.abs(p[:, :, 0][:, left, None]) < 1.5) &
               (np.abs(p[:, :, 0][:, None, right]) < 1.5))
    encounter = np.flatnonzero(((separation < 0.8) & at_gate).any(axis=(1, 2)))
    encounter_step = int(encounter[0]) if len(encounter) else None
    before, after = p[-2], p[-1]
    wall_clear = float(segment_distance(before[:, None], after[:, None],
        walls[None, :, 0], walls[None, :, 1]).min() -
        (cfg['agent_radius'] + cfg['wall_radius'] + cfg['wall_collision_margin']))
    i, j = np.triu_indices(n, 1)
    pair_clear = float(point_segment_distance(np.zeros(2),
        before[i] - before[j], after[i] - after[j]).min() -
        (2 * cfg['agent_radius'] + cfg['agent_collision_margin']))
    cutoff = max(0, len(speed) - max(1, len(speed) // 5))
    post = slice(encounter_step, None) if encounter_step is not None else None
    row = dict(rollout_id=meta['rollout_id'], termination=meta['termination'],
        steps=len(speed), seconds=len(speed)*dt, classification=(
            'C_dynamic_safety_instability' if meta['termination'] == 'collision' else
            'D_other_unadjudicated'), collision_type=(
            'wall_and_agent' if wall_clear <= 0 and pair_clear <= 0 else
            'wall' if wall_clear <= 0 else 'agent' if pair_clear <= 0 else 'unknown'),
        final_wall_clearance=wall_clear, final_agent_clearance=pair_clear,
        minimum_swept_clearance=float(swept.min()),
        start_mean_goal_distance=float(distance[0].mean()),
        final_mean_goal_distance=float(distance[-1].mean()),
        minimum_mean_goal_distance=float(distance.mean(axis=1).min()),
        goal_progress_before_end=float((distance[0] - distance[-1]).mean()),
        total_agent_path_length=float(np.linalg.norm(np.diff(p, axis=0), axis=2).sum()),
        mean_agent_path_length=float(np.linalg.norm(np.diff(p, axis=0), axis=2).sum(axis=0).mean()),
        closest_agent_to_gate_center=float(np.linalg.norm(p, axis=2).min()),
        fraction_agents_entering_gate_region=float(near_gate.any(axis=0).mean()),
        first_gate_region_step=int(first_gate[0]) if len(first_gate) else None,
        first_opposing_gate_encounter_step=encounter_step,
        mean_speed_before_encounter=float(speed[:encounter_step].mean()) if encounter_step else None,
        mean_speed_after_encounter=float(speed[post].mean()) if post is not None else None,
        mean_goal_progress_rate_before_encounter=float(progress[:encounter_step].mean()) if encounter_step else None,
        mean_goal_progress_rate_after_encounter=float(progress[post].mean()) if post is not None else None,
        mean_speed_final_fifth=float(speed[cutoff:].mean()),
        mean_goal_progress_rate_final_fifth=float(progress[cutoff:].mean()),
        final_fifth_low_speed_fraction=float((speed[cutoff:].mean(axis=1) < .05).mean()),
        final_fifth_low_progress_fraction=float((progress[cutoff:].mean(axis=1) < .01).mean()),
        checkpoint=meta['checkpoint'], controller=meta['controller'],
        controller_rng_seed=meta['controller_rng_seed'], episode_seed=meta['seed'],
        trace=str(trace))
    return row


def main():
    OUTPUT.mkdir(exist_ok=True)
    records = []
    for n, root in SPECS:
        summary = json.loads((ROOT/root/'eval_test_mc16_raw/summary.json').read_text())
        rows = []
        for item in summary['rollouts']:
            trace = ROOT/root/'eval_test_mc16_raw/traces'/f"{item['rollout_id']}.npz"
            row = audit(trace); row['N'] = n; rows.append(row); records.append(row)
        (OUTPUT/f'n{n}_test_rollouts.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(n, len(rows), {key:sum(r['collision_type']==key for r in rows)
                             for key in ('wall','agent','wall_and_agent','unknown')},
              'encounter',sum(r['first_opposing_gate_encounter_step'] is not None for r in rows),
              'gate',sum(r['first_gate_region_step'] is not None for r in rows),
              'median_steps',float(np.median([r['steps'] for r in rows])))
    (OUTPUT/'all_test_rollouts.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
