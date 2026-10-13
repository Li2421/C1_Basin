"""Record hashes for the review-facing products of the audit."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCTS = (
    'code_audit.txt', 'final_report.txt', 'reproduce.sh',
    'jacobian_probe.py', 'raw_G_M.npz', 'raw_G_M.json',
    'independent_directional_probes.json', 'stable_only_fixed_Q.json',
    'fixed_q_stable_basin_results.json', 'existing_B15_labels.json',
    'flow_projection_decomposition.json', 'baseline_offsets.json',
    'trajectory_sensitivity_arrays.npz', 'trajectory_sensitivity.json',
    'recurrence_spot_checks.json', 'heldout_predictions.json',
    'rho_gradients.json', 'same_eta_boundary_chain.json',
    'boundary_protocol.json', 'boundary_normals.json',
    'cross_boundary_shift_predictions.json', 'verification.json',
    'video_gap_status.json', 'video_gate.log', 'video_gate_exit_code.txt',
    'figures/action_transfer.pdf', 'figures/fixed_Q_heldout.pdf',
    'figures/trajectory_sensitivity.pdf', 'figures/local_boundary.pdf',
)


def main():
    out = {}
    for name in PRODUCTS:
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        out[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    (HERE / 'artifact_index.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
