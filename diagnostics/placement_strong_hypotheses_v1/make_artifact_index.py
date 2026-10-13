"""Hashes of review-facing products; raw trace directory digests live in verification.json."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRODUCTS=(
    'protocol.json','sobol128_protocol.json','grid9_protocol.json','robust_protocol.json',
    'h1_line_protocol.json','h1_gradient_protocol.json','hybrid_protocol.json','hybrid_grid_protocol.json',
    'curvature_protocol.json','curvature_followup_protocol.json',
    'panel_analysis.json','sobol128_analysis.json','grid9_topology.json','grid9_rho_arrays.npz',
    'robust_analysis.json','h1_roots.json','h1_gradient_analysis.json','h1_boundary_prediction.json',
    'transfer_boundary_rank.json','hybrid_analysis.json','hybrid_grid_analysis.json',
    'curvature_analysis.json','curvature_followup_analysis.json','geometry_summary.json',
    'verification.json','final_report.txt','cache_integration_gap.json',
    'video_gap_status.json','video_gate.log',
    'figures/volume_gain_loss.pdf','figures/volume_qmc_refinement.pdf',
    'figures/grid9_mid_slices.pdf','figures/topology_betti.pdf','figures/robust_gain_loss.pdf',
    'figures/action_family_swap.pdf','figures/hybrid_grid_lines.pdf',
    'figures/boundary_prediction.pdf','figures/curvature_radius.pdf',
    'figures/curvature_followup.pdf')


def main():
    out={}
    for name in PRODUCTS:
        p=HERE/name
        if not p.is_file():raise FileNotFoundError(p)
        b=p.read_bytes();out[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    (HERE/'artifact_index.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
