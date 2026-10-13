"""Review-facing artifact hashes; raw directory digests are in verification.json."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRODUCTS=(
 'protocol.json','rollout_protocol.json','live_full_protocol.json','root_refine_protocol.json',
 'cache_preflight.json','live_full_cache_preflight.json','root_refine_cache_preflight.json',
 'action_state_summary.json','action_rows.json','state_rows.json','cached_geometry_rows.json',
 'boundary_property_summary.json','chain_rule_rows.json','boundary_transversality_rows.json',
 'state_margin_summary.json','state_boundary_refined_summary.json','state_boundary_refined_rows.json',
 'basis_feedback_summary.json','basis_feedback_rows.json','basis_derivative_summary.json',
 'basis_derivative_rows.json','basis_derivative_raw.npz','global_geometry_summary.json',
 'local_geometry_properties.json','nominal_branch_summary.json','counterexamples.json',
 'verification.json','pilot_invalid.json','cache_integration_gap.json','video_gap_status.json',
 'video_gate.log','final_report.txt','figures/property_overview.pdf',
 'figures/action_geometry_state_continuity.pdf','figures/state_margin_boundary_continuity.pdf',
 'figures/state_boundary_refined.pdf','figures/chain_transversality.pdf','figures/basis_feedback.pdf')

def main():
    out={}
    for name in PRODUCTS:
        p=HERE/name
        if not p.is_file():raise FileNotFoundError(p)
        raw=p.read_bytes()
        out[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (HERE/'artifact_index.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
