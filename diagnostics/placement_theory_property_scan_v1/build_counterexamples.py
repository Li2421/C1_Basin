"""Reproducible counterexamples against overly broad theorem assumptions."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'placement_boundary_mechanism_audit_v1'
STRONG=HERE.parent/'placement_strong_hypotheses_v1'

def main():
    actions=json.loads((HERE/'action_rows.json').read_text())
    states=json.loads((HERE/'state_rows.json').read_text())
    basis=json.loads((HERE/'basis_feedback_rows.json').read_text())
    old=json.loads((PRIOR/'rho_gradients.json').read_text())
    volumes=json.loads((HERE/'global_geometry_summary.json').read_text())['volume']
    kink=next(r for r in actions if r['uid'].endswith('820006') and r['eta_name']=='zero' and r['chain']=='TT')
    state_switch=max((r for r in states if not r['active_signature_same']),key=lambda r:r['G_quotient'])
    toy_grad=next(r for r in old if r['uid'].endswith('820013') and r['placement']=='FF' and r['center']=='candidate4')
    same_label=next(r for r in basis if r['uid'].endswith('820002') and r['chain']=='TT' and r['candidate_index']==4)
    examples=[
        dict(property_rejected='A stable symmetric finite-difference G proves differentiability',
            case='Toy UID 820006, TT, eta=0',G_scale_change=kink['G_change_vs_finest'][-2],
            one_sided_relative_gap=max(kink['one_sided_relative_gap_finest']),
            heldout_relative_error=kink['heldout']['relative_error'],
            branch_stable_axes_at_finest=kink['branch_stable_axis_count'][-1],
            evidence='action_results/hard_confirm_toy_1_820006_zero_TT.npz'),
        dict(property_rejected='A global modest state-Lipschitz constant for G',
            case=f"{state_switch['uid']} {state_switch['chain']} {state_switch['eta_name']}",
            state_distance=state_switch['state_distance'],G_difference=state_switch['G_difference'],
            G_quotient=state_switch['G_quotient'],active_signature_same=False,
            evidence='state_rows.json'),
        dict(property_rejected='Success-margin gradient is uniformly stable across event branches',
            case='Toy UID 820013 FF candidate4',gradient_scale_change=toy_grad['gradient_stability'],
            margin_type=toy_grad['baseline_margin_type'],termination=toy_grad['baseline_term'],
            evidence='diagnostics/placement_boundary_mechanism_audit_v1/rho_gradients.json'),
        dict(property_rejected='A large B-feedback trajectory difference must flip a binary label',
            case='Toy UID 820002 TT candidate4 live vs frozen B',
            common_prefix_final_position_difference=same_label['common_prefix_final_position_difference'],
            live_success=same_label['live_success'],frozen_success=same_label['frozen_success'],
            evidence='basis_feedback_rows.json'),
        dict(property_rejected='FF enlarges Success Basin across scenarios',
            case='Toy six-state bounded fixed-seed QMC',
            mean_gain=volumes['toy_give_way']['mean_gain_fraction'],
            mean_loss=volumes['toy_give_way']['mean_loss_fraction'],
            mean_net=volumes['toy_give_way']['mean_net_fraction'],
            evidence='global_geometry_summary.json'),
        dict(property_rejected='A coarse voxel component count proves topology change',
            case='Ring UID 920001 9^3 grid',
            evidence='diagnostics/placement_strong_hypotheses_v1/grid9_topology.json',
            qualification='3^3/5^3 and cubical-versus-6-neighbour conventions do not reproduce the same feature; numerical-failure assignments alter Betti numbers.')]
    (HERE/'counterexamples.json').write_text(json.dumps(examples,indent=2)+'\n')
    print(len(examples))

if __name__=='__main__':main()
