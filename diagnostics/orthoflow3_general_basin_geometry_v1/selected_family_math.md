# No selected general analytic family

The best tested candidate below is **REJECTED**, not a conservative learning label.
Raw fitted parameters are frozen in `family_models_minimal_polyhedral_refined_r7_stable.json`.

Let z=(eta-(0.875,0,0.375))/(0.75,1,0.75), with componentwise division.
For state i the finite candidate is

    S_i = { z in E_bridge : a_ij^T z + b_ij <= 0, j=1..K_i;
                           q_i(z)=z^T A_i z+d_i^T z+e_i >=0 }, K_i<=8.

Unit-normal support faces are selected from the fit-positive hull by optimal
minimum set cover of known exterior non-B63 observations. A single quadratic
rejects the remaining hard negatives. No hull mesh or point-cloud lookup is
stored for membership. The finite deployed formula has at most42 raw coefficients,
or33 meaningful zero-set parameters after normal/scaling redundancies; the
exception beyond the preferred4pieces/32parameters was declared before validation.
The mathematical class can express intrusions, but neither connectedness nor
absence of disconnected pieces is guaranteed by the formula.

Retained construction, delta=0.025 normalized Euclidean units:

    domain unit halfspaces <= -delta;
    a_ij^T z+b_ij <= -delta;
    q_i(z) >= delta*||2 A_i z+d_i||_2 + delta^2*||A_i||_2.

The last inequality is a sufficient Taylor bound for a delta ball to remain
inside the *fitted quadratic inequality*. It is NOT a certificate for actual
closed-loop success. Five final retained samples were confirmed non-B63.

Reproducibility pseudocode (diagnostic only):

    fit_basin(evidence):
        resolve exact acquisition roles from frozen state/eta manifests
        retain non-validation hash-fit B63 + explicitly promoted development B63
        keep all exact non-B63 as hard negative constraints
        compute normalized fit-positive hull, no heldout-positive use
        choose optimal minimal facet cover of exterior negatives
        if more than8 faces required: return UNRESOLVED, never silently enlarge
        fit one bounded-coefficient quadratic against remaining negatives
        return state parameters and provenance snapshot, NOT verified label
    contains(eta): evaluate E_bridge, selected affine faces and q>=0
    retained_contains(eta): evaluate the three eroded inequality groups above
    sample_inside_retained(): frozen scrambled Sobol geometric pool;
        reject outside; freeze6 central/maximin points; exact-Q64 all6

The corrected intended-development fit required9 facets for ep0082 while retaining
all fit positives. This is not a lower bound for a70percent-recall model and does
not prove that a slightly larger family cannot work. No ninth face was silently
added, and no partial corrected fit is selected.
