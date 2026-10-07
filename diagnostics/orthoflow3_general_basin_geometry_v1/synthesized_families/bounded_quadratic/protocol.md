# F1: bounded quadratic support with one quadratic exclusion

Status: hypothesis, NOT accepted. Frozen before fitting this family.

Motivation: the previous unrestricted single quadratic included21/42 fresh retained non-B63 points.20/21 were outside the prior positive convex hull; an enclosing covariance ellipsoid would still include15/21. Seven targeted positive-to-failure segments each showed one sampled transition (six SFFFF, one SSFFF). One deep interior failure remains a pocket/notch question. Thus finite outer support and interior exclusion must be separately adjustable; neither the entire E_bridge nor a positive-only covariance envelope is adequate evidence of support.

Normalized z uses the unchanged frozen affine eta normalization. Let phi(z)=(1,z1,z2,z3,z1^2,z2^2,z3^2,z1z2,z1z3,z2z3).

Full set: E_bridge intersect {g0(z)>=0} intersect {g1(z)>=0}, where gk=wk dot phi. The quadratic matrix of g0 is constrained <=-0.001 I; its positive region is bounded and convex (or empty, explicitly reported). g1 is an unrestricted quadratic exclusion; it can describe an exterior cut, notch, or pocket without presupposing which topology is real. At most20 coefficients,2 learned inequalities. No dense storage or lookup membership.

Fit positives: unchanged hash-fit B63, excluding prospective validation positives. All known exact negatives remain hard constraints. Partition negatives using the convex hull of FIT positives only: exterior negatives constrain g0<=-0.01; interior negatives constrain g1<=-0.01. The hull is a fitting instrument, NOT membership or a label. No outcome-adaptive reassignment. Each quadratic minimizes mean squared positive hinge [max(0,1-g(P))]^2 +0.002||w[1:]||^2, coefficients bounded[-50,50]. g0 uses deterministic convex semidefinite optimization. g1 uses the existing deterministic convex quadratic-feature fit. If there is no interior negative, g1=1. No complexity grid or held-out tuning.

Retained region uses the SAME delta=.025 as the previous quadratic family: erode E_bridge by delta and require gk(z)>=delta||grad gk(z)||+delta^2||Ak||_2 for each active inequality. This is a sufficient Euclidean-ball erosion under the quadratic expansion bound. An empty retained set is explicitly unusable; no silent state removal.

Evaluate unchanged Section17 gates, report each state/scenario separately. Cached positive recall excludes points deliberately sampled inside an earlier fitted retained set. Any future retained validation must use new points and frozen parameters; seed namespace retained-validation-v2 is reserved before outcomes. No repair of the rejected r1 validation claim. Membership cannot establish topology beyond observed evidence.
