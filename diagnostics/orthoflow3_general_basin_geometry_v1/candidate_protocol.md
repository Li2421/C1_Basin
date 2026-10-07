# Warm-start family definitions and limitations

Let z=(eta-[.875,0,.375])/[.75,1,.75], E_bridge given by frozen normalized halfspaces. All models are intersected with E_bridge. No unobserved point is labeled success by these fits. They are falsifiable candidate sets, not verified labels.

## A: common support minus boundary-connected spherical caps

`B=E_bridge intersect_j {||z-c_j|| >= r_j}`, j<=4. Each c_j lies on or beyond a domain facet, so each excluded ball connects to the domain exterior. This tests NOTCHES, not arbitrary enclosed failure holes. The base is literally shared across states; exclusions vary with state. Four meaningful scalar parameters/cap. Negative k-means is deterministic farthest-first; enumerate facets and outward shifts 0,.25,.5,1; radius covers its assigned fit negatives plus .002. Select K by retained fit-positive coverage minus .015K, not heldout performance.

Retained: E_bridge eroded by .025 and each exclusion radius increased by .025. This is a sufficient Euclidean .025 erosion of the full intersection. It is not a proof of dynamical success.

## B: native affine conditional band with at most2 cap cuts

Elliptical tangential support in z1,z2, and independently tilted lower/upper affine z3 bounds. Support covariance is expanded to enclose fit positives. Linear programs choose supporting affine bounds with frozen slope limits [-3,3]; at most2 boundary caps remove remaining fit negatives. Retained tangent factor .85, normal half-width factor .75, cap/domain erosion .025.

## C: rotated asymmetric slab with at most2 cap cuts

State-specific PCA frame; asymmetric coordinate extents enclosing fit positives with .002 padding, plus the same cap cuts. Retained tangent extent factors .85,.85 and thin extent .75, cap/domain erosion .025. This warm start has no curved middle surface; its failure does NOT exhaust every asymmetric slab family.

## D: star-radial body with at most2 cap cuts

Observed robust positive medoid center; enclosing covariance ellipsoid gives analytic directional radius `r(d)=(d^T A d)^(-1/2)`, followed by cap exclusions. Full body without exclusions is star-convex; the excluded version need NOT remain star-convex. Retained radial factor .75 and cap/domain erosion .025. This warm start is not an exhaustive search over all directional radius functions.

## E: one degree2 semialgebraic inequality

`g(z)=w0+w1z1+w2z2+w3z3+w4z1²+w5z2²+w6z3²+w7z1z2+w8z1z3+w9z2z3 >=0`, plus domain. SLSQP minimizes fit-positive squared hinge to g>=1 plus .002 coefficient penalty, with hard fit-negative g<=-.01 and coefficients bounded [-50,50]. Ten scalar coefficients. The family may express indefinite quadrics; topology is not presumed.

For A the symmetric quadratic matrix of g and delta=.025, retained requires `g(z)>=delta*||grad g(z)||+delta²*||A||_2`, plus eroded domain. Taylor's exact quadratic expansion proves that every radius-delta perturbation of a retained point stays in the FULL algebraic set (not necessarily in the true success basin).

## Evidence roles

The initial candidate comparison uses the frozen retrospective hash split for both positive and negative evidence. Known heldout contradictions are reported, not hidden. Original independent Sobol recall is a descriptive provenance-stratum diagnostic; because some of these points enter current fitting, it is NOT called an independent validation of the current fit. Primary heldout recall uses only points not fitted. Final new retained validation must be prospective, separately frozen and never repaired in place.

An optimization failure or one limited parameterization's failure is not proof that an entire mathematical family cannot work. Current results motivate targeted boundary/property probes. Family gate decisions also require topology review and fresh exact-Q64, not just cached scalar metrics.

## Next-round negative constraints (declared while geometry_round1 is running)

Initial and after-common fits withheld hashed negatives to expose out-of-fit precision failures. Those counterexamples are now known hard evidence. After the current targeted geometry batch, refitting must constrain ALL known exact negatives, while keeping the original hashed B63 holdout unchanged. Cached precision will then be a consistency check, NOT an independent precision estimate; only the separately frozen fresh retained-validation batch can supply the latter. This avoids both ignoring known failures and wrongly rejecting an entire class solely because a known failed point was kept out of its fitting constraints. No threshold, positive holdout assignment or erosion factor changes.
