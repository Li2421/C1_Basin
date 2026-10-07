# F2: sparse support cuts plus one quadratic exclusion

Frozen before fitting. NOT accepted.

Measured motivation: bounded-quadratic refinement failed14/48 Toy and8/24 DB fresh retained points. All22 failures lie outside the prior positive convex hull; no new enclosed-hole evidence emerged. On the unchanged eight Toy states, full-recall median improved only2.91percentage points and fresh false-inclusion rate only2.08points; retained recall decreased. The remaining hypothesis is inadequately shaped outer extent, not necessarily more interior components.

Full set: E_bridge intersect at most3 nonparallel affine support halfspaces intersect {g(z)>=0}, where g is quadratic. This is NOT the old RAP parallelotope, positive-ball union, or mesh membership. At most22 raw coefficients,4 learned inequality pieces. All state variation remains in parameters.

Fit uses the unchanged primary hash-fit positives plus disclosed positives from rejected validation rounds. Original primary heldout positives and the predeclared DB interpolation positives remain heldout. Construct candidate supporting facets of the FIT-positive convex hull offline, without retaining the hull as the representation. Greedily choose at most3 facets maximizing newly excluded known exact negatives minus0.05 times the additional fit-positive points lost under delta=.025 erosion; ties by deterministic facet index. Stop if no facet excludes an additional negative. Fit one quadratic to reject every remaining exact negative, using the existing fixed hinge/ridge objective and coefficient bounds. No heldout tuning of the piece count or penalty.

Finite deployed membership stores only selected plane normals/offsets and10 polynomial coefficients. Retained membership erodes every unit-normal support plane and E_bridge by delta=.025 and uses the exact sufficient quadratic erosion bound already frozen. No dense hull/mesh/kNN membership. Empty sets and omitted states remain explicit.

Same Section17 recall, cross-transfer, precision and complexity gates. If cached gates pass, reserve a new prospective validation namespace retained-validation-v4; previous rejected observations are development evidence, never validation of this fit. No controller training.
