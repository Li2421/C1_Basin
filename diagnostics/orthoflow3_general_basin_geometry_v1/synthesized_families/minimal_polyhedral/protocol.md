# F3: minimum-support polytope with one quadratic intrusion constraint

Frozen before fitting; NOT accepted.

F2 failed16/72 fresh points, improving the previous22/72 by8.33percentage points. Therefore the user saturation condition has NOT been reached. An offline exact set-cover diagnosis found that retaining ALL fit positives while excluding exterior negatives requires2–8 supporting facets (median3). The difficult Toy states need4,5,5,6,8; the other Toy states and all DB states need2–3. This is a conditional fitting diagnostic, NOT a lower bound at70%recall.

Same mathematical family for all states: E_bridge intersect K<=8 affine support halfspaces intersect one quadratic inequality g>=0. The quadratic handles negatives internal to the fit-positive envelope; an exterior-connected intrusion and an enclosed pocket are both expressible without asserting either topology. No mesh, kNN, or dense point-cloud membership is stored.

Select the smallest number of FIT-positive convex-hull facets covering ALL exact negatives exterior to that hull. Solve deterministic binary set cover with a fixed lexicographic-scale penalty1e-5 times individual fit-positive erosion loss and1e-9 times facet index. Require an optimal solution and K<=8. A future instance requiring more than8 is unresolved under this frozen family, not silently enlarged. Fit one quadratic to all remaining exact negatives using the unchanged objective/bounds. Primary heldout positives never enter fitting. Disclosed rejected-validation positives can enter development only.

Complexity exception, explicitly justified: at worst8 affine surfaces plus1 quadratic, exceeding the preferred4pieces. Unit-normal affine surfaces have3 geometric degrees each; a quadratic zero-set has9 after positive scaling, so the maximum is33 meaningful geometric parameters (42 stored coefficients), one above the preferred32. Smaller instances use their proven minimum K, not8 by default. Each chosen supporting surface is needed by this minimum-cover solution to exclude an observed exterior negative while retaining all fit positives; do not claim that this complexity is mathematically necessary for every70%recall representation. This finite bounded class is tested because the3-face precision gate failed and the extra directions are measured, not aesthetically chosen.

Retained erosion remains delta=.025 for domain and affine faces, and g>=delta||grad g||+delta^2||A||_2 for the quadratic. All Section17 scientific gates unchanged. Prospective validation namespace retained-validation-v5 is frozen now. No network/controller training.
