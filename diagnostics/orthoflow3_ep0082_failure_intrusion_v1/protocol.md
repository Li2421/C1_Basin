# ep0082 targeted failure-intrusion audit

Only the frozen ep0082 true-t0 state is queried. No controller, Q/J network,
new Basin family, new state/scenario or Fresh-WIDE experiment. Basis SHA and
all continuation semantics are inherited and verified against the completed
general audit. Exact B63 means >=63/64 fixed matched future indices0..63.
Q64 is an empirical label, not a population confidence certificate.

## Existing evidence and coordinates

185 exact-Q64 tuples (117 B63,68 non-B63) are frozen before new execution.
Exact float64 eta deduplication; no screening-only hard labels. The original
ep0082 R0 frame from the manifold audit is reused as coordinates only.
Acquisition categories are joined from frozen manifests, not inferred solely
from the rollout wrapper's phase string. Earlier validation-role correction
is retained. Raw controller/cache files will not be changed.

## Candidate regions and cache-first discovery

Internal candidates: inside the convex hull of existing B63 points and nearest
B63 distance <=2 times the90th-percentile leave-one-out positive5NN distance.
This generous local-support filter avoids treating dense ball-construction
sampling as a uniform density estimate. All hull-interior negatives remain
visible even if a support filter fails. The threshold and local counts are
reported, not used as proof of actual Basin occupancy.

Descriptive negative clustering: radius graph scales0.05,0.10,0.15,0.20,0.30.
Primary working regions use0.20; report scale dependence, never call these
true components. Each region's cached negative medoid represents its first
targeted panel, ties by exact eta key. Every primary region is included.

For all internal points, record three R0 slice views with off-plane bands
0.02 and0.05, plus native views. Approximate-slice ordering is discovery only.
Final transition counts require exact Q64 points on a shared line within
1e-9 geometric floating-point tolerance; no approximate eta cache reuse.

## Round1: corridors and six-direction local lines

For each region medoid, test e1,e2,e3 conditional lines with offsets
-0.10,-0.05,0,+0.05,+0.10 in normalized coordinates. If an offset would leave
E_bridge, replace its magnitude by min(requested,0.95*ray-domain clearance),
deduplicate exact resulting coordinates, and record actual distances. These
six angular directions are also the initial pocket-enclosure test. No point
is interpreted as lying on a plane of successful Basin geometry.

Failure corridors: first build graphs on ALL cached negatives, with edges
<=0.25. Reject candidate edges containing an already-confirmed collinear
B63 observation. Add feasible shortest-normal projections onto E_bridge
facets as candidate outer endpoints. Choose a route minimizing the number
of genuinely uncached segment probes (ties by length/node index), including
direct exit only if it wins over cached waypoint routes. Every segment uses
at least a midpoint and maximum sample spacing0.05; boundary endpoint also
receives exact Q64 unless cached. If any sampled point is B63, that proposed
route is rejected, not silently repaired into a claimed failure corridor.
Finite samples support empirical corridors only, not continuous topology.

## Adaptive refinement and decisions

After round1, exact opposite-label intervals on the registered lines are
bisected until width<=0.01, with at most4 bisections per original bracket.
Prioritize intervals involved in repeated alternation, then one representative
boundary per region/axis, then remaining brackets. No unlimited binary search.
If a corridor fails, seek a second cache-informed piecewise route or test the
specific suspected escape gap; freeze that design before execution.
Additional informative line directions (<=3 per region) are allowed only if
the first exact lines leave the classification genuinely ambiguous.

Plan for <=100 genuinely new exact-Q64 eta overall. Before exceeding100,
record the exact unresolved question and incremental cost. Before approaching
150, perform a full design-insufficiency diagnostic. These are not arbitrary
termination thresholds; topology may remain underresolved in a small audit.

Region-connected counts mean a sampled failure corridor from its tested
representative, not every negative in a geometrically clustered region.
An enclosed hole requires six-direction failure-to-success enclosure and
explicitly tested escape alternatives; mere convex-hull containment is not
enough. Repeated transitions on >=2 independent local lines are necessary
but not sufficient for interleaving: one folded/branched connected notch
must remain a competing explanation. Classification may be UNDERRESOLVED.

## Resources

Initial server check: daytime, no GPU jobs, essentially idle compute; user
daytime/light-load ceiling is2GPU shards. Use2shards,2CPU threads each.
Estimate42worker-seconds perQ64 plus startup; diagnose >3x cost surprise.
No increased shard count without a fresh policy-compatible load/time check.
