# Requirements for the next analytic basin representation search

The continuity audit permits representation search, but does not itself fit a basin. The next representation must respect all accumulated evidence:

- eta is a persistent 3-D OrthoFlow3 parameter selected once at true t0.
- Most observed t0 basins are broad tangentially and narrower normally.
- Existing success-success interpolation was 88/96 B63; explicit hole probes were 16/16 B63.
- Local success/failure boundaries can be sharp.
- Cross-state morphology is `SHARED_MORPHOLOGY_STATE_DEPENDENT_DEFORMATION`.
- A sphere is too conservative; the axis-aligned ellipsoid produced false inclusions.
- A four-ball union had severe recall failure; naive manifold and PACT encodings generalized poorly to independent B63 samples.
- The representation must allow state-dependent orientation and extent, asymmetric thickness, local nonconvexity or occasional gaps, low false inclusion, efficient distance/membership computation, and learnability from h0.
- Target transfer evidence implies that multiple distant robust eta values can be valid for the same local neighborhood, so it must preserve set-valued/multimodal supervision rather than collapse it to a single MSE target.
