# Property diagnosis after A/B/C/D failure

All four predeclared families failed before fresh-interior validation. NCB-AFFINE was closest: it obtained full recall 0.550 and retained recall 0.408, but had 1 cached VAL false inclusion, neighbor-target coverage 0.500, and universal retained coverage 0.773. RACS curvature did not help; RFSE remained low-recall; splitting into two NCB lobes reduced recall.

The observed property is a precision/coverage conflict, not absence of robust eta. Smooth centered supports must globally shrink to avoid sharp local negatives, thereby discarding distant robust modes. Existing 88/96 successful interpolation and 16/16 hole probes argue against pervasive holes, while the new VAL neighbor-mode probes (17/43 B63) show strong state dependence and sharp mode-specific exclusion. The missing capability appeared to be sharp asymmetric clipping rather than more curvature.

Three permitted synthesis cycles tested: (1) a clipped NCB with up to two analytic halfspaces; (2) a rotated asymmetric parallelotope for sharp planar/asymmetric bounds; and (3) a union of two such parallelotopes for limited nonconvexity. None passed the unchanged Section 10 gates. RAP reached full median recall 0.618, but only 5/8 states reached 0.40, retained recall was 0.333, neighbor-target coverage 0.25, and a common retained eta covered 21/22 fitted TRAIN sets. This demonstrates that simply enlarging a sharp-edged family recreates universal-constant degeneracy.

No further form is synthesized: the three-cycle limit is exhausted, and additional analytic flexibility would be unsupported by held-out evidence.
