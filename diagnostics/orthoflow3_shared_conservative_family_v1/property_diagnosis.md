# Property diagnosis after Families A-D

The first oracle simulator was invalid for the requested practical question:
its global maximin order produced only one B63 witness among the first sixteen
labels on every Double-Bottleneck state.  V0 is preserved and V1 uses the same
adaptive robust-anchor policy in both scenarios.

Under V1, the two band/slab forms produced nontrivial reliable regions on
10/12 outer-held-out states, but median independent-B63 recall remained about
0.14 and retained robust-neighbor coverage was zero.  Conversely, a broad
superbody with one cut reached moderate recall (about 0.29) but admitted cached
non-B63 points and left only 4/12 states usable.  Thus the decisive cached
failure is a precision/coverage conflict: a single local slab is conservative
but misses separated transferable robust solutions, whereas one broad body
crosses sharp state-dependent exclusions.

The next discriminating forms are therefore (E) one connected affine capsule
aligned to two robust witnesses, supported by prior success-success path
evidence, with at most one cut; and, only as a topology diagnostic, (F) a union
of two affine conservative superbodies.  Both retain one global equation and
global erosion rule across scenarios, remain below 32 meaningful parameters,
and are evaluated under the unchanged gates.  No rollout is authorized until
a cached-CV candidate passes.
