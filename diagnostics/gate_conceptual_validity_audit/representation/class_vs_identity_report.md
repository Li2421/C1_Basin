# Class information versus state/source identity audit

## Outcome

**CLASS_SIGNAL_GENERALIZES**

The earlier observation that two fixed 64-variant clouds are perfectly separable was **not by itself evidence of an intervention-class rule**; it established state identity.  The independent source-LOGO experiment now supplies the missing evidence: a linear classifier trained only on other source groups generalizes strongly to both the full stable pool and the predefined hard boundary sources.

## Frozen data and leakage controls

- 277 oracle-stable augmented states: 157 gate=0 and 120 gate=1.
- 13 pre-registered difficult stable states from 6 source groups.
- Exactly 64 saved 214-D Flow variants per state; no features or rollouts were regenerated.
- State-held-out probes exclude every variant of the test state. Source-held-out probes exclude every state and variant from the test source.
- Metrics are state-level: probabilities are averaged over the held state's 64 variants.

## Cloud geometry

For cross-source pairs in the full stable pool, the probability that an opposite-label centroid distance exceeds a same-label distance is 0.582.  For the difficult 13-state pool it is 0.461.  Thus label has some global geometric association, but hard-boundary clouds do not form a clean label geometry.

Every difficult-pair cloud distance uses exact 64x64 distances.  For the 38,226 all-stable pairs, centroid and RMS cross-cloud distances are exact; mean distance, energy, and MMD use the same fixed stratified 8x8 variants for tractability.

## Generic label generalization

- Full stable source-LOGO linear probe: BAcc 0.945, AUROC 0.980, accuracy 0.946.
- Difficult-13 source-LOGO linear probe: BAcc 0.917, AUROC 1.000, accuracy 0.923.

The full pool and the hard subset both support a source-generalizable intervention-class direction. This diagnostic uses a fixed, numerically converged L2-logistic probe (`C=1`); it is not the deployed gate and no test threshold was tuned.

## Identity information

- State ID from disjoint Flow variants: 0.824 accuracy across 277 states (chance 0.0036).
- Source ID from disjoint variants of known states: 0.425 across 117 sources (chance 0.0085).
- Source ID with the entire test state excluded, on 12 multi-state groups: 0.169 accuracy, 0.090 group-balanced accuracy (chance 0.083).

State identity is strongly retained, so fixed-pair separability cannot by itself validate a gate representation. Source identity does not generalize nearly as well when an entire state is removed, while intervention labels do generalize in the strict source-LOGO probe.

## Interpretation

Raw Euclidean cloud geometry is identity-dominated: difficult same-label clouds are not systematically closer than opposite-label clouds. Nevertheless, a supervised linear direction learned from other sources separates the labels well. Therefore the old pairwise-cloud argument was logically insufficient, but its representation-sufficiency conclusion is supported by this stronger, leakage-free class probe. The remaining deployed-gate failure is not explained by absence of class information in 214-D.
