# Success-basin gate conceptual-validity audit

## Overall conclusion

**GATE_TARGET_TEMPORAL_MISMATCH**

The previous fixed-pair Flow-cloud result did not by itself prove an
intervention-class rule: state identity is strongly encoded and any two state
clouds can be easy to separate.  However, the stronger leakage-free diagnostic
does show that the existing 214-D input contains source-generalizable class
information.  The dominant conceptual problem exposed here is temporal: the
current label asks whether zero correction for the *entire remaining future*
succeeds, while the deployed gate needs to decide whether correction is
necessary *at the current physical step* when later intervention remains
available.

## 1. Cloud separation: class information or identity?

Pairwise cloud separability mainly proves state identity.  State-ID accuracy
from disjoint Flow variants is 0.824 across 277 states (chance 0.0036), and in
the difficult N=13 set same-label clouds are not systematically closer than
opposite-label clouds: `P(d_opposite > d_same)=0.461`.  Thus the earlier fixed
pair argument was overstated as evidence, although its representation-
sufficiency conclusion is independently supported below.

## 2. Source-generalizable class signal in 214-D

A fixed L2-logistic probe trained only on other source groups achieves:

- all 277 oracle-stable states: BAcc 0.9454, AUROC 0.9801, AUPRC 0.9791;
- difficult N=13: BAcc 0.9167, AUROC 1.000, 12/13 correct.

All 64 variants of every held-out state were excluded from fitting.  Therefore
the 214-D representation does support intervention-class generalization across
unseen states and source groups; the deployed gate's failure is not evidence
that class information is absent.

## 3. Long-horizon label versus current-step decision

They are not aligned.  The causal branch audit executes either the existing
oracle correction or exactly `u_safe` at the current step, then applies the
same saved `eta_best` DiagnosticCorrector/hard-projection continuation from
`t+1` in both branches.  This permits future intervention and changes only the
current action.  It is the closest existing oracle-consistent continuation,
not a receding-horizon eta re-query.

Among the seven difficult `y_long=1` states:

- 6/7 have `Q_N=Q_I=64/64` and are `NOW_INTERVENTION_NOT_NECESSARY`;
- 0/7 are statistically current-step necessary;
- the remaining p018 state is ambiguous at 256 pairs:
  `Q_N=253/256`, `Q_I=256/256`, `Delta_Q=0.01172`, paired bootstrap 95% CI
  `[0, 0.02734]`.

All six difficult `y_long=0` states are also not necessary now, so no reverse
mismatch was found.  Four preregistered easy `y_long=1` controls likewise show
no observed one-step success loss.

## 4. Difficult-state current-step criticality

Of the 13 difficult oracle-stable states:

- current-step intervention necessary: **0**;
- current-step intervention not necessary: **12**;
- ambiguous: **1**.

## 5. Interpretation of the hard-boundary failure

Yes.  A substantial part of the hard-boundary gate failure is an artifact of
asking the wrong temporal classification question.  `y_long=1` can mean that
intervention is needed eventually, not that it must be applied now.  Training
an instantaneous gate against that label forces it to distinguish states that
are often causally equivalent at the current step.

## 6. Decision

**GATE_TARGET_TEMPORAL_MISMATCH**.  The class representation is usable under
strict held-source probes, while the label itself is too conservative for an
instantaneous gate under the tested future-policy semantics.

## 7. Smallest justified next experiment

Run one preregistered, generic **delay-window audit** on a broader
oracle-stable state sample using the same matched downstream continuation at
delays 0/1/2/4.  Its sole purpose should be to locate the temporal intervention
window and define a statistically supported step-aligned target before any new
gate loss, architecture, or correction-head experiment.

