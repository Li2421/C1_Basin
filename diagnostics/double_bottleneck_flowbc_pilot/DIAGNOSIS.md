# Joint 4-Agent Flow-BC Pilot Diagnosis

Date: 2026-09-23

Scope: one CPU-only seed-0 pilot (`2 x 128`, 28,040 parameters, 2,000
updates), followed by raw per-agent-speed-bounded closed-loop evaluation. No
hard-safety projection, eta, `G_phi`, retraining, or hyperparameter sweep was
run.

## Result

Gate D failed closed-loop validation. All 12 predeclared raw rollouts (three
held-out initial-condition families/regimes times four noise seeds) terminated
in wall collision. Success was 0/12; agent-agent collision, deadlock, and
timeout were each 0/12. Consequently this checkpoint is a completed engineering
pilot but not a validated scientific four-agent Flow-BC baseline.

## Training and teacher-forced numerics

- Fixed train CFM loss: `2.207920 -> 0.289067`.
- Fixed validation CFM loss: `2.222850 -> 0.300094`; best step 2,000.
- All logged losses/gradients were finite; checkpoint reload was exact for the
  recorded probe, and the sampled tensor was `[B,4,2]`.
- On 512 held-out validation states with eight samples per state, mean sampled
  joint-action RMSE was `0.033124 m/s`; best-of-eight RMSE was `0.020242 m/s`.

These teacher-forced numbers did not transfer to closed-loop validity.

## Collision timing and location

Collisions occurred at steps 27--80 (`1.35--4.00 s`), with mean step `60.83`
(`3.04 s`). All were wall collisions and no rollout completed a coordination
mode. Endpoint wall-location inspection gives:

- 6/12: `B1` at the right upper entry/exit wall;
- 3/12: `A1` at the left lower outer wall;
- 2/12: `A1` at the left lower entry wall;
- 1/12: `B1` at the right bottleneck floor.

Thus failures were concentrated on the leading `A1`/`B1` agents while
approaching the bottleneck entrances, rather than on inter-agent contact. The
minimum swept wall clearance was `-0.013434 m`, whereas the minimum pair
surface distance stayed positive at `0.219322 m`.

## Closed-loop gap

The learned policy's mean per-step action total variation was `0.058025 m/s`,
versus `0.003366 m/s` for the six held-out expert rollouts: approximately
`17.24x` the expert value. Raw episodes therefore ended after only `60.83`
steps on average, compared with `717.17` expert steps, and their mean nearest
expert phase-aligned position RMSE was `2.6737 m`. No successful crossing
signature was observed, so the pilot supplies no evidence that the policy
reproduces either of the dataset's two valid modes.

## Interpretation and stop decision

The following is an evidence-supported diagnosis, not a proved causal
decomposition: transition-level Flow-BC is trained on expert states but draws a
new 8-D base sample every physical step. Small teacher-forced lateral errors
and the observed high-frequency action variation accumulate, move the closed
loop away from the expert state distribution, and then expose the model to
conditions absent from its teacher-forced validation. Independent per-step
resampling can worsen this drift and can also switch local mode decisions; the
present rollouts collide too early to measure mode switching directly.

The evidence does not identify whether more optimization steps, a persistent
latent/trajectory model, closed-loop data augmentation, or another modeling
change is the appropriate remedy. Per the pilot gate, no additional training
or hard-safety evaluation is justified without master review. Status is
**BLOCKED at Gate D closed-loop validity**, and the system is not ready for
safety-baseline or basin analysis.

Authoritative metrics and per-rollout arrays are in `evaluation_v2/summary.json`
and `evaluation_v2/rollouts/`. The earlier `evaluation/` directory contains the
same deterministic 12 rollouts before wall-versus-agent collision fields and
expert action-variation reference were added; `evaluation_v2/` is the reporting
artifact.
