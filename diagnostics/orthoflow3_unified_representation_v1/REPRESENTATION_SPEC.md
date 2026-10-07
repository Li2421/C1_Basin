# Frozen first representation choice, before learning

Inherit Phase A PARTIAL, Phase B PARTIAL and the accepted contract repair.
R_old is the time-repaired Phase-B representation, not the pre-repair model.
Physical labels come from audited v2. No test labels enter training.

Use one architecture: shared per-agent MLP + shared ordered-pair message MLP
+ shared obstacle-relation MLP. Masked mean/max aggregation produces agent
embeddings; masked mean/max scene pooling plus physical global scalars and
entity counts produces h (128). Both generator and critic use this schema.
No learned scenario adapter, scene one-hot, agent ID, pair ID or obstacle ID.
The generator family remains diagonal squashed Gaussian with the accepted
sigma bounds and loss; the critic remains empirical-Q BCE. K means 16 random
proposals plus the accepted deterministic location candidate, ranked finitely.

Two matched training seeds 17/23; original optimizer budgets and early-stop
rules. Dataset and sampling are scenario/state balanced. Architecture is
chosen for simple shared relations, not confirmation outcomes. No architecture
search is planned. No physical labels are rerun solely for reencoding.

## Contract boundaries

Retain remaining horizon, current raw Flow probe, its bounded physical value,
first-action commitment, all goals/velocities, geometry, safety margins and
causally relevant monitor configuration. Double v2 contains only t0 states:
its monitor history is empty. Intermediate Double without monitor history
must fail parsing, not silently assume an empty history.

Four/Double have fixed upstream policy charts. The position and axes of that
reference chart are expressed in each agent's local frame. These are not
cardinal labels or priorities: they preserve the current physical relationship
to an anisotropic frozen controller. Under a **passive** transformation the
policy chart transforms too. Under an active scene rotation with unchanged
policy chart, h is permitted to change. The accepted contract explicitly does
not certify Four active rotations as continuation equivalences.

Ring's upstream observation/action chart is radial to its physical obstacle
center. Its geometry carries that center; no fixed orientation axis is needed.
Reflection is not imposed: CCW local basis orientation retains chirality.

Agent-list permutation is a reserialization of already associated physical
states/goals/actions. It must not permute MACFlow's original API slots and
silently call that the same controller. Such an active policy permutation is
not contract-certified. This task does not change MACFlow or its adapters.

## Gates

Before training, require masked structural invariance at 2e-5 float32 absolute
embedding tolerance for passive rotations/translations and entity ordering;
raw feature comparisons at 2e-6. Exercise N=2/3/4/5 and variable M, including
zero obstacles. Check exact aliases over the complete train/dev corpus and
report near-neighbor Q overlap without claiming a smoothness theorem.

Train/dev evaluation: fixed subset of previously evaluated validation states,
no test-driven choices. Fresh confirmation only after structural and learning
gates. No adaptation after confirmation. Preserve any upstream-policy frame
limitation explicitly rather than manufacture active-scene invariance.
