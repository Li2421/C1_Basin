# K16 critic closure and physical conditioning canonicalization

## Frozen decisions

- Phase A: `PHASE_A_CRITIC_PARTIAL`.
- Phase B: `CANONICALIZATION_PARTIAL`.
- Overall: `READY_WITH_REPRESENTATION_LIMITATION`.

These phases address different questions. Neither result authorizes an
environment-family generalization claim.

## Phase A: critic only

The generator and its mean+16-stochastic proposal convention stayed fixed.
The critic used audited v2 and complete proposal-aligned train/dev empirical
Q with the inherited BCE objective. Fresh confirmation was completed after
repairing experiment registration, with unchanged frozen proposals and seeds.

| Scenario | Fresh states | Oracle B15 | Old critic | New critic |
|---|---:|---:|---:|---:|
| Double | 24 | 24 | 24 | 24 |
| Four-Way | 24 | 24 | 21 + 3 numerical unresolved | 19 + 5 numerical unresolved |
| Ring | 60 | 60 | 48 | 56 |

Ring misses decreased 12→4, exploitation 6→1, rescue 15→16, and break 9→2
on this same fresh cohort. Double did not regress. Four's numerical unknowns
prevent certification of the joint no-regression gate. The complete record is
`phase_a/confirmation/FK_REPAIR_AND_CONFIRMATION.md`; no adaptation followed
these outcomes.

## Phase B: representation change

See `phase_b/TRANSFORMS.md`. Ring's actual reference Flow vectors now use the
same radial/tangent chart as the observation. Four uses a deterministic
quarter-turn/goal-associated agent representative and transforms all physical
vectors consistently. Relative blocks are rebuilt after sorting. Feature
normalization is shared across equivalent slots and fitted on train only.
Physical context is globally normalized without the explicit scenario one-hot.
Native-dimension learned adapters remain; this phase does not provide a
variable-entity architecture.

The derived `datasets/orthoflow3_basin_dataset_v2_canonical/` preserves all
245 state IDs and byte-identical eta labels. No physical outcomes were replayed
solely for reencoding. The same additional 80 Phase-A train/dev states and Q
evidence were reencoded for the critic. Generator and critic were trained from
scratch using their inherited families, objectives, budgets and three seeds.
Checkpoint selection used validation loss. No Phase-A confirmation state was
used for learning. Configs, histories and selected hashes are in
`phase_b/generator/`, `phase_b/critic/`, and their frozen manifests.

## Metamorphic and development evidence

All 1,280 passive rotation/relabel tests on the primary v2 states had zero
input discrepancy. Active transformations also recompute the frozen MACFlow;
they are not assumed to have equivalent continuation distributions.

| Metric, mean over 12 train/dev states | Previous | Canonical |
|---|---:|---:|
| Four 90° normalized proposal discrepancy | 1.2870 | 0.0685 |
| Four 90° critic top-1 consistency | 33.3% | 75.0% |
| Four cyclic relabel proposal discrepancy | 1.4341 | 0.0451 |
| Four cyclic relabel critic top-1 consistency | 33.3% | 91.7% |
| Ring 90° proposal discrepancy | 0.0803 | 0 |
| Ring 90° critic top-1 consistency | 91.7% | 100% |
| Ring cyclic relabel proposal discrepancy | 0.2264 | 0.0045 |

Ring 180/270° input, proposal and score discrepancies are also zero.
Four's actual Flow still differs under active rotations. This is an upstream
learned-policy limitation, not a failed coordinate transformation.

On 12 fixed validation states per scenario, both old and canonical pipelines
had oracle/selected B15 coverage 12/12 in every scenario. Canonical rescue
counts were Double 12, Four 11 and Ring 2; break and exploitation were zero.
Within-state pairwise Q ranking accuracy was 0.988, 1.000 and 0.781,
respectively. These small validation results do not establish broad
generalization. Full results are `phase_b/development/results.json`.

## New held-out rotation confirmation

After model/development choices were frozen, four new initial states per
primary scenario were drawn from the original development distributions, each
with 0/90/180/270° counterparts. Future random streams were paired by physical
parent anchor. Same eta was held for each continuation. The original
orientation's canonical proposal oracle was 4/4 for both scenarios.

| Scenario | Canonical 0° B15 | 90° | 180° | 270° |
|---|---:|---:|---:|---:|
| Four-Way | 4/4 | 0/4 | 0/4 | 0/4 |
| Ring | 4/4 | 4/4 | 4/4 | 4/4 |

For Four, old selected control was also 0/4 in every rotated orientation.
Canonical back-transformed trajectory RMSE improved from 1.924 to 0.758, but
paired terminal-outcome agreement remained only 12.5%. Reencoding has not
made the frozen Four MACFlow continuation rotation-equivariant.

For Ring, canonical paired terminal outcomes agreed on every certified seed
and mean back-transformed trajectory RMSE was 3.42e-6 (old 0.389). The old
pipeline was 3/4 at 270°, while canonical remained 4/4. No model was adapted
after this confirmation. See `phase_b/rotation_confirmation/results.json`.

## Cache and numerical audit

Development requested 10,368 seed slots: 576 exact cached B0 slots and 9,792
new slots, with 9,804 physical attempts including canonical retries. Four
slots remained numerical-uncertified. Rotation confirmation requested 3,712
logical method slots, corresponding to 3,584 unique physical seed tuples;
deduplication prevented repeated execution of selected/oracle overlap.
There were 3,617 attempts and 11 numerical-uncertified slots. These unknowns
are preserved and never treated as ordinary negatives. Valid rollout
collisions were zero in both batches. All evidence is in the shared DB.

## Accepted scope and next task

The canonical models and derived dataset are validated for the original
scenario chart and for Ring's tested rotations. They remain candidates with
an explicit Four active-rotation limitation. No label is transferred across
physical scenes on the basis of equal neural encodings.

The next authorized task is the closed-loop learning contract audit. It must
test whether the encoding retains the policy-frame and remaining-time
information required by the actual continuation target. The later unified
entity representation must inherit that conclusion rather than enforcing
unsupported invariance.
