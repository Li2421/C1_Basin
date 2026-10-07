# Provisional executed-action deformation audit

## Audit outcome

This audit reused every cached SBMA attempt and launched **zero new rollouts**.  The
logger is an offline reader in this sibling diagnostic directory; no frozen
controller, physics, Flow, projection, event, success, or eta source was edited.

The smallest possible online insertion would be immediately after the second
projection in `success_basin_multimodality/run.py`: compute `delta = u - safe`
after line 52 and accumulate `dt * dot(delta.ravel(), delta.ravel())`.  The
offline implementation is exactly equivalent because lines 56--60 save `safe`
and `u` from that same loop iteration in the same trajectory record.

## Implemented quantity

For each completed cached rollout,

$$
J_{\rm def}(z_t,\eta;\xi)
= 0.05\sum_{k=t}^{T_\eta-1}
  \left\|u_{\mathrm{exec},k}^\eta-u_{\mathrm{safe},k}\right\|_2^2.
$$

The empirical cell value is the arithmetic mean over its stored Flow seeds.
There is no discount, normalization, completion penalty, progress term,
smoothness term, or contribution from raw pre-projection `g`.

## Empirical minima

`Phase-A CRN grid` compares the frozen 80-point lattice using the same 16 Flow
seeds for every eta in a state. `All tested` additionally includes validation,
interpolation, and axis-margin probes; those extra designs use different but
predeclared seed cohorts.

| state | Phase-A eta* | mean J | Phase-A location | all-tested eta* | mean J | paired 95% CI: runner minus Phase-A winner |
|---|---:|---:|---|---:|---:|---:|
| D1_pair231 | (0.75, 0, 0) | 0.28444484 | tested_grid_boundary | (0.75, 0, 0) | 0.28444484 | 0.00570278, 0.01815115 |
| D2_pair228 | (0.5, -0.5, 0) | 0.22102874 | tested_grid_boundary | (0.5, -0.5, 0) | 0.22102874 | 0.05277490, 0.07008052 |
| D4_pair227 | (0.5, -0.5, 0) | 0.21781885 | tested_grid_boundary | (0.5, -0.5, 0) | 0.21781885 | 0.01090487, 0.03047915 |

The best single common Phase-A eta is
`(0.75, 0, 0)`.  Its maximum relative gap from the three
state-specific minima is `29.795%`;
therefore the answer to “same eta within 5% of minimum for all three states” is
**FALSE**.
This 5% label is descriptive only and did not select or classify cells.

All three empirical minima are on the sampled Phase-A box boundary. D1 also
touches both a FAILURE and an UNKNOWN neighbor; D2 touches an UNKNOWN neighbor.
D4's available axis-neighbors are successful, but its minimum remains at the
lowest tested goal/safe/relative corner, so the success-constrained continuous
minimum is not bracketed.

## eta=(1,0,0.25), dedicated validation seeds

| state | mean J | std | min | max | successes |
|---|---:|---:|---:|---:|---:|
| D1_pair231 | 0.47794740 | 0.00879349 | 0.46175088 | 0.49769979 | 32/32 |
| D2_pair228 | 0.47950322 | 0.01056824 | 0.45624096 | 0.49879990 | 32/32 |
| D4_pair227 | 0.47542157 | 0.01028880 | 0.45123071 | 0.49332371 | 32/32 |

## Lowest-cost successful eta cells

The table below shows the ten lowest per state.  The complete table is
`successful_eta_table.csv`, sorted by state and mean J.  J means include every
terminal outcome in a high-confidence successful cell, not only successful
sample paths.

| state | rank | eta | success rate | mean J | std J | mean steps | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1_pair231 | 1 | (0.75, 0, 0) | 1.000 | 0.28444484 | 0.01039513 | 214.4 | 16 |
| D1_pair231 | 2 | (1, 0.25, 0) | 1.000 | 0.29637181 | 0.01063681 | 195.5 | 16 |
| D1_pair231 | 3 | (0.75, -0.25, 0) | 1.000 | 0.29978405 | 0.00801010 | 196.5 | 16 |
| D1_pair231 | 4 | (1.25, 0.5, 0) | 1.000 | 0.30324495 | 0.00859081 | 186.7 | 48 |
| D1_pair231 | 5 | (1.2, 0.4, 0.075) | 1.000 | 0.31029560 | 0.00830074 | 195.0 | 32 |
| D1_pair231 | 6 | (0.75, -0.5, 0) | 1.000 | 0.31127062 | 0.00445888 | 191.5 | 16 |
| D1_pair231 | 7 | (1, 0, 0) | 1.000 | 0.32017929 | 0.00623806 | 183.8 | 16 |
| D1_pair231 | 8 | (1, 0, -0.125) | 1.000 | 0.32278533 | 0.00563725 | 176.2 | 16 |
| D1_pair231 | 9 | (1.25, 0.25, 0) | 1.000 | 0.32655394 | 0.00835299 | 177.6 | 16 |
| D1_pair231 | 10 | (1.25, 0, 0) | 1.000 | 0.34175353 | 0.01085216 | 172.8 | 16 |
| D2_pair228 | 1 | (0.5, -0.5, 0) | 1.000 | 0.22102874 | 0.00958487 | 245.4 | 16 |
| D2_pair228 | 2 | (0.75, 0, 0) | 1.000 | 0.28245644 | 0.01190284 | 215.8 | 16 |
| D2_pair228 | 3 | (1, 0.25, 0) | 1.000 | 0.29688121 | 0.00909743 | 195.2 | 16 |
| D2_pair228 | 4 | (0.75, -0.25, 0) | 1.000 | 0.30081226 | 0.00976476 | 196.5 | 16 |
| D2_pair228 | 5 | (1.25, 0.5, 0) | 1.000 | 0.30172030 | 0.00890054 | 187.2 | 48 |
| D2_pair228 | 6 | (0.75, -0.5, 0) | 1.000 | 0.31128706 | 0.00898335 | 191.9 | 16 |
| D2_pair228 | 7 | (1.2, 0.4, 0.075) | 1.000 | 0.31188717 | 0.01051709 | 195.1 | 32 |
| D2_pair228 | 8 | (1, 0, 0) | 1.000 | 0.31338464 | 0.00796591 | 184.5 | 16 |
| D2_pair228 | 9 | (1, 0, -0.125) | 1.000 | 0.32209222 | 0.00743926 | 176.2 | 16 |
| D2_pair228 | 10 | (1.25, 0.25, 0) | 1.000 | 0.32693171 | 0.00609181 | 177.9 | 16 |
| D4_pair227 | 1 | (0.5, -0.5, 0) | 1.000 | 0.21781885 | 0.01348963 | 246.0 | 16 |
| D4_pair227 | 2 | (0.5, -0.25, 0) | 1.000 | 0.23851086 | 0.01139406 | 275.9 | 16 |
| D4_pair227 | 3 | (0.75, 0, 0) | 1.000 | 0.28271847 | 0.00983669 | 216.2 | 16 |
| D4_pair227 | 4 | (1, 0.25, 0) | 1.000 | 0.29207728 | 0.01058967 | 198.2 | 16 |
| D4_pair227 | 5 | (0.75, -0.25, 0) | 1.000 | 0.30064529 | 0.01035966 | 197.4 | 16 |
| D4_pair227 | 6 | (1.25, 0.5, 0) | 1.000 | 0.30312138 | 0.00833184 | 186.5 | 48 |
| D4_pair227 | 7 | (0.75, -0.5, 0) | 1.000 | 0.30983410 | 0.00921176 | 191.7 | 16 |
| D4_pair227 | 8 | (1.2, 0.4, 0.075) | 1.000 | 0.31131790 | 0.00832241 | 194.6 | 32 |
| D4_pair227 | 9 | (1, 0, 0) | 1.000 | 0.31659056 | 0.00850228 | 184.2 | 16 |
| D4_pair227 | 10 | (1.25, 0.25, 0) | 1.000 | 0.31816629 | 0.00685027 | 178.2 | 16 |

## Pathology audit

Without the success constraint, the smallest J in every state belongs to a
FAILURE_CELL. This is expected for an action-deformation-only objective:
policies that do almost nothing and terminate in deadlock can be extremely
cheap.

| state | lowest-J eta over all cells | class | success rate | mean J | Spearman(J, episode length) within SUCCESS_CELL |
|---|---:|---|---:|---:|---:|
| D1_pair231 | (0.5, 0.5, 0.75) | FAILURE_CELL | 0.000 | 0.00097660 | 0.905 |
| D2_pair228 | (0.375, 0, 0.25) | FAILURE_CELL | 0.000 | 0.00538827 | 0.872 |
| D4_pair227 | (0.5, 0.5, 0.75) | FAILURE_CELL | 0.000 | 0.00672299 | 0.824 |

The strong positive rank correlations show that even inside the success set,
the accumulated metric is materially coupled to episode duration. This is not
an added completion-time penalty--it follows from summing a nonnegative running
cost until policy-dependent termination.

## Sanity checks

- Raw attempts read: **6149**; completed effective rollouts:
  **6144**; unresolved after same-seed repair:
  **0**.
- All source NPZ SHA256 values matched their manifests, and no source file size
  or mtime changed during analysis.
- `J_def == 0` iff every stored `u_exec` equals `u_safe` bit-for-bit passed for
  every cached attempt. Actual all-step-zero rollouts: **0**;
  the explicit equal-array unit check also returned exactly `0.0`.
- Maximum `w - u_safe - g` reconstruction error:
  **2.220e-16**.
- Maximum dynamics reconstruction error:
  **1.110e-16**.
- Maximum vectorized-versus-step-loop J difference:
  **1.776e-15**.
- Existing SBMA independent replay checked 108 projections; its maximum
  recomputation difference was
  **9.700e-07**.

## Interpretation and recommendation

This is only a provisional empirical test of “minimum executed-action
deformation subject to success.”  It measures the action that survives the
hard projection, which is the intended object, and it preserves CRN pairing on
the primary grid.

Potential pathology: the undiscounted sum depends on termination time.  A
quick failure can have a small cost, and among successful policies a faster
completion accumulates fewer terms. Restricting eta selection to the frozen
high-confidence SUCCESS_CELL set prevents the first issue here, but does not
remove the duration coupling.  Also, large raw `g` that is removed by the
second projection is intentionally free under this definition.

**Recommendation: MODIFY before freezing.** Keep the executed-action,
same-state/same-Flow reference as the core deformation term, but do not yet
freeze it as the sole controller objective: the tested minimum must be treated
as finite-design empirical evidence, especially wherever it lies on a sampled
grid/probe boundary, and the termination-time coupling should be addressed or
explicitly accepted in the eventual specification.
