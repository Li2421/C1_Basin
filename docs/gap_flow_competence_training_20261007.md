# Gap1 competence retraining and frozen safety audit

Date: 2026-10-07. This report separates **navigation competence** from the
original opposing-traffic experiment. No eta, critic, generator, MAPF,
path-coordination controller, yielding rule, safety-law change or
opposing-deadlock training example was used. The previous safety audit and
the pre-training root-cause record are in
`docs/gap_flow_safety_deadlock_audit_20261007.md` and
`docs/gap_flow_competence_rootcause_20261007.md`.

## Root cause and exact geometry

The old N=2/N=10 datasets contained successful *sequential* opposing-flow
expert episodes, almost no simultaneous two-agent actions, and no solo or
one-way training task. The old checkpoint imitated held-out sequential
states well, but the safe closed-loop controls were out of distribution.
Old N=2 training had **more** active right-to-left than left-to-right data,
so a crude directional oversample was not indicated. The old gate waypoint
also pointed back to the entrance at the exact expert state from which the
next action should cross the door. The new observer keeps the same eight
features and uses a 0.04 m look-ahead switch; its reflection symmetry and
entry-switch behavior are unit-tested. The old observer remains available
to reproduce old checkpoints.

The *reported* Gap1 N=2/N=10 data use a **0.62 m opening**, not the scalable
template default of 0.50 m. Initial 0.50 m pilot data in
`diagnostics/gap_flow_competence_v1/` and `v2/` are **not accepted** as
evidence for this benchmark. All reported training and evaluation below use
the original 0.62 m geometry and exact N-specific physical fingerprints:
N=2 `8d2b923b2b097d3f81296f997f682f34cab24d5d919fa3cdab0a726917d17565`;
N=10 `50e26f73969abbcaf8a0bfc2a31b4b2c983adf846d36e93a053b645daace39d1`.

The executed nominal law is unchanged except for the retrained checkpoint
and corrected eight-feature observation:

```
o_k = [position, last executed velocity, goal - position,
       corrected public gate waypoint - position] for each robot
u_Flow,k = bounded joint Stage-I MACFlow sample(o_k, xi_k)
u_safe,k = CertifiedHardSafetyFilter(HardProjectionConfig())(x_k, u_Flow,k)
p_(k+1) = p_k + 0.05 u_safe,k
```

MC1, one independent Flow sample at each physical step, is the primary
protocol used in the preceding accepted safety audit. MC16 is a separately
reported sensitivity check. The safety filter and exact-problem retry source
hashes remain `847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79`
and `94a478f7b1dc1897e1e3113a78fe4c4afa134cbb02ce924f4bc19befc6ce2f94`.
The original swept wall/agent collision evaluator, goal tolerance 0.08 m,
N=2 horizon 2000 and N=10 horizon 8000 were not changed.

## Training data and selection

The accepted N=2 data are in
`diagnostics/gap_flow_competence_v3_recovery/dataset/`. A public geometric
waypoint reference plus the **existing** safety projection generated
successful solo and same-direction N=2 rollouts with varied independent
starts/goals. Every base case was mirrored across the symmetric gap and the
two rows were swapped. Goal-local train/dev perturbations from valid
preterminal states teach recovery from overshoot and lateral error. No
opposing task is present. 240 proposed local perturbations were rejected as
physically invalid in mirrored pairs; **no accepted recovery rollout failed**.
The test split was not used for training or checkpoint selection.

| N=2 train source/task | L→R trajectories | R→L trajectories | L→R rollout transitions | R→L rollout transitions |
|---|---:|---:|---:|---:|
| Basic solo | 96 | 96 | 47,218 | 47,218 |
| Basic one-way | 96 | 96 | 51,148 | 51,148 |
| Goal-local solo recovery | 256 | 256 | 4,814 | 4,814 |
| Goal-local one-way recovery | 502 | 502 | 9,450 | 9,450 |
| **Total** | **950** | **950** | **112,630** | **112,630** |

The architecture remains the official joint three-layer 256-unit MACFlow,
10 Euler integration steps and Stage-I conditional flow-matching objective.
The training change is data coverage, equal source sampling, a 30% sample
fraction for states within 1 m of an unfinished goal, and closed-loop dev
checkpoint selection. The final run used 50,000 Adam updates, batch 256,
seed 23. Fixed CFM dev loss went from 1.911 initially to 0.0272 at step
50,000; the numerically lowest CFM dev loss occurred at step 45,000, but
**step 50,000 was selected on navigation dev outcomes**, not CFM or opposing
traffic. Its four N=2 basic dev categories were each 12/12; two dev temporal
release orders were 12/12 and 11/12. No original opposing rollout was run
before the N=2 freeze manifest was written.

For N=10, the same method generated solo, same-direction pair and
same-direction five-agent data, again reflected/permuted for exact
left/right balance. The N=10 accepted train split has **872 trajectories
and 83,110 rollout transitions in each direction**. A simple public
waypoint/safety teacher safely completed sparse controls but did not
complete any of four full ten-agent one-way pilots within 1400 steps.
The N=10 data therefore do not contain a successful dense ten-agent
one-way demonstration. This is a coverage limitation, not evidence that
opposing deadlock was taught. A 100,000-update N=10 recovery run was
performed as a separate attempt; its checkpoint was **not frozen** because
the non-conflicting competence gate failed.

## Pre-freeze N=2 acceptance

The selected N=2 checkpoint was tested once on the untouched competence
test split, with 12 independent cases in each basic category. The temporal
controls used **another 12 independent seeds**. They keep the same two
opposing-side starts and eventual goals, but the waiting agent receives a
nearby same-room staging goal and zero reference velocity until the first
agent reaches its goal; only then is the original goal released. The goal
schedule is an exogenous non-conflicting control, never part of the nominal
opposing controller or training data.

| Held-out category | Safe success | Collision | Median wall-active fraction | Median `‖u_safe-u_ref‖` |
|---|---:|---:|---:|---:|
| Solo L→R | 12/12 | 0 | 0.061 | 0.00089 m/s |
| Solo R→L | 12/12 | 0 | 0.039 | 0.00073 m/s |
| Two-agent one-way L→R | 12/12 | 0 | 0.127 | 0.00197 m/s |
| Two-agent one-way R→L | 12/12 | 0 | 0.107 | 0.00180 m/s |
| Temporal release, left side first | 11/12 | 0 | 0.053 | 0.00082 m/s |
| Temporal release, right side first | 11/12 | 0 | 0.071 | 0.00160 m/s |

The two temporal misses timed out after the gate had been crossed with
final mean-agent goal distances 0.088 m and 0.170 m; neither was a collision
or wall-entrance stall. Every category meets the requested 11/12 threshold,
with no left/right asymmetry in the basic controls. The new model therefore
provides a defensible **N=2 nominal navigation** baseline. Normal N=2
traversal no longer depends on persistent large wall-directed commands;
the old right-to-left safety audit reported median 0.305 m/s correction
and 0.861 wall-active fraction in a comparable single-active control.

The frozen N=2 checkpoint is
`diagnostics/gap_flow_competence_v3_recovery/frozen/gap1_n2_macflow_competence_v1.pkl`,
SHA256 `7ea73d275ba3b928922cd68da24bf8b0c316c098d4e455050828acf8c0ecd6e5`.
Its freeze manifest records the exact training-dataset manifest hash,
training/normalization config hash and unchanged safety-source hashes. It
was not altered after the opposing results.

## Post-freeze original N=2 Gap1

The frozen checkpoint was evaluated on the **original 12 opposing Gap1 test
initial states**, with same-state non-conflicting controls. A second,
independent 12-seed cohort was evaluated after writing the offline
low-progress adjudication code, without changing weights, map, seed policy,
sampling count, safety law or timeout.

| Cohort, MC1 | Solo L→R | Solo R→L remote | One-way L→R | Opposing success | Offline A: bounded safety gridlock | E: other timeout | Collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original test, 12 states | 12/12 | 12/12 | 11/12 | 0/12 | 11 | 1 | 0 |
| Independent replication, 12 states | 12/12 | 12/12 | 12/12 | 1/12 | 9 | 2 | 0 |

The offline A adjudication requires a measured opposing encounter at the
gap, both agents reaching it, matched single-active controls succeeding,
pre-encounter progress above 0.1 m/s, post-encounter progress below
0.01 m/s, less than 0.05 m net mean-goal progress and under 0.25 m
coordinate envelope in the final 20 s, with pair CBF active in over half
those steps. These are **offline evidence rules, not simulator terminal
events**. They were written after inspecting the original cohort and then
held fixed for the independent replication. The original timeout label and
the existing TT 2 s/5 s near-zero-speed deadlock rule were not changed.
The strict TT rule triggered in **0/24** opposing episodes: the agents
often make bounded, centimeter-scale stochastic back-and-forth movements.
The paper must not state that 20 episodes were certified by that old
strict rule. The E episodes are retained as unusual/mixed timeouts, not
silently counted as clean deadlock. One replication episode spontaneously
coordinated and succeeded.

For the 11 original-test A cases, medians were: pre/post opposing-encounter
goal progress **0.268/0.003 m/s**, pre/post mean speed **0.309/0.127 m/s**,
pre/post safety correction **0.0019/0.212 m/s**. In the final 20 s the
pair constraint was active in **97.8%** of steps; Flow requested **0.130
m/s** toward goals while executed goal-direction velocity was approximately
zero. Median absolute net mean-goal progress was **0.0017 m**, with a
maximum-coordinate movement envelope of **0.138 m**. Swept wall/agent
clearances stayed nonnegative. These traces support an interaction-linked
safety-induced progress collapse; they do not prove that any one constraint
row alone caused it. The 9 replication A cases show the same pre/post
progress contrast (**0.276/0.0028 m/s**) and final-20-s pair activation
(median **92.3%**).

The prespecified MC16 sensitivity run used the same frozen weights and
unchanged projector: opposing 0/12, single-active controls 12/12 each,
one-way 8/12. Its lower one-way competence means it is **not** substituted
for the MC1 primary result. Neither protocol had a swept collision.

Complete simulator-state videos and a SHA256 provenance manifest are in
`diagnostics/gap_flow_competence_v3_recovery/videos/`. They include successful
single-agent L→R and R→L, successful one-way traversal, canonical safe
gridlock, an atypical delayed-progress timeout and the spontaneous
opposing success, plus N=10 single-active and one-way failures. Each MP4
contains every stored frame from initial state
to success or timeout at native 20 Hz. There is no eta intervention video
because eta was explicitly prohibited in this task.

## N=10 competence gate and scientific decision

The N=10 recovery checkpoint is only a **candidate**, not frozen. Full
non-conflicting N=10 results are recorded in
`diagnostics/gap_flow_competence_n10_v2_recovery/`; see its complete
summary files for every rollout and continuous safety evidence. The 1500-step
dev pilots at steps 40k/80k/100k had **1/18, 1/18 and 2/18** successes
respectively across solo/pair/group-five categories. A full-horizon held-out
check was then run before any N=10 opposing test:

| N=10 non-conflicting task, MC1 | Runs | Safe success | Timeout | Collision | Median wall-active fraction | Median `‖u_safe-u_ref‖` |
|---|---:|---:|---:|---:|---:|---:|
| Single active L→R, 100k candidate | 12 | 4 | 8 | 0 | 0.079 | 0.00252 m/s |
| Single active R→L, 100k candidate | 12 | 8 | 4 | 0 | 0.138 | 0.00435 m/s |
| Ten agents one-way L→R, 100k candidate | 12 | 0 | 12 | 0 | 0.647 | 0.01638 m/s |

All **24 N=10 failed controls** are non-conflicting navigation/throughput
failures, not opposing-agent deadlocks: 8 single-active L→R, 4 single-active
R→L, and 12 dense one-way. None is a collision. The N=10 opposing-failure
count is intentionally **not measured**, because this candidate failed the
pre-freeze competence gate.

The median final mean-agent goal distance in the ten-agent one-way control
was 2.225 m. Some single-active failures crossed the gate but stopped near
the goal; others did not enter it. The public waypoint plus unchanged safety
filter also failed 4/4 dense ten-agent one-way data-generation pilots at a
1400-step diagnostic horizon, while it usually completed sparse one- and
two-agent N=10 tasks. These observations distinguish the N=10 limitation
from a geometrically closed passage, but do not isolate a single fix.
Larger, more diverse *non-opposing* data and possibly a model that handles
agent permutations and dense queue interactions need to be validated before
N=10 can be a paper baseline. The 100k candidate's held-out CFM loss was
0.135; that scalar alone did not predict closed-loop competence.

**Verdict for the requested N=2 plus N=10 benchmark package: C — Flow
competence remains insufficient at N=10.** The N=2 controller is a strong
and reproducible safety-gridlock baseline under a progress-based offline
classification, but the old strict TT rest detector does not certify those
bounded-chattering episodes. N=10 must not be reported as an external
safety-deadlock result or tested with eta until its single-agent and dense
one-way controls become reliable. The frozen N=2 controller does **not**
need retraining for ordinary navigation. N=10 requires further improvement
in non-conflicting navigation/data coverage; no N=10 opposing test was run
or used for checkpoint selection.
