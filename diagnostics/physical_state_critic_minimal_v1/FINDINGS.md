# Physical-only critic: cache reuse and minimal evaluation

## Direct answer on state and old data

The critic input changed, but old rollout labels are reusable. The old
cross-scene entity input exposed nominal/bounded Flow, policy-frame and
controller/monitor channels. The new input masks them and retains physical
positions, velocities, goals, pair/obstacle geometry, robot dimensions and
padding masks. The old Ring-only physical23 input is different and cannot be
applied directly to Toy/DB/Four. Historical physical entities were masked,
then the **same cached outcome rows** trained Q(s,η) and Q(η).

The newly sampled physical initial states are also different: none of the
672 new Toy/DB/Four/Ring LOSO states per scene exactly matches the old LOSO
source positions/velocities/goals. Thus those old labels cannot stand in for
the new states' exact controller–eta–seed results. The complete shared DB
(2,069,300 rollouts at audit) and all 24 new formal request panels were
scanned by exact UID before any new rollout. The previous Ring FF Q4 512-state
training labels are already fully reusable. Detailed counts are in
`../physical_state_critic_formal_v1/minimal_reuse_db_audit.json`.

## Historical cached FF pilot, zero new rollouts

Strict source-only TRAIN/VAL labels, a frozen stochastic K16 proposal pool,
and the old FF Q16 target states were used. Training labels and checkpoint
selection excluded the held-out scene. Targets were previously opened in
historical studies, so these comparisons are exploratory, not prospective.
The wider eta-only control has eta encoder 32 and downstream widths 128/64,
matching the state critic's downstream head. The derangement diagnostic
retains each eta pool and frozen critic while assigning another state's
physical input.

| Panel | Target states | State B15 | Wider eta-only B15 | Mean B15 after 32 state derangements | Interpretation |
|---|---:|---:|---:|---:|---|
| Within Toy | 48 | 43 | 33 | 31.3 | State-dependent benefit on old target |
| Within DB | 24 | 24 | 24 | 24.0 | Saturated, no discriminating advantage |
| Within Four | 24 | 21 | 21 | 21.0 | No measured advantage; numerical uncertainty |
| Within Ring | 60 | 50 | 49 | 49.7 | No independent state-selection advantage |
| LOSO Toy | 48 | 33 | 30 | 33.0 | No supported adaptive advantage |
| LOSO DB | 24 | 23 | 15 | 23.1 | Gain survives state derangement; adaptation not shown |
| LOSO Four | 24 | 22 | 23 | 22.1 | No gain |
| LOSO Ring | 60 | 48 | 15 | 47.8 | Gain survives state derangement; adaptation not shown |

The larger `joint_enriched` cache was evaluated separately. Its archival
audit explicitly marks it ineligible for strict LOSO because some source
proposals may descend from an all-scene generator. It is a sensitivity
analysis, not cross-scene proof. In both source variants, only Toy within-scene
showed a large performance drop after state derangement. The apparent strict
Ring within-scene advantage over the original narrow eta-only critic (50/60
vs 12/60) nearly vanished with the wider eta-only control (49/60). This is
why the capacity and derangement controls matter.

## Prospective, selected-only Toy FF confirmation

The first 48 states of a previously frozen, independent global Toy TEST panel
were scored by the frozen strict-source critics before outcome generation.
Exact DB preflight found zero reusable slots. Only deduplicated selections and
Safety were evaluated, giving 234 state–eta/Safety pairs × 16 fixed seeds =
3,744 real FF rollouts. No unselected eta was simulated, so this experiment
has no oracle estimate. All six selectors used the same state-specific K16
proposals and seed identities; all observed selections had zero collisions.

| Method | B15 / 48 | Mean rollout success |
|---|---:|---:|
| Within-scene Q(s,η) | 39 | 84.4% |
| Within-scene wider Q(η) | 33 | 81.2% |
| Within-scene Q(s,η), state deranged before selection | 30 | 72.4% |
| LOSO Q(s,η) | 28 | 62.2% |
| LOSO wider Q(η) | 32 | 75.0% |
| Safety | 17 | 48.3% |

Paired physical-state bootstrap: within-scene state critic minus wider
eta-only **+12.5 percentage points B15**, 95% CI **[0.0,+25.0]**;
minus state-deranged version **+18.8 pp**, CI **[+6.2,+31.2]**;
minus Safety **+45.8 pp**, CI **[+29.2,+62.5]**.
The robust state-derangement gap is direct evidence that physical state
changes the Toy selector meaningfully. The state-vs-eta CI touches zero,
so this 48-state prospective sample does not establish a stable incremental
B15 benefit over a strong eta-only model at the conventional 95% level.
LOSO Toy does not beat its wider eta-only control in this fresh panel.

The result supports **within-scene Toy physical-state use**, not a blanket
within-scene four-scenario or cross-scene LOSO claim. DB is ceiling-limited
in the old cached test, Four has numerical uncertainty, and old Four/Ring
source states include intermediate states whose remaining-time information
is intentionally absent from physical-only input. The paused exhaustive
formal four-scene run remains incomplete and has no A–E verdict.

## Reproduction

- `PROTOCOL.md` specifies both cached source variants and the prospective
  selected-only protocol.
- `strict/REPORT.md` and `REPORT.md` contain all eight historical panels,
  capacity controls and derangement results.
- `fresh_toy_48/REPORT.md`, `metrics.json`, `choices.json`,
  `per_state_selected_outcomes.json`, `selected_cases.json` and
  `cache_preflight.json` contain the prospective selections and outcomes.
- Every new seed rollout has its raw JSON in the formal Toy test panel and an
  exact entry in `shared_rollout_db/rollout.sqlite`; no prior artifact was
  deleted. This pure-Q task is exempt from video delivery under `AGENTS.md`.
