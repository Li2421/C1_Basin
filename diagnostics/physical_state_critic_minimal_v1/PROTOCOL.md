# Zero-rollout physical-only critic pilot

This pilot was defined after the shared-database exact-identity audit in
`../physical_state_critic_formal_v1/MINIMAL_REUSE_AUDIT.md` and before new
critic fits. It uses no new simulator rollout and leaves the paused formal
experiment's data/checkpoints intact.

Eight FF panels: within-scene Toy/DB/Four/Ring and four leave-one-scene-out
folds. Both `Q(s,eta)` and `Q(eta)` fit the same cached source rows. The first
uses the existing masked entity encoder with all Flow, policy frame and
controller/global channels zero; no scene ID is added. The other receives only
the identical eta vector. The K16 target candidates and true Q16 outcomes are
the archived frozen FF execution panels (Toy 48, DB 24, Four 24, Ring 60
physical target states). No controller, Flow, safety, candidate generator,
success evaluator, outcome or target state is modified.

Two source variants are separate experiments:

- `strict/`: archived source-only `joint` TRAIN/VAL. Excludes proposals with
  known all-scene-generator ancestry and held-out scene labels. This is the
  primary historical LOSO diagnostic, though target labels were opened in
  older studies and independent-state source coverage is small.
- root directory: cached `joint_enriched` TRAIN/VAL. More state/eta coverage,
  but `enrichment_audit.json` explicitly marks it ineligible for strict LOSO
  because some proposals may descend from an all-scene generator. Its LOSO
  rows are exploratory sensitivity analysis only.

Source TRAIN/VAL splits are the archived independent state/family splits.
Each model uses seed 17, 2400 AdamW steps, a batch of 128 source pairs with
equal scene contribution, and equal-row Bernoulli-count loss on observed
success/failure counts. Checkpoints are selected using source validation
selection regret among its observed candidates; decisive eta pairwise
accuracy and then NLL break near ties. Eta normalization is fitted on source
TRAIN only. All target predictions are saved before target truths are read by
the pilot evaluator; however the target datasets had already been opened in
prior studies, so the result is not a prospective fresh-state confirmation.

The evaluator reports B15, mean rollout success bounds, oracle regret,
collision counts and state-level paired 95% bootstrap intervals. Unknown
numerical outcomes remain unresolved. DB/Four/Ring hard-Safety uses exact
same-state, same-seed 16-rollout cache entries. Toy has no compatible
hard-Safety cache; its historical MAC-only controller is not relabeled Safety.
The 32-derangement diagnostic keeps each target eta pool fixed and changes
only the physical state supplied to the frozen critic. A performance drop
supports real state use; a gain over eta-only without a derangement drop can
still reflect a better global eta tendency.

## Prospective selected-only Toy confirmation

The first 48 physical states of the previously frozen global Toy TEST panel
were fixed before scoring. The strict-source within-Toy and LOSO-Toy state
critics, their wider eta-only controls, and a deterministic state-deranged
within-Toy critic all selected from the same frozen stochastic K16 proposals.
The hard-Safety controller was included. After deduplicating overlapping
choices, 234 state–eta/Safety pairs × 16 frozen seeds = 3,744 exact requests
were preflighted against the entire shared database; zero were previously
available. Only those requests are simulated. This is 48 new independent
states, and paired state-level bootstrap is the inference unit. Because the
other unselected eta candidates are not simulated, this confirmation reports
no oracle. It is an intentionally smaller prospective selection test, not the
paused 128-state exhaustive four-scene formal experiment.
