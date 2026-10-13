# Cache-first physical-only critic pilot: reuse audit

The large formal run was paused without deleting evidence. Completed worker
journals were merged, then all 24 frozen matched/LOSO request panels were
joined by exact rollout UID against the entire shared database (2,069,300
seed rollouts at the time of the scan). The machine-readable counts are in
`minimal_reuse_db_audit.json`. This is an exact state–eta–controller–seed
audit, including numerical failures as observed but unresolved.

The critic input changed: the older cross-scene entity encoder included Flow
and policy/controller channels, while the approved new encoder zeroes those
channels and keeps physical positions, velocities, goals, pair geometry,
obstacles, robot dimensions and masks. That change does **not** invalidate a
rollout label. Historical entity tensors can be masked and their labels used
for training. The raw initial physical states in the new 672-state-per-scene
LOSO files have zero exact position/velocity/goal triples in the historical
`orthoflow3_loso_root_cause_v1/all_source_states.json`; hence historical
rollouts cannot certify these *particular* new requests.
The old encoder's local axes use physical goal direction except at a zero
goal vector, where Flow could be a fallback. The retained source and target
entity tensors have strictly positive goal distance for every active robot
(source minimum 0.0061 physical units; Ring target normalized minimum 1.155),
so zeroing Flow channels does not leave hidden Flow in those local axes.

Exact current-request coverage after the pause:

| Panel | Complete K16 TRAIN states | Complete K16 VAL states | Complete K16+Safety TEST states |
|---|---:|---:|---:|
| Matched Toy TT | 316 | 8 | 0 |
| Matched Toy FF | 32 | 8 | 0 |
| Matched Ring TT | 32 | 8 | 0 |
| Matched Ring FF | 512 | 32 (Safety not cached) | 8 |
| LOSO Toy FF | 122 | 0 | 0 |
| LOSO DB FF | 0 | 0 | 0 |
| LOSO Four FF | 0 | 0 | 0 |
| LOSO Ring FF | 66 | 0 | 0 |

There is another fully cached, historically evaluated FF panel that is more
suitable for a zero-rollout pilot: the old four-scene LOSO target sets have
Toy 48, DB 24, Four 24 and Ring 60 states, each with 16 eta candidates and
up to 16 rollout seeds. The historical `joint_enriched` source table has
train/validation physical states per scene of Toy 48/12, DB 69/16, Four
96/24 and Ring 96/24. Its exact cached labels can be used to retrain both
physical-only and eta-only critics on **identical rows**. Four/Ring source
states include intermediate states; their remaining-time information is
excluded from a physical-only model, so this pilot tests that input limit as
well as transfer. The target labels are historically opened, and the target
state counts are small. Any result is an exploratory screening result, not
the requested fresh 128-state-per-scene formal confirmation.

For the old target states, DB/Four/Ring each have 16 same-seed hard-safety
rollouts per state in the database. Toy's other 16-seed controller is
`MAC_ONLY` with different success semantics, **not** the required hard-safety
baseline; it must not be labeled Safety. No old target outcome may enter
training or checkpoint selection. The pilot uses masked historical physical
entities, historical frozen K16 candidates and old FF execution outcomes;
it performs no new rollout and does not modify the preserved full-run files.
