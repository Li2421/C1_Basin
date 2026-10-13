# Execution notes and deviations

- Initial historical audit discovered the independent state_transfer job19040
  still running. It completed and merged via19041. Its labels were reused;
  neither jobs nor historical source were changed. The independent report and
  §11 updated during this task; new training here has a different state/budget split.
- The first shell used `python`, absent on this host; all executions use the
  existing explicit .venv-c1 interpreter. No outcome was read before first freeze.
- Original native-start pilot failed. The preregistered deeper-bay pilot19071
  completed2048 outcomes and also failed. A separately time-stamped, bounded
  broad-state pilot19088 completed1024, also failed selection. Fixed horizon
  prefix analyses400/200 did not pass. Four native H400 probes19121 reproduced
  original success/first action/termination length semantics. Broad-main never ran.
- Legacy precision supplement19104 added1024missing seeds; fit19120 produced39
  checkpoints. Main metric job19123 completed transfer evaluation, then stopped
  at a historical-unknown assertion in calibration. The51 historical unknowns
  were handled explicitly (NUMERICAL_HANDLING.md); calibration19126 completed.
  Later metric-only jobs added paired Q-MSE and calibration-vs-zero CIs, with no
  refitting or rollout. Earlier output versions/logs are retained in snapshots.
- The EXTRAP diagnostic initially grouped targets under generic .62-only/seen
  widths. It was corrected to its preregistered .59/.62 target set using saved
  predictions; source .41/.47/.53 and all fitted weights unchanged. Original
  metric output is retained. This was a target-index correction, not test tuning.
- Factor protocol and features froze before outcomes. Probe19125 executed8
  preplanned state0 cases, merged them, re-preflight found8 exact plus3064missing.
  Full array19127 skipped cached8 and completed all3072attempts. Full trajectories
  retained; no videos required under pure-Q exemption.
- Factor fit19143 stopped on one TRAIN numerical failure before any fits.
  Its row/trace/error remain. Known-outcome likelihood excludes only that unknown;
  final test has no unknown. Retry19153 fits the critics, NOT another rollout.
  It produced90primary checkpoints;19145 evaluated and merged journals.
- GPU fit19153 waited behind another user's allocated shards. An attempted
  reduction4→2 was rejected because the job had just started as resources freed;
  actual primary allocation remained4. No other user's job was altered.
- Label-free source feature audit found source-constant target coordinates
 100/200/2250 standardized units away. A separately frozen exploratory physical-
 unit normalization diagnostic19155 added9fits, no rollout. It does not replace
 the original mechanism comparisons. Total new fits=138.
- CPU JAX initialization logs contain CUDA_ERROR_NO_DEVICE warnings while
 JAX_PLATFORMS=cpu is selected. They are not simulator unknown labels; actual
 exceptions are counted from per-rollout numerical_failure records.
- Slurm accounting storage is disabled. Costs are measured per-rollout and
 per-fit elapsed sums and live submission/resource records, not utilization
 integrals. No cleanup, deletion, Git push, external publication or historical
 evidence overwrite was performed.
