# Part 3 reproducibility log — 2026-10-12

Environment: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`, `JAX_PLATFORMS=cpu`, one local CPU process at a time. No Slurm job, GPU shard, new rollout, model fit, or evaluator modification.

Historical evidence inspected before running: `docs/part1_implementation_theory_audit_20261012.md`, `diagnostics/part2_closed_loop_sensitivity_validation_20261012/`, `diagnostics/controller_q_transfer_round3_v1/{REPORT.md,P1_LABELS.npz,P1_PROTOCOL.json,p1_features/}`, the external frozen field source and hard-field confirmation state manifest. `shared_rollout_db/README.md` checked; no rollout request was planned, hence no cache preflight or journal was required.

Commands, in order (all exited 0 on the final run):

```bash
JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_theory_validation_20261012/reuse_analysis.py
JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_theory_validation_20261012/h_bias_probe.py
JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_theory_validation_20261012/closed_loop_bias_probe.py
JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m py_compile diagnostics/part3_theory_validation_20261012/*.py
```

The first H attempt stopped before writing results because `BasisFields` exposes `.values`, not `.matrix`; the script was corrected and rerun. The first closed-loop attempt stopped before writing results because the Toy config does not expose `num_agents`; the script now infers the state dimension from `env.positions`. JAX printed a CUDA plugin discovery warning on the CPU workstation even with `JAX_PLATFORMS=cpu`; all final processes completed on CPU.

Outputs: `results.json`, `pairs.json`, `h_bias_results.json`, `closed_loop_results.json`. The input P1 label SHA256 is recorded in `results.json`. The external archived source directories remain required for the two H scripts. Unknown P1 labels were retained as intervals. The 24 P1 states and all 91 pairs remain in the pair output; no pair was dropped after viewing outcomes.
