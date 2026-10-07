# Mode-free generator K-sweep, coverage audit, and online latency

The frozen state-conditioned unimodal generator and three-seed continuous-Q ensemble were evaluated on the same 200 outcome-blind WIDE-IC Toy states, with the same 16 matched continuation seeds. K-prefixes use one frozen proposal stream; no model, critic, or threshold was retrained. B15 means at least 15/16 successes. The first four proposals and all baseline outcomes were reused from the prior experiment. The new samples 4–15 were evaluated with the frozen true-t0 controller.

## Success versus computation

| K | Oracle B15 | Critic B15 | Coverage | Oracle–critic gap | GPU batched p50 / p95 (ms) | Theoretical selection Hz, p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 131/200 | 131/200 | 65.5% | 0 | 0.084 / 0.128 | 11,875 |
| 2 | 166/200 | 161/200 | 83.0% | 5 | 0.125 / 0.171 | 7,995 |
| 4 | 173/200 | 167/200 | 86.5% | 6 | 0.124 / 0.171 | 8,084 |
| 8 | 184/200 | 177/200 | 92.0% | 7 | 0.126 / 0.177 | 7,911 |
| 16 | 194/200 | 181/200 | 97.0% | 13 | 0.127 / 0.182 | 7,848 |

`Coverage` is the fraction with at least one B15 proposal, so it equals oracle B15 here. One random first proposal succeeds on 131/200; mean success rate of individual draws over the 16-proposal panel is 67.1%. The frozen generator mean succeeds on 142/200, fixed common eta on 180/200, and old selector anchor on 199/200. K=16 critic rescues 41 generator-mean failures but breaks 2 mean successes; against fixed common eta, it rescues 15 but breaks 14. Thus K=16 improves over the mean, but does not materially outperform the fixed common eta and remains 18 B15 states below the old selector. The oracle–critic gap grows from 7 states at K=8 to 13 at K=16: proposal coverage improves faster than critic selection.

K=8 to K=16 gains 10 oracle-covered states and 4 critic-selected B15 states. GPU batched p95 rises only 0.005 ms; CPU p95 rises from 0.110 to 0.173 ms. K=16 is Pareto-efficient when robustness has priority and is the preferred GPU setting. K=8 remains a reasonable CPU option with a strict latency budget. K=32 was not evaluated: despite cheap inference, exact matched Q16 testing of 16 additional proposals per state would require 51,200 new continuations. The observed coverage curve does **not** saturate at K=4 or K=8.

## The 27 K=4 proposal-coverage failures

The prespecified audit rule classified 21/27 as `LOW_PROBABILITY_BASIN`: the generator mean was B15 in 2, and/or at least one of samples 4–15 reached B15. The remaining 6 failed to produce a B15 sample at K=16: 4 `MISSING_BASIN_COVERAGE` and 2 `UNDERRESOLVED`. Of the 27, fixed common eta is B15 on 22 and old selector anchor on 26. In the four missing-coverage cases, every compatible known B15 eta lies more than three predicted latent standard deviations from the generator center; this is evidence of mass placed away from known success regions, not proof of disconnected basin topology. In the two underresolved cases, available evidence does not identify the local basin width or a reliable alternative region. All per-state eta distances, generator sigma, known robust references, and classification rationale are in `k4_failure_state_audit.csv`.

Increasing K addresses most original misses, but the six K=16 failures merit distribution calibration/local basin study. The present data do not justify claiming that a multimodal generator is necessary: a shifted or better-calibrated low-entropy unimodal distribution could still help. Separately, the growing K=16 oracle–critic gap argues for improving finite-proposal ranking before adding still more samples.

## Synchronized latency benchmark

The online benchmark used 1,000 warmed, synchronized JAX calls per K and a vectorized K-proposal batch with a three-critic ensemble. GPU: NVIDIA RTX PRO 6000 Blackwell; CPU: Intel Core Ultra 7 270K Plus. Neural inference is float32; physical safety projection is float64. The GPU p50/p95 figures above are generator-plus-critic-plus-argmax **after** the state/history feature vector is available. Component timings (encoding, generator, critic, selection) are separately recorded in `latency_gpu.json` and `latency_cpu.json`; they should not be summed, because separate timed JIT calls have different launch and fusion overhead from the end-to-end compiled call. GPU state/history encoding p50 is 0.160 ms; base Flow plus safety projection p50 is 0.458 ms. CPU values are 0.137 and 0.676 ms.

At K=16, CPU batched proposal selection is 0.142/0.173/0.311 ms at p50/p95/p99, versus GPU 0.127/0.182/0.261 ms. A deliberately sequential K=16 diagnostic takes 2.466 ms p50 on GPU and 0.715 ms on CPU, confirming the importance of batching. On GPU, the hypothetical per-step sum of base/safety + feature encoding + K=16 proposal selection is about 0.745 ms p50, or 1,342 Hz; CPU is about 0.955 ms, or 1,047 Hz. These are compute-only approximations, not certified hard real-time rates. The current controller chooses eta **once at true t0** and latches it, so proposal selection adds one decision delay; it does **not** reduce the per-step low-level safety/control frequency. Only a future receding-eta architecture would pay this cost each physical timestep.

## Cache and numerical audit

The planned request contained 44,800 continuations: 6,400 exact reused, 38,400 initially missing. All 38,400 new attempts are journaled and merged into the global rollout database. Postflight finds 44,784 exact reusable seed outcomes, 15 partial seeds, and one genuinely missing scientifically valid seed. That one request, episode 187/sample 12/future seed 15, encountered a certified safety-projection numerical solver failure. The other 15 seeds for that pair succeeded; hence its B15 membership is **certain** under either possible value of the censored seed. Its Q16 is reported as [15/16, 1], never silently certified as exactly 1. The censored pair is neither the K=16 critic choice nor the oracle best on that state; all headline B15 counts and selected/oracle mean Q16 in the sweep are unaffected. The failure remains visible in `numerical_censoring.json` and database postflight, rather than being overwritten.

## Decision

`LARGER_K_WORTH_LATENCY`. K=16 substantially improves coverage over K=4 with little batched inference cost, but the critic leaves 13 oracle-covered states unselected and the method remains behind the old selector anchor. The next focused step is to diagnose/improve proposal ranking and the six K=16 coverage failures; do not train a new multimodal generator solely from this audit.
