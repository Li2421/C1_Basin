# New deadlock benchmarks provenance

## Pre-work snapshot

- Repository HEAD: `aba7d6d284e479cb0678b6270cc79f9b617dc654`
- Canonical Toy Give-Way snapshot: `6c88e06bd4f167ad5ce6b75ff670247cb916310913c7e37d1459e93edb0f9bf9`
- Canonical Double-Bottleneck snapshot: `1e0e98a3ddb91dc123ba2bdeb75476fa41583382e34da1febd159fc307de39ab`
- Canonical single-integrator snapshot: `f26814a9616d9bb7d17a731656e93721031570fb5e271315d3607ea62f728437`
- Canonical shared-control snapshot: `3accf5da696f083a684694beb668797f28f660f3d2d942790c41cacd3cd983c9`

The worktree was already dirty before this task. Existing unrelated changes
were preserved. New scenario implementation remained under
`four_way_intersection/`, `ring_exchange/`, `new_benchmark_common/`, and their
diagnostic roots.

## Post-work isolation check

The four canonical snapshot hashes above were rechecked during execution and
remained identical. No canonical Toy Give-Way, Double-Bottleneck,
single-integrator, or shared-control source was edited by this task. Final HEAD
remains `aba7d6d284e479cb0678b6270cc79f9b617dc654`.

Maintained regressions passed as follows:

- shared/single-integrator suite: 58/58;
- Toy Give-Way regression: 2/2;
- Double-Bottleneck integration: 19/19;
- new shared/Four-Way/Ring/eta3 suite: 49/49 plus 3 subtests.

The only runtime warning was the known unavailable-CUDA-plugin message; JAX
fell back to CPU.

## Formal safety and eta3 phase

The later frozen evaluation performed the requested formal hard-safety and
OrthoFlow3 eta3 studies for Four-Way and Ring. It did not retrain MACFlow,
train `G_phi`, expand eta, redesign OrthoFlow3, tune the eta domain, or modify
hard safety. The four canonical source snapshots above were rechecked after
the eta3 work and remained byte-identical.
