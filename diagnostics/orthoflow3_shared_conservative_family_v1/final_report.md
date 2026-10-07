# Shared conservative success-set family audit

**Classification:** `NO_SHARED_CONSERVATIVE_FAMILY_EVIDENCED`  
**READY_FOR_MARGIN_LOSS_TRAINING:** `NO`

Two compatible scenarios and 12 adequately sampled states were analyzed from 2,374 reused exact-Q64 tuples. The only new acquisition was the frozen shared-core test: 24 exact-Q64 state–eta evaluations, 1,536 continuations, 590,011 physical steps, and 398 s critical rollout time on two GPU shards. No controller was trained.

## Family results

|ID|Family|Toy usable /8|DB usable /4|Total /12|Median independent B63 recall|Median transfer coverage|Median retained diameter|Gate|
|---|---|---:|---:|---:|---:|---:|---:|---|
|A|affine superbody|5|1|6|0.199|0.071|1.001|FAIL|
|B|superbody + ≤2 cuts|3|1|4|0.293|0.238|0.941|FAIL|
|C|native conditional band|6|4|10|0.144|0.000|0.919|FAIL|
|D|rotated asymmetric slab|7|3|10|0.141|0.000|0.937|FAIL|
|E|affine capsule + ≤1 cut|7|2|9|0.134|0.000|0.869|FAIL|
|F|two-superbody union|3|0|3|0.233|0.143|1.521|FAIL|
|G|domain minus boundary caps|6|4|10|0.141|0.000|0.943|FAIL|

A-D failed the unchanged gate. E uses the connected analytic capsule

`(|max(|u1|-l,0)/r1|^p + |u2/r2|^p + |u3/r3|^p <= 1)`

with at most one cut; F is a union of two affine p-superbodies; G is `E_bridge` minus at most four boundary-open spherical caps. E/F and G formed two synthesis refinement rounds; neither improved the joint reliability/coverage result by five percentage points. All tested forms also failed at 48 labels/state, so the result is not explained solely by the ≤32-label target.

The candidate equations, batched membership, homothetic erosion, and violation code passed 84 model-instance tests (`retained ⊆ full`, `V=0 iff retained membership`, retained anchor, and finite batched evaluation). Their fitted parameters were finite and numerically conditioned, but that does not rescue the failed geometric gates or establish parameter learnability.

## Limited-oracle fitting

The table below reports descriptive performance of each predeclared family at its frozen final global hyperparameters as `usable states / median independent-B63 recall`. The unbiased family gate above uses outer-fold results.

|Family|16 labels|24 labels|32 labels|
|---|---:|---:|---:|
|A|6 / 0.192|6 / 0.160|8 / 0.147|
|B|7 / 0.233|3 / 0.276|4 / 0.333|
|C|8 / 0.114|8 / 0.132|11 / 0.144|
|D|6 / 0.081|10 / 0.132|10 / 0.141|

Increasing the oracle budget did not monotonically solve the precision/coverage tradeoff. At 48 labels/state, none of A-D or G passed.

## Common-core result

None of the three eta that were B63 on all4 Double-Bottleneck states reached the cross-scenario threshold after exact evaluation on all8 Toy states. Existing Toy-common coordinates were already0/4 B63 on Double-Bottleneck. Scenario-specific common cores exist on the sampled panels, but no cross-scenario common eta is supported.

Each of the three DB-common eta was B63 on only 3/8 Toy states, hence 7/12 overall—well below the preregistered 11/12 overall and 7/8 Toy requirement. This rules out the tested shared-core special case; it does not prove that no other common eta exists.

## Fresh retained-set validation

**NOT RUN.** No family passed the cached state-level gate, so sampling fresh interior points would not have been a valid route to rescue it. This is not a claim of zero fresh false inclusions.

## Answers

- **Does one shared analytic conservative success-set family work across compatible scenarios?** No family passed the reliability, state-coverage, moderate-recall, multimodal-support, and size gates together.
- **Is it large enough for margin-loss supervision?** No.
- **Can its parameters be obtained without remapping the full Basin?** Not under the tested 16/24/32-label protocol; 48 labels also did not produce a passing family.
- **Would a new scenario need only new parameters?** Current evidence is insufficient; assuming only theta changes is not justified.

The single next step is to collect a small cross-scenario panel of verified local inner sets around each scenario-specific robust core and compare their normalized support functions before any margin-loss training.
