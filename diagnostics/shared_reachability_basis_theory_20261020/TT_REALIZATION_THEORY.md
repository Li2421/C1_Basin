# TT realization after the actual safety projection

## Three distinct objects

Freeze physical state, controller and Flow key. The original TT chain is `a_s=Π_U π_c`, `δ=B_cη`, `z=a_s+δ`, `u_TT=Π_U z`. Raw correction, **total projection input**, and **executed safe action** are different. `U(x)` is the frozen full CBF affine-row intersection with per-robot speed balls; the Toy primary backend is the SLSQP QP with same-problem retries, while the general Ring backend is Clarabel. Mathematical statements below concern exact Euclidean projection onto the closed convex set; numerical use requires a solver-error allowance. FF/TF are not represented by this single TT projection.

Let `E_m(x)={u∈U(x): C_m(x)u≥d_m(x)}` be a predeclared stage action set. The statewise target is exactly `A_c(x)∩E_m(x)≠∅`, with `A_c={Π_U(a_s+B_cη):η∈Θ}`. This is a criterion, not a theorem that it holds. For any candidate `η,u∈U`, exact projection gives the alternative test

`u=Π_U(a_s+B_cη) ⇔ a_s+B_cη−u∈N_U(u)`,

where `N_U(u)={v:v·(w−u)≤0 for all w∈U}`. This exposes how a normal component can disappear. It applies with active speed balls and redundant CBF rows; no stable active-set assumption is needed. It may be hard to solve globally in `η` because the projector is nonlinear.

## Rechecked cancellation and established sufficient bound

For `η†=(.5,−1/κ,.55)∈Θ`, frozen Perp gives

`a_s+Bη†=Q a_s+.5G+.55R`, with `Q_i=G_iG_iᵀ/(||G_i||²+ε²)`.

When `G_i=0`, `Q_i=0`. For nonzero `G_i`, `Q_i=λ_iP_{G_i}` with `λ_i=||G_i||²/(||G_i||²+ε²)<1`; it is **not** the exact orthogonal projector. Put `v=.5G+.55R`, `u₀=Π_U v`, `d_c=||P_Ga_s||` with zero blocks at `G_i=0`. Euclidean projection nonexpansiveness gives `||Π_U(v+Qa_s)−u₀||≤||Qa_s||≤d_c`. If every stage row has `C_j u₀−d_j>||C_j||d_c`, then `η†` executes inside `E_m`. The earlier `diagnostics/shared_escape_cross_controller_audit_20261019/THEOREM_AUDIT.md` establishes this one-step result. It does not assert that the inequality holds on real controllers; archived `η†` executions are absent.

## A less restrictive adaptive reference-distance certificate

**Proposition (PROVEN).** At a fixed `(x,c,ξ)`, choose any legal `η∈Θ` and any safe reference action `v∈U(x)` with stage row margins `μ_j=C_jv−d_j>0`. If

`||a_s+B_cη−v|| < min_j μ_j/||C_j||`,

then `u_TT∈E_m(x)`. More sharply, it suffices to check each row with its own bound `μ_j>||C_j||·||a_s+B_cη−v||`. Proof: `Π_U v=v`; nonexpansiveness bounds `||u_TT−v||≤||a_s+B_cη−v||`, then Cauchy–Schwarz preserves all strict row margins. □

This permits **stage-specific η and reference actions**, and uses the actual distance of the total input to a feasible action. It does not require all walls to be inactive, a fixed active set, a universal controller bound, or the particular `η†`. It is a conservative sufficient certificate; failure gives no impossibility conclusion. The archived Toy yield/pass references supply feasible `v` on their domains, but no saved matched `(a_s,B)` panel there, so this inequality is **not yet numerically certified** on those domains. A useful optimization form is `min_{η∈Θ}||a_s+Bη−v||`, a three-variable convex box-constrained least-squares problem. For a finite set of references, taking the smallest bound remains only a sufficient search.

A second exact, potentially weaker *computational* test is simply to solve `max_{η∈Θ} min_j(C_jΠ_U(a_s+Bη)−d_j)` with certified global bounds; its nonnegative optimum is necessary and sufficient for stage intersection. The continuous box matters. No such global TT optimization is claimed in the current archive.

## Open-domain and temporal conditions

If at a state/key/controller the exact TT action has strict **stage row margins**, and the anchor, basis, CBF coefficients, and projector vary continuously locally, the same `η` works on a local open neighborhood. The TT action may be on an active CBF face. A sufficient regularity route is continuous constraints, compact speed balls, and a uniform local Slater feasible point, which gives continuous feasible sets/projectors near the witness. The archived Toy first-action witness has strict yield/pass output margins; its contact-free geometric neighborhood is archived. This yields a **conditional local open-domain** theorem, not a measured volume or a controller-population probability. The controller behavior class is an open neighborhood of the observed anchor at fixed key; arbitrary nominal policies need not fall in it.

Single-step membership in `E_m` does **not** imply complete escape. For a closed-loop stage theorem one must additionally certify: the resulting state stays in its stage tube, the next key/controller action satisfies the condition at each reached state, a progress coordinate grows by a uniform amount while the exit guard is false, the guard enters the next certified tube, and the remaining horizon and termination monitor permit the required steps. If `q⁺−q≥Δtν>0`, at most `ceil((q_exit−q_start)/(Δtν))` steps are needed, subject to boundary-step handling. The archived Toy reference verifies these properties for its **reference controls**, not for TT.

At the actual Toy collinear pair contact with `z=0`, pair CBF implies `u₁x−u₂x≤0` for every `u∈U`, despite positive raw pass correction and positive total-input pass component in a saved constructed probe. This is an exact failure of unconditional projection progress preservation. A separate archived state has positive raw yielding while the anchor makes the total TT input and output yielding negative. Safety, raw expression, total input, one-step TT progress, and multi-step escape must therefore be reported separately.
