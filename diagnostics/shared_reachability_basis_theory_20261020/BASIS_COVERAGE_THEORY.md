# Coverage of stage demands by the original parameter box

## Exact raw feasibility theorem

At fixed `(x,ξ,c)` let a finite, predeclared stage requirement be `Cδ≥b`, with `δ=B_cη` and original compact box `Θ=[ℓ,h]`. Put `H=CB_c`. Define `m*(H,b)=max_{η∈Θ} min_j (H_jη−b_j)`. Then

`∃η∈Θ: Hη≥b  ⇔  m*≥0`, and a common strict row margin exists iff `m*>0`.

This is a three-variable linear program, including all tradeoffs among rows and columns. A second exact form is

`m*=min_{λ≥0, 1ᵀλ=1} [σ_Θ(Hᵀλ)−λᵀb]`,

where `σ_Θ(v)=Σ_k (h_k v_k if v_k≥0 else ℓ_k v_k)`. Proof: `min_j r_j=min_{λ∈simplex}λᵀr`; interchange max over compact convex `Θ` and min over compact simplex for the bilinear expression, then maximize each box coordinate. For one stage row, feasibility is simply `b≤σ_Θ(H_j)`. The sign of each `C_jG,C_jL,C_jR` determines whether that row prefers the high or low bound. Different rows can prefer incompatible bounds; checking them separately is not sufficient. If a controller anchor is included in a **total preprojection input** condition, replace `b` by `d−Ca_s`; do not call that raw-correction feasibility.

For a specified stage subset `E_m(x)={u∈U(x):Cu≥d}`, this exact raw test says only what the three-column correction can encode. It does not imply `E_m` contains an action, that the projected action satisfies the rows, or that future states escape.

## Nondegenerate local coverage: strongest supported form

The archived Toy `K_pass` is a positive-volume four-dimensional set on which the legal `η*=(.5,0,.5)` has both raw yield and raw pass margins stated in `BASIS_DERIVATION.md`. Thus for the **specified two-row raw demand** with thresholds below those margins, `m*>0` throughout this domain, for **every** safe anchor/controller because `η_L=0`. This is a structural, state-volume-positive raw-expression result. It is not a claim that every such controller executes those actions.

For TT first-action coverage, the archived frozen Flow action has strict positive stage coordinates after projection. At this **fixed state**, keep `η=(.5,0,.5)` and vary the safe anchor. Goal and Relative do not change. Projection nonexpansiveness gives `||u_TT(a)−u_TT(a₀)||≤||a−a₀||`. The two coordinate rows have Euclidean norms `1` and `√2`; with archived output margins `.12013649` and `.08658147 m/s`, **every feasible safe anchor within `0.06 m/s` in joint Euclidean norm of the archived anchor** retains both strictly positive outputs. This is a quantitative, nonempty relative-open controller-behavior ball at one state/key, not a population claim. If `(x,ξ,c)↦a_s` is continuous near that witness, `U(x)` is locally regular with a continuous Euclidean projector (for example, locally feasible with a uniform Slater point and continuous CBF rows), and the stage rows vary continuously, those strict inequalities also persist on an open state/anchor/η neighborhood. This state-open extension is **conditional** on the stated regularity; its radius and probability have not been measured. A nonzero `η_L` sufficiently near zero remains admissible because zero is interior to its coefficient interval. This certifies local participation of the *full original family*, but does not prove that Perp was needed.

## Structural interpretation and failures

The image `BΘ` is a compact convex zonotope translated away from the origin in the Goal direction. It is **not** automatically a cone. For any finite stage rows its progress image `CBΘ` is likewise compact and convex, and the `m*` formula measures exact row feasibility. Direction-cone language applies only to a separately defined local tangent or normalized direction analysis; it must not replace the bounded-speed, bounded-parameter test. If `rank(CB)` is small, the progress image can miss required dimensions. Rank alone cannot guarantee coverage because offsets and box bounds matter.

Concrete failure mechanisms are: `G` opposes yielding while `η_G≥.5`; `R` has zero or adverse lateral component; pair contributions cancel for `N>2`; the safe anchor is zero so `L=0`; `L` aligns with a stage-conflicting direction; parameter rows demand incompatible coefficient ends; or the stage set lies beyond the correction's bounded image. These are **raw** limitations. Even raw feasibility may be lost after adding an adverse anchor or projecting onto a CBF face. The archived collinear pair-contact state proves the latter for positive longitudinal gap: every safe action has `u₁x−u₂x≤0`. A selected finite η menu failing is not proof that `m*<0` over continuous `Θ`.

## What “large fraction” would mean

State-volume coverage requires a declared measurable region `X₀` and a measure, e.g. normalized 4D Lebesgue measure on Toy `Ω_conf`: `vol({x∈X₀:∃η, Bη meets raw rows})/vol(X₀)`. Controller-action coverage instead requires a predeclared distribution `D` on `(x,c,ξ)` and a stage rule fixed independently of success labels:

`q_D=Pr_D[∃η∈Θ: Π_U(a_s+B_cη)∈E_m(x)]`.

Repeated physical steps within an episode and multiple η arms on one state are clustered observations, not independent draws. An estimate would need a sampling protocol, exact anchor/B reconstruction, continuous-box feasibility or a labeled lower-bound menu, and cluster-level confidence intervals. The 68 archived Toy actions include no rows in the certified `K_entry/K_pass`; one controller and duplicated selected states cannot estimate `q_D`. The old Ring rotation domain excludes all 73,358 archived episodes by fixed-goal geometry; the later broader candidate is exploratory and does not store precorrection anchor/B. Its high reference feasibility and sparse observed candidate-cone actions do not estimate original three-column TT capacity. **No defensible `q>0.5`, or other population lower bound, follows.**
