# Requirement-informed derivation of the frozen three columns

## Exact implementation

For `N≥2` planar robots and maximum speed `V`, write `clip_V(v)=v min(1,V/||v||)` (zero maps to zero). At fixed state `x`, Flow/noise key `ξ`, and safe anchor `a_s=Π_U π_c(x,ξ)`, the frozen implementation in `shared_control/basis_families.py` and `shared_control/diagnostic_corrector.py` is

`G_i=clip_V(g_i−p_i)`, `R_i=(N−1)⁻¹ Σ_{j≠i}clip_V(p_i−p_j)`,

`L_i=κ[a_{s,i}− (a_{s,i}·G_i)/(||G_i||²+ε²) G_i]`,

where `κ=3.303687238760696`, `ε=10⁻⁶V`. The original `B=[G,L,R]` and candidate box `Θ=[.5,1.25]×[−.5,.5]×[0,.75]` are retained. In particular `η_G>0`; deleting Goal is a **diagnostic ablation outside this box**, whereas setting `η_L=0` or `η_R=0` uses legal box faces. The zero-correction baseline `η=0` is outside `Θ`.

## What the reachability requirements motivate

**Goal.** `G_i` points along each robot's displacement to its own goal and is bounded in speed. It supplies a common task-progress direction after conflict relief. During yielding it may point against the needed motion; positive `η_G` can consume a clearance or passing budget. Goal is a design response to eventual task completion, not a theorem that goal motion is safe or locally useful at every stage.

**Relative.** `R_i` points away from other robots after each pair contribution is bounded. With two robots, `R_2=−R_1`, so its lateral component can enlarge `z` and its longitudinal component can either help or oppose a pass, depending on order. Pair-safety change uses `(p_i−p_j)·(u_i−u_j)`; `R` is aligned with that term for an isolated pair, giving a geometric rationale. For `N>2`, the average can cancel, may be dominated by uninvolved neighbors, and cannot command a selected pair independently. Positive `η_R` is also one-sided and cannot reverse the entire relative field.

**Perp.** Set `P_{G_i}=G_iG_iᵀ/||G_i||²` when `G_i≠0` and `Q_i=I−G_iG_iᵀ/(||G_i||²+ε²)`. The implementation has `L_i=κQ_i a_{s,i}`. For `G_i≠0`, `Q_i=(I−P_{G_i})+[ε²/(||G_i||²+ε²)]P_{G_i}`. It retains the anchor's component transverse to Goal and a tiny regularized parallel component; it is **not** exact orthogonal projection. At `G_i=0`, `Q_i=I` but the word perpendicular has no defined goal direction. If `a_{s,i}=0`, that robot's Perp row is zero. Because `η_L` has either sign, it can strengthen or weaken anchor-dependent transverse motion; its direction is not arbitrary lateral motion and it can be blocked by CBF projection. This is a controller-behavior interface rather than a uniquely implied reachability direction.

These three links are **construction motivations**. The mathematical content is the exact formulas plus the stage feasibility and TT criteria below. Reachability does not uniquely derive these columns, establish minimality, or imply global coverage.

## Analytic Toy subdomain and complementary roles

The archived full-constraint Toy proof defines a positive-volume `K_pass`. For the legal `η*=(.5,0,.5)`, it analytically bounds the **raw correction** on *every* point of this domain:

`δ₂y≥.023512274 m/s`, `δ₁x−δ₂x≥.043710055 m/s`.

This is a nontrivial Goal+Relative result independent of the safe anchor. It does not imply that `a_s+δ` or the projected TT action has either bound. The domain has width and clearance, so this is more than an isolated point. It also shows that Perp is **not indispensable** for these two raw inequalities. In contrast, wherever `G` and `R` jointly fail a lateral requirement but `C_m L` points into the missing margin with a legal `η_L`, Perp adds expressivity. This last statement is an algebraic conditional result; the archive does not certify a positive-measure real-controller domain requiring Perp.

The archived Toy action at `p₁=(−.30,0),p₂=(.02,.13)`, same frozen Flow checkpoint/key, and `η=(.5,0,.5)` has raw yield/pass coordinates `(.03591948,.17830289) m/s` and TT executed coordinates `(.12013649,.08658147) m/s`. It certifies a first-action witness with positive margins. The zero baseline already progresses, so this observation does not attribute full executed progress to correction. Under continuity of the frozen action map and local feasible-set regularity, the strict margins extend to a small open neighborhood of states and anchor behaviors; the historical finite probe itself does not measure that neighborhood's size or probability.

## Theory ablation protocol

At a predeclared state/behavior domain and fixed stage rows, solve the same feasibility criterion for `Θ_GR={η∈Θ:η_L=0}`, `Θ_GL={η∈Θ:η_R=0}`, and full `Θ`. Both faces are legal subfamilies, so their feasible stage sets are subsets of the full feasible set. A one-column Goal diagnostic uses `η_L=η_R=0`. A Relative-only or Perp-only diagnostic necessarily sets `η_G=0` and must be labeled **outside-box structural analysis**. Strict improvement by Perp or Relative requires a witnessed stage demand feasible in the full box and infeasible on the corresponding face; the present matched frozen data do not establish such a distribution-wide improvement. No column is proved globally necessary, unique, optimal, or minimal.
