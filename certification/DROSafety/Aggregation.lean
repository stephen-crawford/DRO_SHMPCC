import DROSafety.Basic
import DROSafety.Ambiguity

set_option linter.style.header false

namespace DROSafety

variable {K : ℕ}

/--
If the true mode distribution belongs to the ambiguity set,
each conditional violation is bounded by ε,
and every distribution in the ambiguity set satisfies the robust
weighted ε bound, then the true weighted violation is bounded.
-/
theorem dro_risk_bound
    (U : AmbiguitySet K)
    (p V ε : ModeVec K)
    (εTarget : ℝ)
    (hpProb : IsProbabilityVector p)
    (hpU : p ∈ U)
    (hConditional : ∀ i, V i ≤ ε i)
    (hRobust : ∀ q, q ∈ U → dot q ε ≤ εTarget) :
    dot p V ≤ εTarget := by
  have hModeBound : dot p V ≤ dot p ε :=
    dot_le_dot_of_nonneg_left p V ε hpProb.1 hConditional
  have hWorstCase : dot p ε ≤ εTarget :=
    hRobust p hpU
  exact le_trans hModeBound hWorstCase

end DROSafety
