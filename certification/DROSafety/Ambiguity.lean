import DROSafety.Basic

set_option linter.style.header false

namespace DROSafety

variable {K : ℕ}

/-- A set of possible mode distributions. -/
abbrev AmbiguitySet (K : ℕ) :=
  Set (ModeVec K)

end DROSafety
