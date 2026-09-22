import DROSafety.SparseBudget

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Sparse Sample-Complexity Optimization

This file formalizes optimization over both

* the set of omitted modes, and
* the number of scenarios assigned to each mode.

A sparse allocation is feasible if there exist sampled-risk and
omitted-mass budgets whose sum satisfies the global target.
-/

open scoped BigOperators

namespace DROSafety

variable {K : ℕ}

/--
A sparse allocation is feasible if there exist values
`εSampled` and `η` that certify it through
`FeasibleSparseAllocation`.
-/
def SparseAllocationFeasible
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ) :
    Prop :=
  ∃ εSampled η,
    FeasibleSparseAllocation
      U
      omitted
      S
      nBar
      β
      εSampled
      η
      εTarget

/--
Every feasible sparse allocation satisfies the complete robust
sparse risk constraint.
-/
theorem SparseAllocationFeasible.is_safe
    {U : AmbiguitySet K}
    {omitted : Finset (Fin K)}
    {S nBar : Fin K → ℕ}
    {β : ModeVec K}
    {εTarget : ℝ}
    (h :
      SparseAllocationFeasible
        U omitted S nBar β εTarget) :
    RobustSparseScenarioAllocation
      U omitted S nBar β εTarget := by
  rcases h with ⟨εSampled, η, hFeasible⟩

  exact
    feasibleSparseAllocation_is_safe
      U
      omitted
      S
      nBar
      β
      εSampled
      η
      εTarget
      hFeasible

/--
A feasible sparse allocation assigns zero scenarios to every
omitted mode.
-/
theorem SparseAllocationFeasible.zero_on_omitted
    {U : AmbiguitySet K}
    {omitted : Finset (Fin K)}
    {S nBar : Fin K → ℕ}
    {β : ModeVec K}
    {εTarget : ℝ}
    (h :
      SparseAllocationFeasible
        U omitted S nBar β εTarget) :
    ZeroSamplesOnOmitted omitted S := by
  rcases h with ⟨εSampled, η, hFeasible⟩
  exact hFeasible.1

/--
Every sampled mode in a feasible sparse allocation satisfies
the support-count requirement.
-/
theorem SparseAllocationFeasible.valid_sampled_counts
    {U : AmbiguitySet K}
    {omitted : Finset (Fin K)}
    {S nBar : Fin K → ℕ}
    {β : ModeVec K}
    {εTarget : ℝ}
    (h :
      SparseAllocationFeasible
        U omitted S nBar β εTarget) :
    ValidSampledScenarioCounts omitted S nBar := by
  rcases h with ⟨εSampled, η, hFeasible⟩
  exact hFeasible.2.1

/--
A sparse allocation `(omittedStar, Sstar)` is sample-minimal if

1. it is feasible, and
2. no other feasible choice of omitted modes and sample counts
   uses fewer total scenarios.
-/
def IsSparseSampleMinimal
    (U : AmbiguitySet K)
    (omittedStar : Finset (Fin K))
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ) :
    Prop :=
  SparseAllocationFeasible
      U
      omittedStar
      Sstar
      nBar
      β
      εTarget ∧
  ∀ omitted S,
    SparseAllocationFeasible
        U
        omitted
        S
        nBar
        β
        εTarget →
    totalSamples Sstar ≤ totalSamples S

/--
A sample-minimal sparse allocation is safe because feasibility is
part of the definition of minimality.
-/
theorem sparse_sample_minimal_is_safe
    (U : AmbiguitySet K)
    (omittedStar : Finset (Fin K))
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hMinimal :
      IsSparseSampleMinimal
        U
        omittedStar
        Sstar
        nBar
        β
        εTarget) :
    RobustSparseScenarioAllocation
      U
      omittedStar
      Sstar
      nBar
      β
      εTarget := by
  exact hMinimal.1.is_safe

/--
A sample-minimal sparse allocation assigns zero scenarios to
every mode that it chooses to omit.
-/
theorem sparse_sample_minimal_zero_on_omitted
    (U : AmbiguitySet K)
    (omittedStar : Finset (Fin K))
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hMinimal :
      IsSparseSampleMinimal
        U
        omittedStar
        Sstar
        nBar
        β
        εTarget) :
    ZeroSamplesOnOmitted omittedStar Sstar := by
  exact hMinimal.1.zero_on_omitted

end DROSafety
