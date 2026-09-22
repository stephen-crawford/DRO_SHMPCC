import DROSafety.DRO.BinomialUpperTailStrictMonotonicity
import DROSafety.DRO.BinomialTailSupport
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Monotonicity of upper binomial tails in the count index

For fixed `m` and `x ∈ [0,1]`, increasing the threshold decreases the
upper-tail probability:

    k₁ ≤ k₂
      =>
    T_{m,k₂}(x) ≤ T_{m,k₁}(x).

This follows directly from event inclusion

    {X ≥ k₂} ⊆ {X ≥ k₁}.

Combining this with strict monotonicity of each tail polynomial in `x`
gives monotonicity of the exact tail-level roots:

    k₁ ≤ k₂
      =>
    r_{m,k₁}(α) ≤ r_{m,k₂}(α).

This is exactly the count-index monotonicity needed for the
Clopper-Pearson lower endpoint.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
If `k₁ ≤ k₂`, then the upper-tail event at `k₂` is contained in the
upper-tail event at `k₁`.
-/
theorem Ici_subset_Ici_of_le
    {k₁ k₂ : ℕ}
    (hk :
      k₁ ≤ k₂) :
    Set.Ici k₂
      ⊆
    Set.Ici k₁ := by

  intro n hn

  exact
    le_trans
      hk
      hn

/--
For fixed `m` and `x ∈ [0,1]`, the upper binomial-tail polynomial is
antitone in the count threshold.
-/
theorem upperBinomialTailPolynomial_antitone_index
    (m k₁ k₂ : ℕ)
    (x : ℝ)
    (hk :
      k₁ ≤ k₂)
    (hx :
      x ∈ Set.Icc (0 : ℝ) 1) :
    upperBinomialTailPolynomial
        m
        k₂
        x
      ≤
    upperBinomialTailPolynomial
        m
        k₁
        x := by

  let p : unitInterval :=
    ⟨x, hx⟩

  have hSubset :
      Set.Ici k₂
        ⊆
      Set.Ici k₁ :=
    Ici_subset_Ici_of_le
      hk

  have hMeasure :
      (
        ProbabilityTheory.binomial
          m
          p
      ).real
          (Set.Ici k₂)
        ≤
      (
        ProbabilityTheory.binomial
          m
          p
      ).real
          (Set.Ici k₁) :=
    MeasureTheory.measureReal_mono
      hSubset

  rw [
    binomial_real_Ici_eq_upperTailPolynomial
      m
      p
      k₂,
    binomial_real_Ici_eq_upperTailPolynomial
      m
      p
      k₁
  ] at hMeasure

  simpa [p] using hMeasure

/--
Specialized form of index antitonicity at an exact upper-tail root.
-/
theorem upperBinomialTailPolynomial_at_root_index_le
    (m k₁ k₂ : ℕ)
    (α : ℝ)
    (hk :
      k₁ ≤ k₂)
    (hk₂Pos :
      0 < k₂)
    (hk₂m :
      k₂ ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailPolynomial
        m
        k₂
        (
          upperBinomialTailLevelRoot
            m
            k₂
            α
        )
      ≤
    upperBinomialTailPolynomial
        m
        k₁
        (
          upperBinomialTailLevelRoot
            m
            k₂
            α
        ) := by

  have hRootMem :
      upperBinomialTailLevelRoot
          m
          k₂
          α
        ∈
      Set.Icc (0 : ℝ) 1 :=
    upperBinomialTailLevelRoot_mem_Icc
      m
      k₂
      α
      hk₂Pos
      hk₂m
      hα0
      hα1

  exact
    upperBinomialTailPolynomial_antitone_index
      m
      k₁
      k₂
      (
        upperBinomialTailLevelRoot
          m
          k₂
          α
      )
      hk
      hRootMem

/--
For an admissible tail level, exact upper-tail roots are monotone in the
count index.

If

    0 < k₁ ≤ k₂ ≤ m,

then

    root(m,k₁,α) ≤ root(m,k₂,α).
-/
theorem upperBinomialTailLevelRoot_mono_index
    (m k₁ k₂ : ℕ)
    (α : ℝ)
    (hk₁Pos :
      0 < k₁)
    (hk₁₂ :
      k₁ ≤ k₂)
    (hk₂m :
      k₂ ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailLevelRoot
        m
        k₁
        α
      ≤
    upperBinomialTailLevelRoot
        m
        k₂
        α := by

  have hk₂Pos :
      0 < k₂ :=
    lt_of_lt_of_le
      hk₁Pos
      hk₁₂

  have hk₁m :
      k₁ ≤ m :=
    le_trans
      hk₁₂
      hk₂m

  have hRoot₁Mem :
      upperBinomialTailLevelRoot
          m
          k₁
          α
        ∈
      Set.Icc (0 : ℝ) 1 :=
    upperBinomialTailLevelRoot_mem_Icc
      m
      k₁
      α
      hk₁Pos
      hk₁m
      hα0
      hα1

  have hRoot₂Mem :
      upperBinomialTailLevelRoot
          m
          k₂
          α
        ∈
      Set.Icc (0 : ℝ) 1 :=
    upperBinomialTailLevelRoot_mem_Icc
      m
      k₂
      α
      hk₂Pos
      hk₂m
      hα0
      hα1

  have hRoot₁Level :
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₁
              α
          )
        =
      α :=
    upperBinomialTailLevelRoot_eq
      m
      k₁
      α
      hk₁Pos
      hk₁m
      hα0
      hα1

  have hRoot₂Level :
      upperBinomialTailPolynomial
          m
          k₂
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          )
        =
      α :=
    upperBinomialTailLevelRoot_eq
      m
      k₂
      α
      hk₂Pos
      hk₂m
      hα0
      hα1

  have hIndex :
      upperBinomialTailPolynomial
          m
          k₂
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          )
        ≤
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          ) :=
    upperBinomialTailPolynomial_antitone_index
      m
      k₁
      k₂
      (
        upperBinomialTailLevelRoot
          m
          k₂
          α
      )
      hk₁₂
      hRoot₂Mem

  have hAtRoot₂ :
      α
        ≤
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          ) := by

    calc
      α
          =
        upperBinomialTailPolynomial
          m
          k₂
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          ) :=
        hRoot₂Level.symm

      _ ≤
        upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          ) :=
        hIndex

  by_contra hNot

  have hRoot₂ltRoot₁ :
      upperBinomialTailLevelRoot
          m
          k₂
          α
        <
      upperBinomialTailLevelRoot
          m
          k₁
          α :=
    lt_of_not_ge
      hNot

  have hStrict :
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          )
        <
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₁
              α
          ) :=
    (
      strictMonoOn_upperBinomialTailPolynomial
        m
        k₁
        hk₁Pos
        hk₁m
    )
      hRoot₂Mem
      hRoot₁Mem
      hRoot₂ltRoot₁

  have hStrictAlpha :
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          )
        <
      α := by

    calc
      upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₂
              α
          )
          <
        upperBinomialTailPolynomial
          m
          k₁
          (
            upperBinomialTailLevelRoot
              m
              k₁
              α
          ) :=
        hStrict

      _ = α :=
        hRoot₁Level

  exact
    (not_lt_of_ge hAtRoot₂)
      hStrictAlpha

/--
For fixed `m` and admissible `α`, the upper-tail root function is
monotone on the valid count range `{1,...,m}`.
-/
theorem upperBinomialTailLevelRoot_monoOn_index
    (m : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    MonotoneOn
      (fun k : ℕ =>
        upperBinomialTailLevelRoot
          m
          k
          α)
      {
        k : ℕ |
          0 < k ∧ k ≤ m
      } := by

  intro k₁ hk₁ k₂ hk₂ hk₁₂

  exact
    upperBinomialTailLevelRoot_mono_index
      m
      k₁
      k₂
      α
      hk₁.1
      hk₁₂
      hk₂.2
      hα0
      hα1

end DROSafety.DRO
