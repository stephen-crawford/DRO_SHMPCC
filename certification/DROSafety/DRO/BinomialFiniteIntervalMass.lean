import DROSafety.DRO.BinomialSingletonMass
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Finite interval masses for the binomial distribution

This file turns the exact singleton-mass formula into finite-sum formulas
over bounded sets of counts.

We deliberately stop at bounded finite intervals here. The next file will
prove that the unbounded tail events `Set.Ici k` and `Set.Iic k` reduce to
these bounded intervals because `Bin(m,p)` has support in `{0,...,m}`.

That separation keeps the support argument independent of the algebraic
binomial-PMF sum.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Real-valued binomial PMF.
-/
def binomialPMFReal
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    ℝ :=
  (m.choose k : ℝ)
    *
  (p : ℝ) ^ k
    *
  (1 - (p : ℝ)) ^ (m - k)

/--
The real measure of any finite set of counts is the sum of the real
singleton masses over that set.
-/
theorem binomial_real_finset_eq_sum_singletons
    (m : ℕ)
    (p : unitInterval)
    (s : Finset ℕ) :
    (ProbabilityTheory.binomial m p).real
        (s : Set ℕ)
      =
    ∑ k ∈ s,
      (ProbabilityTheory.binomial m p).real
        ({k} : Set ℕ) := by

  symm

  exact
    MeasureTheory.sum_measureReal_singleton
      (μ := ProbabilityTheory.binomial m p)
      s

/--
The real binomial mass of a finite set is the sum of the PMF over that set.
-/
theorem binomial_real_finset_eq_sum_pmf
    (m : ℕ)
    (p : unitInterval)
    (s : Finset ℕ) :
    (ProbabilityTheory.binomial m p).real
        (s : Set ℕ)
      =
    ∑ k ∈ s,
      binomialPMFReal
        m
        p
        k := by

  rw [
    binomial_real_finset_eq_sum_singletons
      m
      p
      s
  ]

  apply Finset.sum_congr rfl

  intro k hk

  rw [
    binomial_real_singleton_mass
      m
      p
      k
  ]

  rfl

/--
Mass of a bounded count interval `[a,b]`.
-/
theorem binomial_real_Icc_eq_sum_pmf
    (m : ℕ)
    (p : unitInterval)
    (a b : ℕ) :
    (ProbabilityTheory.binomial m p).real
        ((Finset.Icc a b : Finset ℕ) : Set ℕ)
      =
    ∑ k ∈ Finset.Icc a b,
      binomialPMFReal
        m
        p
        k := by

  exact
    binomial_real_finset_eq_sum_pmf
      m
      p
      (Finset.Icc a b)

/--
Mass of the finite lower interval `{0,...,k}`.
-/
theorem binomial_real_finset_Iic_eq_sum_pmf
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        ((Finset.Iic k : Finset ℕ) : Set ℕ)
      =
    ∑ j ∈ Finset.Iic k,
      binomialPMFReal
        m
        p
        j := by

  exact
    binomial_real_finset_eq_sum_pmf
      m
      p
      (Finset.Iic k)

/--
Expanded form of the bounded upper-tail sum.
-/
theorem binomial_real_Icc_eq_explicit_sum
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        ((Finset.Icc k m : Finset ℕ) : Set ℕ)
      =
    ∑ j ∈ Finset.Icc k m,
      (m.choose j : ℝ)
        *
      (p : ℝ) ^ j
        *
      (1 - (p : ℝ)) ^ (m - j) := by

  rw [
    binomial_real_Icc_eq_sum_pmf
      m
      p
      k
      m
  ]

  rfl

/--
Expanded form of the bounded lower-tail sum.
-/
theorem binomial_real_finset_Iic_eq_explicit_sum
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        ((Finset.Iic k : Finset ℕ) : Set ℕ)
      =
    ∑ j ∈ Finset.Iic k,
      (m.choose j : ℝ)
        *
      (p : ℝ) ^ j
        *
      (1 - (p : ℝ)) ^ (m - j) := by

  rw [
    binomial_real_finset_Iic_eq_sum_pmf
      m
      p
      k
  ]

  rfl

/--
The PMF vanishes above the number of trials.
-/
theorem binomialPMFReal_eq_zero_of_lt
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ)
    (hmk :
      m < k) :
    binomialPMFReal
        m
        p
        k
      =
    0 := by

  unfold binomialPMFReal

  have hChoose :
      m.choose k = 0 :=
    Nat.choose_eq_zero_of_lt
      hmk

  rw [hChoose]

  ring

end DROSafety.DRO
