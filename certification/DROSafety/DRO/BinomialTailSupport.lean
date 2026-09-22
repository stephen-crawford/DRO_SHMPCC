import DROSafety.DRO.BinomialFiniteIntervalMass
import Mathlib.Probability.Distributions.Binomial
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Support reduction for binomial tails

A binomial random variable with `m` trials is almost surely bounded by `m`.

Therefore

    Bin(m,p) [k,∞)
      =
    Bin(m,p) [k,m].

For natural-valued counts, the lower-tail set

    (-∞,k]

is already finite and agrees exactly with `Finset.Iic k`.

Combining these support facts with the finite-set PMF formulas proved in
`BinomialFiniteIntervalMass.lean` yields explicit finite polynomial
expressions for both binomial tails.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Under the binomial measure itself, the identity random variable is almost
surely bounded by the number of trials.
-/
theorem binomial_ae_le_trials
    (m : ℕ)
    (p : unitInterval) :
    ∀ᵐ x : ℕ ∂(ProbabilityTheory.binomial m p),
      x ≤ m := by

  have hLaw :
      HasLaw
        id
        (ProbabilityTheory.binomial m p)
        (ProbabilityTheory.binomial m p) :=
    ProbabilityTheory.HasLaw.id

  simpa using
    (ProbabilityTheory.ae_le_of_hasLaw_binomial
      (n := m)
      (p := p)
      hLaw)

/--
Membership in the unbounded upper-tail event agrees almost everywhere with
membership in the bounded interval `[k,m]`.
-/
theorem binomial_Ici_ae_iff_Icc
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    ∀ᵐ x : ℕ ∂(ProbabilityTheory.binomial m p),
      x ∈ Set.Ici k
        ↔
      x ∈ (((Finset.Icc k m : Finset ℕ) : Set ℕ)) := by

  filter_upwards [
    binomial_ae_le_trials m p
  ] with x hx

  simp only [
    Set.mem_Ici,
    Finset.mem_coe,
    Finset.mem_Icc
  ]

  constructor

  · intro hk

    exact
      ⟨hk, hx⟩

  · intro hk

    exact hk.1

/--
The binomial mass of the unbounded upper tail equals the mass of the
bounded interval `[k,m]`.
-/
theorem binomial_Ici_eq_bounded_Icc
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    ProbabilityTheory.binomial m p
        (Set.Ici k)
      =
    ProbabilityTheory.binomial m p
        (((Finset.Icc k m : Finset ℕ) : Set ℕ)) := by

  apply MeasureTheory.measure_congr

  filter_upwards [
    binomial_Ici_ae_iff_Icc m p k
  ] with x hx

  exact propext hx

/--
Real-valued version of the upper-tail support reduction.
-/
theorem binomial_real_Ici_eq_bounded_Icc
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        (Set.Ici k)
      =
    (ProbabilityTheory.binomial m p).real
        (((Finset.Icc k m : Finset ℕ) : Set ℕ)) := by

  unfold Measure.real

  rw [
    binomial_Ici_eq_bounded_Icc
      m
      p
      k
  ]

/--
For natural numbers, the lower order interval `Set.Iic k` is exactly the
finite set `Finset.Iic k`.
-/
theorem nat_Iic_eq_finset_Iic
    (k : ℕ) :
    Set.Iic k
      =
    ((Finset.Iic k : Finset ℕ) : Set ℕ) := by

  ext x

  simp

/--
The binomial lower-tail mass can therefore be written using the finite
lower interval directly.
-/
theorem binomial_Iic_eq_finite_Iic
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    ProbabilityTheory.binomial m p
        (Set.Iic k)
      =
    ProbabilityTheory.binomial m p
        (((Finset.Iic k : Finset ℕ) : Set ℕ)) := by

  rw [
    nat_Iic_eq_finset_Iic
      k
  ]

/--
Real-valued version of the lower-tail finite-set identity.
-/
theorem binomial_real_Iic_eq_finite_Iic
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        (Set.Iic k)
      =
    (ProbabilityTheory.binomial m p).real
        (((Finset.Iic k : Finset ℕ) : Set ℕ)) := by

  rw [
    nat_Iic_eq_finset_Iic
      k
  ]

/--
Explicit finite polynomial formula for the real-valued upper binomial tail.

No condition `k ≤ m` is needed: when `m < k`, `Finset.Icc k m` is empty.
-/
theorem binomial_real_Ici_eq_explicit_sum
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        (Set.Ici k)
      =
    ∑ j ∈ Finset.Icc k m,
      (m.choose j : ℝ)
        *
      (p : ℝ) ^ j
        *
      (1 - (p : ℝ)) ^ (m - j) := by

  rw [
    binomial_real_Ici_eq_bounded_Icc
      m
      p
      k
  ]

  exact
    binomial_real_Icc_eq_explicit_sum
      m
      p
      k

/--
Explicit finite polynomial formula for the real-valued lower binomial tail.
-/
theorem binomial_real_Iic_eq_explicit_sum
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        (Set.Iic k)
      =
    ∑ j ∈ Finset.Iic k,
      (m.choose j : ℝ)
        *
      (p : ℝ) ^ j
        *
      (1 - (p : ℝ)) ^ (m - j) := by

  rw [
    binomial_real_Iic_eq_finite_Iic
      m
      p
      k
  ]

  exact
    binomial_real_finset_Iic_eq_explicit_sum
      m
      p
      k

/--
Above the support, the upper binomial tail has zero mass.
-/
theorem binomial_Ici_eq_zero_of_trials_lt
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ)
    (hmk :
      m < k) :
    ProbabilityTheory.binomial m p
        (Set.Ici k)
      =
    0 := by

  rw [
    binomial_Ici_eq_bounded_Icc
      m
      p
      k
  ]

  have hEmpty :
      Finset.Icc k m = ∅ :=
    Finset.Icc_eq_empty_of_lt
      hmk

  rw [hEmpty]

  simp

end DROSafety.DRO
