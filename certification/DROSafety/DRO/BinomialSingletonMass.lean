import DROSafety.DRO.BetaMeasureTailIntegral
import Mathlib.Probability.Distributions.Binomial
import Mathlib.Probability.Distributions.SetBernoulli
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Exact singleton masses of the binomial measure

Mathlib defines

    binomial m p

as the pushforward of the Bernoulli random subset of `Set.Iio m` under
`Set.ncard`.

The SetBernoulli API already proves the exact mass of an `ncard`
singleton.  This file packages that result directly for the binomial
measure.

These singleton formulas are the discrete building blocks needed to turn
binomial tails into finite sums.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Exact mass of one count under the binomial distribution:

    P[X = k]
      =
    choose(m,k) p^k (1-p)^(m-k).

The result is expressed in `ENNReal`, matching the codomain of a measure.
-/
theorem binomial_singleton_mass
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    ProbabilityTheory.binomial m p
        ({k} : Set ℕ)
      =
    ENNReal.ofReal
      (
        (m.choose k : ℝ)
          *
        (p : ℝ) ^ k
          *
        (1 - (p : ℝ)) ^ (m - k)
      ) := by

  simpa [ProbabilityTheory.binomial] using
    (ProbabilityTheory.map_ncard_setBernoulli_singleton
      (u := Set.Iio m)
      (Set.finite_Iio m)
      p
      k)

/--
Counts strictly greater than the number of trials have zero binomial mass.
-/
theorem binomial_singleton_eq_zero_of_lt
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ)
    (hmk :
      m < k) :
    ProbabilityTheory.binomial m p
        ({k} : Set ℕ)
      =
    0 := by

  rw [
    binomial_singleton_mass
      m
      p
      k
  ]

  have hChoose :
      m.choose k = 0 :=
    Nat.choose_eq_zero_of_lt
      hmk

  rw [hChoose]

  simp

/--
The real-valued singleton probability has the usual binomial PMF form.
-/
theorem binomial_real_singleton_mass
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        ({k} : Set ℕ)
      =
    (m.choose k : ℝ)
      *
    (p : ℝ) ^ k
      *
    (1 - (p : ℝ)) ^ (m - k) := by

  rw [
    MeasureTheory.measureReal_def,
    binomial_singleton_mass
      m
      p
      k
  ]

  rw [ENNReal.toReal_ofReal]

  have hp0 :
      0 ≤ (p : ℝ) :=
    p.property.1

  have hp1 :
      (p : ℝ) ≤ 1 :=
    p.property.2

  positivity

/--
The real-valued singleton probability is zero above the support.
-/
theorem binomial_real_singleton_eq_zero_of_lt
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ)
    (hmk :
      m < k) :
    (ProbabilityTheory.binomial m p).real
        ({k} : Set ℕ)
      =
    0 := by

  rw [
    binomial_real_singleton_mass
      m
      p
      k
  ]

  have hChoose :
      m.choose k = 0 :=
    Nat.choose_eq_zero_of_lt
      hmk

  rw [hChoose]

  ring

end DROSafety.DRO
