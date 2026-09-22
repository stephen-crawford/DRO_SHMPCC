import DROSafety.DRO.FiniteEnumeratorExactClopperPearsonCoverage
import Mathlib.Probability.Distributions.SetBernoulli
import Mathlib.Probability.Distributions.Binomial
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Bernoulli random-set counts have binomial law

Mathlib defines the binomial distribution by taking a Bernoulli random
subset of the trial-index set `Iio m` and pushing that measure forward
through `Set.ncard`.

Consequently, if a random set `X : Ω → Set ℕ` has law

    setBernoulli (Set.Iio m) p,

then its cardinality has law

    binomial m p.

This is the bridge needed before deriving the Bernoulli random-set law
from i.i.d. categorical observations.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

noncomputable section

/--
The cardinality map itself has binomial law under the corresponding
Bernoulli random-set measure.
-/
theorem hasLaw_ncard_setBernoulli
    (m : ℕ)
    (p : unitInterval) :
    HasLaw
      Set.ncard
      (ProbabilityTheory.binomial m p)
      (ProbabilityTheory.setBernoulli (Set.Iio m) p) := by

  refine
    {
      aemeasurable := ?_
      map_eq := ?_
    }

  · measurability

  · rfl

/--
If `X` is a `p`-Bernoulli random subset of the first `m` natural
indices, then `X.ncard` has the binomial distribution `Binomial(m,p)`.
-/
theorem hasLaw_ncard_binomial_of_isSetBernoulli
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    (m : ℕ)
    (p : unitInterval)
    (X : Ω → Set ℕ)
    (hX :
      IsSetBernoulli
        X
        (Set.Iio m)
        p
        μ) :
    HasLaw
      (fun ω : Ω => (X ω).ncard)
      (ProbabilityTheory.binomial m p)
      μ := by

  have hNcard :
      HasLaw
        Set.ncard
        (ProbabilityTheory.binomial m p)
        (ProbabilityTheory.setBernoulli (Set.Iio m) p) :=
    hasLaw_ncard_setBernoulli
      m
      p

  have hComp :
      HasLaw
        (Set.ncard ∘ X)
        (ProbabilityTheory.binomial m p)
        μ :=
    hNcard.comp hX

  simpa [Function.comp_def] using hComp

/--
A version where the random count is supplied separately and proved
pointwise equal to the cardinality of the Bernoulli random set.
-/
theorem hasLaw_binomial_of_count_eq_ncard_isSetBernoulli
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    (m : ℕ)
    (p : unitInterval)
    (X : Ω → Set ℕ)
    (count : Ω → ℕ)
    (hX :
      IsSetBernoulli
        X
        (Set.Iio m)
        p
        μ)
    (hCount :
      ∀ ω : Ω,
        count ω = (X ω).ncard) :
    HasLaw
      count
      (ProbabilityTheory.binomial m p)
      μ := by

  have hCard :
      HasLaw
        (fun ω : Ω => (X ω).ncard)
        (ProbabilityTheory.binomial m p)
        μ :=
    hasLaw_ncard_binomial_of_isSetBernoulli
      μ
      m
      p
      X
      hX

  apply hCard.congr

  filter_upwards with ω

  exact hCount ω

/--
The set of observation indices among the first `m` trials at which
mode `i` occurs.
-/
def categoricalModeHitSet
    {Ω : Type*}
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (ω : Ω) :
    Set ℕ :=
  {
    t |
      t < m
        ∧
      observation ω t = i
  }

/--
The number of observations among the first `m` trials at which mode
`i` occurs.
-/
def categoricalModeCount
    {Ω : Type*}
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (ω : Ω) :
    ℕ :=
  (
    categoricalModeHitSet
      m
      observation
      i
      ω
  ).ncard

/--
The count definition is definitionally the cardinality of the
corresponding mode-hit set.
-/
theorem categoricalModeCount_eq_ncard
    {Ω : Type*}
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (ω : Ω) :
    categoricalModeCount
        m
        observation
        i
        ω
      =
    (
      categoricalModeHitSet
        m
        observation
        i
        ω
    ).ncard := by

  rfl

/--
If the trial-index hit set for mode `i` is a Bernoulli random subset
of `Iio m` with success probability `p`, then the observed number of
occurrences of mode `i` is binomial.
-/
theorem categoricalModeCount_hasLaw_binomial_of_isSetBernoulli
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (p : unitInterval)
    (hHit :
      IsSetBernoulli
        (
          categoricalModeHitSet
            m
            observation
            i
        )
        (Set.Iio m)
        p
        μ) :
    HasLaw
      (
        categoricalModeCount
          m
          observation
          i
      )
      (ProbabilityTheory.binomial m p)
      μ := by

  exact
    hasLaw_ncard_binomial_of_isSetBernoulli
      μ
      m
      p
      (
        categoricalModeHitSet
          m
          observation
          i
      )
      hHit

end

end DROSafety.DRO
