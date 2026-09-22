import DROSafety.SingleMode.IID
import DROSafety.SingleMode.Definitions
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Fixed-Decision IID Bound

For a fixed decision `θ`, if its true violation probability is greater
than `ε`, then the probability that `k` independent samples all happen
to satisfy `θ` is at most

    (1 - ε)^k.

This is the basic probability lemma used later in the fixed-support
scenario theorem.
-/

namespace DROSafety.SingleMode

/--
The uncertainty realizations that do not violate `θ`.
-/
def safeSet
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (θ : Θ) :
    Set Ξ :=
  (violationSet violates θ)ᶜ

/--
The event that all `k` sampled uncertainty realizations are safe for
the fixed decision `θ`.
-/
def AllSafeEvent
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (θ : Θ)
    (k : ℕ) :
    Set (Sample k Ξ) :=
  Set.univ.pi
    (fun _ : Fin k => safeSet violates θ)

/--
Under the IID product measure, the probability that all `k` samples
are safe is the `k`th power of the one-sample safe probability.
-/
theorem allSafeEvent_measure
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (θ : Θ)
    (k : ℕ) :
    iidSampleMeasure P k
        (AllSafeEvent violates θ k) =
      (P (safeSet violates θ)) ^ k := by
  unfold iidSampleMeasure AllSafeEvent
  rw [MeasureTheory.Measure.pi_pi]
  simp

/--
If the true violation probability of `θ` exceeds `ε`, then its safe
probability is at most `1 - ε`.
-/
theorem safeProbability_le_one_sub
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (θ : Θ)
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (hMeas :
      MeasurableSet (violationSet violates θ))
    (hBad :
      ENNReal.ofReal ε <
        violationProbability P violates θ) :
    P (safeSet violates θ) ≤
      ENNReal.ofReal (1 - ε) := by

  have hBad' :
      ENNReal.ofReal ε <
        P (violationSet violates θ) := by
    simpa [violationProbability] using hBad

  have hSafe :
      P (violationSet violates θ)ᶜ <
        1 - ENNReal.ofReal ε :=
    MeasureTheory.prob_compl_lt_one_sub_of_lt_prob
      hBad'
      hMeas

  have hSub :
      (1 : ENNReal) - ENNReal.ofReal ε =
        ENNReal.ofReal (1 - ε) := by
    symm
    simpa using
      (ENNReal.ofReal_sub
        (p := (1 : ℝ))
        hε)

  rw [safeSet]
  rw [hSub] at hSafe

  exact le_of_lt hSafe

/--
Fixed-decision IID probability bound.

If `V(θ) > ε`, then the probability that all `k` independent samples
are nonviolating is at most `(1 - ε)^k`.
-/
theorem fixedDecision_allSafe_le
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (θ : Θ)
    (k : ℕ)
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (hMeas :
      MeasurableSet (violationSet violates θ))
    (hBad :
      ENNReal.ofReal ε <
        violationProbability P violates θ) :
    iidSampleMeasure P k
        (AllSafeEvent violates θ k) ≤
      (ENNReal.ofReal (1 - ε)) ^ k := by

  rw [allSafeEvent_measure]

  exact
    pow_le_pow_left'
      (safeProbability_le_one_sub
        P
        violates
        θ
        ε
        hε
        hMeas
        hBad)
      k

end DROSafety.SingleMode
