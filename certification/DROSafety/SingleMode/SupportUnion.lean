import DROSafety.SingleMode.FixedSupport
import Mathlib.Data.Finset.Powerset
import DROSafety.SingleMode.SupportTransfer
import DROSafety.SingleMode.Regularity
import Mathlib.Data.Finset.Powerset
import DROSafety.ScenarioBound

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Union over Support Sets

For a fixed support size `n`, there are `S.choose n` possible
support subsets.

The bad event for support size `n` is the union of the bad events
corresponding to every such subset.
-/

namespace DROSafety.SingleMode

/--
All subsets of `{0, ..., S-1}` containing exactly `n` indices.
-/
def supportSets
    (S n : ℕ) :
    Finset (Finset (Fin S)) :=
  Finset.powersetCard n Finset.univ

/--
There are exactly `S.choose n` support subsets of cardinality `n`.
-/
theorem supportSets_card
    (S n : ℕ) :
    (supportSets S n).card =
      Nat.choose S n := by
  simp [supportSets]

/--
The event that some support set of size `n` produces a bad
fixed-support certificate.
-/
def SupportSizeBadEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (n : ℕ)
    (ε : ℝ) :
    Set (Sample S Ξ) :=
  {z |
    ∃ I,
      I ∈ supportSets S n ∧
      z ∈ FixedSupportBadEvent
        P violates reconstruct I ε}

/--
The probability bound we ultimately want after taking the finite union
over every support set of size `n`.

If every fixed support event is bounded by `(1-ε)^(S-n)`, then this
event should be bounded by

    choose(S,n) * (1-ε)^(S-n).
-/
def SupportSizeProbabilityBound
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (n : ℕ)
    (ε : ℝ) :
    Prop :=
  iidSampleMeasure P S
      (SupportSizeBadEvent
        P violates reconstruct n ε)
    ≤
      (Nat.choose S n : ENNReal) *
        (ENNReal.ofReal (1 - ε)) ^
          (S - n)

/--
The bad event for support size `n` is exactly the finite union of the
fixed-support bad events over all support sets of cardinality `n`.
-/
theorem supportSizeBadEvent_eq_union
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (n : ℕ)
    (ε : ℝ) :
    SupportSizeBadEvent
        P violates reconstruct n ε =
      ⋃ I ∈ supportSets S n,
        FixedSupportBadEvent
          P violates reconstruct I ε := by
  ext z
  simp [SupportSizeBadEvent]

/--
Union-bound estimate for a fixed support size.

For every support set `I` with `|I| = n`,

    P(B_I) ≤ (1 - ε)^(S-n).

There are `choose(S,n)` such support sets, hence

    P(B_n)
      ≤ choose(S,n) * (1 - ε)^(S-n).
-/
theorem supportSizeBadEvent_measure_le
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (n : ℕ)
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (hReg :
      ScenarioRegularity
        P violates reconstruct) :
    iidSampleMeasure P S
        (SupportSizeBadEvent
          P violates reconstruct n ε)
      ≤
    (Nat.choose S n : ENNReal) *
      (ENNReal.ofReal (1 - ε)) ^
        (S - n) := by

  rw [
    supportSizeBadEvent_eq_union
      P violates reconstruct n ε
  ]

  calc
    iidSampleMeasure P S
        (⋃ I ∈ supportSets S n,
          FixedSupportBadEvent
            P violates reconstruct I ε)
      ≤
        ∑ I ∈ supportSets S n,
          iidSampleMeasure P S
            (FixedSupportBadEvent
              P violates reconstruct I ε) := by
      exact
        MeasureTheory.measure_biUnion_finset_le
          (supportSets S n)
          (fun I =>
            FixedSupportBadEvent
              P violates reconstruct I ε)

    _ ≤
        ∑ _I ∈ supportSets S n,
          (ENNReal.ofReal (1 - ε)) ^
            (S - n) := by
      apply Finset.sum_le_sum
      intro I hI

      have hI' :
          I ∈
            Finset.powersetCard
              n
              (Finset.univ : Finset (Fin S)) := by
        simpa [supportSets] using hI

      have hCard :
          I.card = n :=
        (Finset.mem_powersetCard.mp hI').2

      have hBound :=
        fixedSupportBadEvent_measure_le
          P
          violates
          reconstruct
          I
          ε
          hε
          hReg.violationSets
          (hReg.splitBadEvent I ε)

      simpa [hCard] using hBound

    _ =
        (Nat.choose S n : ENNReal) *
          (ENNReal.ofReal (1 - ε)) ^
            (S - n) := by
      simp [supportSets_card]

/--
For each admissible support size `n`, the probability of the
corresponding bad event is at most `β / S`.
-/
theorem supportSizeScenarioFailure_measure_le
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (n : ℕ)
    (β : ℝ)
    (hnS : n < S)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hReg :
      ScenarioRegularity
        P violates reconstruct) :
    iidSampleMeasure P S
        (SupportSizeBadEvent
          P
          violates
          reconstruct
          n
          (scenarioEpsilon S n β))
      ≤
    ENNReal.ofReal (β / (S : ℝ)) := by

  have hUnion :=
    supportSizeBadEvent_measure_le
      P
      violates
      reconstruct
      n
      (scenarioEpsilon S n β)
      (DROSafety.scenarioEpsilon_nonneg
        S n β hnS hβ0 hβ1)
      hReg

  calc
    iidSampleMeasure P S
        (SupportSizeBadEvent
          P
          violates
          reconstruct
          n
          (scenarioEpsilon S n β))
      ≤
        (Nat.choose S n : ENNReal) *
          (ENNReal.ofReal
            (1 - scenarioEpsilon S n β)) ^
              (S - n) :=
      hUnion

    _ =
        ENNReal.ofReal (β / (S : ℝ)) :=
      DROSafety.choose_mul_one_sub_scenarioEpsilon_pow_ennreal
        S n β hnS hβ0

end DROSafety.SingleMode
