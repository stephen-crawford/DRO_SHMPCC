import DROSafety.SingleMode.SupportUnion

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Union Over All Support Sizes

For every admissible support size `n < S`, the previous development
proves

    P(B_n) ≤ β / S.

There are exactly `S` possible support sizes

    n = 0, ..., S - 1.

A finite union bound therefore gives

    P(⋃ n < S, B_n) ≤ β.

This is the final combinatorial probability step before deriving the
generic single-mode scenario theorem.
-/

namespace DROSafety.SingleMode

/--
The bad event obtained by taking the union over every admissible
support size `n < S`.

At each support size, the risk threshold is the corresponding
scenario bound `scenarioEpsilon S n β`.
-/
def AllSupportSizesBadEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (β : ℝ) :
    Set (Sample S Ξ) :=
  ⋃ n ∈ Finset.range S,
    SupportSizeBadEvent
      P
      violates
      reconstruct
      n
      (DROSafety.scenarioEpsilon S n β)

/--
The probability of failure at any admissible support size is at most
`β`.

The proof consists of:

1. a finite union bound over `n = 0, ..., S - 1`;
2. the previously proved bound `P(B_n) ≤ β / S`;
3. summing `S` identical terms `β / S`.

The `S = 0` case is handled separately.
-/theorem allSupportSizesBadEvent_measure_le
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hReg :
      ScenarioRegularity
        P violates reconstruct) :
    iidSampleMeasure P S
        (AllSupportSizesBadEvent
          P violates reconstruct β)
      ≤
    ENNReal.ofReal β := by

  unfold AllSupportSizesBadEvent

  calc
    iidSampleMeasure P S
        (⋃ n ∈ Finset.range S,
          SupportSizeBadEvent
            P
            violates
            reconstruct
            n
            (DROSafety.scenarioEpsilon S n β))
      ≤
        ∑ n ∈ Finset.range S,
          iidSampleMeasure P S
            (SupportSizeBadEvent
              P
              violates
              reconstruct
              n
              (DROSafety.scenarioEpsilon S n β)) := by
      exact
        MeasureTheory.measure_biUnion_finset_le
          (Finset.range S)
          (fun n =>
            SupportSizeBadEvent
              P
              violates
              reconstruct
              n
              (DROSafety.scenarioEpsilon S n β))

    _ ≤
        ∑ _n ∈ Finset.range S,
          ENNReal.ofReal
            (β / (S : ℝ)) := by
      apply Finset.sum_le_sum
      intro n hn

      have hnS : n < S :=
        Finset.mem_range.mp hn

      exact
        supportSizeScenarioFailure_measure_le
          P
          violates
          reconstruct
          n
          β
          hnS
          hβ0
          hβ1
          hReg

    _ ≤ ENNReal.ofReal β := by
      by_cases hS0 : S = 0

      · subst S
        simp

      · have hSReal :
            (S : ℝ) ≠ 0 := by
          exact_mod_cast hS0

        have hSNonneg :
            0 ≤ (S : ℝ) := by
          positivity

        have hMul :
            (S : ℝ) * (β / (S : ℝ)) = β := by
          field_simp [hSReal]

        have hEq :
            (∑ _n ∈ Finset.range S,
              ENNReal.ofReal
                (β / (S : ℝ)))
              =
            ENNReal.ofReal β := by

          calc
            (∑ _n ∈ Finset.range S,
                ENNReal.ofReal
                  (β / (S : ℝ)))
                =
              (S : ENNReal) *
                ENNReal.ofReal
                  (β / (S : ℝ)) := by
              simp [nsmul_eq_mul]

            _ =
              ENNReal.ofReal
                ((S : ℝ) *
                  (β / (S : ℝ))) := by
              rw [
                ENNReal.ofReal_mul
                  hSNonneg
              ]
              simp

            _ =
              ENNReal.ofReal β := by
              rw [hMul]

        exact hEq.le

end DROSafety.SingleMode
