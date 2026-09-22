import DROSafety.DRO.ClopperPearsonExactUpperEndpoint
import DROSafety.DRO.BinomialBetaLowerMeasure
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Real-valued exact Clopper-Pearson quantile equations

The exact lower and upper endpoints have already been constructed and
proved monotone.

This file proves their Beta-tail equations in `Measure.real`.

For the lower endpoint `L(k)` and `0 < k ≤ m`,

    Beta(k,m-k+1).real (-∞,L(k)] = α.

For the upper endpoint `U(k)` and `k < m`,

    Beta(k+1,m-k).real [U(k),∞) = α.

The following file will lift these real-valued equations back to
`ENNReal`, which is the form required by
`BetaClopperPearsonEndpointSpec`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

/--
The exact lower Clopper-Pearson endpoint satisfies its Beta lower-tail
equation in real-valued measure form.
-/
theorem betaMeasure_real_Iic_exactLowerEndpoint
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    (
      ProbabilityTheory.betaMeasure
        (k : ℝ)
        (
          (m : ℝ)
            -
          (k : ℝ)
            +
          1
        )
    ).real
      (
        Set.Iic
          (
            exactClopperPearsonLowerEndpoint
              m
              α
              k
          )
      )
      =
    α := by

  let L : ℝ :=
    exactClopperPearsonLowerEndpoint
      m
      α
      k

  have hLMem :
      L ∈ Set.Icc (0 : ℝ) 1 := by

    constructor

    · exact
        exactClopperPearsonLowerEndpoint_nonneg
          m
          k
          α
          hα0
          hα1

    · exact
        exactClopperPearsonLowerEndpoint_le_one
          m
          k
          α
          hα0
          hα1

  let pL : unitInterval :=
    ⟨L, hLMem⟩

  have hBeta :
      (
        ProbabilityTheory.binomial
          m
          pL
      ).real
        (Set.Ici k)
        =
      (
        ProbabilityTheory.betaMeasure
          (k : ℝ)
          (
            ((m - k : ℕ) : ℝ)
              +
            1
          )
      ).real
        (Set.Iic L) := by

    simpa [pL] using
      (
        binomial_real_Ici_eq_betaMeasure_real_Iic
          m
          k
          pL
          hk
          hkm
      )

  have hBinomialPolynomial :
      (
        ProbabilityTheory.binomial
          m
          pL
      ).real
        (Set.Ici k)
        =
      upperBinomialTailPolynomial
        m
        k
        L := by

    simpa [pL] using
      (
        binomial_real_Ici_eq_upperTailPolynomial
          m
          pL
          k
      )

  have hLevel :
      upperBinomialTailPolynomial
          m
          k
          L
        =
      α := by

    simpa [L] using
      (
        upperBinomialTailPolynomial_exactLowerEndpoint
          m
          k
          α
          hk
          hkm
          hα0
          hα1
      )

  have hCastSub :
      ((m - k : ℕ) : ℝ)
        =
      (m : ℝ) - (k : ℝ) := by

    rw [
      Nat.cast_sub
        hkm
    ]

  calc
    (
      ProbabilityTheory.betaMeasure
        (k : ℝ)
        (
          (m : ℝ)
            -
          (k : ℝ)
            +
          1
        )
    ).real
      (
        Set.Iic
          (
            exactClopperPearsonLowerEndpoint
              m
              α
              k
          )
      )
        =
      (
        ProbabilityTheory.betaMeasure
          (k : ℝ)
          (
            ((m - k : ℕ) : ℝ)
              +
            1
          )
      ).real
        (Set.Iic L) := by

          rw [hCastSub]

    _ =
      (
        ProbabilityTheory.binomial
          m
          pL
      ).real
        (Set.Ici k) :=
      hBeta.symm

    _ =
      upperBinomialTailPolynomial
        m
        k
        L :=
      hBinomialPolynomial

    _ = α :=
      hLevel

/--
At the exact upper endpoint, the binomial upper tail beginning at
`k+1` has real probability `1-α`.
-/
theorem binomial_real_Ici_succ_exactUpperEndpoint
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    let U :=
      exactClopperPearsonUpperEndpoint
        m
        α
        k
    let pU : unitInterval :=
      ⟨
        U,
        ⟨
          exactClopperPearsonUpperEndpoint_nonneg
            m k α hα0 hα1,
          exactClopperPearsonUpperEndpoint_le_one
            m k α hα0 hα1
        ⟩
      ⟩
    (
      ProbabilityTheory.binomial
        m
        pU
    ).real
      (Set.Ici (k + 1))
      =
    1 - α := by

  dsimp

  let U : ℝ :=
    exactClopperPearsonUpperEndpoint
      m
      α
      k

  have hUMem :
      U ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨
      exactClopperPearsonUpperEndpoint_nonneg
        m
        k
        α
        hα0
        hα1,
      exactClopperPearsonUpperEndpoint_le_one
        m
        k
        α
        hα0
        hα1
    ⟩

  let pU : unitInterval :=
    ⟨U, hUMem⟩

  calc
    (
      ProbabilityTheory.binomial
        m
        pU
    ).real
      (Set.Ici (k + 1))
        =
      upperBinomialTailPolynomial
        m
        (k + 1)
        U := by

          simpa [pU] using
            (
              binomial_real_Ici_eq_upperTailPolynomial
                m
                pU
                (k + 1)
            )

    _ =
      1 - α := by

        simpa [U] using
          (
            upperBinomialTailPolynomial_exactUpperEndpoint
              m
              k
              α
              hkm
              hα0
              hα1
          )

/--
At the exact upper endpoint, the real-valued lower binomial tail through
`k` equals `α`.
-/
theorem binomial_real_Iic_exactUpperEndpoint
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    let U :=
      exactClopperPearsonUpperEndpoint
        m
        α
        k
    let pU : unitInterval :=
      ⟨
        U,
        ⟨
          exactClopperPearsonUpperEndpoint_nonneg
            m k α hα0 hα1,
          exactClopperPearsonUpperEndpoint_le_one
            m k α hα0 hα1
        ⟩
      ⟩
    (
      ProbabilityTheory.binomial
        m
        pU
    ).real
      (Set.Iic k)
      =
    α := by

  dsimp

  let U : ℝ :=
    exactClopperPearsonUpperEndpoint
      m
      α
      k

  have hUMem :
      U ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨
      exactClopperPearsonUpperEndpoint_nonneg
        m
        k
        α
        hα0
        hα1,
      exactClopperPearsonUpperEndpoint_le_one
        m
        k
        α
        hα0
        hα1
    ⟩

  let pU : unitInterval :=
    ⟨U, hUMem⟩

  have hUpper :
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
        (Set.Ici (k + 1))
        =
      1 - α := by

    calc
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
        (Set.Ici (k + 1))
          =
        upperBinomialTailPolynomial
          m
          (k + 1)
          U := by

            simpa [pU] using
              (
                binomial_real_Ici_eq_upperTailPolynomial
                  m
                  pU
                  (k + 1)
              )

      _ =
        1 - α := by

          simpa [U] using
            (
              upperBinomialTailPolynomial_exactUpperEndpoint
                m
                k
                α
                hkm
                hα0
                hα1
            )

  have hComplement :
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
          (Set.Iic k)
        +
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
          (Set.Ici (k + 1))
        =
      1 := by

    have h :=
      MeasureTheory.probReal_add_probReal_compl
        (μ :=
          ProbabilityTheory.binomial
            m
            pU)
        (s := Set.Iic k)
        measurableSet_Iic

    rw [
      nat_compl_Iic_eq_Ici_succ
        k
    ] at h

    exact h

  rw [hUpper] at hComplement

  linarith

/--
The exact upper Clopper-Pearson endpoint satisfies its Beta upper-tail
equation in real-valued measure form.
-/
theorem betaMeasure_real_Ici_exactUpperEndpoint
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    (
      ProbabilityTheory.betaMeasure
        (
          (k : ℝ)
            +
          1
        )
        (
          (m : ℝ)
            -
          (k : ℝ)
        )
    ).real
      (
        Set.Ici
          (
            exactClopperPearsonUpperEndpoint
              m
              α
              k
          )
      )
      =
    α := by

  let U : ℝ :=
    exactClopperPearsonUpperEndpoint
      m
      α
      k

  have hUMem :
      U ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨
      exactClopperPearsonUpperEndpoint_nonneg
        m
        k
        α
        hα0
        hα1,
      exactClopperPearsonUpperEndpoint_le_one
        m
        k
        α
        hα0
        hα1
    ⟩

  let pU : unitInterval :=
    ⟨U, hUMem⟩

  have hBinomial :
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
        (Set.Iic k)
        =
      α := by

    have hUpper :
        (
          ProbabilityTheory.binomial
            m
            pU
        ).real
          (Set.Ici (k + 1))
          =
        1 - α := by

      calc
        (
          ProbabilityTheory.binomial
            m
            pU
        ).real
          (Set.Ici (k + 1))
            =
          upperBinomialTailPolynomial
            m
            (k + 1)
            U := by

              simpa [pU] using
                (
                  binomial_real_Ici_eq_upperTailPolynomial
                    m
                    pU
                    (k + 1)
                )

        _ =
          1 - α := by

            simpa [U] using
              (
                upperBinomialTailPolynomial_exactUpperEndpoint
                  m
                  k
                  α
                  hkm
                  hα0
                  hα1
              )

    have hComplement :
        (
          ProbabilityTheory.binomial
            m
            pU
        ).real
            (Set.Iic k)
          +
        (
          ProbabilityTheory.binomial
            m
            pU
        ).real
            (Set.Ici (k + 1))
          =
        1 := by

      have h :=
        MeasureTheory.probReal_add_probReal_compl
          (μ :=
            ProbabilityTheory.binomial
              m
              pU)
          (s := Set.Iic k)
          measurableSet_Iic

      rw [
        nat_compl_Iic_eq_Ici_succ
          k
      ] at h

      exact h

    rw [hUpper] at hComplement

    linarith

  have hBeta :
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
        (Set.Iic k)
        =
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
        (Set.Ici U) := by

    simpa [pU] using
      (
        binomial_real_Iic_eq_betaMeasure_real_Ici
          m
          k
          pU
          hkm
      )

  have hkmLe :
      k ≤ m :=
    Nat.le_of_lt
      hkm

  have hCastSub :
      ((m - k : ℕ) : ℝ)
        =
      (m : ℝ) - (k : ℝ) := by

    rw [
      Nat.cast_sub
        hkmLe
    ]

  calc
    (
      ProbabilityTheory.betaMeasure
        (
          (k : ℝ)
            +
          1
        )
        (
          (m : ℝ)
            -
          (k : ℝ)
        )
    ).real
      (
        Set.Ici
          (
            exactClopperPearsonUpperEndpoint
              m
              α
              k
          )
      )
        =
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
        (Set.Ici U) := by

          rw [
            Nat.cast_add,
            Nat.cast_one,
            hCastSub
          ]

    _ =
      (
        ProbabilityTheory.binomial
          m
          pU
      ).real
        (Set.Iic k) :=
      hBeta.symm

    _ = α :=
      hBinomial

end DROSafety.DRO
