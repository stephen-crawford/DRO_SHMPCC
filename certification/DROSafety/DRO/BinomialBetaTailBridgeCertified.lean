import DROSafety.DRO.ClopperPearsonBetaBridge
import DROSafety.DRO.BinomialBetaLowerMeasure
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Certified binomial-Beta tail bridge

`ClopperPearsonBetaBridge.lean` introduced `BinomialBetaTailBridge`
as an abstract analytic assumption containing three facts:

1. the upper binomial tail equals the corresponding lower Beta tail;
2. the lower binomial tail equals the corresponding upper Beta tail;
3. the upper binomial tail is zero above the binomial support.

The preceding development has now proved all three facts.

This file packages those results into an unconditional
`BinomialBetaTailBridge`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

/--
The binomial-Beta tail bridge is fully certified.

For every number of trials `m` and Bernoulli parameter `p`:

* for `0 < k ≤ m`,

      Bin(m,p) [k,∞)
        =
      Beta(k,m-k+1) (-∞,p];

* for `k < m`,

      Bin(m,p) (-∞,k]
        =
      Beta(k+1,m-k) [p,∞);

* for `m < k`,

      Bin(m,p) [k,∞) = 0.
-/
theorem certifiedBinomialBetaTailBridge
    (m : ℕ)
    (p : unitInterval) :
    BinomialBetaTailBridge
      m
      p := by

  refine
    {
      upperBinomialTail := ?_
      lowerBinomialTail := ?_
      upperTail_above_support := ?_
    }

  · intro k hk hkm

    have h :=
      binomial_Ici_eq_betaMeasure_Iic
        m
        k
        p
        hk
        hkm

    simpa [
      Nat.cast_add,
      Nat.cast_one,
      Nat.cast_sub hkm
    ] using h

  · intro k hkm

    have hkle :
        k ≤ m :=
      Nat.le_of_lt
        hkm

    have h :=
      binomial_Iic_eq_betaMeasure_Ici
        m
        k
        p
        hkm

    simpa [
      Nat.cast_add,
      Nat.cast_one,
      Nat.cast_sub hkle
    ] using h

  · intro k hmk

    exact
      binomial_Ici_eq_zero_of_trials_lt
        m
        p
        k
        hmk

/--
Convenient specialization: the certified bridge may be supplied directly
to any theorem that previously required an explicit
`BinomialBetaTailBridge m p` hypothesis.
-/
theorem binomialBetaTailBridge
    (m : ℕ)
    (p : unitInterval) :
    BinomialBetaTailBridge
      m
      p :=
  certifiedBinomialBetaTailBridge
    m
    p

end DROSafety.DRO
