import DROSafety.SingleMode.SupportUnion
import DROSafety.ScenarioTheorem

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Single-Mode Scenario Theorem

This file is the final single-mode scenario-theory layer.

The target is

    Pr[V > ε(S,n,β)] ≤ β.

The proof will be assembled from:

1. the fixed-support bound,
2. a union over all support sets of size `n`,
3. the scenario-epsilon algebra,
4. a union over possible support sizes.
-/

namespace DROSafety.SingleMode

/--
Bad event over all possible support sizes.

For some `n < S`, there exists a support subset of size `n`
whose reconstructed solution

* has true violation probability greater than
  `scenarioEpsilon S n β`, and
* nevertheless satisfies all non-support sampled constraints.
-/
def ScenarioBadEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (β : ℝ) :
    Set (Sample S Ξ) :=
  {z |
    ∃ n,
      n < S ∧
      z ∈ SupportSizeBadEvent
        P
        violates
        reconstruct
        n
        (DROSafety.scenarioEpsilon S n β)}

/--
The final single-mode probability statement.

This proposition is what must ultimately be proved from IID sampling
and the compression assumptions.
-/
def SingleModeScenarioBound
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (β : ℝ) :
    Prop :=
  iidSampleMeasure P S
      (ScenarioBadEvent
        P violates reconstruct β)
    ≤ ENNReal.ofReal β

end DROSafety.SingleMode
