import DROSafety.SingleMode.Definitions
import Mathlib.MeasureTheory.Constructions.Pi

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# IID Scenario Samples

Finite product probability measure describing `S` independent
samples from one uncertainty distribution.
-/

namespace DROSafety.SingleMode

/--
The IID product measure on `S` copies of `P`.
-/
noncomputable def iidSampleMeasure
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (S : ℕ) :
    MeasureTheory.Measure (Sample S Ξ) :=
  MeasureTheory.Measure.pi
    (fun _ : Fin S => P)

/--
If `P` is a probability measure, then the finite IID sample measure
is also a probability measure.
-/
theorem iidSampleMeasure_isProbability
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (S : ℕ)
    [MeasureTheory.IsProbabilityMeasure P] :
    MeasureTheory.IsProbabilityMeasure
      (iidSampleMeasure P S) := by
  unfold iidSampleMeasure
  infer_instance

end DROSafety.SingleMode
