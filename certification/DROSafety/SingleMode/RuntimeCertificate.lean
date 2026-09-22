import DROSafety.SingleMode.ScenarioMonotonicityCertified

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Runtime Safe-Horizon certificate interface

The C++ controller reports `CERTIFIED` only after checking, among other
conditions, that:

* enough scenarios were sampled,
* the returned plan satisfies the sampled constraints,
* fallback was not used,
* multi-homotopy recovery was not used,
* support accounting was evaluated, and
* the realized support size does not exceed the configured support limit.

This file introduces an abstract Lean representation of those runtime facts
and proves the support-count consequences needed by the scenario theory.

This does not yet prove that the C++ implementation satisfies this model.
That implementation-refinement step is separate.
-/

namespace DROSafety.SingleMode

/--
Abstract information relevant to the Safe-Horizon certificate gate.

The proposition-valued fields represent facts established by the runtime
before it reports a certified solution.
-/
structure RuntimeCertificateSnapshot where
  sampleCountSufficient : Prop
  sampledConstraintsSatisfied : Prop
  usedFallback : Prop
  multiHomotopyRecovery : Prop
  supportEvaluated : Prop
  supportSize : ℕ
  supportLimit : ℕ

/--
The logical conjunction corresponding to the conditions under which the
ordinary Safe-Horizon certificate may be reported.
-/
def RuntimeCertified
    (r : RuntimeCertificateSnapshot) :
    Prop :=
  r.sampleCountSufficient ∧
  r.sampledConstraintsSatisfied ∧
  ¬ r.usedFallback ∧
  ¬ r.multiHomotopyRecovery ∧
  r.supportEvaluated ∧
  r.supportSize ≤ r.supportLimit

/--
A runtime-certified result necessarily satisfies its support cap.
-/
theorem runtimeCertified_support_le_limit
    (r : RuntimeCertificateSnapshot)
    (hCertified : RuntimeCertified r) :
    r.supportSize ≤ r.supportLimit := by

  rcases hCertified with
    ⟨_,
     _,
     _,
     _,
     _,
     hSupport⟩

  exact hSupport

/--
If the runtime support counter agrees with the mathematical `supportSize`
and the runtime limit agrees with `nBar`, then certification of one realized
sample gives the mathematical support bound for that sample.
-/
theorem support_bound_of_runtimeCertified
    {S : ℕ}
    {Ξ : Type}
    (supportSize : Sample S Ξ → ℕ)
    (runtime :
      Sample S Ξ → RuntimeCertificateSnapshot)
    (nBar : ℕ)
    (z : Sample S Ξ)
    (hCertified :
      RuntimeCertified (runtime z))
    (hSupportSize :
      (runtime z).supportSize =
        supportSize z)
    (hSupportLimit :
      (runtime z).supportLimit =
        nBar) :
    supportSize z ≤ nBar := by

  have hRuntime :
      (runtime z).supportSize ≤
        (runtime z).supportLimit :=
    runtimeCertified_support_le_limit
      (runtime z)
      hCertified

  calc
    supportSize z =
        (runtime z).supportSize := by
          symm
          exact hSupportSize

    _ ≤ (runtime z).supportLimit :=
      hRuntime

    _ = nBar :=
      hSupportLimit

/--
If the configured support limit is strictly below the sample count, every
runtime-certified realized sample also has support size strictly below `S`.
-/
theorem support_lt_samples_of_runtimeCertified
    {S : ℕ}
    {Ξ : Type}
    (supportSize : Sample S Ξ → ℕ)
    (runtime :
      Sample S Ξ → RuntimeCertificateSnapshot)
    (nBar : ℕ)
    (z : Sample S Ξ)
    (hCertified :
      RuntimeCertified (runtime z))
    (hSupportSize :
      (runtime z).supportSize =
        supportSize z)
    (hSupportLimit :
      (runtime z).supportLimit =
        nBar)
    (hnBarLtS :
      nBar < S) :
    supportSize z < S := by

  have hSupport :
      supportSize z ≤ nBar :=
    support_bound_of_runtimeCertified
      supportSize
      runtime
      nBar
      z
      hCertified
      hSupportSize
      hSupportLimit

  exact
    lt_of_le_of_lt
      hSupport
      hnBarLtS

/--
For a valid confidence parameter, a certified realized support count has a
scenario threshold no larger than the configured `nBar` threshold.
-/
theorem scenarioEpsilon_le_nBar_of_runtimeCertified
    {S : ℕ}
    {Ξ : Type}
    (supportSize : Sample S Ξ → ℕ)
    (runtime :
      Sample S Ξ → RuntimeCertificateSnapshot)
    (nBar : ℕ)
    (β : ℝ)
    (z : Sample S Ξ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hCertified :
      RuntimeCertified (runtime z))
    (hSupportSize :
      (runtime z).supportSize =
        supportSize z)
    (hSupportLimit :
      (runtime z).supportLimit =
        nBar) :
    DROSafety.scenarioEpsilon
        S
        (supportSize z)
        β
      ≤
    DROSafety.scenarioEpsilon
        S
        nBar
        β := by

  have hSupport :
      supportSize z ≤ nBar :=
    support_bound_of_runtimeCertified
      supportSize
      runtime
      nBar
      z
      hCertified
      hSupportSize
      hSupportLimit

  have hMono :
      ScenarioEpsilonMonotoneUpTo
        S nBar β :=
    scenarioEpsilonMonotoneUpTo_of_confidence
      S
      nBar
      β
      hβ0
      hβ1

  exact
    hMono
      (supportSize z)
      hSupport

end DROSafety.SingleMode
