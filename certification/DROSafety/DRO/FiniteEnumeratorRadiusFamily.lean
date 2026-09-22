import DROSafety.DRO.FiniteEnumeratorRadius
import DROSafety.DRO.ConfidencePolytopeWasserstein
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Pointwise finite-enumerator radius family

The Clopper-Pearson coverage theorem is stated over a sample space `Ω`.
For each realization `ω`, the interval endpoints, nominal center, and
therefore the finite box-simplex enumeration may be different.

This file lifts the deterministic finite-enumerator certificate pointwise.

For every `ω` we assume finite enumerator data

    data ω :
      FiniteEnumeratorTransportData
        D
        (center ω)
        (lower ω)
        (upper ω).

We then define

    rho(ω) = finiteEnumeratorRadius (data ω)

and prove pointwise that

    ConfidencePolytope (lower ω) (upper ω)
      ⊆
    FiniteWassersteinAmbiguity D (center ω) (rho ω).

Thus the abstract pointwise `hRadiusCover` assumption used by the
Clopper-Pearson theorem follows directly from finite enumerator data.
-/

namespace DROSafety.DRO

/--
Radius produced pointwise from finite box-simplex enumerator data.
-/
noncomputable def finiteEnumeratorRadiusFamily
    {Ω : Type*}
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : Ω → ModeDistribution d}
    {lower upper : Ω → Fin d → ℝ}
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω)) :
    Ω → ℝ :=
  fun ω =>
    finiteEnumeratorRadius
      (data ω)

/--
The finite-enumerator radius is nonnegative for every sample realization.
-/
theorem finiteEnumeratorRadiusFamily_nonneg
    {Ω : Type*}
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : Ω → ModeDistribution d}
    {lower upper : Ω → Fin d → ℝ}
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω)) :
    ∀ ω : Ω,
      0 ≤ finiteEnumeratorRadiusFamily data ω := by

  intro ω

  unfold finiteEnumeratorRadiusFamily

  exact
    finiteEnumeratorRadius_nonneg
      (data ω)

/--
Main pointwise radius-cover theorem.

For every sample realization, the radius computed from the complete finite
enumerator data covers the entire corresponding box-simplex confidence
polytope.
-/
theorem radiusCover_family_of_finiteEnumeratorData
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω))
    (hd :
      0 < d)
    (hLower :
      ∀ ω : Ω,
        ∀ i : Fin d,
          0 ≤ lower ω i) :
    ∀ ω : Ω,
      RadiusCoversConfidencePolytope
        D
        (center ω)
        (lower ω)
        (upper ω)
        (finiteEnumeratorRadiusFamily data ω) := by

  intro ω

  unfold finiteEnumeratorRadiusFamily

  exact
    radiusCoversConfidencePolytope_of_finiteEnumeratorData
      D
      (center ω)
      (lower ω)
      (upper ω)
      (data ω)
      hd
      (hLower ω)

/--
Pointwise membership form.

Any distribution inside the confidence polytope for realization `ω`
belongs to the Wasserstein ambiguity ball whose radius is obtained from
that realization's finite enumerator data.
-/
theorem confidencePoint_mem_ambiguity_of_finiteEnumeratorFamily
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω))
    (hd :
      0 < d)
    (hLower :
      ∀ ω : Ω,
        ∀ i : Fin d,
          0 ≤ lower ω i)
    (ω : Ω)
    {q : ModeDistribution d}
    (hq :
      q ∈
        ConfidencePolytope
          (lower ω)
          (upper ω)) :
    q ∈
      FiniteWassersteinAmbiguity
        D
        (center ω)
        (finiteEnumeratorRadiusFamily data ω) := by

  unfold finiteEnumeratorRadiusFamily

  exact
    confidencePoint_mem_ambiguity_of_finiteEnumeratorData
      D
      (center ω)
      (lower ω)
      (upper ω)
      (data ω)
      hd
      (hLower ω)
      hq

/--
The pointwise enumerator transport-radius certificate itself is also
available directly for each realization.
-/
theorem enumeratorCertificate_family_of_finiteEnumeratorData
    {Ω : Type*}
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : Ω → ModeDistribution d}
    {lower upper : Ω → Fin d → ℝ}
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω)) :
    ∀ ω : Ω,
      EnumeratorTransportRadiusCertificate
        D
        (center ω)
        (lower ω)
        (upper ω)
        (finiteEnumeratorRadiusFamily data ω) := by

  intro ω

  unfold finiteEnumeratorRadiusFamily

  exact
    finiteEnumeratorRadius_certificate
      (data ω)

end DROSafety.DRO
