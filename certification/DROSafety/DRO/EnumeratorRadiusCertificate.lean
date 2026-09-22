import DROSafety.DRO.BoxSimplexEnumeratorShape
import DROSafety.DRO.VertexRadiusCover
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Wasserstein radius certificate for box-simplex enumerator points

The preceding files prove

    ConfidencePolytope lower upper
      ⊆
    convexHull ℝ (BoxSimplexEnumeratorPoints lower upper).

Therefore, since a fixed-center finite Wasserstein ambiguity ball is convex,
it is sufficient to prove that every enumerator point lies inside that ball.

This file packages the remaining numerical obligation in the form used by
the C++ radius calibration:

for every enumerated candidate `q`, produce a feasible transport plan from
the chosen center to `q` whose cost is at most `rho`.

From those candidate certificates we derive

    RadiusCoversConfidencePolytope D center lower upper rho.

This removes the abstract `hRadiusCover` assumption from the geometric part
of the Clopper-Pearson/Wasserstein coverage argument.
-/

namespace DROSafety.DRO

/--
Implementation-level certificate for the radius computed from the
box-simplex candidates.

For every enumerator candidate `q`, there must exist a valid transport plan
from `center` to `q` having transport cost at most `rho`.
-/
def EnumeratorTransportRadiusCertificate
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (rho : ℝ) :
    Prop :=
  ∀ q : ModeDistribution d,
    BoxSimplexEnumeratorCandidate
        lower
        upper
        q →
    ∃ π : TransportPlan center q,
      transportCost D π ≤ rho

/--
A certified enumerator candidate lies in the finite Wasserstein ambiguity
ball.
-/
theorem enumeratorCandidate_mem_finiteWassersteinAmbiguity
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    {rho : ℝ}
    (hCertificate :
      EnumeratorTransportRadiusCertificate
        D
        center
        lower
        upper
        rho)
    {q : ModeDistribution d}
    (hCandidate :
      BoxSimplexEnumeratorCandidate
        lower
        upper
        q) :
    q ∈
      FiniteWassersteinAmbiguity
        D
        center
        rho := by

  have hPoly :
      q ∈
        ConfidencePolytope
          lower
          upper :=
    hCandidate.1

  have hProb :
      IsProbabilityVector q :=
    hPoly.1

  have hTransport :
      ∃ π : TransportPlan center q,
        transportCost D π ≤ rho :=
    hCertificate
      q
      hCandidate

  exact
    ⟨hProb, hTransport⟩

/--
Every point in the abstract C++ enumerator set lies in the Wasserstein ball
when the enumerator transport-radius certificate holds.
-/
theorem enumeratorPoints_subset_finiteWassersteinAmbiguity
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    {rho : ℝ}
    (hCertificate :
      EnumeratorTransportRadiusCertificate
        D
        center
        lower
        upper
        rho) :
    BoxSimplexEnumeratorPoints
        lower
        upper
      ⊆
    FiniteWassersteinAmbiguity
      D
      center
      rho := by

  intro q hq

  exact
    enumeratorCandidate_mem_finiteWassersteinAmbiguity
      hCertificate
      hq

/--
Main radius-cover bridge.

If

* the mode dimension is nonzero;
* confidence lower bounds are nonnegative;
* every C++ enumerator candidate has a transport plan of cost at most `rho`;

then the entire box-simplex confidence polytope is contained in the
finite Wasserstein ambiguity ball of radius `rho`.
-/
theorem radiusCoversConfidencePolytope_of_enumeratorCertificate
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (rho : ℝ)
    (hd :
      0 < d)
    (hLower :
      ∀ i,
        0 ≤ lower i)
    (hCertificate :
      EnumeratorTransportRadiusCertificate
        D
        center
        lower
        upper
        rho) :
    RadiusCoversConfidencePolytope
      D
      center
      lower
      upper
      rho := by

  have hGenerate :
      VerticesGenerateConfidencePolytope
        (BoxSimplexEnumeratorPoints
          lower
          upper)
        lower
        upper :=
    enumeratorPoints_generate_confidencePolytope
      lower
      upper
      hd
      hLower

  have hCovered :
      BoxSimplexEnumeratorPoints
          lower
          upper
        ⊆
      FiniteWassersteinAmbiguity
        D
        center
        rho :=
    enumeratorPoints_subset_finiteWassersteinAmbiguity
      hCertificate

  exact
    radiusCoversConfidencePolytope_of_vertices
      D
      center
      lower
      upper
      rho
      (BoxSimplexEnumeratorPoints
        lower
        upper)
      hGenerate
      hCovered

/--
Equivalent pointwise form of the main bridge.

Every distribution inside the confidence polytope belongs to the ambiguity
ball when all enumerator candidates are transport-certified.
-/
theorem confidencePoint_mem_ambiguity_of_enumeratorCertificate
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (rho : ℝ)
    (hd :
      0 < d)
    (hLower :
      ∀ i,
        0 ≤ lower i)
    (hCertificate :
      EnumeratorTransportRadiusCertificate
        D
        center
        lower
        upper
        rho)
    {q : ModeDistribution d}
    (hq :
      q ∈
        ConfidencePolytope
          lower
          upper) :
    q ∈
      FiniteWassersteinAmbiguity
        D
        center
        rho := by

  have hCover :
      RadiusCoversConfidencePolytope
        D
        center
        lower
        upper
        rho :=
    radiusCoversConfidencePolytope_of_enumeratorCertificate
      D
      center
      lower
      upper
      rho
      hd
      hLower
      hCertificate

  exact
    hCover
      q
      hq

end DROSafety.DRO
