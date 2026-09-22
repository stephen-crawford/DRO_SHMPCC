import DROSafety.DRO.FiniteTransportConvexity
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Vertex certificates for Wasserstein confidence-polytope coverage

The finite-sample calibration implementation enumerates vertices of the
Clopper-Pearson box-simplex confidence polytope and chooses a Wasserstein
radius large enough to contain every enumerated vertex.

This file proves the abstract geometric step behind that construction.

If

    C ⊆ convexHull(V)

and every vertex in `V` lies in the finite Wasserstein ambiguity ball,

    V ⊆ Q_rho(p),

then, because `Q_rho(p)` is convex,

    C ⊆ Q_rho(p).

Therefore a complete vertex enumeration together with a radius covering
every enumerated vertex is sufficient to establish

    RadiusCoversConfidencePolytope.

The only remaining implementation-specific theorem is then that the vertex
enumerator used by the calibration routine actually generates all vertices
needed to represent the box-simplex confidence polytope.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
The finite Wasserstein ambiguity set is convex in the candidate
distribution.

This packages the binary-mixture result proved in
`FiniteTransportConvexity.lean` into Mathlib's standard `Convex` predicate.
-/
theorem finiteWassersteinAmbiguity_convex
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    (ρ : ℝ) :
    Convex ℝ
      (FiniteWassersteinAmbiguity
        D
        p
        ρ) := by

  rw [convex_iff_add_mem]

  intro q₁ hq₁
  intro q₂ hq₂
  intro a b
  intro ha hb hab

  have ha1 :
      a ≤ 1 := by
    linarith

  have hMix :
      mixDistribution a q₁ q₂
        ∈
      FiniteWassersteinAmbiguity
        D
        p
        ρ :=
    finiteWassersteinAmbiguity_closed_under_mix
      D
      hq₁
      hq₂
      ha
      ha1

  have hbEq :
      b = 1 - a := by
    linarith

  have hEq :
      a • q₁ + b • q₂
        =
      mixDistribution a q₁ q₂ := by

    funext i

    simp [
      mixDistribution,
      hbEq
    ]

  rw [hEq]

  exact hMix

/--
If all elements of `vertices` lie in a finite Wasserstein ambiguity set,
then the entire convex hull of those vertices lies in the ambiguity set.
-/
theorem convexHull_vertices_subset_finiteWassersteinAmbiguity
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    (ρ : ℝ)
    (vertices : Set (ModeDistribution d))
    (hVertices :
      vertices ⊆
        FiniteWassersteinAmbiguity
          D
          p
          ρ) :
    convexHull ℝ vertices
      ⊆
    FiniteWassersteinAmbiguity
      D
      p
      ρ := by

  have hConvex :
      Convex ℝ
        (FiniteWassersteinAmbiguity
          D
          p
          ρ) :=
    finiteWassersteinAmbiguity_convex
      D
      p
      ρ

  exact
    (hConvex.convexHull_subset_iff).2
      hVertices

/--
A set of candidate vertices generates a confidence polytope when every
point of the polytope belongs to their convex hull.
-/
def VerticesGenerateConfidencePolytope
    {d : ℕ}
    (vertices : Set (ModeDistribution d))
    (lower upper : Fin d → ℝ) :
    Prop :=
  ConfidencePolytope lower upper
    ⊆
  convexHull ℝ vertices

/--
If a vertex set generates the confidence polytope and every vertex lies
inside the Wasserstein ambiguity ball, then the entire confidence polytope
lies inside the ambiguity ball.
-/
theorem confidencePolytope_subset_ambiguity_of_vertices
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (ρ : ℝ)
    (vertices : Set (ModeDistribution d))
    (hGenerate :
      VerticesGenerateConfidencePolytope
        vertices
        lower
        upper)
    (hVerticesCovered :
      vertices ⊆
        FiniteWassersteinAmbiguity
          D
          center
          ρ) :
    ConfidencePolytope
        lower
        upper
      ⊆
    FiniteWassersteinAmbiguity
      D
      center
      ρ := by

  have hHullCovered :
      convexHull ℝ vertices
        ⊆
      FiniteWassersteinAmbiguity
        D
        center
        ρ :=
    convexHull_vertices_subset_finiteWassersteinAmbiguity
      D
      center
      ρ
      vertices
      hVerticesCovered

  exact
    Set.Subset.trans
      hGenerate
      hHullCovered

/--
The previous theorem is exactly the radius-coverage property required by
`ConfidencePolytopeWasserstein.lean`.
-/
theorem radiusCoversConfidencePolytope_of_vertices
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (ρ : ℝ)
    (vertices : Set (ModeDistribution d))
    (hGenerate :
      VerticesGenerateConfidencePolytope
        vertices
        lower
        upper)
    (hVerticesCovered :
      vertices ⊆
        FiniteWassersteinAmbiguity
          D
          center
          ρ) :
    RadiusCoversConfidencePolytope
      D
      center
      lower
      upper
      ρ := by

  intro q hq

  exact
    confidencePolytope_subset_ambiguity_of_vertices
      D
      center
      lower
      upper
      ρ
      vertices
      hGenerate
      hVerticesCovered
      hq

/--
Pointwise version useful for the random confidence intervals generated
from observations.

For every realization `ω`, it is sufficient to provide

1. a set of enumerated vertices;
2. proof that those vertices generate the realized confidence polytope;
3. proof that every such vertex lies in the realized Wasserstein ball.
-/
theorem radiusCover_family_of_vertexCertificates
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (ρ : Ω → ℝ)
    (vertices :
      Ω → Set (ModeDistribution d))
    (hGenerate :
      ∀ ω,
        VerticesGenerateConfidencePolytope
          (vertices ω)
          (lower ω)
          (upper ω))
    (hVerticesCovered :
      ∀ ω,
        vertices ω ⊆
          FiniteWassersteinAmbiguity
            D
            (center ω)
            (ρ ω)) :
    ∀ ω,
      RadiusCoversConfidencePolytope
        D
        (center ω)
        (lower ω)
        (upper ω)
        (ρ ω) := by

  intro ω

  exact
    radiusCoversConfidencePolytope_of_vertices
      D
      (center ω)
      (lower ω)
      (upper ω)
      (ρ ω)
      (vertices ω)
      (hGenerate ω)
      (hVerticesCovered ω)

end DROSafety.DRO
