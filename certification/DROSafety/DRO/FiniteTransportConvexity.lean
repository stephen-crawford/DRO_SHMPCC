import DROSafety.DRO.ConfidencePolytopeWasserstein
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Convexity of the finite Wasserstein ambiguity set

For a fixed nominal distribution `p`, ground-cost matrix `D`, and radius `ρ`,
the finite Wasserstein ambiguity set

    Qρ(p)
      =
    { q :
        q is a probability vector and
        there exists a coupling π from p to q
        with transportCost π ≤ ρ }

is convex in `q`.

This is a key ingredient in justifying the radius-calibration algorithm.

The C++ implementation enumerates vertices of the Clopper-Pearson
box-simplex confidence polytope and chooses a radius large enough to contain
every vertex.

Because the Wasserstein ambiguity ball is convex, if the confidence polytope
is the convex hull of those vertices, then containing every vertex implies
containing the entire confidence polytope.

This file proves the transport side of that argument.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
Pointwise convex mixture of two categorical distributions.
-/
def mixDistribution
    {d : ℕ}
    (t : ℝ)
    (q₁ q₂ : ModeDistribution d) :
    ModeDistribution d :=
  fun i =>
    t * q₁ i +
    (1 - t) * q₂ i

/--
A convex mixture of two probability vectors is again a probability vector.
-/
theorem mixDistribution_isProbabilityVector
    {d : ℕ}
    {q₁ q₂ : ModeDistribution d}
    {t : ℝ}
    (hq₁ :
      IsProbabilityVector q₁)
    (hq₂ :
      IsProbabilityVector q₂)
    (ht0 :
      0 ≤ t)
    (ht1 :
      t ≤ 1) :
    IsProbabilityVector
      (mixDistribution t q₁ q₂) := by

  classical

  constructor

  · intro i

    unfold mixDistribution

    have hOneMinusT :
        0 ≤ 1 - t :=
      sub_nonneg.mpr ht1

    exact
      add_nonneg
        (mul_nonneg
          ht0
          (hq₁.1 i))
        (mul_nonneg
          hOneMinusT
          (hq₂.1 i))

  · unfold mixDistribution

    calc
      (∑ i,
          (t * q₁ i +
           (1 - t) * q₂ i))
          =
        t * (∑ i, q₁ i) +
          (1 - t) * (∑ i, q₂ i) := by

            rw [Finset.sum_add_distrib]

            rw [
              ← Finset.mul_sum,
              ← Finset.mul_sum
            ]

      _ =
        t * 1 +
          (1 - t) * 1 := by

            rw [
              hq₁.2,
              hq₂.2
            ]

      _ = 1 := by
        ring

/--
Convex mixture of two transport plans having the same source distribution.

If

    π₁ : p -> q₁

and

    π₂ : p -> q₂,

then their pointwise convex mixture transports `p` to

    t q₁ + (1-t) q₂.
-/
def mixTransportPlan
    {d : ℕ}
    {p q₁ q₂ : ModeDistribution d}
    (π₁ : TransportPlan p q₁)
    (π₂ : TransportPlan p q₂)
    (t : ℝ)
    (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) :
    TransportPlan
      p
      (mixDistribution t q₁ q₂) where

  mass :=
    fun i j =>
      t * π₁.mass i j +
      (1 - t) * π₂.mass i j

  nonneg := by
    intro i j

    have hOneMinusT :
        0 ≤ 1 - t :=
      sub_nonneg.mpr ht1

    exact
      add_nonneg
        (mul_nonneg
          ht0
          (π₁.nonneg i j))
        (mul_nonneg
          hOneMinusT
          (π₂.nonneg i j))

  row_marginal := by
    intro i
    classical

    calc
      (∑ j,
          (t * π₁.mass i j +
           (1 - t) * π₂.mass i j))
          =
        t * (∑ j, π₁.mass i j) +
          (1 - t) * (∑ j, π₂.mass i j) := by

            rw [Finset.sum_add_distrib]

            rw [
              ← Finset.mul_sum,
              ← Finset.mul_sum
            ]

      _ =
        t * p i +
          (1 - t) * p i := by

            rw [
              π₁.row_marginal i,
              π₂.row_marginal i
            ]

      _ = p i := by
        ring

  col_marginal := by
    intro j
    classical

    calc
      (∑ i,
          (t * π₁.mass i j +
           (1 - t) * π₂.mass i j))
          =
        t * (∑ i, π₁.mass i j) +
          (1 - t) * (∑ i, π₂.mass i j) := by

            rw [Finset.sum_add_distrib]

            rw [
              ← Finset.mul_sum,
              ← Finset.mul_sum
            ]

      _ =
        t * q₁ j +
          (1 - t) * q₂ j := by

            rw [
              π₁.col_marginal j,
              π₂.col_marginal j
            ]

      _ =
        mixDistribution t q₁ q₂ j := by
          rfl

/--
Transport cost is affine under convex mixtures of transport plans.
-/
theorem transportCost_mixTransportPlan
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q₁ q₂ : ModeDistribution d}
    (π₁ : TransportPlan p q₁)
    (π₂ : TransportPlan p q₂)
    (t : ℝ)
    (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) :
    transportCost
        D
        (mixTransportPlan
          π₁
          π₂
          t
          ht0
          ht1)
      =
    t * transportCost D π₁ +
      (1 - t) * transportCost D π₂ := by

  classical

  unfold transportCost

  change
    (∑ i, ∑ j,
      D.cost i j *
        (t * π₁.mass i j +
         (1 - t) * π₂.mass i j))
      =
    t *
        (∑ i, ∑ j,
          D.cost i j * π₁.mass i j)
      +
    (1 - t) *
        (∑ i, ∑ j,
          D.cost i j * π₂.mass i j)

  calc
    (∑ i, ∑ j,
      D.cost i j *
        (t * π₁.mass i j +
         (1 - t) * π₂.mass i j))
        =
      ∑ i, ∑ j,
        (t *
            (D.cost i j * π₁.mass i j) +
         (1 - t) *
            (D.cost i j * π₂.mass i j)) := by

          apply Finset.sum_congr rfl
          intro i hi

          apply Finset.sum_congr rfl
          intro j hj

          ring

    _ =
      ∑ i,
        (
          t *
            (∑ j,
              D.cost i j * π₁.mass i j)
          +
          (1 - t) *
            (∑ j,
              D.cost i j * π₂.mass i j)
        ) := by

          apply Finset.sum_congr rfl
          intro i hi

          rw [Finset.sum_add_distrib]

          rw [
            ← Finset.mul_sum,
            ← Finset.mul_sum
          ]

    _ =
      t *
          (∑ i, ∑ j,
            D.cost i j * π₁.mass i j)
        +
      (1 - t) *
          (∑ i, ∑ j,
            D.cost i j * π₂.mass i j) := by

          rw [Finset.sum_add_distrib]

          rw [
            ← Finset.mul_sum,
            ← Finset.mul_sum
          ]

/--
The finite Wasserstein ambiguity ball is closed under convex mixtures.

This is the central convexity theorem needed by the radius calibration.
-/
theorem finiteWassersteinAmbiguity_closed_under_mix
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q₁ q₂ : ModeDistribution d}
    {ρ t : ℝ}
    (hq₁ :
      q₁ ∈
        FiniteWassersteinAmbiguity
          D
          p
          ρ)
    (hq₂ :
      q₂ ∈
        FiniteWassersteinAmbiguity
          D
          p
          ρ)
    (ht0 :
      0 ≤ t)
    (ht1 :
      t ≤ 1) :
    mixDistribution t q₁ q₂
      ∈
    FiniteWassersteinAmbiguity
      D
      p
      ρ := by

  rcases hq₁ with
    ⟨hq₁Prob, π₁, hCost₁⟩

  rcases hq₂ with
    ⟨hq₂Prob, π₂, hCost₂⟩

  have hMixedProb :
      IsProbabilityVector
        (mixDistribution t q₁ q₂) :=
    mixDistribution_isProbabilityVector
      hq₁Prob
      hq₂Prob
      ht0
      ht1

  let πMix :
      TransportPlan
        p
        (mixDistribution t q₁ q₂) :=
    mixTransportPlan
      π₁
      π₂
      t
      ht0
      ht1

  refine
    ⟨hMixedProb, πMix, ?_⟩

  have hOneMinusT :
      0 ≤ 1 - t :=
    sub_nonneg.mpr ht1

  have hWeightedCost :
      t * transportCost D π₁ +
          (1 - t) * transportCost D π₂
        ≤
      t * ρ +
          (1 - t) * ρ := by

    exact
      add_le_add
        (mul_le_mul_of_nonneg_left
          hCost₁
          ht0)
        (mul_le_mul_of_nonneg_left
          hCost₂
          hOneMinusT)

  have hCostIdentity :
      transportCost D πMix
        =
      t * transportCost D π₁ +
        (1 - t) * transportCost D π₂ := by

    dsimp [πMix]

    exact
      transportCost_mixTransportPlan
        D
        π₁
        π₂
        t
        ht0
        ht1

  calc
    transportCost D πMix
        =
      t * transportCost D π₁ +
        (1 - t) * transportCost D π₂ :=
      hCostIdentity

    _ ≤
      t * ρ +
        (1 - t) * ρ :=
      hWeightedCost

    _ = ρ := by
      ring

/--
Custom segment-closure notion for sets of mode distributions.

This avoids depending on any particular Mathlib `Convex` API in the later
calibration proof.
-/
def DistributionSegmentClosed
    {d : ℕ}
    (S : Set (ModeDistribution d)) :
    Prop :=
  ∀ q₁ q₂,
    q₁ ∈ S →
    q₂ ∈ S →
    ∀ t : ℝ,
      0 ≤ t →
      t ≤ 1 →
      mixDistribution t q₁ q₂ ∈ S

/--
Every finite Wasserstein ambiguity ball is segment-closed.
-/
theorem finiteWassersteinAmbiguity_segmentClosed
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    (ρ : ℝ) :
    DistributionSegmentClosed
      (FiniteWassersteinAmbiguity
        D
        p
        ρ) := by

  intro q₁ q₂
  intro hq₁ hq₂
  intro t ht0 ht1

  exact
    finiteWassersteinAmbiguity_closed_under_mix
      D
      hq₁
      hq₂
      ht0
      ht1

end DROSafety.DRO
