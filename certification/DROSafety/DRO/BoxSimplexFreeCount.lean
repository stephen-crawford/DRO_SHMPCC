import DROSafety.DRO.BoxSimplexDecomposition
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Strictly-free coordinate count

For a point in the box-simplex confidence polytope, define

    F(q) = #{ i : lower_i < q_i < upper_i }.

The boundary decomposition from `BoxSimplexDecomposition.lean` takes two
strictly-free coordinates `i` and `j` and constructs two feasible points

    qPlus
    qMinus.

Each maximal perturbation reaches at least one interval bound.

This file proves that both perturbations strictly reduce `F`.

That gives the well-founded measure required for the later strong-induction
proof showing that every confidence-polytope point lies in the convex hull
of points with at most one strictly-free coordinate.
-/

namespace DROSafety.DRO

/--
The finite set of coordinates strictly inside their confidence intervals.
-/
noncomputable def strictlyFreeCoordinates
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d) :
    Finset (Fin d) := by

  classical

  exact
    Finset.univ.filter
      (fun i =>
        StrictlyFree
          lower
          upper
          q
          i)

/--
Number of strictly-free coordinates.
-/
noncomputable def strictlyFreeCount
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d) :
    ℕ :=
  (strictlyFreeCoordinates
    lower
    upper
    q).card

/--
Membership in `strictlyFreeCoordinates` is exactly `StrictlyFree`.
-/
theorem mem_strictlyFreeCoordinates
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i : Fin d} :
    i ∈
        strictlyFreeCoordinates
          lower
          upper
          q
      ↔
    StrictlyFree
      lower
      upper
      q
      i := by

  classical

  simp [
    strictlyFreeCoordinates
  ]

/--
A coordinate that is strictly free belongs to the free-coordinate finset.
-/
theorem mem_strictlyFreeCoordinates_of_free
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i : Fin d}
    (hi :
      StrictlyFree
        lower
        upper
        q
        i) :
    i ∈
      strictlyFreeCoordinates
        lower
        upper
        q := by

  exact
    mem_strictlyFreeCoordinates.mpr
      hi

/--
If a coordinate equals its upper bound, it is not strictly free.
-/
theorem not_strictlyFree_of_eq_upper
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i : Fin d}
    (h :
      q i = upper i) :
    ¬ StrictlyFree
        lower
        upper
        q
        i := by

  intro hi

  unfold StrictlyFree at hi

  linarith

/--
If a coordinate equals its lower bound, it is not strictly free.
-/
theorem not_strictlyFree_of_eq_lower
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i : Fin d}
    (h :
      q i = lower i) :
    ¬ StrictlyFree
        lower
        upper
        q
        i := by

  intro hi

  unfold StrictlyFree at hi

  linarith

/--
No new strictly-free coordinate is created by the positive boundary
perturbation.

Coordinates other than `i` and `j` are unchanged, while `i` and `j`
were already strictly free in the original point.
-/
theorem strictlyFreeCoordinates_plus_subset
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCoordinates
        lower
        upper
        (plusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      ⊆
    strictlyFreeCoordinates
      lower
      upper
      q := by

  intro k hk

  have hkPlus :
      StrictlyFree
        lower
        upper
        (plusBoundaryDistribution
          lower
          upper
          q
          i
          j)
        k :=
    mem_strictlyFreeCoordinates.mp
      hk

  apply
    mem_strictlyFreeCoordinates.mpr

  by_cases hki :
      k = i

  · subst k
    exact hi

  · by_cases hkj :
        k = j

    · subst k
      exact hj

    · have hEq :
          plusBoundaryDistribution
              lower
              upper
              q
              i
              j
              k
            =
          q k := by

        unfold plusBoundaryDistribution

        exact
          transferDistribution_at_other
            q
            i
            j
            k
            _
            hki
            hkj

      unfold StrictlyFree at hkPlus ⊢

      rw [hEq] at hkPlus

      exact hkPlus

/--
No new strictly-free coordinate is created by the negative boundary
perturbation.
-/
theorem strictlyFreeCoordinates_minus_subset
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCoordinates
        lower
        upper
        (minusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      ⊆
    strictlyFreeCoordinates
      lower
      upper
      q := by

  intro k hk

  have hkMinus :
      StrictlyFree
        lower
        upper
        (minusBoundaryDistribution
          lower
          upper
          q
          i
          j)
        k :=
    mem_strictlyFreeCoordinates.mp
      hk

  apply
    mem_strictlyFreeCoordinates.mpr

  by_cases hki :
      k = i

  · subst k
    exact hi

  · by_cases hkj :
        k = j

    · subst k
      exact hj

    · have hEq :
          minusBoundaryDistribution
              lower
              upper
              q
              i
              j
              k
            =
          q k := by

        unfold minusBoundaryDistribution

        exact
          transferDistribution_at_other
            q
            i
            j
            k
            _
            hki
            hkj

      unfold StrictlyFree at hkMinus ⊢

      rw [hEq] at hkMinus

      exact hkMinus

/--
The positive perturbation loses at least one previously strictly-free
coordinate.
-/
theorem strictlyFreeCoordinates_plus_ne
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCoordinates
        lower
        upper
        (plusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      ≠
    strictlyFreeCoordinates
      lower
      upper
      q := by

  intro hEq

  have hHit :=
    plusBoundaryDistribution_hits_bound
      lower
      upper
      q
      i
      j
      hij

  rcases hHit with
    hUpper | hLower

  · have hiOld :
        i ∈
          strictlyFreeCoordinates
            lower
            upper
            q :=
      mem_strictlyFreeCoordinates_of_free
        hi

    have hiNew :
        i ∈
          strictlyFreeCoordinates
            lower
            upper
            (plusBoundaryDistribution
              lower
              upper
              q
              i
              j) := by

      rw [hEq]

      exact hiOld

    have hiFree :
        StrictlyFree
          lower
          upper
          (plusBoundaryDistribution
            lower
            upper
            q
            i
            j)
          i :=
      mem_strictlyFreeCoordinates.mp
        hiNew

    exact
      (not_strictlyFree_of_eq_upper
        hUpper)
        hiFree

  · have hjOld :
        j ∈
          strictlyFreeCoordinates
            lower
            upper
            q :=
      mem_strictlyFreeCoordinates_of_free
        hj

    have hjNew :
        j ∈
          strictlyFreeCoordinates
            lower
            upper
            (plusBoundaryDistribution
              lower
              upper
              q
              i
              j) := by

      rw [hEq]

      exact hjOld

    have hjFree :
        StrictlyFree
          lower
          upper
          (plusBoundaryDistribution
            lower
            upper
            q
            i
            j)
          j :=
      mem_strictlyFreeCoordinates.mp
        hjNew

    exact
      (not_strictlyFree_of_eq_lower
        hLower)
        hjFree

/--
The negative perturbation loses at least one previously strictly-free
coordinate.
-/
theorem strictlyFreeCoordinates_minus_ne
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCoordinates
        lower
        upper
        (minusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      ≠
    strictlyFreeCoordinates
      lower
      upper
      q := by

  intro hEq

  have hHit :=
    minusBoundaryDistribution_hits_bound
      lower
      upper
      q
      i
      j
      hij

  rcases hHit with
    hLower | hUpper

  · have hiOld :
        i ∈
          strictlyFreeCoordinates
            lower
            upper
            q :=
      mem_strictlyFreeCoordinates_of_free
        hi

    have hiNew :
        i ∈
          strictlyFreeCoordinates
            lower
            upper
            (minusBoundaryDistribution
              lower
              upper
              q
              i
              j) := by

      rw [hEq]

      exact hiOld

    have hiFree :
        StrictlyFree
          lower
          upper
          (minusBoundaryDistribution
            lower
            upper
            q
            i
            j)
          i :=
      mem_strictlyFreeCoordinates.mp
        hiNew

    exact
      (not_strictlyFree_of_eq_lower
        hLower)
        hiFree

  · have hjOld :
        j ∈
          strictlyFreeCoordinates
            lower
            upper
            q :=
      mem_strictlyFreeCoordinates_of_free
        hj

    have hjNew :
        j ∈
          strictlyFreeCoordinates
            lower
            upper
            (minusBoundaryDistribution
              lower
              upper
              q
              i
              j) := by

      rw [hEq]

      exact hjOld

    have hjFree :
        StrictlyFree
          lower
          upper
          (minusBoundaryDistribution
            lower
            upper
            q
            i
            j)
          j :=
      mem_strictlyFreeCoordinates.mp
        hjNew

    exact
      (not_strictlyFree_of_eq_upper
        hUpper)
        hjFree

/--
The positive boundary perturbation strictly reduces the number of
strictly-free coordinates.
-/
theorem strictlyFreeCount_plus_lt
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCount
        lower
        upper
        (plusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      <
    strictlyFreeCount
      lower
      upper
      q := by

  unfold strictlyFreeCount

  have hSubset :
      strictlyFreeCoordinates
          lower
          upper
          (plusBoundaryDistribution
            lower
            upper
            q
            i
            j)
        ⊆
      strictlyFreeCoordinates
        lower
        upper
        q :=
    strictlyFreeCoordinates_plus_subset
      hij
      hi
      hj

  have hNe :
      strictlyFreeCoordinates
          lower
          upper
          (plusBoundaryDistribution
            lower
            upper
            q
            i
            j)
        ≠
      strictlyFreeCoordinates
        lower
        upper
        q :=
    strictlyFreeCoordinates_plus_ne
      hij
      hi
      hj

  have hStrict :
      strictlyFreeCoordinates
          lower
          upper
          (plusBoundaryDistribution
            lower
            upper
            q
            i
            j)
        ⊂
      strictlyFreeCoordinates
        lower
        upper
        q := by

    exact
      Finset.ssubset_iff_subset_ne.mpr
        ⟨hSubset, hNe⟩

  exact
    Finset.card_lt_card
      hStrict

/--
The negative boundary perturbation strictly reduces the number of
strictly-free coordinates.
-/
theorem strictlyFreeCount_minus_lt
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCount
        lower
        upper
        (minusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      <
    strictlyFreeCount
      lower
      upper
      q := by

  unfold strictlyFreeCount

  have hSubset :
      strictlyFreeCoordinates
          lower
          upper
          (minusBoundaryDistribution
            lower
            upper
            q
            i
            j)
        ⊆
      strictlyFreeCoordinates
        lower
        upper
        q :=
    strictlyFreeCoordinates_minus_subset
      hij
      hi
      hj

  have hNe :
      strictlyFreeCoordinates
          lower
          upper
          (minusBoundaryDistribution
            lower
            upper
            q
            i
            j)
        ≠
      strictlyFreeCoordinates
        lower
        upper
        q :=
    strictlyFreeCoordinates_minus_ne
      hij
      hi
      hj

  have hStrict :
      strictlyFreeCoordinates
          lower
          upper
          (minusBoundaryDistribution
            lower
            upper
            q
            i
            j)
        ⊂
      strictlyFreeCoordinates
        lower
        upper
        q := by

    exact
      Finset.ssubset_iff_subset_ne.mpr
        ⟨hSubset, hNe⟩

  exact
    Finset.card_lt_card
      hStrict

/--
Both branches of the boundary decomposition strictly reduce the induction
measure.
-/
theorem boundaryDistributions_reduce_strictlyFreeCount
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hij :
      i ≠ j)
    (hi :
      StrictlyFree
        lower
        upper
        q
        i)
    (hj :
      StrictlyFree
        lower
        upper
        q
        j) :
    strictlyFreeCount
        lower
        upper
        (plusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      <
    strictlyFreeCount
        lower
        upper
        q
      ∧
    strictlyFreeCount
        lower
        upper
        (minusBoundaryDistribution
          lower
          upper
          q
          i
          j)
      <
    strictlyFreeCount
        lower
        upper
        q := by

  constructor

  · exact
      strictlyFreeCount_plus_lt
        hij
        hi
        hj

  · exact
      strictlyFreeCount_minus_lt
        hij
        hi
        hj

/--
Terminal geometry for the decomposition induction.

A point is terminal when at most one coordinate remains strictly inside its
interval. All other coordinates are therefore at interval boundaries,
provided the point lies in the confidence polytope.
-/
def AtMostOneStrictlyFree
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d) :
    Prop :=
  strictlyFreeCount
      lower
      upper
      q
    ≤ 1

end DROSafety.DRO
