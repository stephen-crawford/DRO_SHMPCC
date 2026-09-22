import DROSafety.DRO.BoxSimplexFreeCount
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Convex-hull generation of the box-simplex confidence polytope

Let

    C =
      { q in Delta_d :
          lower_i <= q_i <= upper_i }.

Define

    F(q) =
      #{ i : lower_i < q_i < upper_i }.

The boundary decomposition proved previously shows that whenever `F(q) > 1`,
there exist feasible points `qPlus` and `qMinus` such that

    q = theta qPlus + (1-theta) qMinus

with

    0 < theta < 1,

and

    F(qPlus) < F(q),
    F(qMinus) < F(q).

Strong induction on `F(q)` therefore proves that every feasible point lies
in the convex hull of feasible points having at most one strictly-free
coordinate.

Those terminal points are precisely the geometric form targeted by the
C++ box-simplex vertex enumeration: all but at most one coordinate are on
lower or upper bounds.
-/

namespace DROSafety.DRO

/--
Terminal feasible points for the box-simplex decomposition.

A point is terminal when it lies in the confidence polytope and has at
most one coordinate strictly inside its interval.
-/
noncomputable def TerminalConfidencePoints
    {d : ℕ}
    (lower upper : Fin d → ℝ) :
    Set (ModeDistribution d) :=
  {
    q |
      q ∈ ConfidencePolytope lower upper ∧
      AtMostOneStrictlyFree lower upper q
  }

/--
If a point is not terminal, then it has two distinct strictly-free
coordinates.
-/
theorem exists_two_strictlyFree_of_not_atMostOne
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    (hNotTerminal :
      ¬ AtMostOneStrictlyFree
          lower
          upper
          q) :
    ∃ i j : Fin d,
      i ≠ j ∧
      StrictlyFree lower upper q i ∧
      StrictlyFree lower upper q j := by

  classical

  have hCard :
      1 <
        (strictlyFreeCoordinates
          lower
          upper
          q).card := by

    unfold AtMostOneStrictlyFree at hNotTerminal
    unfold strictlyFreeCount at hNotTerminal

    omega

  rcases Finset.one_lt_card.mp hCard with
    ⟨i, hiMem, j, hjMem, hij⟩

  have hi :
      StrictlyFree
        lower
        upper
        q
        i :=
    mem_strictlyFreeCoordinates.mp
      hiMem

  have hj :
      StrictlyFree
        lower
        upper
        q
        j :=
    mem_strictlyFreeCoordinates.mp
      hjMem

  exact
    ⟨i, j, hij, hi, hj⟩

/--
Our `mixDistribution` is the usual affine combination in the function
vector space.
-/
theorem mixDistribution_eq_smul_add_smul
    {d : ℕ}
    (t : ℝ)
    (q₁ q₂ : ModeDistribution d) :
    mixDistribution t q₁ q₂
      =
    t • q₁ +
      (1 - t) • q₂ := by

  funext i

  simp [
    mixDistribution
  ]

/--
Strong-induction theorem.

Every point whose strictly-free count equals `n` lies in the convex hull
of terminal confidence-polytope points.
-/
theorem confidencePoint_mem_convexHull_terminal_by_count
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (hLower :
      ∀ i,
        0 ≤ lower i) :
    ∀ n : ℕ,
      ∀ q : ModeDistribution d,
        q ∈ ConfidencePolytope lower upper →
        strictlyFreeCount lower upper q = n →
        q ∈
          convexHull ℝ
            (TerminalConfidencePoints
              lower
              upper) := by

  classical

  intro n

  induction n using Nat.strong_induction_on with
  | h n ih =>

      intro q hq hCount

      by_cases hTerminal :
          AtMostOneStrictlyFree
            lower
            upper
            q

      · have hTerminalMem :
            q ∈
              TerminalConfidencePoints
                lower
                upper := by

          exact
            ⟨hq, hTerminal⟩

        exact
          (subset_convexHull
            ℝ
            (TerminalConfidencePoints
              lower
              upper))
            hTerminalMem

      · rcases
          exists_two_strictlyFree_of_not_atMostOne
            hTerminal
        with
          ⟨i, j, hij, hi, hj⟩

        have hPlusMem :
            plusBoundaryDistribution
                lower
                upper
                q
                i
                j
              ∈
            ConfidencePolytope
              lower
              upper :=
          plusBoundaryDistribution_mem_confidencePolytope
            hq
            hLower
            hij
            hi
            hj

        have hMinusMem :
            minusBoundaryDistribution
                lower
                upper
                q
                i
                j
              ∈
            ConfidencePolytope
              lower
              upper :=
          minusBoundaryDistribution_mem_confidencePolytope
            hq
            hLower
            hij
            hi
            hj

        have hPlusCountLt :
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
            n := by

          rw [← hCount]

          exact
            strictlyFreeCount_plus_lt
              hij
              hi
              hj

        have hMinusCountLt :
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
            n := by

          rw [← hCount]

          exact
            strictlyFreeCount_minus_lt
              hij
              hi
              hj

        have hPlusHull :
            plusBoundaryDistribution
                lower
                upper
                q
                i
                j
              ∈
            convexHull ℝ
              (TerminalConfidencePoints
                lower
                upper) := by

          exact
            ih
              (strictlyFreeCount
                lower
                upper
                (plusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j))
              hPlusCountLt
              (plusBoundaryDistribution
                lower
                upper
                q
                i
                j)
              hPlusMem
              rfl

        have hMinusHull :
            minusBoundaryDistribution
                lower
                upper
                q
                i
                j
              ∈
            convexHull ℝ
              (TerminalConfidencePoints
                lower
                upper) := by

          exact
            ih
              (strictlyFreeCount
                lower
                upper
                (minusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j))
              hMinusCountLt
              (minusBoundaryDistribution
                lower
                upper
                q
                i
                j)
              hMinusMem
              rfl

        let theta : ℝ :=
          boundaryMixWeight
            (plusBoundaryStep
              lower
              upper
              q
              i
              j)
            (minusBoundaryStep
              lower
              upper
              q
              i
              j)

        have hThetaBounds :
            0 < theta ∧
            theta < 1 := by

          dsimp [theta]

          exact
            confidencePoint_boundaryMixWeight_between
              hi
              hj

        have hThetaNonneg :
            0 ≤ theta :=
          le_of_lt
            hThetaBounds.1

        have hOneMinusThetaNonneg :
            0 ≤ 1 - theta := by

          exact
            sub_nonneg.mpr
              (le_of_lt
                hThetaBounds.2)

        have hWeightsSum :
            theta + (1 - theta) = 1 := by

          ring

        have hHullConvex :
            Convex ℝ
              (convexHull ℝ
                (TerminalConfidencePoints
                  lower
                  upper)) :=

          convex_convexHull
            ℝ
            (TerminalConfidencePoints
              lower
              upper)

        have hWeightedMem :
            theta •
                (plusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j)
              +
            (1 - theta) •
                (minusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j)
              ∈
            convexHull ℝ
              (TerminalConfidencePoints
                lower
                upper) := by

          have hConvexAdd :=
            (convex_iff_add_mem.mp
              hHullConvex)

          exact
            hConvexAdd
              hPlusHull
              hMinusHull
              hThetaNonneg
              hOneMinusThetaNonneg
              hWeightsSum

        have hMixMem :
            mixDistribution
                theta
                (plusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j)
                (minusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j)
              ∈
            convexHull ℝ
              (TerminalConfidencePoints
                lower
                upper) := by

          rw [
            mixDistribution_eq_smul_add_smul
          ]

          exact hWeightedMem

        have hQEq :
            q =
              mixDistribution
                theta
                (plusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j)
                (minusBoundaryDistribution
                  lower
                  upper
                  q
                  i
                  j) := by

          dsimp [theta]

          exact
            confidencePoint_eq_mix_boundaryDistributions
              hi
              hj

        rw [hQEq]

        exact hMixMem

/--
Main geometric theorem.

Every point in the box-simplex confidence polytope belongs to the convex
hull of feasible points having at most one strictly-free coordinate.
-/
theorem confidencePolytope_subset_convexHull_terminal
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (hLower :
      ∀ i,
        0 ≤ lower i) :
    ConfidencePolytope
        lower
        upper
      ⊆
    convexHull ℝ
      (TerminalConfidencePoints
        lower
        upper) := by

  intro q hq

  exact
    confidencePoint_mem_convexHull_terminal_by_count
      lower
      upper
      hLower
      (strictlyFreeCount
        lower
        upper
        q)
      q
      hq
      rfl

/--
The terminal points generate the complete confidence polytope in the sense
required by `VertexRadiusCover.lean`.
-/
theorem terminalPoints_generate_confidencePolytope
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (hLower :
      ∀ i,
        0 ≤ lower i) :
    VerticesGenerateConfidencePolytope
      (TerminalConfidencePoints
        lower
        upper)
      lower
      upper := by

  exact
    confidencePolytope_subset_convexHull_terminal
      lower
      upper
      hLower

end DROSafety.DRO
