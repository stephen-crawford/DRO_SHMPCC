import DROSafety.DRO.BoxSimplexVertexGeometry
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Boundary decomposition for box-simplex confidence polytopes

Suppose `q` lies in the confidence polytope

    q in Delta_d,
    lower_i <= q_i <= upper_i,

and two distinct coordinates `i` and `j` are strictly inside their bounds.

We construct two feasible points:

* `qPlus`, obtained by moving probability mass from `j` to `i`;
* `qMinus`, obtained by moving probability mass from `i` to `j`.

The step sizes are maximal, so each new point activates at least one
additional box constraint.

We then prove that the original point is a strict convex combination

    q = theta * qPlus + (1-theta) * qMinus

with

    0 < theta < 1.

This is the key decomposition used later to show that every point of the
box-simplex confidence polytope belongs to the convex hull of the
C++-enumerated boundary points.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
Direction that transfers one unit of probability mass from coordinate `j`
to coordinate `i`.
-/
def transferDirection
    {d : ℕ}
    (i j : Fin d)
    (k : Fin d) :
    ℝ :=
  if k = i then
    1
  else if k = j then
    -1
  else
    0

/--
Transfer `delta` units of probability mass from `j` to `i`.

Positive `delta` moves mass from `j` to `i`.
Negative `delta` moves mass in the opposite direction.
-/
def transferDistribution
    {d : ℕ}
    (q : ModeDistribution d)
    (i j : Fin d)
    (delta : ℝ) :
    ModeDistribution d :=
  fun k =>
    q k +
      delta * transferDirection i j k

/--
At coordinate `i`, a transfer increases the value by `delta`.
-/
theorem transferDistribution_at_i
    {d : ℕ}
    (q : ModeDistribution d)
    (i j : Fin d)
    (delta : ℝ)
    (_hij : i ≠ j) :
    transferDistribution q i j delta i
      =
    q i + delta := by

  simp [
    transferDistribution,
    transferDirection
  ]

/--
At coordinate `j`, the same transfer decreases the value by `delta`.
-/
theorem transferDistribution_at_j
    {d : ℕ}
    (q : ModeDistribution d)
    (i j : Fin d)
    (delta : ℝ)
    (hij : i ≠ j) :
    transferDistribution q i j delta j
      =
    q j - delta := by

  have hji :
      j ≠ i :=
    Ne.symm hij

  simp [
    transferDistribution,
    transferDirection,
    hji,
    sub_eq_add_neg
  ]

/--
Every coordinate other than `i` and `j` is unchanged.
-/
theorem transferDistribution_at_other
    {d : ℕ}
    (q : ModeDistribution d)
    (i j k : Fin d)
    (delta : ℝ)
    (hki : k ≠ i)
    (hkj : k ≠ j) :
    transferDistribution q i j delta k
      =
    q k := by

  simp [
    transferDistribution,
    transferDirection,
    hki,
    hkj
  ]

/--
The transfer direction has total sum zero.
-/
theorem sum_transferDirection
    {d : ℕ}
    (i j : Fin d)
    (hij : i ≠ j) :
    (∑ k, transferDirection i j k) = 0 := by

  classical

  have hPointwise :
      ∀ k : Fin d,
        transferDirection i j k
          =
        (if k = i then (1 : ℝ) else 0)
          +
        (if k = j then (-1 : ℝ) else 0) := by

    intro k

    by_cases hki : k = i

    · subst k

      simp [
        transferDirection,
        hij
      ]

    · by_cases hkj : k = j

      · subst k

        have hji :
            j ≠ i :=
          Ne.symm hij

        simp [
          transferDirection,
          hji
        ]

      · simp [
          transferDirection,
          hki,
          hkj
        ]

  calc
    (∑ k, transferDirection i j k)
        =
      ∑ k,
        (
          (if k = i then (1 : ℝ) else 0)
            +
          (if k = j then (-1 : ℝ) else 0)
        ) := by

      apply Finset.sum_congr rfl

      intro k hk

      exact hPointwise k

    _ =
      (∑ k,
        if k = i then (1 : ℝ) else 0)
        +
      (∑ k,
        if k = j then (-1 : ℝ) else 0) := by

      rw [Finset.sum_add_distrib]

    _ =
      1 + (-1) := by
      simp

    _ = 0 := by
      norm_num

/--
Transferring mass between two distinct coordinates preserves total mass.
-/
theorem transferDistribution_sum
    {d : ℕ}
    (q : ModeDistribution d)
    (i j : Fin d)
    (delta : ℝ)
    (hij : i ≠ j) :
    (∑ k, transferDistribution q i j delta k)
      =
    ∑ k, q k := by

  classical

  unfold transferDistribution

  calc
    (∑ k,
      (q k +
        delta * transferDirection i j k))
        =
      (∑ k, q k) +
        ∑ k,
          delta * transferDirection i j k := by

      rw [Finset.sum_add_distrib]

    _ =
      (∑ k, q k) +
        delta *
          (∑ k, transferDirection i j k) := by

      rw [Finset.mul_sum]

    _ =
      ∑ k, q k := by

      rw [
        sum_transferDirection
          i
          j
          hij
      ]

      ring

/--
Positive-boundary perturbation.
-/
def plusBoundaryDistribution
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    ModeDistribution d :=
  transferDistribution
    q
    i
    j
    (plusBoundaryStep
      lower
      upper
      q
      i
      j)

/--
Negative-boundary perturbation.
-/
def minusBoundaryDistribution
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    ModeDistribution d :=
  transferDistribution
    q
    i
    j
    (-
      minusBoundaryStep
        lower
        upper
        q
        i
        j)

/--
The positive perturbation stays inside the confidence polytope.

The assumption `0 <= lower k` matches the Clopper-Pearson construction,
whose interval endpoints are probabilities.
-/
theorem plusBoundaryDistribution_mem_confidencePolytope
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hq :
      q ∈
        ConfidencePolytope
          lower
          upper)
    (hLower :
      ∀ k,
        0 ≤ lower k)
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
    plusBoundaryDistribution
        lower
        upper
        q
        i
        j
      ∈
    ConfidencePolytope
      lower
      upper := by

  have hDeltaPos :
      0 <
        plusBoundaryStep
          lower
          upper
          q
          i
          j :=
    plusBoundaryStep_pos
      hi
      hj

  have hDeltaNonneg :
      0 ≤
        plusBoundaryStep
          lower
          upper
          q
          i
          j :=
    le_of_lt hDeltaPos

  constructor

  · constructor

    · intro k

      by_cases hki : k = i

      · subst k

        rw [
          plusBoundaryDistribution,
          transferDistribution_at_i
            q
            i
            j
            _
            hij
        ]

        exact
          add_nonneg
            (hq.1.1 i)
            hDeltaNonneg

      · by_cases hkj : k = j

        · subst k

          rw [
            plusBoundaryDistribution,
            transferDistribution_at_j
              q
              i
              j
              _
              hij
          ]

          have hStep :
              plusBoundaryStep
                  lower
                  upper
                  q
                  i
                  j
                ≤
              q j - lower j :=
            plusBoundaryStep_le_lowerSlack
              lower
              upper
              q
              i
              j

          have hLowerJ :
              0 ≤ lower j :=
            hLower j

          linarith

        · rw [
            plusBoundaryDistribution,
            transferDistribution_at_other
              q
              i
              j
              k
              _
              hki
              hkj
          ]

          exact hq.1.1 k

    · rw [
        plusBoundaryDistribution,
        transferDistribution_sum
          q
          i
          j
          _
          hij,
        hq.1.2
      ]

  · intro k

    by_cases hki : k = i

    · subst k

      rw [
        plusBoundaryDistribution,
        transferDistribution_at_i
          q
          i
          j
          _
          hij
      ]

      constructor

      · have hOldLower :
            lower i ≤ q i :=
          (hq.2 i).1

        linarith

      · have hStep :
            plusBoundaryStep
                lower
                upper
                q
                i
                j
              ≤
            upper i - q i :=
          plusBoundaryStep_le_upperSlack
            lower
            upper
            q
            i
            j

        linarith

    · by_cases hkj : k = j

      · subst k

        rw [
          plusBoundaryDistribution,
          transferDistribution_at_j
            q
            i
            j
            _
            hij
        ]

        constructor

        · have hStep :
              plusBoundaryStep
                  lower
                  upper
                  q
                  i
                  j
                ≤
              q j - lower j :=
            plusBoundaryStep_le_lowerSlack
              lower
              upper
              q
              i
              j

          linarith

        · have hOldUpper :
              q j ≤ upper j :=
            (hq.2 j).2

          linarith

      · rw [
          plusBoundaryDistribution,
          transferDistribution_at_other
            q
            i
            j
            k
            _
            hki
            hkj
        ]

        exact hq.2 k

/--
The negative perturbation stays inside the confidence polytope.
-/
theorem minusBoundaryDistribution_mem_confidencePolytope
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hq :
      q ∈
        ConfidencePolytope
          lower
          upper)
    (hLower :
      ∀ k,
        0 ≤ lower k)
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
    minusBoundaryDistribution
        lower
        upper
        q
        i
        j
      ∈
    ConfidencePolytope
      lower
      upper := by

  have hDeltaPos :
      0 <
        minusBoundaryStep
          lower
          upper
          q
          i
          j :=
    minusBoundaryStep_pos
      hi
      hj

  have hDeltaNonneg :
      0 ≤
        minusBoundaryStep
          lower
          upper
          q
          i
          j :=
    le_of_lt hDeltaPos

  constructor

  · constructor

    · intro k

      by_cases hki : k = i

      · subst k

        rw [
          minusBoundaryDistribution,
          transferDistribution_at_i
            q
            i
            j
            _
            hij
        ]

        have hStep :
            minusBoundaryStep
                lower
                upper
                q
                i
                j
              ≤
            q i - lower i :=
          minusBoundaryStep_le_lowerSlack
            lower
            upper
            q
            i
            j

        have hLowerI :
            0 ≤ lower i :=
          hLower i

        linarith

      · by_cases hkj : k = j

        · subst k

          rw [
            minusBoundaryDistribution,
            transferDistribution_at_j
              q
              i
              j
              _
              hij
          ]

          have hQNonneg :
              0 ≤ q j :=
            hq.1.1 j

          linarith

        · rw [
            minusBoundaryDistribution,
            transferDistribution_at_other
              q
              i
              j
              k
              _
              hki
              hkj
          ]

          exact hq.1.1 k

    · rw [
        minusBoundaryDistribution,
        transferDistribution_sum
          q
          i
          j
          _
          hij,
        hq.1.2
      ]

  · intro k

    by_cases hki : k = i

    · subst k

      rw [
        minusBoundaryDistribution,
        transferDistribution_at_i
          q
          i
          j
          _
          hij
      ]

      constructor

      · have hStep :
            minusBoundaryStep
                lower
                upper
                q
                i
                j
              ≤
            q i - lower i :=
          minusBoundaryStep_le_lowerSlack
            lower
            upper
            q
            i
            j

        linarith

      · have hOldUpper :
            q i ≤ upper i :=
          (hq.2 i).2

        linarith

    · by_cases hkj : k = j

      · subst k

        rw [
          minusBoundaryDistribution,
          transferDistribution_at_j
            q
            i
            j
            _
            hij
        ]

        constructor

        · have hOldLower :
              lower j ≤ q j :=
            (hq.2 j).1

          linarith

        · have hStep :
              minusBoundaryStep
                  lower
                  upper
                  q
                  i
                  j
                ≤
              upper j - q j :=
            minusBoundaryStep_le_upperSlack
              lower
              upper
              q
              i
              j

          linarith

      · rw [
          minusBoundaryDistribution,
          transferDistribution_at_other
            q
            i
            j
            k
            _
            hki
            hkj
        ]

        exact hq.2 k

/--
Convex-combination weight used to reconstruct the original point.

For positive boundary distances `deltaPlus` and `deltaMinus`,

    theta = deltaMinus / (deltaPlus + deltaMinus).
-/
noncomputable def boundaryMixWeight
    (deltaPlus deltaMinus : ℝ) :
    ℝ :=
  deltaMinus /
    (deltaPlus + deltaMinus)

/--
The reconstruction weight is strictly positive.
-/
theorem boundaryMixWeight_pos
    {deltaPlus deltaMinus : ℝ}
    (hPlus :
      0 < deltaPlus)
    (hMinus :
      0 < deltaMinus) :
    0 <
      boundaryMixWeight
        deltaPlus
        deltaMinus := by

  unfold boundaryMixWeight

  exact
    div_pos
      hMinus
      (add_pos hPlus hMinus)

/--
The reconstruction weight is strictly less than one.
-/
theorem boundaryMixWeight_lt_one
    {deltaPlus deltaMinus : ℝ}
    (hPlus :
      0 < deltaPlus)
    (hMinus :
      0 < deltaMinus) :
    boundaryMixWeight
        deltaPlus
        deltaMinus
      < 1 := by

  unfold boundaryMixWeight

  have hDen :
      0 <
        deltaPlus + deltaMinus :=
    add_pos hPlus hMinus

  apply (div_lt_one hDen).2

  linarith

/--
The chosen convex weight exactly balances the two perturbation distances.
-/
theorem boundaryMixWeight_balance
    {deltaPlus deltaMinus : ℝ}
    (hPlus :
      0 < deltaPlus)
    (hMinus :
      0 < deltaMinus) :
    boundaryMixWeight
          deltaPlus
          deltaMinus
        * deltaPlus
      =
    (1 -
        boundaryMixWeight
          deltaPlus
          deltaMinus)
      * deltaMinus := by

  unfold boundaryMixWeight

  have hDen :
      deltaPlus + deltaMinus ≠ 0 :=
    ne_of_gt
      (add_pos hPlus hMinus)

  field_simp [hDen]

  ring

/--
If the weighted positive and negative transfer distances balance, mixing
the two perturbed distributions recovers the original distribution.
-/
theorem mix_transferDistributions_eq_original
    {d : ℕ}
    (q : ModeDistribution d)
    (i j : Fin d)
    (deltaPlus deltaMinus theta : ℝ)
    (hBalance :
      theta * deltaPlus
        =
      (1 - theta) * deltaMinus) :
    mixDistribution
        theta
        (transferDistribution
          q
          i
          j
          deltaPlus)
        (transferDistribution
          q
          i
          j
          (-deltaMinus))
      =
    q := by

  funext k

  unfold mixDistribution
  unfold transferDistribution

  calc
    theta *
          (q k +
            deltaPlus *
              transferDirection i j k)
        +
        (1 - theta) *
          (q k +
            (-deltaMinus) *
              transferDirection i j k)
        =
      q k +
        (
          theta * deltaPlus -
          (1 - theta) * deltaMinus
        ) *
        transferDirection i j k := by

      ring

    _ =
      q k := by

      rw [hBalance]

      ring

/--
The original confidence-polytope point is a strict convex combination of
its two maximal boundary perturbations.
-/
theorem confidencePoint_eq_mix_boundaryDistributions
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
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
    let deltaPlus :=
      plusBoundaryStep
        lower
        upper
        q
        i
        j

    let deltaMinus :=
      minusBoundaryStep
        lower
        upper
        q
        i
        j

    let theta :=
      boundaryMixWeight
        deltaPlus
        deltaMinus

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

  dsimp only

  have hPlus :
      0 <
        plusBoundaryStep
          lower
          upper
          q
          i
          j :=
    plusBoundaryStep_pos
      hi
      hj

  have hMinus :
      0 <
        minusBoundaryStep
          lower
          upper
          q
          i
          j :=
    minusBoundaryStep_pos
      hi
      hj

  have hBalance :
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
          *
          plusBoundaryStep
            lower
            upper
            q
            i
            j
        =
      (1 -
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
              j))
        *
        minusBoundaryStep
          lower
          upper
          q
          i
          j :=
    boundaryMixWeight_balance
      hPlus
      hMinus

  symm

  exact
    mix_transferDistributions_eq_original
      q
      i
      j
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
      (boundaryMixWeight
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
          j))
      hBalance

/--
The mixing coefficient used in the boundary decomposition is genuinely a
strict convex-combination coefficient.
-/
theorem confidencePoint_boundaryMixWeight_between
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
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
    0 <
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
      ∧
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
        < 1 := by

  have hPlus :
      0 <
        plusBoundaryStep
          lower
          upper
          q
          i
          j :=
    plusBoundaryStep_pos
      hi
      hj

  have hMinus :
      0 <
        minusBoundaryStep
          lower
          upper
          q
          i
          j :=
    minusBoundaryStep_pos
      hi
      hj

  exact
    ⟨
      boundaryMixWeight_pos
        hPlus
        hMinus,
      boundaryMixWeight_lt_one
        hPlus
        hMinus
    ⟩

/--
The positive boundary perturbation activates at least one of the two
coordinates.
-/
theorem plusBoundaryDistribution_hits_bound
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d)
    (hij : i ≠ j) :
    plusBoundaryDistribution
          lower
          upper
          q
          i
          j
          i
        =
        upper i
      ∨
    plusBoundaryDistribution
          lower
          upper
          q
          i
          j
          j
        =
        lower j := by

  rw [
    plusBoundaryDistribution,
    transferDistribution_at_i
      q i j _ hij,
    transferDistribution_at_j
      q i j _ hij
  ]

  exact
    plusBoundaryStep_hits_bound
      lower
      upper
      q
      i
      j

/--
The negative boundary perturbation activates at least one of the two
coordinates.
-/
theorem minusBoundaryDistribution_hits_bound
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d)
    (hij : i ≠ j) :
    minusBoundaryDistribution
          lower
          upper
          q
          i
          j
          i
        =
        lower i
      ∨
    minusBoundaryDistribution
          lower
          upper
          q
          i
          j
          j
        =
        upper j := by

  rw [
    minusBoundaryDistribution,
    transferDistribution_at_i
      q i j _ hij,
    transferDistribution_at_j
      q i j _ hij
  ]

  simpa [sub_eq_add_neg] using
    (minusBoundaryStep_hits_bound
      lower
      upper
      q
      i
      j)

end DROSafety.DRO
