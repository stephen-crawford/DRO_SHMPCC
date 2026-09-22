import DROSafety.DRO.VertexRadiusCover
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Geometry of box-simplex confidence polytopes

The Clopper-Pearson confidence polytope has the form

    C =
      { q in Delta_d :
          lower_i <= q_i <= upper_i }.

The C++ calibration routine enumerates candidates by choosing one coordinate
as free and placing every other coordinate at either its lower or upper bound.

To justify that enumeration, we will prove constructively that any feasible
point having at least two coordinates strictly inside their bounds can be
split into feasible points closer to the boundary.

The first step is purely scalar:

if coordinates `i` and `j` are both strictly inside their intervals, then
probability mass can be moved from `j` to `i` until either

    q_i = upper_i

or

    q_j = lower_j.

Similarly, mass can be moved in the opposite direction until either

    q_i = lower_i

or

    q_j = upper_j.

These two boundary-reaching perturbations will later be used to express an
interior point as a convex combination of two feasible points with more
active box constraints.
-/

namespace DROSafety.DRO

/--
Coordinate `i` is strictly inside its interval.
-/
def StrictlyFree
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i : Fin d) :
    Prop :=
  lower i < q i ∧
  q i < upper i

/--
Maximum positive step that can move mass from coordinate `j` to coordinate
`i` without violating either

    q_i <= upper_i

or

    lower_j <= q_j.
-/
def plusBoundaryStep
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    ℝ :=
  min
    (upper i - q i)
    (q j - lower j)

/--
Maximum positive step that can move mass from coordinate `i` to coordinate
`j` without violating either

    lower_i <= q_i

or

    q_j <= upper_j.
-/
def minusBoundaryStep
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    ℝ :=
  min
    (q i - lower i)
    (upper j - q j)

/--
If `i` and `j` are both strictly interior, the positive-direction
boundary step is strictly positive.
-/
theorem plusBoundaryStep_pos
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hi :
      StrictlyFree lower upper q i)
    (hj :
      StrictlyFree lower upper q j) :
    0 <
      plusBoundaryStep
        lower
        upper
        q
        i
        j := by

  unfold plusBoundaryStep
  unfold StrictlyFree at hi hj

  apply lt_min

  · linarith [hi.2]

  · linarith [hj.1]

/--
If `i` and `j` are both strictly interior, the negative-direction
boundary step is strictly positive.
-/
theorem minusBoundaryStep_pos
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hi :
      StrictlyFree lower upper q i)
    (hj :
      StrictlyFree lower upper q j) :
    0 <
      minusBoundaryStep
        lower
        upper
        q
        i
        j := by

  unfold minusBoundaryStep
  unfold StrictlyFree at hi hj

  apply lt_min

  · linarith [hi.1]

  · linarith [hj.2]

/--
The positive-direction step is no larger than the remaining upper slack
at coordinate `i`.
-/
theorem plusBoundaryStep_le_upperSlack
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    plusBoundaryStep lower upper q i j
      ≤
    upper i - q i := by

  unfold plusBoundaryStep

  exact min_le_left _ _

/--
The positive-direction step is no larger than the remaining lower slack
at coordinate `j`.
-/
theorem plusBoundaryStep_le_lowerSlack
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    plusBoundaryStep lower upper q i j
      ≤
    q j - lower j := by

  unfold plusBoundaryStep

  exact min_le_right _ _

/--
The negative-direction step is no larger than the remaining lower slack
at coordinate `i`.
-/
theorem minusBoundaryStep_le_lowerSlack
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    minusBoundaryStep lower upper q i j
      ≤
    q i - lower i := by

  unfold minusBoundaryStep

  exact min_le_left _ _

/--
The negative-direction step is no larger than the remaining upper slack
at coordinate `j`.
-/
theorem minusBoundaryStep_le_upperSlack
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    minusBoundaryStep lower upper q i j
      ≤
    upper j - q j := by

  unfold minusBoundaryStep

  exact min_le_right _ _

/--
After taking the full positive boundary step, at least one of the two
affected coordinates reaches a box constraint:

* coordinate `i` reaches its upper bound, or
* coordinate `j` reaches its lower bound.
-/
theorem plusBoundaryStep_hits_bound
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    q i +
          plusBoundaryStep
            lower
            upper
            q
            i
            j
        =
        upper i
      ∨
    q j -
          plusBoundaryStep
            lower
            upper
            q
            i
            j
        =
        lower j := by

  unfold plusBoundaryStep

  by_cases h :
      upper i - q i
        ≤
      q j - lower j

  · left

    rw [min_eq_left h]

    ring

  · right

    have hReverse :
        q j - lower j
          ≤
        upper i - q i := by

      exact
        le_of_lt
          (lt_of_not_ge h)

    rw [min_eq_right hReverse]

    ring

/--
After taking the full negative boundary step, at least one of the two
affected coordinates reaches a box constraint:

* coordinate `i` reaches its lower bound, or
* coordinate `j` reaches its upper bound.
-/
theorem minusBoundaryStep_hits_bound
    {d : ℕ}
    (lower upper : Fin d → ℝ)
    (q : ModeDistribution d)
    (i j : Fin d) :
    q i -
          minusBoundaryStep
            lower
            upper
            q
            i
            j
        =
        lower i
      ∨
    q j +
          minusBoundaryStep
            lower
            upper
            q
            i
            j
        =
        upper j := by

  unfold minusBoundaryStep

  by_cases h :
      q i - lower i
        ≤
      upper j - q j

  · left

    rw [min_eq_left h]

    ring

  · right

    have hReverse :
        upper j - q j
          ≤
        q i - lower i := by

      exact
        le_of_lt
          (lt_of_not_ge h)

    rw [min_eq_right hReverse]

    ring

/--
Convenient combined statement for two strictly-free coordinates.

There is a strictly positive feasible step in both mass-transfer directions,
and each maximal step reaches at least one interval bound.
-/
theorem two_strictlyFree_have_boundary_steps
    {d : ℕ}
    {lower upper : Fin d → ℝ}
    {q : ModeDistribution d}
    {i j : Fin d}
    (hi :
      StrictlyFree lower upper q i)
    (hj :
      StrictlyFree lower upper q j) :
    0 <
        plusBoundaryStep
          lower
          upper
          q
          i
          j
      ∧
    0 <
        minusBoundaryStep
          lower
          upper
          q
          i
          j
      ∧
    (
      q i +
            plusBoundaryStep
              lower
              upper
              q
              i
              j
          =
          upper i
        ∨
      q j -
            plusBoundaryStep
              lower
              upper
              q
              i
              j
          =
          lower j
    )
      ∧
    (
      q i -
            minusBoundaryStep
              lower
              upper
              q
              i
              j
          =
          lower i
        ∨
      q j +
            minusBoundaryStep
              lower
              upper
              q
              i
              j
          =
          upper j
    ) := by

  constructor

  · exact
      plusBoundaryStep_pos
        hi
        hj

  constructor

  · exact
      minusBoundaryStep_pos
        hi
        hj

  constructor

  · exact
      plusBoundaryStep_hits_bound
        lower
        upper
        q
        i
        j

  · exact
      minusBoundaryStep_hits_bound
        lower
        upper
        q
        i
        j

end DROSafety.DRO
