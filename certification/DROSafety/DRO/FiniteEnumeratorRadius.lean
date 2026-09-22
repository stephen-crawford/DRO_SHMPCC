import DROSafety.DRO.EnumeratorRadiusCertificate
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Finite enumerator radius

The C++ calibration routine constructs a finite collection of feasible
box-simplex candidates and computes a transport cost for each candidate.
Its radius is the maximum of those candidate costs.

For the safety proof, we do not yet need to prove that each reported
transport cost is the *optimal* transport cost. It is sufficient to provide,
for every candidate, a feasible transport plan whose cost is no greater
than the reported candidate cost.

Therefore a finite certificate consists of:

* a finite candidate set;
* completeness with respect to `BoxSimplexEnumeratorCandidate`;
* one reported transport-cost upper bound per candidate;
* a feasible transport-plan witness achieving at most that bound.

We define

    rho = max (0, all reported candidate costs)

and prove automatically that this `rho` satisfies
`EnumeratorTransportRadiusCertificate`.

Combining with `EnumeratorRadiusCertificate.lean` then proves that the
entire Clopper-Pearson confidence polytope is contained in the finite
Wasserstein ambiguity ball of radius `rho`.
-/

namespace DROSafety.DRO

/--
Finite data corresponding to the output of the box-simplex enumeration and
transport computations.

`complete` states that the finite set contains exactly all abstract
enumerator candidates.

`candidateCost q` is the reported transport-cost upper bound for `q`.

`transportWitness` certifies that the reported value is safe: there exists
a feasible transport plan whose actual transport cost is no larger than the
reported value.
-/
structure FiniteEnumeratorTransportData
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ) where

  candidates :
    Finset (ModeDistribution d)

  complete :
    ∀ q : ModeDistribution d,
      BoxSimplexEnumeratorCandidate
          lower
          upper
          q
        ↔
      q ∈ candidates

  candidateCost :
    ModeDistribution d → ℝ

  transportWitness :
    ∀ q : ModeDistribution d,
      q ∈ candidates →
      ∃ π : TransportPlan center q,
        transportCost D π ≤ candidateCost q

/--
Finite set of values over which the certified radius is computed.

We explicitly insert zero. This makes the set nonempty and ensures that the
resulting radius is nonnegative even in degenerate cases.
-/
noncomputable def finiteEnumeratorCosts
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper) :
    Finset ℝ := by

  classical

  exact
    insert
      0
      (data.candidates.image
        data.candidateCost)

/--
The finite cost set is always nonempty because it contains zero.
-/
theorem finiteEnumeratorCosts_nonempty
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper) :
    (finiteEnumeratorCosts data).Nonempty := by

  classical

  unfold finiteEnumeratorCosts

  simp

/--
Certified finite-enumerator Wasserstein radius.

This is the maximum of zero and all reported candidate transport costs.
-/
noncomputable def finiteEnumeratorRadius
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper) :
    ℝ :=
  (finiteEnumeratorCosts data).max'
    (finiteEnumeratorCosts_nonempty data)

/--
Every reported candidate cost occurs in the finite cost set.
-/
theorem candidateCost_mem_finiteEnumeratorCosts
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    {q : ModeDistribution d}
    (hq :
      q ∈ data.candidates) :
    data.candidateCost q
      ∈
    finiteEnumeratorCosts data := by

  classical

  unfold finiteEnumeratorCosts

  apply Finset.mem_insert.mpr
  right

  apply Finset.mem_image.mpr

  exact
    ⟨q, hq, rfl⟩

/--
Every reported candidate cost is bounded above by the finite maximum.
-/
theorem candidateCost_le_finiteEnumeratorRadius
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    {q : ModeDistribution d}
    (hq :
      q ∈ data.candidates) :
    data.candidateCost q
      ≤
    finiteEnumeratorRadius data := by

  classical

  have hMem :
      data.candidateCost q
        ∈
      finiteEnumeratorCosts data :=
    candidateCost_mem_finiteEnumeratorCosts
      data
      hq

  exact
    Finset.le_max'
      (finiteEnumeratorCosts data)
      (data.candidateCost q)
      hMem

/--
The certified finite radius is nonnegative.
-/
theorem finiteEnumeratorRadius_nonneg
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper) :
    0 ≤ finiteEnumeratorRadius data := by

  classical

  have hZero :
      (0 : ℝ) ∈
        finiteEnumeratorCosts data := by

    unfold finiteEnumeratorCosts

    simp

  exact
    Finset.le_max'
      (finiteEnumeratorCosts data)
      0
      hZero

/--
Every abstract enumerator candidate belongs to the finite candidate set.
-/
theorem enumeratorCandidate_mem_finiteCandidates
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    {q : ModeDistribution d}
    (hq :
      BoxSimplexEnumeratorCandidate
        lower
        upper
        q) :
    q ∈ data.candidates := by

  exact
    (data.complete q).mp
      hq

/--
Every element of the finite candidate set is an abstract enumerator
candidate.
-/
theorem finiteCandidate_is_enumeratorCandidate
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    {q : ModeDistribution d}
    (hq :
      q ∈ data.candidates) :
    BoxSimplexEnumeratorCandidate
      lower
      upper
      q := by

  exact
    (data.complete q).mpr
      hq

/--
For every abstract enumerator candidate there exists a feasible transport
plan whose cost is bounded by the finite maximum radius.
-/
theorem finiteEnumeratorTransport_bound
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    {q : ModeDistribution d}
    (hq :
      BoxSimplexEnumeratorCandidate
        lower
        upper
        q) :
    ∃ π : TransportPlan center q,
      transportCost D π
        ≤
      finiteEnumeratorRadius data := by

  have hqFinite :
      q ∈ data.candidates :=
    enumeratorCandidate_mem_finiteCandidates
      data
      hq

  rcases
    data.transportWitness
      q
      hqFinite
  with
    ⟨π, hπ⟩

  have hCandidateCost :
      data.candidateCost q
        ≤
      finiteEnumeratorRadius data :=
    candidateCost_le_finiteEnumeratorRadius
      data
      hqFinite

  exact
    ⟨
      π,
      le_trans
        hπ
        hCandidateCost
    ⟩

/--
The finite maximum automatically produces the enumerator-radius certificate
required by `EnumeratorRadiusCertificate.lean`.
-/
theorem finiteEnumeratorRadius_certificate
    {d : ℕ}
    {D : FiniteGroundCost d}
    {center : ModeDistribution d}
    {lower upper : Fin d → ℝ}
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper) :
    EnumeratorTransportRadiusCertificate
      D
      center
      lower
      upper
      (finiteEnumeratorRadius data) := by

  intro q hq

  exact
    finiteEnumeratorTransport_bound
      data
      hq

/--
Main finite-radius coverage theorem.

If the finite enumeration is complete and every reported candidate cost has
a feasible transport witness, then the radius obtained by taking the maximum
of those costs covers the entire confidence polytope.
-/
theorem radiusCoversConfidencePolytope_of_finiteEnumeratorData
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    (hd :
      0 < d)
    (hLower :
      ∀ i,
        0 ≤ lower i) :
    RadiusCoversConfidencePolytope
      D
      center
      lower
      upper
      (finiteEnumeratorRadius data) := by

  have hCertificate :
      EnumeratorTransportRadiusCertificate
        D
        center
        lower
        upper
        (finiteEnumeratorRadius data) :=
    finiteEnumeratorRadius_certificate
      data

  exact
    radiusCoversConfidencePolytope_of_enumeratorCertificate
      D
      center
      lower
      upper
      (finiteEnumeratorRadius data)
      hd
      hLower
      hCertificate

/--
Pointwise form of the finite-radius coverage theorem.
-/
theorem confidencePoint_mem_ambiguity_of_finiteEnumeratorData
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (data :
      FiniteEnumeratorTransportData
        D
        center
        lower
        upper)
    (hd :
      0 < d)
    (hLower :
      ∀ i,
        0 ≤ lower i)
    {q : ModeDistribution d}
    (hq :
      q ∈ ConfidencePolytope
        lower
        upper) :
    q ∈
      FiniteWassersteinAmbiguity
        D
        center
        (finiteEnumeratorRadius data) := by

  have hCover :
      RadiusCoversConfidencePolytope
        D
        center
        lower
        upper
        (finiteEnumeratorRadius data) :=
    radiusCoversConfidencePolytope_of_finiteEnumeratorData
      D
      center
      lower
      upper
      data
      hd
      hLower

  exact
    hCover
      q
      hq

end DROSafety.DRO
