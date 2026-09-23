import DROSafety.DRO.IndependentModeHitsToBinomial
import Mathlib.Probability.Distributions.Bernoulli
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# IID categorical observations imply binomial mode counts

For a finite mode set `Fin d`, an i.i.d. categorical observation model
with parameter vector `pCoord` is specified by:

1. measurability of every observation;
2. mutual independence of the observations;
3. for every trial `t` and mode `i`,

       P[Y_t = i] = pCoord i.

For a fixed mode `i`, the event `Y_t = i` is therefore a Bernoulli
trial with success probability `pCoord i`.

The previous development proves that the number of such hits among the
first `m` independent observations has law

    Binomial(m, pCoord i).

Thus the binomial marginal required by the Clopper-Pearson calibration
is derived from the IID categorical observation model.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

noncomputable section

/--
An i.i.d. categorical mode-observation model.

`pCoord i` is the probability assigned to mode `i`.

We state the singleton probabilities using `ENNReal`, which is the
native codomain of a Lean `Measure`.  The assertion

    μ {ω | observation ω t = i}
      = ↑(unitInterval.toNNReal (pCoord i))

is exactly the statement

    P[Y_t = i] = pCoord i.
-/
structure IIDCategoricalModeObservations
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (observation : Ω → ℕ → Fin d)
    (pCoord : Fin d → unitInterval) :
    Prop where

  measurable :
    ∀ t : ℕ,
      Measurable
        (fun ω : Ω =>
          observation ω t)

  independent :
    iIndepFun
      (fun t : ℕ =>
        fun ω : Ω =>
          observation ω t)
      μ

  modeProbability :
    ∀ (t : ℕ) (i : Fin d),
      μ
        {
          ω : Ω |
            observation ω t = i
        }
        =
      (unitInterval.toNNReal (pCoord i) : ENNReal)

/--
The event that observation `t` realizes mode `i`.
-/
def categoricalModeEvent
    {Ω : Type*}
    {d : ℕ}
    (observation : Ω → ℕ → Fin d)
    (t : ℕ)
    (i : Fin d) :
    Set Ω :=
  {
    ω |
      observation ω t = i
  }

/--
A categorical mode event is measurable.
-/
theorem measurableSet_categoricalModeEvent
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (observation : Ω → ℕ → Fin d)
    (t : ℕ)
    (i : Fin d)
    (hMeas :
      Measurable
        (fun ω : Ω =>
          observation ω t)) :
    MeasurableSet
      (
        categoricalModeEvent
          observation
          t
          i
      ) := by

  unfold categoricalModeEvent

  exact
    MeasurableSet.preimage
      (MeasurableSet.singleton i)
      hMeas

/--
The probability of a mode event is exactly its corresponding
categorical coordinate.
-/
theorem categoricalModeEvent_probability
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (observation : Ω → ℕ → Fin d)
    (pCoord : Fin d → unitInterval)
    (hIID :
      IIDCategoricalModeObservations
        μ
        observation
        pCoord)
    (t : ℕ)
    (i : Fin d) :
    μ
        (
          categoricalModeEvent
            observation
            t
            i
        )
      =
    (unitInterval.toNNReal (pCoord i) : ENNReal) := by

  exact
    hIID.modeProbability
      t
      i

/--
The singleton `{False}` is the complement of the singleton `{True}`
in the measurable space `Prop`.
-/
theorem singleton_false_eq_compl_singleton_true :
    ({False} : Set Prop)
      =
    ({True} : Set Prop)ᶜ := by

  ext a

  by_cases ha : a

  · have haTrue :
        a = True := by

      apply propext

      constructor

      · intro _
        trivial

      · intro _
        exact ha

    rw [haTrue]

    simp

  · have haFalse :
        a = False := by

      apply propext

      constructor

      · intro h
        exact False.elim (ha h)

      · intro h
        exact False.elim h

    rw [haFalse]

    simp

/--
Under the IID categorical model, the proposition-valued random
variable

    Y_t = i

has Bernoulli law with success probability `pCoord i`.

This is proved directly from the two singleton masses on `Prop`.
No indicator convenience theorem is required.
-/
theorem categoricalModeHit_hasLaw_bernoulli
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (observation : Ω → ℕ → Fin d)
    (pCoord : Fin d → unitInterval)
    (hIID :
      IIDCategoricalModeObservations
        μ
        observation
        pCoord)
    (t : ℕ)
    (i : Fin d) :
    HasLaw
      (
        fun ω : Ω =>
          observation ω t = i
      )
      (
        ProbabilityTheory.bernoulliMeasure
          True
          False
          (pCoord i)
      )
      μ := by

  let f : Ω → Prop :=
    fun ω : Ω =>
      observation ω t = i

  have hfMeas :
      Measurable f := by

    dsimp [f]

    exact
      (hIID.measurable t).eq_const i

  have hTruePreimage :
      f ⁻¹' ({True} : Set Prop)
        =
      categoricalModeEvent
        observation
        t
        i := by

    ext ω

    simp [
      f,
      categoricalModeEvent
    ]

  have hTrueMass :
      (
        Measure.map
          f
          μ
      ) {True}
        =
      (
        ProbabilityTheory.bernoulliMeasure
          True
          False
          (pCoord i)
      ) {True} := by

    calc
      (
        Measure.map
          f
          μ
      ) {True}
          =
        μ
          (
            f ⁻¹'
              ({True} : Set Prop)
          ) := by

            exact
              Measure.map_apply
                hfMeas
                (MeasurableSet.singleton True)

      _ =
        μ
          (
            categoricalModeEvent
              observation
              t
              i
          ) := by

            rw [hTruePreimage]

      _ =
        (unitInterval.toNNReal (pCoord i) : ENNReal) := by

            exact
              categoricalModeEvent_probability
                μ
                observation
                pCoord
                hIID
                t
                i

      _ =
        (
          ProbabilityTheory.bernoulliMeasure
            True
            False
            (pCoord i)
        ) {True} := by

            symm

            rw [
              ProbabilityTheory.bernoulliMeasure_apply
                (pCoord i)
                (MeasurableSet.singleton True)
            ]

            simp

  refine
    {
      aemeasurable :=
        hfMeas.aemeasurable

      map_eq := ?_
    }

  apply
    Measure.ext_of_singleton

  intro a

  by_cases ha :
      a

  · have haTrue :
        a = True := by

      apply propext

      constructor

      · intro _
        trivial

      · intro _
        exact ha

    rw [haTrue]

    exact hTrueMass

  · have haFalse :
        a = False := by

      apply propext

      constructor

      · intro h
        exact False.elim (ha h)

      · intro h
        exact False.elim h

    rw [haFalse]

    rw [
      singleton_false_eq_compl_singleton_true
    ]

    calc
      (
        Measure.map
          f
          μ
      )
        (({True} : Set Prop)ᶜ)
          =
        1 -
          (
            Measure.map
              f
              μ
          ) {True} := by

            exact
              MeasureTheory.prob_compl_eq_one_sub
                (MeasurableSet.singleton True)

      _ =
        1 -
          (
            ProbabilityTheory.bernoulliMeasure
              True
              False
              (pCoord i)
          ) {True} := by

            rw [hTrueMass]

      _ =
        (
          ProbabilityTheory.bernoulliMeasure
            True
            False
            (pCoord i)
        )
          (({True} : Set Prop)ᶜ) := by

            symm

            exact
              MeasureTheory.prob_compl_eq_one_sub
                (MeasurableSet.singleton True)

/--
Main categorical-to-binomial result.

For every mode `i`, the number of occurrences of `i` among the first
`m` IID categorical observations has the exact binomial marginal

    Binomial(m, pCoord i).
-/
theorem categoricalModeCount_hasLaw_binomial_of_iidCategorical
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (pCoord : Fin d → unitInterval)
    (hIID :
      IIDCategoricalModeObservations
        μ
        observation
        pCoord)
    (i : Fin d) :
    HasLaw
      (
        categoricalModeCount
          m
          observation
          i
      )
      (
        ProbabilityTheory.binomial
          m
          (pCoord i)
      )
      μ := by

  apply
    categoricalModeCount_hasLaw_binomial_of_iIndep
      μ
      m
      observation
      i
      (pCoord i)

  · exact
      hIID.measurable

  · exact
      hIID.independent

  · intro t

    exact
      categoricalModeHit_hasLaw_bernoulli
        μ
        observation
        pCoord
        hIID
        t
        i

/--
All coordinate counts simultaneously have their required binomial
marginal laws.

No independence between the counts themselves is asserted or needed.

Indeed, the counts are generally dependent because

    sum_i C_i = m.

Bonferroni requires only the individual marginal confidence bounds, so
this dependence creates no problem for the calibration theorem.
-/
theorem categoricalModeCounts_haveLaw_binomial_of_iidCategorical
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (pCoord : Fin d → unitInterval)
    (hIID :
      IIDCategoricalModeObservations
        μ
        observation
        pCoord) :
    ∀ i : Fin d,
      HasLaw
        (
          categoricalModeCount
            m
            observation
            i
        )
        (
          ProbabilityTheory.binomial
            m
            (pCoord i)
        )
        μ := by

  intro i

  exact
    categoricalModeCount_hasLaw_binomial_of_iidCategorical
      μ
      m
      observation
      pCoord
      hIID
      i

end

end DROSafety.DRO
