import DROSafety.DRO.SetBernoulliBinomialCount
import Mathlib.Probability.Independence.InfinitePi
import Mathlib.Probability.Distributions.Bernoulli
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Independent categorical-mode hits give binomial counts

Fix one mode `i`.

Suppose the mode observations

    Y_t : Ω → Fin d

are mutually independent and, for every trial `t`, the hit indicator

    Y_t = i

has Bernoulli law with success probability `p`.

For the first `m` observations, define the hit set

    { t | t < m ∧ Y_t = i }.

This file proves that this random set has law

    setBernoulli (Set.Iio m) p,

and therefore its cardinality has law

    binomial m p.

The only remaining step after this file is to derive the Bernoulli
hit-law premise from a common categorical marginal distribution.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

noncomputable section

/--
The Prop-valued hit process for mode `i`, truncated to the first `m`
trials.
-/
def truncatedModeHit
    {Ω : Type*}
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (t : ℕ)
    (ω : Ω) :
    Prop :=
  t < m ∧ observation ω t = i

/--
The truncated hit process is mutually independent whenever the
underlying mode observations are mutually independent.

Independence is preserved by applying, separately at each time index,
the measurable map

    y ↦ (t < m ∧ y = i).
-/
theorem iIndepFun_truncatedModeHit
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (hIndep :
      iIndepFun
        (fun t : ℕ =>
          fun ω : Ω =>
            observation ω t)
        μ) :
    iIndepFun
      (fun t : ℕ =>
        fun ω : Ω =>
          truncatedModeHit
            m
            observation
            i
            t
            ω)
      μ := by

  have hComp :=
    hIndep.comp
      (
        fun t : ℕ =>
          fun y : Fin d =>
            t < m ∧ y = i
      )
      (by
        intro t

        exact
          measurable_const.and
            (
              measurable_id.eq_const
                i
            ))

  simpa [
    Function.comp_def,
    truncatedModeHit
  ] using hComp

/--
Measurability of every coordinate of the truncated hit process.
-/
theorem measurable_truncatedModeHit
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (hMeas :
      ∀ t : ℕ,
        Measurable
          (fun ω : Ω =>
            observation ω t)) :
    ∀ t : ℕ,
      Measurable
        (
          fun ω : Ω =>
            truncatedModeHit
              m
              observation
              i
              t
              ω
        ) := by

  intro t

  exact
    measurable_const.and
      (
        (hMeas t).eq_const
          i
      )

/--
Outside the first `m` trials, the truncated hit process is constantly
false and therefore has Dirac law at `False`.
-/
theorem hasLaw_truncatedModeHit_of_not_lt
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (t : ℕ)
    (ht :
      ¬ t < m) :
    HasLaw
      (
        fun ω : Ω =>
          truncatedModeHit
            m
            observation
            i
            t
            ω
      )
      (Measure.dirac False)
      μ := by

  rw [
    ProbabilityTheory.hasLaw_dirac_iff
  ]

  filter_upwards with ω

  simp [
    truncatedModeHit,
    ht
  ]

/--
Each coordinate of the truncated hit process has the exact Bernoulli
law required by `setBernoulli (Iio m) p`.

Inside `Iio m`, the law is `Ber(True, False, p)`.
Outside `Iio m`, both Bernoulli atoms are `False`, so the law reduces
to `dirac False`.
-/
theorem hasLaw_truncatedModeHit
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (p : unitInterval)
    (hHitLaw :
      ∀ t : ℕ,
        HasLaw
          (
            fun ω : Ω =>
              observation ω t = i
          )
          (
            ProbabilityTheory.bernoulliMeasure
              True
              False
              p
          )
          μ) :
    ∀ t : ℕ,
      HasLaw
        (
          fun ω : Ω =>
            truncatedModeHit
              m
              observation
              i
              t
              ω
        )
        (
          ProbabilityTheory.bernoulliMeasure
            (t ∈ Set.Iio m)
            False
            p
        )
        μ := by

  intro t

  by_cases ht :
      t < m

  · have h :=
      hHitLaw t

    simpa [
      truncatedModeHit,
      Set.mem_Iio,
      ht
    ] using h

  · have hFalse :
        HasLaw
          (
            fun ω : Ω =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
          )
          (Measure.dirac False)
          μ :=
      hasLaw_truncatedModeHit_of_not_lt
        μ
        m
        observation
        i
        t
        ht

    simpa [
      Set.mem_Iio,
      ht,
      ProbabilityTheory.bernoulliMeasure_self_eq_dirac
    ] using hFalse

/--
The entire truncated hit process has the product Bernoulli law on
`ℕ`.

This is the product-valued statement immediately before converting a
Prop-valued process into a random set.
-/
theorem hasLaw_truncatedModeHit_process
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (p : unitInterval)
    (hMeas :
      ∀ t : ℕ,
        Measurable
          (fun ω : Ω =>
            observation ω t))
    (hIndep :
      iIndepFun
        (fun t : ℕ =>
          fun ω : Ω =>
            observation ω t)
        μ)
    (hHitLaw :
      ∀ t : ℕ,
        HasLaw
          (
            fun ω : Ω =>
              observation ω t = i
          )
          (
            ProbabilityTheory.bernoulliMeasure
              True
              False
              p
          )
          μ) :
    HasLaw
      (
        fun ω : Ω =>
          fun t : ℕ =>
            truncatedModeHit
              m
              observation
              i
              t
              ω
      )
      (
        Measure.infinitePi
          (
            fun t : ℕ =>
              ProbabilityTheory.bernoulliMeasure
                (t ∈ Set.Iio m)
                False
                p
          )
      )
      μ := by

  have hIndepHit :
      iIndepFun
        (
          fun t : ℕ =>
            fun ω : Ω =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
        )
        μ :=
    iIndepFun_truncatedModeHit
      μ
      m
      observation
      i
      hIndep

  have hCoordinateLaw :
      ∀ t : ℕ,
        HasLaw
          (
            fun ω : Ω =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
          )
          (
            ProbabilityTheory.bernoulliMeasure
              (t ∈ Set.Iio m)
              False
              p
          )
          μ :=
    hasLaw_truncatedModeHit
      μ
      m
      observation
      i
      p
      hHitLaw

  have hHitMeas :
      ∀ t : ℕ,
        Measurable
          (
            fun ω : Ω =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
          ) :=
    measurable_truncatedModeHit
      m
      observation
      i
      hMeas

  have hJointMeas :
      Measurable
        (
          fun ω : Ω =>
            fun t : ℕ =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
        ) := by

    exact
      measurable_pi_lambda
        (
          fun ω : Ω =>
            fun t : ℕ =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
        )
        hHitMeas

  exact
    (
      ProbabilityTheory.iIndepFun_iff_hasLaw_Pi_infinitePi
        hCoordinateLaw
        hJointMeas.aemeasurable
    ).mp
      hIndepHit

/--
The random set of indices at which mode `i` occurs among the first
`m` independent observations has the exact `setBernoulli (Iio m) p`
law.
-/
theorem categoricalModeHitSet_isSetBernoulli
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (p : unitInterval)
    (hMeas :
      ∀ t : ℕ,
        Measurable
          (fun ω : Ω =>
            observation ω t))
    (hIndep :
      iIndepFun
        (fun t : ℕ =>
          fun ω : Ω =>
            observation ω t)
        μ)
    (hHitLaw :
      ∀ t : ℕ,
        HasLaw
          (
            fun ω : Ω =>
              observation ω t = i
          )
          (
            ProbabilityTheory.bernoulliMeasure
              True
              False
              p
          )
          μ) :
    IsSetBernoulli
      (
        categoricalModeHitSet
          m
          observation
          i
      )
      (Set.Iio m)
      p
      μ := by

  have hProcess :
      HasLaw
        (
          fun ω : Ω =>
            fun t : ℕ =>
              truncatedModeHit
                m
                observation
                i
                t
                ω
        )
        (
          Measure.infinitePi
            (
              fun t : ℕ =>
                ProbabilityTheory.bernoulliMeasure
                  (t ∈ Set.Iio m)
                  False
                  p
            )
        )
        μ :=
    hasLaw_truncatedModeHit_process
      μ
      m
      observation
      i
      p
      hMeas
      hIndep
      hHitLaw

  have hSetMap :
      HasLaw
        (
          fun b : ℕ → Prop =>
            {t : ℕ | b t}
        )
        (
          ProbabilityTheory.setBernoulli
            (Set.Iio m)
            p
        )
        (
          Measure.infinitePi
            (
              fun t : ℕ =>
                ProbabilityTheory.bernoulliMeasure
                  (t ∈ Set.Iio m)
                  False
                  p
            )
        ) := by

    refine
      {
        aemeasurable :=
          measurable_setOfPred.aemeasurable
        map_eq := ?_
      }

    exact
      (
        ProbabilityTheory.setBernoulli_eq_map
          (Set.Iio m)
          p
      ).symm

  have hSetLaw :
      HasLaw
        (
          (
            fun b : ℕ → Prop =>
              {t : ℕ | b t}
          )
          ∘
          (
            fun ω : Ω =>
              fun t : ℕ =>
                truncatedModeHit
                  m
                  observation
                  i
                  t
                  ω
          )
        )
        (
          ProbabilityTheory.setBernoulli
            (Set.Iio m)
            p
        )
        μ :=
    hSetMap.comp
      hProcess

  change
    HasLaw
      (
        categoricalModeHitSet
          m
          observation
          i
      )
      (
        ProbabilityTheory.setBernoulli
          (Set.Iio m)
          p
      )
      μ

  apply hSetLaw.congr

  filter_upwards with ω

  ext t

  simp [
    Function.comp_def,
    categoricalModeHitSet,
    truncatedModeHit
  ]

/--
Main result of this file.

For independent observations, if the event that a trial realizes mode
`i` has Bernoulli success probability `p` at every trial, then the
number of occurrences of mode `i` among the first `m` observations has
the exact binomial law `Binomial(m,p)`.
-/
theorem categoricalModeCount_hasLaw_binomial_of_iIndep
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {d : ℕ}
    (m : ℕ)
    (observation : Ω → ℕ → Fin d)
    (i : Fin d)
    (p : unitInterval)
    (hMeas :
      ∀ t : ℕ,
        Measurable
          (fun ω : Ω =>
            observation ω t))
    (hIndep :
      iIndepFun
        (fun t : ℕ =>
          fun ω : Ω =>
            observation ω t)
        μ)
    (hHitLaw :
      ∀ t : ℕ,
        HasLaw
          (
            fun ω : Ω =>
              observation ω t = i
          )
          (
            ProbabilityTheory.bernoulliMeasure
              True
              False
              p
          )
          μ) :
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
          p
      )
      μ := by

  exact
    categoricalModeCount_hasLaw_binomial_of_isSetBernoulli
      μ
      m
      observation
      i
      p
      (
        categoricalModeHitSet_isSetBernoulli
          μ
          m
          observation
          i
          p
          hMeas
          hIndep
          hHitLaw
      )

end

end DROSafety.DRO
