import DROSafety.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Scenario-Based Risk Bounds

This file formalizes the support-dependent scenario risk bound used
by Safe-Horizon MPC.

For sample count `S`, support size `n`, and confidence parameter `β`,
the bound is

    ε(S,n,β)
      = 1 - ( β / (S * choose(S,n)) )^(1/(S-n))

when `n < S`.

If `n ≥ S`, we conservatively return `1`.

The probabilistic scenario theorem itself is not proved here yet.
This file defines the deterministic risk bound that will later be
connected to that theorem.
-/

namespace DROSafety

variable {K : ℕ}

/--
Support-dependent violation-probability bound.

Mathematically,

    ε(S,n,β)
      = 1 - (β / (S * C(S,n)))^(1/(S-n))

for `n < S`.

If the support size is not strictly smaller than the number of
samples, no nontrivial certificate is returned, so the risk bound
is set to `1`.
-/
noncomputable def scenarioEpsilon
    (S n : ℕ)
    (β : ℝ) : ℝ :=
  if n < S then
    1 -
      Real.rpow
        (β / ((S : ℝ) * (Nat.choose S n : ℝ)))
        ((1 : ℝ) / ((S - n : ℕ) : ℝ))
  else
    1

/--
If the support size is not strictly less than the sample count,
the scenario bound is the trivial bound `1`.
-/
theorem scenarioEpsilon_of_not_lt
    (S n : ℕ)
    (β : ℝ)
    (h : ¬ n < S) :
    scenarioEpsilon S n β = 1 := by
  simp [scenarioEpsilon, h]

/--
The mode-wise vector of scenario bounds.

Each mode can have its own:

* sample count `S m`,
* support count `n m`,
* confidence parameter `β m`.
-/
noncomputable def modeScenarioEpsilon
    (S n : Fin K → ℕ)
    (β : ModeVec K) :
    ModeVec K :=
  fun m =>
    scenarioEpsilon
      (S m)
      (n m)
      (β m)

/--
A mode-wise scenario certificate means that the actual conditional
violation probability of every mode is bounded by its corresponding
scenario bound.
-/
def ModewiseScenarioCertified
    (S n : Fin K → ℕ)
    (β V : ModeVec K) :
    Prop :=
  ∀ m,
    V m ≤
      scenarioEpsilon
        (S m)
        (n m)
        (β m)

/--
Unpack a mode-wise scenario certificate into the pointwise inequality
needed by the DRO aggregation theorem.
-/
theorem modewiseScenarioCertified_bound
    (S n : Fin K → ℕ)
    (β V : ModeVec K)
    (hcert :
      ModewiseScenarioCertified S n β V) :
    ∀ m,
      V m ≤ modeScenarioEpsilon S n β m := by
  intro m
  exact hcert m

/--
For `n < S`, raising `1 - scenarioEpsilon S n β` to the
`S - n` power recovers the base appearing in the definition.
-/
theorem one_sub_scenarioEpsilon_pow
    (S n : ℕ)
    (β : ℝ)
    (hnS : n < S)
    (hβ : 0 ≤ β) :
    (1 - scenarioEpsilon S n β) ^ (S - n) =
      β / ((S : ℝ) * (Nat.choose S n : ℝ)) := by

  have hk : S - n ≠ 0 := by
    omega

  have hnle : n ≤ S :=
    Nat.le_of_lt hnS

  have hSpos : 0 < S := by
    omega

  have hChoosePos :
      0 < Nat.choose S n :=
    Nat.choose_pos hnle

  have hDenPos :
      0 <
        (S : ℝ) *
          (Nat.choose S n : ℝ) := by
    positivity

  have hBaseNonneg :
      0 ≤
        β /
          ((S : ℝ) *
            (Nat.choose S n : ℝ)) := by
    positivity

  have hRoot :
      (Real.rpow
          (β /
            ((S : ℝ) *
              (Nat.choose S n : ℝ)))
          (((S - n : ℕ) : ℝ)⁻¹)) ^
        (S - n)
        =
      β /
        ((S : ℝ) *
          (Nat.choose S n : ℝ)) := by
    exact
      Real.rpow_inv_natCast_pow
        hBaseNonneg
        hk

  simpa [
    scenarioEpsilon,
    hnS,
    one_div
  ] using hRoot


/--
For `n < S`, multiplying by the number of support sets cancels the
binomial factor in the scenario bound.
-/
theorem choose_mul_one_sub_scenarioEpsilon_pow
    (S n : ℕ)
    (β : ℝ)
    (hnS : n < S)
    (hβ : 0 ≤ β) :
    (Nat.choose S n : ℝ) *
        (1 - scenarioEpsilon S n β) ^ (S - n)
      =
    β / (S : ℝ) := by

  rw [
    one_sub_scenarioEpsilon_pow
      S n β hnS hβ
  ]

  have hS :
      (S : ℝ) ≠ 0 := by
    have : 0 < S := by omega
    positivity

  have hChoose :
      (Nat.choose S n : ℝ) ≠ 0 := by
    have hnle : n ≤ S := Nat.le_of_lt hnS
    have hc : Nat.choose S n ≠ 0 :=
      Nat.choose_ne_zero hnle
    exact_mod_cast hc

  field_simp [hS, hChoose]

/--
ENNReal form of the support-size cancellation identity.
-/
theorem choose_mul_one_sub_scenarioEpsilon_pow_ennreal
    (S n : ℕ)
    (β : ℝ)
    (hnS : n < S)
    (hβ : 0 ≤ β) :
    (Nat.choose S n : ENNReal) *
        (ENNReal.ofReal
          (1 - scenarioEpsilon S n β)) ^ (S - n)
      =
    ENNReal.ofReal (β / (S : ℝ)) := by

  have hChooseNonneg :
      0 ≤ (Nat.choose S n : ℝ) := by
    positivity

  have hOneSubNonneg :
      0 ≤ 1 - scenarioEpsilon S n β := by

    have hBaseNonneg :
        0 ≤
          β /
            ((S : ℝ) *
              (Nat.choose S n : ℝ)) := by
      positivity

    have hrpow :
        0 ≤
          Real.rpow
            (β /
              ((S : ℝ) *
                (Nat.choose S n : ℝ)))
            ((1 : ℝ) /
              ((S - n : ℕ) : ℝ)) := by
      exact
        Real.rpow_nonneg
          hBaseNonneg
          _

    simpa [
      scenarioEpsilon,
      hnS,
      one_div
    ] using hrpow

  calc
    (Nat.choose S n : ENNReal) *
        (ENNReal.ofReal
          (1 - scenarioEpsilon S n β)) ^ (S - n)
        =
      ENNReal.ofReal (Nat.choose S n : ℝ) *
        (ENNReal.ofReal
          (1 - scenarioEpsilon S n β)) ^ (S - n) := by
      simp

    _ =
      ENNReal.ofReal (Nat.choose S n : ℝ) *
        ENNReal.ofReal
          ((1 - scenarioEpsilon S n β) ^ (S - n)) := by
      rw [
        ENNReal.ofReal_pow
          hOneSubNonneg
          (S - n)
      ]

    _ =
      ENNReal.ofReal
        ((Nat.choose S n : ℝ) *
          (1 - scenarioEpsilon S n β) ^ (S - n)) := by
      rw [
        ENNReal.ofReal_mul
          hChooseNonneg
      ]

    _ =
      ENNReal.ofReal (β / (S : ℝ)) := by
      rw [
        choose_mul_one_sub_scenarioEpsilon_pow
          S n β hnS hβ
      ]

/--
For a valid confidence parameter and `n < S`, the scenario bound is
nonnegative.
-/
theorem scenarioEpsilon_nonneg
    (S n : ℕ)
    (β : ℝ)
    (hnS : n < S)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1) :
    0 ≤ scenarioEpsilon S n β := by

  have hnle : n ≤ S :=
    Nat.le_of_lt hnS

  have hSpos : 0 < S := by
    omega

  have hChoosePos :
      0 < Nat.choose S n :=
    Nat.choose_pos hnle

  have hDenPos :
      0 <
        (S : ℝ) *
          (Nat.choose S n : ℝ) := by
    positivity

  have hSone :
      1 ≤ (S : ℝ) := by
    exact_mod_cast hSpos

  have hChooseOne :
      1 ≤ (Nat.choose S n : ℝ) := by
    exact_mod_cast hChoosePos

  have hDenOne :
      1 ≤
        (S : ℝ) *
          (Nat.choose S n : ℝ) := by
    nlinarith

  have hBaseNonneg :
      0 ≤
        β /
          ((S : ℝ) *
            (Nat.choose S n : ℝ)) := by
    positivity

  have hBaseLe :
      β /
          ((S : ℝ) *
            (Nat.choose S n : ℝ))
        ≤ 1 := by
    apply (div_le_one hDenPos).2
    exact le_trans hβ1 hDenOne

  have hExpNonneg :
      0 ≤
        (1 : ℝ) /
          ((S - n : ℕ) : ℝ) := by
    positivity

  have hrpow :
      Real.rpow
          (β /
            ((S : ℝ) *
              (Nat.choose S n : ℝ)))
          ((1 : ℝ) /
            ((S - n : ℕ) : ℝ))
        ≤ 1 :=
    Real.rpow_le_one
      hBaseNonneg
      hBaseLe
      hExpNonneg

  rw [scenarioEpsilon]
  simp only [hnS]

  exact sub_nonneg.mpr hrpow

end DROSafety
