import DROSafety.DRO.BinomialTailLevelExistence
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Exact roots of binomial tail levels

The previous file proved existence of a point in `[0,1]` attaining any
tail level `α ∈ [0,1]`.

Here we turn those existence statements into total noncomputable
functions using `Classical.choose`.

We deliberately do not claim monotonicity yet.  An arbitrary selected
root is useful only after we establish the comparison/uniqueness facts
needed for the Clopper-Pearson endpoint monotonicity proof.
-/

namespace DROSafety.DRO

/--
A noncomputable exact root of the upper binomial-tail polynomial.

When

    0 < k ≤ m
    0 ≤ α ≤ 1,

this selects a point `x ∈ [0,1]` satisfying

    upperBinomialTailPolynomial m k x = α.

Outside those hypotheses we define the value to be `0`.
-/
noncomputable def upperBinomialTailLevelRoot
    (m k : ℕ)
    (α : ℝ) :
    ℝ :=
  if h :
      0 < k ∧
      k ≤ m ∧
      0 ≤ α ∧
      α ≤ 1
  then
    Classical.choose
      (
        exists_upperBinomialTailPolynomial_eq
          m
          k
          α
          h.1
          h.2.1
          h.2.2.1
          h.2.2.2
      )
  else
    0

/--
A noncomputable exact root of the lower binomial-tail polynomial.

When

    k < m
    0 ≤ α ≤ 1,

this selects a point `x ∈ [0,1]` satisfying

    lowerBinomialTailPolynomial m k x = α.

Outside those hypotheses we define the value to be `1`.
-/
noncomputable def lowerBinomialTailLevelRoot
    (m k : ℕ)
    (α : ℝ) :
    ℝ :=
  if h :
      k < m ∧
      0 ≤ α ∧
      α ≤ 1
  then
    Classical.choose
      (
        exists_lowerBinomialTailPolynomial_eq
          m
          k
          α
          h.1
          h.2.1
          h.2.2
      )
  else
    1

/--
Under the admissible upper-tail hypotheses, the selected root belongs
to `[0,1]`.
-/
theorem upperBinomialTailLevelRoot_mem_Icc
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailLevelRoot
        m
        k
        α
      ∈
    Set.Icc (0 : ℝ) 1 := by

  have h :
      0 < k ∧
      k ≤ m ∧
      0 ≤ α ∧
      α ≤ 1 :=
    ⟨hk, hkm, hα0, hα1⟩

  have hExists :=
    exists_upperBinomialTailPolynomial_eq
      m
      k
      α
      hk
      hkm
      hα0
      hα1

  have hSpec :=
    Classical.choose_spec
      hExists

  simpa [
    upperBinomialTailLevelRoot,
    h
  ] using hSpec.1

/--
The selected upper-tail root attains the requested level exactly.
-/
theorem upperBinomialTailLevelRoot_eq
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailPolynomial
        m
        k
        (
          upperBinomialTailLevelRoot
            m
            k
            α
        )
      =
    α := by

  have h :
      0 < k ∧
      k ≤ m ∧
      0 ≤ α ∧
      α ≤ 1 :=
    ⟨hk, hkm, hα0, hα1⟩

  have hExists :=
    exists_upperBinomialTailPolynomial_eq
      m
      k
      α
      hk
      hkm
      hα0
      hα1

  have hSpec :=
    Classical.choose_spec
      hExists

  simpa [
    upperBinomialTailLevelRoot,
    h
  ] using hSpec.2

/--
The upper selected root is nonnegative.
-/
theorem upperBinomialTailLevelRoot_nonneg
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    0 ≤
      upperBinomialTailLevelRoot
        m
        k
        α :=
  (
    upperBinomialTailLevelRoot_mem_Icc
      m
      k
      α
      hk
      hkm
      hα0
      hα1
  ).1

/--
The upper selected root is at most one.
-/
theorem upperBinomialTailLevelRoot_le_one
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailLevelRoot
        m
        k
        α
      ≤
    1 :=
  (
    upperBinomialTailLevelRoot_mem_Icc
      m
      k
      α
      hk
      hkm
      hα0
      hα1
  ).2

/--
Under the admissible lower-tail hypotheses, the selected root belongs
to `[0,1]`.
-/
theorem lowerBinomialTailLevelRoot_mem_Icc
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    lowerBinomialTailLevelRoot
        m
        k
        α
      ∈
    Set.Icc (0 : ℝ) 1 := by

  have h :
      k < m ∧
      0 ≤ α ∧
      α ≤ 1 :=
    ⟨hkm, hα0, hα1⟩

  have hExists :=
    exists_lowerBinomialTailPolynomial_eq
      m
      k
      α
      hkm
      hα0
      hα1

  have hSpec :=
    Classical.choose_spec
      hExists

  simpa [
    lowerBinomialTailLevelRoot,
    h
  ] using hSpec.1

/--
The selected lower-tail root attains the requested level exactly.
-/
theorem lowerBinomialTailLevelRoot_eq
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    lowerBinomialTailPolynomial
        m
        k
        (
          lowerBinomialTailLevelRoot
            m
            k
            α
        )
      =
    α := by

  have h :
      k < m ∧
      0 ≤ α ∧
      α ≤ 1 :=
    ⟨hkm, hα0, hα1⟩

  have hExists :=
    exists_lowerBinomialTailPolynomial_eq
      m
      k
      α
      hkm
      hα0
      hα1

  have hSpec :=
    Classical.choose_spec
      hExists

  simpa [
    lowerBinomialTailLevelRoot,
    h
  ] using hSpec.2

/--
The lower selected root is nonnegative.
-/
theorem lowerBinomialTailLevelRoot_nonneg
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    0 ≤
      lowerBinomialTailLevelRoot
        m
        k
        α :=
  (
    lowerBinomialTailLevelRoot_mem_Icc
      m
      k
      α
      hkm
      hα0
      hα1
  ).1

/--
The lower selected root is at most one.
-/
theorem lowerBinomialTailLevelRoot_le_one
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    lowerBinomialTailLevelRoot
        m
        k
        α
      ≤
    1 :=
  (
    lowerBinomialTailLevelRoot_mem_Icc
      m
      k
      α
      hkm
      hα0
      hα1
  ).2

end DROSafety.DRO
