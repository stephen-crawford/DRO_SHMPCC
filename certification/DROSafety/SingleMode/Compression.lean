import DROSafety.SingleMode.Definitions

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Scenario Compression Certificates

A scenario solution may be reconstructed from a smaller subset of
support scenarios.

This file formalizes:

* support subsets,
* restriction of a full sample to a support subset,
* reconstruction from support scenarios,
* consistency with all remaining scenarios.
-/

namespace DROSafety.SingleMode

/--
The samples indexed by a support set `I`.
-/
abbrev SupportSample
    {S : ℕ}
    (I : Finset (Fin S))
    (Ξ : Type*) :=
  ↥I → Ξ

/--
Restrict a full scenario sample to the indices contained in `I`.
-/
def restrictSample
    {S : ℕ}
    {Ξ : Type*}
    (z : Sample S Ξ)
    (I : Finset (Fin S)) :
    SupportSample I Ξ :=
  fun i => z i.1

/--
A scenario optimizer maps the full set of scenarios to a decision.
-/
abbrev Solver
    (S : ℕ)
    (Ξ Θ : Type*) :=
  Sample S Ξ → Θ

/--
A reconstruction rule builds a decision using only the scenarios
contained in a selected support subset.
-/
abbrev Reconstructor
    (S : ℕ)
    (Ξ Θ : Type*) :=
  (I : Finset (Fin S)) →
    SupportSample I Ξ →
    Θ

/--
Decision reconstructed from support set `I`.
-/
def reconstructedDecision
    {S : ℕ}
    {Ξ Θ : Type*}
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (z : Sample S Ξ) :
    Θ :=
  reconstruct I (restrictSample z I)

/--
The decision is consistent with every scenario outside the support set.

That is, every non-support scenario is satisfied.
-/
def ConsistentOutside
    {S : ℕ}
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (I : Finset (Fin S))
    (z : Sample S Ξ)
    (θ : Θ) :
    Prop :=
  ∀ j, j ∉ I →
    ¬ violates θ (z j)

/--
A full scenario solution has a compression certificate of size `n`
when there exists a support subset `I` of exactly size `n` such that

1. reconstruction from `I` reproduces the solver output, and
2. that reconstructed decision satisfies all scenarios outside `I`.
-/
def HasCompressionCertificate
    {S : ℕ}
    {Ξ Θ : Type*}
    (solver : Solver S Ξ Θ)
    (reconstruct : Reconstructor S Ξ Θ)
    (violates : Θ → Ξ → Prop)
    (n : ℕ)
    (z : Sample S Ξ) :
    Prop :=
  ∃ I : Finset (Fin S),
    I.card = n ∧
    reconstructedDecision reconstruct I z = solver z ∧
    ConsistentOutside
      violates I z (solver z)

end DROSafety.SingleMode
