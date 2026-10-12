# Independent anchor pilot: implementation scope

The user's supplied feedback explicitly requested the two-sample construction
and the M={0,5,10,20,40,80,160} fixed-h10 sweep. The candidate change implements
that mathematical behavior, using the existing sampler and anchor repair.

`ScenarioSamplingSettings::anchor_pilot_samples` defaults to zero. YAML parsing
accepts the same key. Validation rejects negative values and pilot activation
outside ordinary non-DRO SH sampling. In `MPCController::solve`, the existing
pre-sampling reference/backup anchor is repaired using a local pilot batch before
the ordinary certification draw. The repaired anchor is then frozen. The pilot
is never passed to the scenario optimization or support counter.

This changes the anchor and feasible halfspaces when explicitly enabled. The
current certification batch does not choose or repair the anchor. Defaults,
sample-complexity functions, confidence settings, numerical tolerances, removal
logic and the hybrid branch retain their previous behavior. RNG streams use
separate domains; the original certification stream is unchanged by pilot
sampling. Counterfactual clones reconstruct a shared pilot at a context.

The fixed-budget conditional argument requires a valid implementation theorem
for every eligible fixed pilot/context. Under that assumption, conditioning on
history and the pilot, applying the fresh-sample theorem, and averaging preserves
beta. No additional beta allocation is needed merely for pilot preparation.
This is an assumption-dependent argument, not a proof that the current
approximate numerical implementation satisfies all hypotheses.

The four audit obligations remain separate: pre-certification anchor
measurability; numerical physical feasibility; theorem support dominated by
logged support (including facets, removals and returns); and conditional
adaptive budget selection. No new safety guarantee is claimed here.

See [the executed sweep](../results/anchor-pilot-sweep/README.md). The short
experiment did not identify a reliably successful pilot size. Any further
protected anchor/constraint change requires its own concrete diagnosis and
explicit authorization; a learner and full budget-response repeat are premature.
