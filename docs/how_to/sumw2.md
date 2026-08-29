# Select or change sumw2 storage

For a weighted histogram bin, the nominal content is the sum of event weights;
sumw2 is the sum of their squares. A selected `<family>_sumw2` companion stores
the scalar Standard Model (`WC = 0`) statistical second moment on the same
processing axes as its nominal sibling. It is not another nominal yield, and
the current contract does not model nonzero-WC quartic variance.

The workflow keeps five concepts separate:

- **mode**: a named selection strategy such as `production`;
- **default**: the mode used when configuration does not select one;
- **resolved policy**: the concrete `(dataset, process, family)` targets after
  samples, runtime families, rules, analysis mode, and consumers are known;
- **provenance schema**: the serialized policy contract in the metadata
  sidecar, currently version 2;
- **content manifest**: independent readback evidence of nominal/companion
  processes physically present and required in each family.

This guide covers configuration and developer changes. See the
[sumw2 explanation](../explanation/sumw2.md) for the statistical and artifact
model and the [sumw2 reference](../reference/sumw2.md) for exact symbols,
serialized fields, and failure contracts.

## Select an existing mode

Set `sumw2_storage` in the top-level YAML mapping passed through
`run_analysis.py --options FILE`:

```yaml
sumw2_storage:
  mode: production
  rules:
    - process_prefixes: [data, TTTo]
      variables: [njets, lj0pt]
```

Rule selector fields are `dataset_names`, `dataset_prefixes`, `process_names`,
`process_prefixes`, and optional `variables`. Exact selectors compare complete
names; prefix selectors use string prefixes. Omitting `variables` selects all
runtime histogram families matched by the rule. Every list must contain unique
non-empty strings.

Choose a mode according to its supported responsibility:

| Mode | Use | Rules and coupling |
| --- | --- | --- |
| `production` | maintained private-signal production | an explicit request requires non-empty rules; an entirely absent block uses the consumer-derived implicit production default |
| `production_central` | maintained central-signal production | non-empty rules and the certified central sample profile |
| `taufitter` | input for the maintained fake-tau fitter | non-empty rules and `analysis_mode: taufitter` must be selected together |
| `full_diagnostics` | diagnostics needing every runtime target | rules are forbidden; storage can be large |
| `disabled` | workflow with no supported companion consumer | rules are forbidden; consumers requiring a companion reject the artifact |
| `full_custom` | operator-owned nonstandard selection | non-empty rules; the operator must cover every intended consumer |

When the whole `sumw2_storage` block is absent, the current default is
`production` and the producer uses active consumer requirements to resolve
targets. A present empty block or one omitting `mode` also resolves the mode to
`production`, but an explicit `production` block must satisfy explicit rule
semantics. Prefer a complete explicit block for a nonstandard run.

The two production modes do not choose samples. The sample cfg/JSON remains
authoritative; the mode certifies whether the active signal universe is private
or central. Switching only the sample cfg or only the mode is rejected.

Before running, verify the resolved sample universe, histogram families,
consumer, and expected storage. `--pretend` exercises the policy boundary
without event processing. Unknown policy/rule fields, unknown or inactive
families, selectors matching nothing, structurally duplicate rules, overlapping
concrete targets, mode/analysis mismatch, sample-profile mismatch, and uncovered
consumer targets fail before artifact publication.

The legacy `--no-sumw2`, YAML `no_sumw2`, and YAML `do_errors` surfaces are
migration-only. Explicit legacy true maps to `disabled`; explicit legacy false
maps to `full_diagnostics`, both with warnings. Modern and legacy settings may
not be combined. New configurations use `sumw2_storage`.

## Check propagation through artifacts

Resolution occurs before `AnalysisProcessor` allocation. For every selected
target, the processor fills a `<family>_sumw2` sibling on the same physical
processing axis and compatible sparse categories as nominal content. The
sidecar records:

1. schema-v2 resolved policy identity, including source, requested/resolved
   mode, signal profile, normalized rules, runtime universe, and concrete
   targets; and
2. a separate version-1 content manifest derived from the written histogram
   mapping, including observed and required processes for each family.

That separation prevents an intended policy from certifying missing or extra
content. Reopen the PKL and sidecar together and confirm the policy, manifest,
and actual process/family content agree.

Data-driven transformation derives its source and generated companion
requirements from the resolved transformation contract, preserves the input
sumw2 policy provenance, and publishes a new manifest. Merge consumers require
one compatible policy identity and schema and validate required content in
every input. Plotting and cards apply the same channel selection and exact
binning aggregation to nominal and companion histograms. The fake-tau fitter
requires `tau0Fpt_sumw2` and `tau0Tpt_sumw2` and has no count-based fallback.

Consumers fail closed when a required family/process companion is absent,
orphaned, attached to the wrong sidecar identity, inconsistent with its nominal
axis/categories, missing from one merged input, or unsupported by provenance.
Do not repair this by copying/renaming a companion, editing the content manifest,
or estimating variance from nominal yields. Reproduce the earliest artifact
whose resolved policy failed to select the requirement.

## Add a new mode

Adding a mode changes policy code, even if users select it through YAML:

1. Define the mode's purpose, supported analysis/sample profile, whether it
   accepts rules, and how it covers declared consumers.
2. Add it to `SUMW2_MODES` in `topeft/modules/sumw2_policy.py`; add it to
   `RULE_MODES` only if rules are part of its contract, and add a production
   signal-profile mapping only when it certifies that exact profile.
3. Extend `resolve_sumw2_storage_mode` for source/default/legacy-independent
   resolution and `resolve_sumw2_storage_policy` for target selection,
   analysis-mode coupling, rules, and consumer coverage.
4. Ensure `resolved_sumw2_policy.to_provenance()` and
   `resolved_policy_from_provenance()` serialize and reopen the mode in canonical
   deterministic form. A mode expressible by schema 2 does not require a schema
   change.
5. Thread the mode through `run_analysis.py` preflight and production sample
   certification without changing cfg authority.
6. Update producer allocation/fill, artifact publication/readback,
   transformation, merge, plotting, card, and specialist consumers affected by
   its intended scope.
7. Add positive cases plus unknown field/mode, missing/empty rules, zero match,
   overlap, sample mismatch, mode mismatch, and missing consumer tests.

Adding a mode does not change what an absent block means. Do not combine mode
addition with a default change merely for convenience.

## Change the default mode

The default is owned by `resolve_sumw2_storage_mode` in two source paths: an
absent `sumw2_storage` block and a present mapping whose `mode` is omitted.
Change those branches consistently and decide whether they should still have
the same rule behavior. Then:

1. inventory maintained option files that omit the block or `mode`;
2. recalculate their resolved target sets and storage impact;
3. update implicit consumer requirements and private/central sample
   certification if the new default requires it;
4. update warning/source labels and default-resolution tests;
5. validate producer output, sidecar identity, data-driven products, merges,
   plotting, cards, and fake-tau requirements for omitted configuration; and
6. plan new PKL production for campaigns whose resolved companions change.

A default change can alter artifacts even though no campaign YAML changes. It
does not add a mode and does not automatically require a provenance version
bump when schema 2 can represent the new resolved policy.

## Evolve the provenance schema

Schema evolution is necessary only when the current exact field contract cannot
represent the new policy semantics or compatibility boundary. It is not a tool
for renaming a mode or changing the default.

1. Increment `SUMW2_PROVENANCE_SCHEMA_VERSION`; keep the old value as an
   explicit legacy constant if readback remains supported.
2. Define the exact required/forbidden field set and canonical ordering for the
   new version. Reject unknown, missing, noncanonical, or semantically
   inconsistent data.
3. Implement version-specific `resolved_policy_from_provenance` behavior.
   Never infer fields absent from an old sidecar or silently rewrite old
   metadata as current.
4. Decide which old artifacts are readable, mergeable, transformable, and
   consumable. Today schema 1 is bounded read compatibility and cannot encode
   `production_central`; it is not silently promoted to schema 2.
5. Update artifact writers/validators, transformation certification, merge
   identity, content-manifest validation, and every consumer gate.
6. Add current/legacy acceptance, cross-version rejection, tampering,
   canonical round-trip, and regenerate-required tests.

If an old artifact cannot prove the newly required contract, fail with a
regeneration instruction rather than adding a compatibility shim.

## Validation matrix

Run the focused owners appropriate to the change:

| Contract | Tests |
| --- | --- |
| mode, rules, defaults, provenance parsing | `tests/test_sumw2_policy.py` |
| CLI/YAML preflight and default output selection | `tests/test_run_analysis_preflight.py`, `tests/test_run_analysis_hist_outputs.py` |
| physical SM-point second moment and processor companions | `tests/test_analysis_processor_eft_sumw2.py` |
| sidecar policy/manifest/identity/transform/merge | `tests/test_histogram_artifact_sidecars.py` |
| data-driven requirements and output | `tests/test_data_driven_products.py`, `tests/test_run_data_driven*.py` |
| card merge and selective requirements | `tests/test_datacard_tools_selective_sumw2.py`, `tests/test_make_cards_multi_pkl.py` |
| fake-tau coupling | `tests/test_fake_tau_sf_taufitter_policy.py`, fake-tau fitter tests |
| nominal/companion late aggregation | `tests/test_datacard_late_rebin.py`, plotting tests |

Also validate one representative resolved policy and artifact readback for each
maintained mode changed. A passing policy unit test alone does not establish
processor storage cost, transformed companion propagation, card readiness, or
production reproducibility.
