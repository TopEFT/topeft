# Analysis workflow contract changes

## Summary

This branch is an intentionally broad integration update. Relative to
`run3_test_mmerged_anpicci`, it changes histogram representation and provenance,
sample identities and input endpoints, Run 3 JME diagnostics, datacard and
binning contracts, remote-environment validation, nonprompt production, and the
documentation that connects those surfaces. The domains are coupled through
shared artifact loaders and maintained production entry points; this record is
a navigation and migration guide, not a claim that the branch is one narrow
feature.

## Semantic domains

| Domain | Main behavior change | Validation basis |
| --- | --- | --- |
| Histogram provenance and plotting | Split nominal families, PKL-sidecar identity, compatible multi-input composition, combined-era plotting, and repeatable plotter inputs | Focused sidecar-merge tests, ten plotter-wrapper tests, a six-command dry run, and exact readback of certified Run 3 artifacts |
| Sample and process identity | Canonical input endpoints and selected process labels, plus an exact fail-closed repair route for known historical labels | Redirector checks, fourteen repair-bootstrap tests, and nineteen warning/label boundary tests |
| Run 3 JME and diagnostics | JERC tag alignment and nominal-only before/after jet-veto-map eta-phi diagnostics | Twenty-one focused diagnostic tests, tag/schema checks, and bounded sample validation |
| Datacards, missing partons, binning, and EFT scaling | Exact processing/fitting bin ownership, late rebinning, complete shape-pair gates, missing-parton payload handling, negative-template sanitation, and scaling preservation | Payload derivation/closure checks, exact certification of five Run 3 artifact pairs, the accepted 555-bin map, and scaling readback |
| Environments and correction boundaries | Fingerprinted environment archives and NumPy/scalar correctionlib arguments | Focused archive/cache tests, correction-boundary tests, and zero-difference bounded file probes |
| Nonprompt and production workflows | Canonical prompt-family resolution, fail-closed coverage, streaming transformation lineage, maintained campaign profiles, resume accounting, and bounded legacy-sidecar normalization | 266 policy tests plus an independent contract validator, focused execution-coverage tests, legacy-contract regressions, and exact transformed-artifact replay |
| CLI contract alignment | Seven documented entry-point fixes spanning production, analysis options, cards, post-processing counts, and plotter dry-run behavior | Eighty-one focused tests, shell syntax checks, and source/documentation review |
| Documentation | Role-separated tutorials, how-to guides, explanation, and source-aligned reference pages | Static link, navigation, and source-consistency review |

The validation above is targeted to the named surfaces. It is not a claim that
full repository tests or a new production campaign were run for this record.

## Impact classification

- `invocation_change`: **optional or workflow-specific**. Normal direct
  `run_analysis.py` use remains available; repeatable plotting inputs and newer
  recovery/validation controls are selected only when needed. Maintained
  production profiles have their own required arguments.
- `configuration_change`: **mixed by workflow**. Existing unaffected direct
  commands need no cosmetic rewrite. Affected production, sample, missing-parton,
  binning, and nonprompt workflows must use the current canonical configuration.
- `artifact_contract_change`: **migration required for affected artifacts**.
  Maintained consumers treat a PKL and its adjacent sidecar as one unit and
  validate schema, identity, provenance, and required companions.
- `existing_artifact_status`: **mixed by artifact class**; see the table below.
- `downstream_workflow_change`: **mixed by consumer**. Plotting can opt into
  multiple inputs, while current datacard/nonprompt paths require compatible
  artifacts and complete inputs.
- `operational_practice_change`: **required** for affected workflows. Preserve
  sidecars, record both repository revisions, use maintained profiles or exact
  direct interfaces, and do not repair metadata by hand.

## Affected entry points

| Entry point | Invocation | Configuration and output impact |
| --- | --- | --- |
| `analysis/topeft_run2/run_analysis.py` | Core input, executor, output-name, output-path, year, and systematics patterns remain supported; new controls are optional | Resolves current sample/policy/environment contracts and publishes a validated PKL-sidecar pair; optional inline nonprompt output has its own lineage |
| `analysis/topeft_run2/analysis_processor.py` | Normal construction remains owned by `run_analysis.py`; the obsolete internal `rebin` constructor argument is removed | Histogram schema, processing bins, JME diagnostics, and selected physics-policy inputs changed; call it through the maintained CLI unless developing the processor itself |
| `analysis/topeft_run2/run_cr.sh` | Maintained profiles require a profile, fresh output directory, and campaign tag | Owns block plans, state, resume checks, environment resolution, and deferred transformations; a dry run may still resolve/validate an environment |
| `analysis/topeft_run2/run_plotter.sh` and `make_cr_and_sr_plots.py` | A single input remains valid; repeated `-f` inputs or a list file are available | Every input needs a compatible sidecar and is merged through the maintained loader before plotting |
| `analysis/topeft_run2/run_data_driven.py` | The source PKL is required; lower-level compatibility controls are explicit | Consumes a certified processor pair and publishes a separate lineage-bound transformed pair |
| `analysis/topeft_run2/make_cards.py` | Positional PKLs or one list file; optional missing-parton, coverage, and binning controls | Requires coherent merged inputs, second-moment companions, complete shape pairs, and fitting-axis/EFT consistency |
| `analysis/topeft_run2/datacards_post_processing.py` | The topology selector remains explicit | Finalizes selected card copies and channel-aware scalings; EFTFit/Combine later owns card combination and workspace construction |

## Artifact and compatibility contract

- Keep each histogram PKL beside its `.metadata.json` sidecar. A filename alone
  is not provenance.
- Current processor and transformed outputs use the split nominal-family and
  current transformation contracts. Supported older transformation records have
  bounded read compatibility; they are not silently rewritten as current.
- Merge compatibility includes schema, policy, production identity, axes,
  contributions, lineage, requested products, and required second moments.
  Diagnostic warning text is retained as a sorted union rather than used as a
  semantic-identity gate.
- Process/sample names are contract data. Use the exact maintained repair helper
  only for supported historical variants; fuzzy or hand-edited relabeling is
  refused.
- Current environment archives require an adjacent manifest, archive digest,
  and matching resolved fingerprint. Do not edit an archive or manifest in
  place; resolve a new cache key after relevant package inputs change.

## Existing artifacts

| Artifact class | Status | Required action |
| --- | --- | --- |
| Current PKL-sidecar pair that passes the matching validator | Reusable as-is for its certified purpose | Keep the pair together and retain its recorded repository revisions |
| Supported older transformation record | Bounded read compatibility | Use only the documented compatibility path; regenerate before claiming current production provenance |
| Known historical process-label variant accepted by the repair helper | Repair required | Create and validate a corrected copy; preserve the original |
| Missing, mismatched, malformed, or unsupported sidecar | Unsupported by maintained consumers | Recover the matching sidecar from authoritative evidence or rerun the producer; do not invent metadata |
| Artifact needed to claim refreshed JME/JER behavior or new JVM diagnostics | Rerun required | Reproduce with compatible `topcoffea` and current configuration |
| The separately certified five-block Run 3 card-input set | Reusable as-is for that certified card purpose | Preserve exact files, sidecars, and lineage; do not generalize this result to other historical PKLs |
| Environment archive with matching manifest and fingerprint | Reusable as-is | Validate before use |
| Stale or unverifiable historical environment archive | Repair is not supported; rebuild for strict use | Snapshot inspection may preserve historical evidence but does not relax integrity checks |

## Downstream workflows

- **Plotting:** supply all intended fragments explicitly. The plotter validates
  sidecars, merges compatible contributions, and rejects mixed Run 2/Run 3
  inputs or ambiguous channel authority.
- **Datacards:** use fitting binning by default, provide the complete compatible
  artifact set, retain required second moments, and resolve missing-parton and
  shape-pair requirements before card production.
- **Scalings:** retain the finalized channel order and EFT coefficients emitted
  with the individual cards.
- **EFTFit/Combine:** this repository owns the cards, templates, scaling handoff,
  and compatible channel namespace. The external workflow owns card combination
  and workspace construction; no broader external-runtime compatibility claim
  is made here.

## Operational practice

1. Record the exact `topeft` and `topcoffea` revisions used for production.
2. Install or deploy the compatible `topcoffea` update before running dependent
   topeft JME/correction workflows.
3. Preserve PKL-sidecar pairs and validate them at each consumer boundary.
4. Use fresh campaign namespaces or a compatible recorded resume; never infer
   completion from filenames alone.
5. Treat historical repair as a copy-and-readback operation. Never hand-edit a
   PKL or sidecar to make it look current.

## Unchanged interfaces

- The normal `run_analysis.py` input, executor/resource, output-name,
  output-path, year, and systematics call pattern remains supported.
- `run_analysis.py` remains the owner that constructs `AnalysisProcessor`.
- A single plotter input remains supported; repeated inputs extend that route.
- `topcoffea` continues to own reusable correction factories and payload
  mechanisms, while this repository owns concrete era/tag and analysis policy.

These are deliberately narrow stability statements. They do not mean that all
CLIs, configuration, artifact schemas, or operating practices are unchanged.

## Known limitations and deferred follow-ups

This integration does not include axis-merge prevalidation alignment,
family-mapping insertion-order changes, period-name cleanup, a campaign-level
CR plotting/output interface, generated API documentation, or TAU work. It also
does not claim completion of Run 2 plotting, cross-era output consolidation, new
production, new datacards, or external workspace construction.
