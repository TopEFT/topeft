# Why sumw2 is a policy and artifact contract

For a weighted histogram, the nominal bin content is the sum of event weights.
The statistical companion is the sum of their squares, conventionally called
sumw2. It supplies the variance information needed by plotting, card creation,
and other consumers at the Standard Model point.

Storing every possible companion can make production artifacts unnecessarily
large, while omitting a companion required by a consumer makes the result
statistically incomplete. `topeft.modules.sumw2_policy` resolves that tradeoff
before execution as a concrete set of dataset, process, and histogram-family
targets.

## Five distinct concepts

- A **mode** is a named selection strategy such as `production` or
  `full_diagnostics`.
- The **default** is the mode selected when configuration does not request one;
  it is currently `production`.
- The **resolved policy** is the concrete target set after sample and family
  resolution.
- The **provenance schema** defines how that resolution is serialized; the
  current schema is version 2.
- The **content manifest** records which companions were actually written and
  which are required.

These concepts evolve independently. Adding a mode does not change the
default. Changing the default does not imply a schema change. A schema change
requires explicit compatibility behavior rather than a silent reinterpretation
of old sidecars.

## From a mode to concrete content

The mode resolver first distinguishes an explicit request from a defaulted
request and any supported legacy key. The policy resolver then combines that
mode with the active sample universe, runtime histogram families, signal
profile, and optional exact/prefix selector rules. Its output is no longer an
abstract name: it is a set of concrete `(dataset, process, family)` targets.

That resolution happens before the processor runs. Unknown modes or rule keys,
unknown families, selectors that match nothing, overlapping rules, incompatible
signal profiles, and policies that cannot satisfy a declared consumer fail at
this boundary. A warning is used only for explicitly supported compatibility
paths, not to turn an unresolved policy into production content.

The current modes serve different scopes:

- `production` selects the maintained private-signal production contract;
- `production_central` selects the central-signal variant;
- `taufitter` selects the companion scope required by the tau-fitter path;
- `full_diagnostics` retains broad diagnostic content;
- `disabled` requests no companions;
- `full_custom` makes the caller responsible for a complete validated rule set.

These names describe storage policy. They are not alternative physical
definitions of a squared event weight.

## Relationship to nominal and EFT content

The stored companion is the scalar Standard Model/Wilson-coefficient-zero sum
of squared event weights for a selected nominal family. It uses the same
processing axis and compatible sparse categories as that family. Nonzero-WC
quartic variance modeling is not part of the current contract.

Late transformations must keep the relationship intact. When nominal content
is grouped, selected, or exactly rebinned, the corresponding sumw2 content must
follow the same semantic mapping. Matching array sizes are insufficient if the
process identity, physical axes, or category association differ.

## Provenance and content are checked separately

The schema-v2 provenance records the source and resolved mode, signal profile,
normalized rules, runtime families, concrete targets, warnings, and policy
identity. The content manifest is derived from the output histogram mapping and
records what was actually stored.

A producer therefore cannot certify success by serializing its intended
policy. Readback compares the policy, manifest, and payload. A consumer also
declares the companions it requires for its chosen operation. Plotting can
request a different family scope from card production, and a transformed
nonprompt product derives its requirements from the transformation contract.

Schema version 1 remains a readable legacy format under explicit restrictions.
Readback does not silently rewrite it into version 2 or infer fields that were
never serialized. Mixed or ambiguous policy identity fails rather than being
normalized from filenames.

## Three developer changes with different consequences

Adding a new mode extends the registry and resolver semantics; it must specify
target selection, signal-profile coupling, rule behavior, consumer coverage,
serialization, and tests. Changing the default changes what an absent YAML
block means and can change production artifacts without changing any explicit
configuration. Changing the provenance schema changes the compatibility
boundary and needs a version bump plus explicit old-version readback rules.

Keeping these operations separate prevents a convenience change in one owner
from silently redefining artifact compatibility in another.

## Propagation and failure boundary

The processor builds `<family>_sumw2` siblings on the same processing axes as
their nominal families. The policy identity and content manifest travel in the
metadata sidecar. Transform, merge, plot, and card stages resolve their own
requirements and fail when a required companion is absent, inconsistent, or
attached to the wrong artifact identity.

This fail-closed path is why the policy is not merely a memory optimization.
It is part of the reproducible statistical meaning of the artifact.

The main validation owners cover registry and rule resolution, run-analysis
preflight, processor output construction, artifact sidecars and transformations,
merges, plots, and datacards. A documentation check can verify that these
owners are named consistently; it cannot replace their behavior tests or a
regenerated production/card validation after a policy change.

See the [sumw2 how-to](../how_to/sumw2.md) for selection and extension tasks
and the [sumw2 reference](../reference/sumw2.md) for modes, symbols, schemas,
and test owners.
