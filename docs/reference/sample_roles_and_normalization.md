# Sample roles and normalization

Sample metadata identifies a dataset and supplies numeric properties used for
normalization. Sample-role policy classifies how that dataset participates in
overlap removal, systematic handling, EFT treatment, or data-driven products.
The two concepts are related but not interchangeable.

## Metadata and active production universe

The resolved sample JSON/cfg supplies dataset identity, era, data/MC status,
file inputs, cross section, generated-event sums, and optional EFT metadata.
Production-profile certification constrains which configured samples are part
of the maintained active universe and prevents ambiguous central/private
combinations. `run_analysis.py` and `AnalysisProcessor` consume the resolved
metadata; they do not derive missing sample identity from a filename.

Numeric cross sections and generator sums remain sample metadata authorities.
Shared luminosity values are owned by `topcoffea/params/params.json` and are
selected by the consuming era.

## `lo_xsec_samples` is a role set

`topeft/params/params.json` defines `lo_xsec_samples` as a list of sample names.
The main and specialist processors test membership to select a maintained
rate/systematic treatment. The list does not contain numeric cross sections and
must not be used as their authority. Membership establishes current processing
behavior; the repository does not establish the scientific motivation for
every membership choice.

## ttgamma source roles

The ttgamma policy partitions relevant samples into maintained prompt,
conversion, decay, inclusive-ttbar, and inclusive-ttgamma roles so the
processor can apply the selected overlap contract. The direct CLI exposes the
supported policy choice, while the production wrapper pins the maintained
production behavior. Role masks apply consistently across the enabled
variations for a sample.

`topeft/modules/ttgamma_photon_history.py`, parameter role sets, the CLI
resolver, and processor call sites jointly establish implementation authority.
They establish the partition, not its unrecorded motivation.

## Data-driven source roles

Nonprompt and charge-flip products depend on certified source and target
identities. Data contributions, prompt or conversion subtraction sets, and
target names are policy inputs to the transformation contract. A process name
alone is not sufficient authority for a transformation; the product policy
and artifact sidecar must agree.

See [data-driven estimation](data_driven_estimation.md).

## Normalization flow

For MC, sample cross section, luminosity, and generated-event sums contribute
to the nominal normalization before scale factors and systematic modifiers.
The maintained sum-of-weights processor produces normalization information for
the selected samples; it is a specialist normalization interface, not the main
selected-event processor. See [specialist interfaces](specialist_interfaces.md).

Normalization choices affect every weighted yield and downstream rate. Sample
roles additionally control which overlap masks, source transformations, EFT
treatment, or rate/systematic branches apply. Execution controls such as
workers, chunks, and output paths do not define either contract.

## Defaults, role example, and modification route

The current role authorities are
[`params.json`](../../topeft/params/params.json),
[`ttgamma_photon_history.py`](../../topeft/modules/ttgamma_photon_history.py),
and the selected repository sample cfg/JSON. The direct ttgamma policy defaults
to `split`; `run2_nlo_inclusive` is the explicit supported Run-2 diagnostic
alternative. `lo_xsec_samples` is represented by a list of sample names and
has no numeric-xsec default.

Use [change sample roles or normalization inputs](../how_to/sample_roles_and_normalization.md)
for sample records, role sets, ttgamma policy, active-universe certification,
and separately authorized sum-of-weights refreshes.
