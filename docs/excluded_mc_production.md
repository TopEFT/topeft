# Excluded-MC producer

The dedicated public entry point is `analysis/topeft_run2/run_excluded_mc.sh`. It
calls `fullR3_run_excluded_mc.sh`, which selects the dedicated
`run_analysis_excluded_mc.py` entry point. The neutral file names reflect the
existing cross-era producer; the wrapper and entry point were renamed from
their earlier `ul18` names, not copied from the standard production chain.

The producer selects native SR categories. Its active background-only input
CFGs are `input_samples/cfgs/excluded_mc_run2_sr.cfg` and
`input_samples/cfgs/excluded_mc_run3_sr.cfg`. Each is required to equal the
active background CR JSON paths minus the active background SR JSON paths for
each included era, with no path shared with a signal-named CFG. Run3 includes
the four existing `TTto4Q_NDSkim_<era>.json` inputs. The producer changes no
source JSON metadata. Run2 uses the available excluded backgrounds across its
four eras; unavailable `TTToHadronic` remains a physical-coverage caveat.

## Interface

Use `--block all` for an a-then-d campaign with either `--run run2` or
`--run run3`. `--block a` and `--block d` select one unit. The required options are
`--campaign-tag`, `--output-root`, `--env-file`, `--wq-staging-base`,
`--wq-quota-root`, `--wq-free-warning-bytes`, `--wq-min-free-bytes`, and
`--wq-poll-seconds`. All paths must be absolute. Supply distinct `--port-a`
and `--port-d` for `--block all`, or `--port` for one block. The optional
single-unit `--manager-name` only checks the name derived from the campaign
tag, run and block; it cannot change it.

`--snapshot` requires an explicit `--env-file`. In snapshot mode the dedicated
entrypoint validates the archive and its adjacent manifest, then uses that
worker environment without resolving the dirty current checkout as an
environment request. Ordinary mode retains strict current-source resolution
and rejects a dirty checkout or incompatible archive. Dry-run prints the
resolved environment mode and both wrapper and entrypoint argv; it does not
validate a synthetic archive.

The tarball snapshots the worker environment. The manager entrypoint and its
imports still come from the active checkout, and Work Queue separately receives
`analysis_processor.py` from that checkout as an extra input file.

Select exactly one of `--dry-run` and `--execute`. Help or missing mode is
nonexecuting. Dry-run resolves both units through the actual dedicated wrapper
and prints each final Python command without making output or starting a
manager. Synthetic dry-run arguments are for interface tests only. Execute
requires an existing environment archive. It refuses pre-existing PKLs and
sidecars and rechecks each unit's output paths before launch. The native
producer refuses existing Work Queue log files. No overwrite, retry, resume,
or tag change is automatic.

Each unit uses four separate eras (Run2: UL16, UL16APV, UL17, UL18; Run3:
2022, 2022EE, 2023, 2023BPix), 50000-event chunks, MC-only
`processor_output`, `--do-systs`, `--all-analysis`,
`--np-postprocess=skip`, and `--ttgamma-sample-role-policy split`. Block a selects
`2l` and `2los_1tau` with `njets lj0pt ptz lt`; block d selects
`3l_onZ_tau` with `njets lj0pt ptz lt`. The wider `run3_prod` a block also
selects `2lss_1tau`, whose on-Z channels can fill `ptz_wtau`. Neither selected
excluded-MC a category has a channel that fills that family, so omitting it
loses no physical contribution from this slice. Each requested family has an
explicit full-custom sumw2 companion, including `njets_sumw2`. The `_njets`
output basename is the unchanged campaign identifier, not a histogram limit.
Run2 uses the same applicable SR-full histogram families and category groups.
The first plot wave remains `njets` only; it does not narrow producer output.
Work Queue requests one processing core and
32000 MB plus two accumulating cores. These are task requests, not worker
factory sizing or available-capacity findings.

The standard `run3_prod` SR source uses `--do-np --defer-np` and implicit
production sumw2. Its `--do-np` resolves and certifies same-year data and prompt
MC contributors before the executor, which also determines the implicit
sumw2 targets; the separate `run_data_driven.py` step then runs after the
source. The excluded CFG deliberately contains backgrounds only, so it has no
same-year data contributors and cannot satisfy that nonprompt source contract.
The dedicated producer therefore does not request `--do-np` or start a data
driven stage. It retains `--np-postprocess=skip`; no nonprompt output or
nonprompt certification is claimed. With no enabled data driven product, the
implicit production policy would select no sumw2 targets for these backgrounds.
Explicit `--sumw2-family` for each requested distribution preserves the
nominal MC statistical companions consumed by the SR plotter. This changes
statistical sidecar provenance relative to standard `run3_prod` and includes
background sumw2 targets regardless of that profile's contributor selection.
The event selection, nominal weights and systematic variation switch use the
same source flags for these units, while the background-only input universe
remains intentional.

The foreground producer inherits stdout and stderr directly from the invoking
terminal, as in `run_cr.sh`. The launcher waits for its native return code and
starts d only if a returns zero. There is no display helper or runner log.
The native producer owns PKL and sidecar creation, archive validation, Work
Queue staging, and native logs. A terminal Ctrl-C reaches the foreground
process group as it does for `run_cr.sh`; partial products remain for review.

## Campaign identity and preflight

The caller selects `--campaign-tag` and `--output-root` for each campaign.
`run_excluded_mc.sh` derives a manager name
`<campaign-tag>-<run>-block-<a|d>` and writes
`<output-root>/<run>/excluded_mc_<run>_block_<a|d>_njets.pkl.gz`
with its adjacent metadata sidecar. Native logs belong below that run's
`work_queue_logs/<outname>/source_native/` directory. The tag must satisfy
the launcher's safe-name checks, include `excludedmc` in the derived manager
name, and avoid the current user name. `--manager-name` is an assertion of
this derived identity, not an override.

Choose the archive and adjacent manifest explicitly; no historical campaign
path is a default. Before execution, inspect the dry-run commands, exact output
collisions, environment binding, staging/quota thresholds, manager names,
ports, and externally provisioned worker resources. An interrupted attempt
requires a separate inspection of its processes, artifacts, logs and staging
before any recovery decision. This interface does not report the live state
or completion of a particular campaign.
