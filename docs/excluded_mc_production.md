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
source JSON metadata. Run2 remains on hold because `TTToHadronic` is not
available in the active CR inputs.

## Interface

Use `--run run3 --block all` for the a-then-d campaign. `--block a`
and `--block d` select one unit; these single-unit forms also resolve Run2,
but the current campaign does not run Run2. The required options are
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

Each unit uses all four Run3 eras (2022, 2022EE, 2023, 2023BPix), 50000-event
chunks, `njets` with explicit full-custom `njets_sumw2`, MC-only
`processor_output`, `--np-postprocess=skip`, and
`--ttgamma-sample-role-policy split`. Block a selects `2l` and `2los_1tau`;
block d selects `3l_onZ_tau`. Work Queue requests one processing core and
32000 MB plus two accumulating cores. These are task requests, not worker
factory sizing or available-capacity findings.

The foreground producer inherits stdout and stderr directly from the invoking
terminal, as in `run_cr.sh`. The launcher waits for its native return code and
starts d only if a returns zero. There is no display helper or runner log.
The native producer owns PKL and sidecar creation, archive validation, Work
Queue staging, and native logs. A terminal Ctrl-C reaches the foreground
process group as it does for `run_cr.sh`; partial products remain for review.

## Immediate Run3 campaign identity

The planned tag is
`top26006-excludedmc-cross-era-njets-20261018-001`. Derived managers are
`top26006-excludedmc-cross-era-njets-20261018-001-run3-block-a` and
`top26006-excludedmc-cross-era-njets-20261018-001-run3-block-d`; the selector
is `^top26006-excludedmc-cross-era-njets-20261018-001-run3-block-(a|d)$`.
These are the identities for the owner-run replacement of the interrupted
attempt. Lane HTCondor's existing factory selector matches these two names.

The prospective output root is
`/groups/klannon/apiccine/top26006_excluded_mc_njets_corrected_20261018_run3_001/`.
The aborted output root and exact old Run3/a staging directory were removed
after bounded ownership and process checks. Capacity, environment binding,
staging/quota paths, free-space thresholds, ports, and worker readiness require
fresh production preflight. The prior
`top26006_excluded_mc_ul18_njets_probe_001/cross_era` products are immutable
historical evidence and are not part of this corrected campaign.

The existing archive and adjacent manifest currently reside under
`/groups/klannon/apiccine/top26006_excluded_mc_env_build_20261019_001/topeft-envs/`.
The desired `analysis/topeft_run2/topeft-envs/` location is a separate storage
follow-up. Neither archive nor manifest was moved for this launcher change.
The owner runs the campaign manually from `analysis/topeft_run2` in the active
Conda environment. Lane HTCondor owns its worker factory.
