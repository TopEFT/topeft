# State of the embedded sumw2 consumer chain

The v3 candidate supports source publication, certified nonprompt/flips
transformation, homogeneous composition, cards, yields and the maintained plot
renderer. Qualification is bounded to targeted fixtures and the recorded
nominal real-event comparison; production adoption remains a separate decision.

The paired provider is `lazdlie/topcoffea` branch `feature/multicell-clean`,
commit `a91e141b1ea316589e4008605658513a2e1f0567`. This includes moment recording
and origin transport plus Weight-storage emptiness handling. Source-branch
synchronization is checked separately; local test results alone do not prove it.
The dedicated Python 3.11 environment is specified in `environment-multicell.yml`.
Its Numba 0.57.1 and llvmlite 0.40.1 pins avoid the NanoEvents LLVM IR failure
observed with Numba 0.68.1 and awkward 1.10.3.

There are two support stacks. Existing v2 products remain in their bound legacy
code/environment pairs; there is no universal legacy baseline or requirement to
read them through the new pipeline. The new candidate uses v3 with global
`embedded` or `none`. TOPCOFFEA retains its legacy backend and opt-in MultiCell
backend in the same repository. A parallel upstream TOPEFT line, its base,
retargeting and integration require a separate owner decision.

V3 certification preserves source sample variants, year-specific contributors,
product applicability, exact input identity and lineage. Nonprompt subtraction
uses the maintained independent-contribution convention and the SM-point EFT
projection. Inline and deferred DD use the same producer and artifact writer;
streaming and materialized paths retain numerical and recording evidence.

Cards and plotting use the existing bounded consumer view in `nominal_schema`,
which extracts only recorded nominal Q and retains coefficients/flows. This view
is transient and is never published as a v2 artifact. `YieldTools` consumes
embedded Q directly; yield-only requests can omit uncertainty, including
nonzero-WC yields. Q denotes the SM second moment, not generic EFT variance.

The specialist diboson processor and standalone historical diboson-SF reader
remain separate legacy paths. Diboson process contributions in the maintained
v3 card/plot chain use the common consumer boundary; the separate legacy reader
is not claimed migrated. No arbitrary pickle-reading scripts are certified.

The 40,969-event private ttH UL17 comparison covers nominal ht, lj0pt, njets
and ptz, 253 category keys and 506 Y/Q arrays including flow. It does not cover
real Data, non-EFT MC, 2D or all systematic variations; those receive small
fixtures only. Buffer inspection distinguishes actual owned allocations from
skipped Q fills, without RSS or full-scale performance claims. No whole-project
suite, production campaign, fit or scientific acceptance is implied.
