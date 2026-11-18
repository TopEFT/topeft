# State of the embedded sumw2 consumer chain

The v3 candidate supports source publication, certified nonprompt/flips
transformation, homogeneous composition, cards, yields and the maintained plot
renderer. Qualification is bounded to targeted fixtures and the recorded
nominal real-event comparison; production adoption remains a separate decision.

[TOPCOFFEA #97](https://github.com/TopEFT/topcoffea/pull/97) integrated
`ef941d2cf7d93182c85ff45eb93af18fc9e6bce2`;
[TOPEFT #549](https://github.com/TopEFT/topeft/pull/549) integrated
`54dce429d98529a374ba993f5c3cf2f5d91a12a1` on `top-26-006_multicell`.
The [README installation recipe](README.md#installation-orientation) pins the
public integrated provider. TOPCOFFEA keeps its legacy backend as default;
the TOPEFT v3 producer explicitly selects MultiCell.

The qualified R4 run used TOPCOFFEA
`a91e141b1ea316589e4008605658513a2e1f0567` and TOPEFT
`73aa5f50c9535f5e0ae7181240b8bae3b1dc5098`.
Their trees match the respective merges exactly:
`2ee69b862c4f5ed13984470e771d1d4b51cb3a9c` (provider) and
`6af9e40fd46ea5f71d58f616463deb8812a2355f` (consumer).
Tree equality identifies tracked content, not execution history or environment.
The dedicated Python 3.11 environment is specified in `environment-multicell.yml`.
Its Numba 0.57.1 and llvmlite 0.40.1 pins avoid the NanoEvents LLVM IR failure
observed with Numba 0.68.1 and awkward 1.10.3.

There are two support stacks. Existing v2 products remain in their bound legacy
code/environment pairs; there is no universal legacy baseline or requirement to
read them through the new pipeline. The new candidate uses v3 with global
`embedded` or `none`. TOPCOFFEA retains its legacy backend and opt-in MultiCell
backend in the same repository. The parallel upstream TOPEFT line is
`top-26-006_multicell`; production adoption and propagation to other analysis lines remain separate decisions.

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

## R4 consumer evidence and limits

The retained R4 report is
`TOP26006_L12_MULTICELL_V3_CONSUMER_COMPLETION_20261117_R4_report.md`
(SHA-256 `ca3d18b95925e8e9f5b697e1caccff560569dcccc4f926bad23726b1b3e9a70a`),
in the coordination workspace's matching `reports/diagnostics/` directory.
It is local evidence, not a file distributed with this repository.
The immutable [consumer fixtures](https://github.com/TopEFT/topeft/blob/54dce429d98529a374ba993f5c3cf2f5d91a12a1/tests/test_v3_multicell_consumers.py)
define the controlled scope. All 22 focused v3 cases passed in R4; the wider
latest-result accounting is 545 passed, four retained legacy/matrix failures
and two skips, not a wholly passing suite.

The source/DD roundtrip and streaming/materialized fixtures exercise actual
nonprompt/flips publication and lineage. Their independent contributions give
Y = 3 - 2 - 1 = 0 and Q = 9 + 4 + 1 = 14; scaling by -2 gives Q = 56.
Yield fixtures cover recorded SM Q and explicit yield-only evaluation at
nonzero WC. Card fixtures write ROOT/TXT/scalings and check EFT decomposition
and independent renorm/fact nuisance shapes. Renderer fixtures produce stat
and total PNG output with diboson contributions; they do not quantify numerical
band widths or qualify the specialist historical diboson reader.

The R3 nominal 1D comparison reused by R4 is a separate population: 40,969
private simulated ttH UL17 events, four observable families, 253 keys and
506 Y/Q arrays including flow. Neither population establishes campaign-scale
performance or production acceptance.
