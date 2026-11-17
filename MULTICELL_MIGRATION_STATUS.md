# State of the embedded sumw2 migration

The candidate remains incomplete and is not qualified for source-branch
synchronization or production adoption.

The tested provider is `lazdlie/topcoffea` branch `feature/multicell-clean`,
local commit `307c15ac8dc2381cf8d6fd43c08c44a369bf7931`. This local continuation
has not been pushed. Install the exact paired provider and TOPEFT candidates
editably into the dedicated environment described by `environment-multicell.yml`.
Its Numba and llvmlite pins avoid the LLVM IR failure observed in NanoEvents
with Numba 0.68.1 and awkward 1.10.3.

Completed behavior includes MultiCell provider parity, explicit moment recording
and origin through transforms, processor-output v3 reading and validation, and
homogeneous processor v3 merges. The v2 companion route is retained. Mixed
layouts and uncertified transformed v3 declarations are rejected.

The targeted provider, backend, schema, processor-allocation and artifact tests
have raw validation evidence. On the 40,969-event private ttH UL17 CI input,
nominal SM yields and second moments agree with the historical companion oracle
for ht, lj0pt, njets and ptz, including flow bins and matching categorical keys.
The oracle contains no coefficient or raw-count arrays. This comparison does
not cover Data, non-EFT MC, 2D or varied-systematic real events.

Remaining implementation and qualification include policy-certified production
sample and data-driven contracts in v3 sidecars; nonprompt/flips transformation
and lineage; actual cards/yields/plots/diboson entry points; and their streaming
versus materialized equivalence. Inline v3 nonprompt remains unsupported. These
limits must not be inferred closed from provider or artifact tests. No whole
project suite, campaign, performance qualification or production adoption is
claimed. Publication of either candidate remains conditional on full pair
qualification and the exact live permission/ref gates.
