# B-tag scale-factor payloads

## Packaged UL payloads

`topeft/data/btagSF/UL/` contains the AK4 DeepJet payloads used by the UL
correction path:

| Era | Scale-factor CSV selected by current source | MC-efficiency payload |
| --- | --- | --- |
| `2016` | `DeepJet_106XUL16postVFPSF_v2.csv` | `btagMCeff_2016.pkl.gz` |
| `2016APV` | `wp_deepJet_106XUL16preVFP_v2.csv` | `btagMCeff_2016APV.pkl.gz` |
| `2017` | `wp_deepJet_106XUL17_v3.csv` | `btagMCeff_2017.pkl.gz` |
| `2018` | `wp_deepJet_106XUL18_v2.csv` | `btagMCeff_2018.pkl.gz` |

The current correction source selects these exact paths in
`topeft/modules/corrections.py`. The packaged CSVs correspond to the DeepJet
working-point scale-factor interface; the local historical note identifies the
combined measurement and medium working point, but does not provide complete
payload-generation provenance. Do not infer a newer recommendation or payload
identity from that note.

The retained source note cites the CMS UL recommendations for
[2017](https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL17)
and
[2018](https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL18).
These links record package provenance; they were not queried during this local
documentation pass and do not supersede the checked-in files selected by
current source.

Run 3 MC-efficiency payloads are selected from `topeft/data/btagSF/Run3/` by the
current correction source. The `btagMCeff_2022*.pkl.gz` files present in the UL
directory are not the authority for that Run 3 lookup.

The older reproduction procedure is retained as the
[historical b-tag MC-efficiency how-to](../how_to/historical/btag_mc_efficiency.md);
it does not define the current payload contract.

## Source authority and limits

- `topeft/modules/corrections.py` owns runtime file selection.
- `topeft/data/btagSF/UL/` owns the installed UL files.
- File replacement is a correction-payload change, not a documentation edit.
- The former package-local README did not establish checksums or full source
  manifests; those provenance limits remain explicit.

## Consumer interfaces

The functions below are developer-facing and have signature authority in
`topeft.modules.corrections`.

| Symbol | Parameters/defaults and return | Contract and failures |
| --- | --- | --- |
| `corrections.GetMCeffFunc` | Year; `wp="medium"`, `btagalgo="btagDeepFlavB"`, flavor default `b` → lookup callable | Selects the era/algo MC-efficiency pickle, reads `jetptetaflav`, and returns numerator/denominator efficiency lookup over pT, absolute eta, and hadron flavor. Unknown years fail; untrusted pickle inputs must not be substituted. |
| `corrections.GetBtagEff` | Jets, year; medium DeepJet defaults → per-jet efficiency array | Calls the MC-efficiency lookup on `jets.pt`, `abs(jets.eta)`, and `jets.hadronFlavour`. |
| `corrections.GetBTagSF` | Jets, UL year; `wp="MEDIUM"`, `syst="central"` → scale-factor array or annotated jets for variations | Selects the exact UL CSV above. UL16 light flavor intentionally uses the UL16APV evaluator. Unsupported years fail. |

These interfaces read installed payloads and may annotate the passed jet
collection for systematic variations. They do not generate or update payload
files. The correction functions and processor are their callers; correction
and selection tests own behavior beyond file-selection documentation.

Return to the [software-reference map](README.md) for the production,
artifact, and processor interfaces that consume correction results.
