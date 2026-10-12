"""Generate the source-derived nuisance and standardizer reference page."""

import argparse
import contextlib
import difflib
import io
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUTPUT = REPO_ROOT / "docs/reference/systematics_nuisance_catalog.md"


def _mapping(value, name):
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Unsupported {name} registry structure")
    return value


def _text(value):
    if not isinstance(value, str):
        raise ValueError(f"Expected text, got {value!r}")
    return value


def _rows_unique(rows, name):
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate normalized {name} rows")
    return tuple(sorted(rows))


def build_catalog_model():
    """Read maintained runtime/config authorities and normalize all static rows."""
    from topeft.modules.datacard_tools import (
        DatacardMaker, FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES,
        cross_run_mva_identities, lepton_mva_correlation_modes,
        resolve_shape_nuisance_identity, run2_luminosity_excluded_processes,
        run2_luminosity_shape_factors, run3_luminosity_shape_factors,
    )
    from analysis.topeft_run2.standardize_systematics import SYST_NAMING_MAP

    with contextlib.redirect_stdout(io.StringIO()):
        maker = DatacardMaker(hists={}, do_nuisance=False, verbose=False)

    year_rows = []
    for base, edges in _mapping(maker.syst_year_corr, "year correlation").items():
        groups = []
        for period, partners in _mapping(edges, "year correlation edges").items():
            if not isinstance(partners, list) or not partners or not all(isinstance(p, str) for p in partners):
                raise ValueError(f"Unsupported year correlation partners: {base}")
            group = frozenset((period, *partners))
            if len(group) != len(partners) + 1:
                raise ValueError(f"Duplicate year correlation partner: {base}")
            for partner in partners:
                if partner not in edges or period not in edges[partner]:
                    raise ValueError(f"Contradictory year correlation: {base}")
            groups.append(group)
        for group in set(groups):
            for period in group:
                if frozenset((period, *edges[period])) != group:
                    raise ValueError(f"Contradictory year correlation group: {base}")
            year_rows.append((base, ", ".join(sorted(group))))
    year_rows = _rows_unique(year_rows, "year correlation")

    if not isinstance(maker.run_decorrelate, list) or not all(isinstance(x, str) for x in maker.run_decorrelate):
        raise ValueError("Unsupported run decorrelation registry")
    run_rows = _rows_unique([
        (base,
         resolve_shape_nuisance_identity(base, "_run2", maker.run_decorrelate)[0],
         resolve_shape_nuisance_identity(base, "_run3", maker.run_decorrelate)[0])
        for base in maker.run_decorrelate
    ], "run decorrelation")
    if len(run_rows) != len(maker.run_decorrelate):
        raise ValueError("Duplicate run decorrelation base")
    mva_rows = _rows_unique([
        (mode, base,
         resolve_shape_nuisance_identity(base, "_run2", maker.run_decorrelate, mode)[0],
         resolve_shape_nuisance_identity(base, "_run3", maker.run_decorrelate, mode)[0])
        for mode in lepton_mva_correlation_modes
        for base in cross_run_mva_identities
    ], "lepton MVA correlation")

    shape_rows = []
    for base, rules in _mapping(maker.syst_shape_decorrelate, "shape decorrelation").items():
        if not isinstance(rules, list):
            raise ValueError(f"Unsupported shape decorrelation rules: {base}")
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {"matches", "group"} or not isinstance(rule["matches"], list):
                raise ValueError(f"Unsupported shape decorrelation rule: {base}")
            for pattern in rule["matches"]:
                group = _text(rule["group"])
                final_identity = f"{base}_{group}" if group else f"{base}_<matched process>"
                shape_rows.append((base, _text(pattern), group, final_identity))
    shape_rows = _rows_unique(shape_rows, "shape decorrelation")

    tau_prefixes = FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES
    if not isinstance(tau_prefixes, tuple) or not all(isinstance(prefix, str) and prefix for prefix in tau_prefixes):
        raise ValueError("Unsupported fully specified Tau prefixes")
    tau_rows = _rows_unique(
        [(prefix, "preserve base; bypass generic run suffix") for prefix in tau_prefixes],
        "fully specified Tau precedence",
    )
    if len(tau_rows) != len(tau_prefixes):
        raise ValueError("Duplicate fully specified Tau prefix")
    if not isinstance(run2_luminosity_excluded_processes, (set, frozenset)) or not all(
        isinstance(process, str) for process in run2_luminosity_excluded_processes
    ):
        raise ValueError("Unsupported luminosity process exclusions")
    luminosity_exclusion_rows = _rows_unique(
        [(process, "all maintained Run 2 and Run 3 luminosity shape components")
         for process in run2_luminosity_excluded_processes],
        "luminosity exclusions",
    )

    luminosity_rows = {}
    for run, factors in (("run2", run2_luminosity_shape_factors), ("run3", run3_luminosity_shape_factors)):
        rows = []
        for identity, periods in _mapping(factors, "luminosity factors").items():
            for period, factor in _mapping(periods, "luminosity periods").items():
                if not isinstance(factor, (float, int)) or isinstance(factor, bool):
                    raise ValueError("Unsupported luminosity factor")
                rows.append((identity, period, str(factor)))
        luminosity_rows[run] = _rows_unique(rows, "luminosity")

    rate_rows = {}
    correlation_rows = {}
    diboson_rows = {}
    for run in ("run2", "run3"):
        path = REPO_ROOT / f"topeft/params/rate_systs_{run}.json"
        config = _mapping(json.loads(path.read_text(encoding="utf-8")), "rate JSON")
        if set(config) != {"rate_uncertainties", "correlations", "diboson_njets"}:
            raise ValueError(f"Unsupported rate JSON sections: {run}")
        correlations = _mapping(config["correlations"], "rate correlations")
        corr_rows = []
        for process, families in correlations.items():
            for family, tag in _mapping(families, "rate correlation families").items():
                corr_rows.append((process, family, _text(tag)))
        correlation_rows[run] = _rows_unique(corr_rows, "rate correlation")
        rows = []
        for family, processes in _mapping(config["rate_uncertainties"], "rate uncertainties").items():
            for process, uncertainty in _mapping(processes, "rate uncertainty processes").items():
                if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float, str)):
                    raise ValueError("Unsupported rate uncertainty")
                tag = correlations.get(process, {}).get(family, "")
                identity = f"{family}_{tag}" if tag else family
                rows.append((family, process, tag, identity, str(uncertainty)))
        rate_rows[run] = _rows_unique(rows, "rate uncertainty")
        special = []
        for process, bins in _mapping(config["diboson_njets"], "diboson njets").items():
            for njets, factor in _mapping(bins, "diboson njets bins").items():
                if not isinstance(factor, (int, float)) or isinstance(factor, bool):
                    raise ValueError("Unsupported diboson njets factor")
                special.append((process, njets, str(factor)))
        diboson_rows[run] = _rows_unique(special, "diboson njets")

    maps = _mapping(SYST_NAMING_MAP, "standardizer")
    if set(maps) != {"run2", "run3"}:
        raise ValueError("Unsupported standardizer run domains")
    standardizer_rows = {}
    for run in ("run2", "run3"):
        standardizer_rows[run] = _rows_unique([
            (_text(source), _text(target), "yes" if source != target else "no")
            for source, target in _mapping(maps[run], "standardizer map").items()
        ], "standardizer")

    payloads = _mapping(DatacardMaker.MISSING_PARTON_DEFAULT_PAYLOADS, "missing parton payloads")
    if set(payloads) != {"run2", "run3"}:
        raise ValueError("Unsupported missing parton run domains")
    return {
        "year": year_rows, "run": run_rows, "mva": mva_rows, "shape": shape_rows,
        "tau_precedence": tau_rows, "luminosity_exclusions": luminosity_exclusion_rows,
        "luminosity": luminosity_rows, "rates": rate_rows,
        "correlations": correlation_rows, "diboson": diboson_rows,
        "standardizer": standardizer_rows,
        "missing_parton": (_text(DatacardMaker.MISSING_PARTON_NUISANCE_NAME),
                           tuple((run, _text(payloads[run])) for run in ("run2", "run3"))),
    }


def _cell(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>").replace("`", "\\`")


def _table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "".join(
        "| " + " | ".join(_cell(x) for x in row) + " |\n" for row in rows
    )


def render_catalog(model):
    """Render one deterministic Markdown materialized view."""
    index = [
        ("Year correlation", "Run 2 / Run 3", len(model["year"]), "Groups source periods", "DatacardMaker.syst_year_corr"),
        ("Run decorrelation", "Run 2 / Run 3", len(model["run"]), "Resolves final base by run", "DatacardMaker.run_decorrelate; resolve_shape_nuisance_identity"),
        ("Lepton MVA correlation", "Run 2 / Run 3", len(model["mva"]), "Selects per-run or shared MVA identity", "resolve_shape_nuisance_identity"),
        ("Fully specified Tau precedence", "Both runs", len(model["tau_precedence"]), "Preserves already specified nuisance bases", "FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES; resolve_shape_nuisance_identity"),
        ("Process shape decorrelation", "Both runs", len(model["shape"]), "Matches processes and group tokens", "DatacardMaker.syst_shape_decorrelate"),
        ("Luminosity process exclusions", "Run 2 / Run 3", len(model["luminosity_exclusions"]), "Excludes logical processes from luminosity templates", "run2_luminosity_excluded_processes"),
    ]
    for run in ("run2", "run3"):
        label = "Run 2" if run == "run2" else "Run 3"
        index.extend((
            ("Luminosity shapes", label, len(model["luminosity"][run]), "Component factors by period", f"{run}_luminosity_shape_factors"),
            ("Rate uncertainties", label, len(model["rates"][run]), "Configured process values", f"rate_systs_{run}.json"),
            ("Rate correlation tags", label, len(model["correlations"][run]), "Configured process-family tags", f"rate_systs_{run}.json"),
            ("Diboson njets", label, len(model["diboson"][run]), "Configured jet-bin factors", f"rate_systs_{run}.json"),
            ("Standardizer map", label, len(model["standardizer"][run]), "Final identity rename or explicit identity mapping", "SYST_NAMING_MAP"),
        ))
    parts = [
        "# Systematics nuisance catalog\n\n",
        "**Generated file: do not edit manually.** Canonical scientific and software authorities remain "
        "`topeft/modules/datacard_tools.py`, `topeft/params/rate_systs_run2.json`, "
        "`topeft/params/rate_systs_run3.json`, and `analysis/topeft_run2/standardize_systematics.py`. "
        "Regenerate with `python tools/generate_systematics_nuisance_catalog.py --write`; "
        "verify freshness with `python tools/generate_systematics_nuisance_catalog.py --check`. "
        "DatacardMaker policy and standardizer mapping are separate layers.\n\n",
        "## Physicist quick index\n\n",
        _table(("Mechanism", "Run/domain", "Entries", "Controls", "Canonical authority"), index), "\n",
        "## DatacardMaker year-correlation groups\n\n",
        "Each row is one normalized group; no particular textual year label is guaranteed to survive card construction.\n\n",
        _table(("Source nuisance base", "Period group", "Mechanism/authority"),
               ((base, group, "syst_year_corr") for base, group in model["year"])), "\n",
        "## DatacardMaker run-domain decorrelation\n\n",
        _table(("Source base", "Run 2 final base", "Run 3 final base"), model["run"]), "\n",
        "## Selectable electron and muon MVA correlation\n\n",
        "`per-run` is the default. `cross-run` shares each flavor's nuisance across Run 2 and Run 3; "
        "electron and muon remain separate. DatacardMaker selects the identity before TXT and ROOT writing. "
        "The original TXT/ROOT and standardized TXT/ROOT use the same MVA identities: unsuffixed "
        "names are shared, while run-suffixed names are per-run. The standardizer infers this from "
        "source identities and rejects mixed topology within either flavor.\n\n",
        _table(("Mode", "Source base", "Run 2 final base", "Run 3 final base"), model["mva"]), "\n",
        "## Fully specified Tau identity precedence\n\n",
        "A nuisance base beginning with a maintained prefix is already fully specified. "
        "`resolve_shape_nuisance_identity` preserves that base before the generic `run_decorrelate` suffix path.\n\n",
        _table(("Maintained prefix", "Final identity rule"), model["tau_precedence"]), "\n",
        "## Process-conditioned shape decorrelation\n\n",
        "An empty group token appends the matched process name; unmatched processes retain the original systematic identity.\n\n",
        _table(("Systematic base", "Process match pattern", "Group token", "Final identity rule"), model["shape"]), "\n",
        "## Luminosity shape nuisances\n\n",
        "The maintained luminosity components are not applied to these logical process identities. "
        "The exclusion authority is shared by the Run 2 and Run 3 template paths.\n\n",
        _table(("Excluded logical process", "Applies to"), model["luminosity_exclusions"]), "\n",
    ]
    for run in ("run2", "run3"):
        label = "Run 2" if run == "run2" else "Run 3"
        parts += [f"### {label} luminosity components\n\n", _table(("Nuisance identity", "Period", "Factor"), model["luminosity"][run]), "\n"]
    parts += ["## Configured rate nuisances\n\n"]
    for run in ("run2", "run3"):
        label = "Run 2" if run == "run2" else "Run 3"
        parts += [f"### {label} rate uncertainties\n\n", _table(("Source/config family", "Process", "Correlation tag", "Resolved DatacardMaker nuisance identity", "Uncertainty"), model["rates"][run]), "\n",
                  f"### {label} correlation tags\n\n", _table(("Process", "Nuisance family", "Tag"), model["correlations"][run]), "\n",
                  f"### {label} diboson_njets\n\n", _table(("Process", "Jet count", "Factor"), model["diboson"][run]), "\n"]
    missing_name, payloads = model["missing_parton"]
    parts += ["### Missing-parton rate nuisance\n\n",
              f"Static nuisance identity: `{_cell(missing_name)}`. DatacardMaker selects the run-era payload and owns the card value; "
              "the ROOT payload owns channel-dependent values. Those dynamic values are not in this catalog.\n\n",
              _table(("Run", "Default payload authority"), payloads), "\n",
              "## Complete standardizer mapping\n\n"]
    for run in ("run2", "run3"):
        label = "Run 2" if run == "run2" else "Run 3"
        parts += [f"### {label} SYST_NAMING_MAP\n\n", _table(("Input final nuisance identity", "Standardized nuisance identity", "Renamed"), model["standardizer"][run]), "\n"]
    parts += ["## Scope boundary\n\n",
              "This generated catalog is exhaustive for the current active static nuisance policy authorities rendered here "
              "and for `SYST_NAMING_MAP`. Nuisance labels discovered only from concrete input histograms or artifacts "
              "are not presented as a universal static inventory. Alternative nuisance representations intentionally "
              "not written to current datacards are outside the active inventory. Actual package and card nuisance populations are artifacts, "
              "not source registries. Standardization does not decide scientific correlation topology.\n"]
    return "".join(parts)


def write_catalog(output=OUTPUT):
    content = render_catalog(build_catalog_model())
    encoded = content.encode("utf-8")
    if not output.exists() or output.read_bytes() != encoded:
        output.write_bytes(encoded)
        return "written"
    return "unchanged"


def check_catalog(output=OUTPUT, *, stream=sys.stdout):
    expected = render_catalog(build_catalog_model())
    actual_bytes = output.read_bytes() if output.exists() else b""
    if actual_bytes == expected.encode("utf-8"):
        return True
    actual = actual_bytes.decode("utf-8", errors="replace")
    diff = difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                fromfile=str(output), tofile="generated expectation")
    for line in list(diff)[:80]:
        stream.write(line)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        print(write_catalog())
        return 0
    return 0 if check_catalog() else 1


if __name__ == "__main__":
    raise SystemExit(main())
