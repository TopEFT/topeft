"""Coverage and freshness for the generated nuisance reference."""

import io
import json

from analysis.topeft_run2.standardize_systematics import SYST_NAMING_MAP
from topeft.modules.datacard_tools import (
    DatacardMaker, FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES,
    cross_run_mva_identities, lepton_mva_correlation_modes,
    resolve_shape_nuisance_identity, run2_luminosity_excluded_processes,
    run2_luminosity_shape_factors, run3_luminosity_shape_factors,
)
from tools.generate_systematics_nuisance_catalog import (
    OUTPUT, REPO_ROOT, build_catalog_model, check_catalog, render_catalog,
)


def test_render_is_deterministic_and_committed_page_is_fresh():
    model = build_catalog_model()
    first = render_catalog(model)
    assert first == render_catalog(build_catalog_model())
    assert first == OUTPUT.read_text(encoding="utf-8")
    assert check_catalog(stream=io.StringIO())


def test_standardizer_maps_are_complete():
    model = build_catalog_model()
    for run in ("run2", "run3"):
        assert model["standardizer"][run] == tuple(sorted(
            (source, target, "yes" if source != target else "no")
            for source, target in SYST_NAMING_MAP[run].items()
        ))


def test_datacard_static_registries_are_complete():
    maker = DatacardMaker(hists={}, do_nuisance=False, verbose=False)
    model = build_catalog_model()
    expected_groups = set()
    for base, edges in maker.syst_year_corr.items():
        for period, partners in edges.items():
            expected_groups.add((base, ", ".join(sorted((period, *partners)))))
    assert set(model["year"]) == expected_groups
    assert model["run"] == tuple(sorted(
        (base,
         resolve_shape_nuisance_identity(base, "_run2", maker.run_decorrelate)[0],
         resolve_shape_nuisance_identity(base, "_run3", maker.run_decorrelate)[0])
        for base in maker.run_decorrelate
    ))
    assert model["mva"] == tuple(sorted(
        (mode, base,
         resolve_shape_nuisance_identity(base, "_run2", maker.run_decorrelate, mode)[0],
         resolve_shape_nuisance_identity(base, "_run3", maker.run_decorrelate, mode)[0])
        for mode in lepton_mva_correlation_modes
        for base in cross_run_mva_identities
    ))
    page = OUTPUT.read_text(encoding="utf-8")
    for mode, base, run2, run3 in model["mva"]:
        assert f"| {mode} | {base} | {run2} | {run3} |" in page
        if mode == "cross-run":
            assert run2 == run3 == base
        else:
            assert (run2, run3) == (f"{base}_13TeV", f"{base}_13p6TeV")
    assert "_13TeV_13p6TeV" not in page
    assert model["shape"] == tuple(sorted(
        (base, pattern, rule["group"],
         f"{base}_{rule['group']}" if rule["group"] else f"{base}_<matched process>")
        for base, rules in maker.syst_shape_decorrelate.items()
        for rule in rules for pattern in rule["matches"]
    ))
    assert model["tau_precedence"] == tuple(sorted(
        (prefix, "preserve base; bypass generic run suffix")
        for prefix in FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES
    ))
    for prefix in FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES:
        assert resolve_shape_nuisance_identity(f"{prefix}exampleUp", "_run3", maker.run_decorrelate) == (
            f"{prefix}example", f"{prefix}exampleUp"
        )
    assert model["luminosity_exclusions"] == tuple(sorted(
        (process, "all maintained Run 2 and Run 3 luminosity shape components")
        for process in run2_luminosity_excluded_processes
    ))
    for run, factors in (("run2", run2_luminosity_shape_factors),
                         ("run3", run3_luminosity_shape_factors)):
        assert model["luminosity"][run] == tuple(sorted(
            (identity, period, str(factor))
            for identity, periods in factors.items()
            for period, factor in periods.items()
        ))


def test_rate_json_sections_are_complete():
    model = build_catalog_model()
    for run in ("run2", "run3"):
        config = json.loads((REPO_ROOT / f"topeft/params/rate_systs_{run}.json").read_text())
        assert model["rates"][run] == tuple(sorted(
            (family, process, tag, f"{family}_{tag}" if tag else family, str(value))
            for family, processes in config["rate_uncertainties"].items()
            for process, value in processes.items()
            for tag in (config["correlations"].get(process, {}).get(family, ""),)
        ))
        assert model["correlations"][run] == tuple(sorted(
            (process, family, tag)
            for process, families in config["correlations"].items()
            for family, tag in families.items()
        ))
        assert model["diboson"][run] == tuple(sorted(
            (process, njets, str(value))
            for process, bins in config["diboson_njets"].items()
            for njets, value in bins.items()
        ))
    assert model["missing_parton"][0] == DatacardMaker.MISSING_PARTON_NUISANCE_NAME
    assert dict(model["missing_parton"][1]) == DatacardMaker.MISSING_PARTON_DEFAULT_PAYLOADS


def test_generated_policy_tables_match_normalized_model():
    model = build_catalog_model()
    page = OUTPUT.read_text(encoding="utf-8")
    for prefix, rule in model["tau_precedence"]:
        assert f"| {prefix} | {rule} |" in page
    for process, applicability in model["luminosity_exclusions"]:
        assert f"| {process} | {applicability} |" in page
    for base, pattern, group, identity in model["shape"]:
        assert f"| {base} | {pattern} | {group} | {identity} |" in page
    for run in ("run2", "run3"):
        for family, process, tag, identity, uncertainty in model["rates"][run]:
            assert f"| {family} | {process} | {tag} | {identity} | {uncertainty} |" in page


def test_check_detects_stale_and_missing_without_mutation(tmp_path):
    page = tmp_path / "catalog.md"
    page.write_text("stale\n", encoding="utf-8")
    before = page.read_bytes()
    output = io.StringIO()
    assert not check_catalog(page, stream=output)
    assert "stale" in output.getvalue()
    assert page.read_bytes() == before
    page.write_bytes(render_catalog(build_catalog_model()).replace("\n", "\r\n").encode())
    assert not check_catalog(page, stream=io.StringIO())
    page.unlink()
    assert not check_catalog(page, stream=io.StringIO())
    assert not page.exists()


def test_markdown_escaping_and_ordering():
    from tools.generate_systematics_nuisance_catalog import _rows_unique, _table

    rows = _rows_unique([("z.*", "", "0.9/1.1"), ("a|b", "x`y", "\\")], "synthetic")
    assert rows[0][0] == "a|b"
    rendered = _table(("Pattern", "Group", "Uncertainty"), rows)
    assert "a\\|b" in rendered
    assert "x\\`y" in rendered
    assert "z.*" in rendered and "0.9/1.1" in rendered
    assert "\\\\" in rendered
