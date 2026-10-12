"""Exact Tau correlation schema and process-conditioned legacy transport."""

import json
import os

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.corrections import (
    get_tau_energy_variation_specs,
    get_tau_eta_variation_specs,
    get_tau_vsjet_variation_specs,
)
from topeft.modules.datacard_tools import DatacardMaker
from analysis.topeft_run2.standardize_systematics import stale_run2_tau_apv_bases


RUN2_PERIODS = ("2016APV", "2016", "2017", "2018")
RUN3_PERIODS = ("2022", "2022EE", "2023", "2023BPix")
DMS = (0, 1, 10, 11)
RUN2_VSE_ETA = ("abseta0to1p5", "abseta1p5to2p3")
RUN2_VSMU_ETA = (
    "abseta0to0p4", "abseta0p4to0p8", "abseta0p8to1p2",
    "abseta1p2to1p7", "abseta1p7to2p3",
)
RUN3_VSE_ETA = ("abseta0to1p46", "abseta1p46to1p56", "abseta1p56to2p5")
RUN3_VSMU_ETA = (
    "abseta0to0p4", "abseta0p4to0p8", "abseta0p8to1p2",
    "abseta1p2to1p7", "abseta1p7to2p4",
)


def maker(hists=None, years=None, *, enable_legacy_run3_vsmu_recovery=False):
    return DatacardMaker(
        hists={} if hists is None else hists,
        year_lst=[] if years is None else years,
        var_lst=["x"], do_nuisance=True,
        skip_missing_parton_rate_syst=True, verbose=False,
        enable_legacy_run3_vsmu_recovery=enable_legacy_run3_vsmu_recovery,
    )


def test_legacy_recovery_flag_requires_bool():
    card = DatacardMaker(
        hists={}, year_lst=[], var_lst=["x"], do_nuisance=True,
        skip_missing_parton_rate_syst=True, verbose=False,
    )
    assert card.enable_legacy_run3_vsmu_recovery is False

    for value in (False, True):
        card = maker(enable_legacy_run3_vsmu_recovery=value)
        assert card.enable_legacy_run3_vsmu_recovery is value


@pytest.mark.parametrize("value", (
    "False", "True", "0", "1", 0, 1, None, [], {},
))
def test_legacy_recovery_flag_rejects_non_bool(value):
    with pytest.raises(
        TypeError, match="enable_legacy_run3_vsmu_recovery",
    ):
        maker(enable_legacy_run3_vsmu_recovery=value)


def semantic_groups(registry):
    groups = set()
    for base, rules in registry.items():
        if not base.startswith(("CMS_eff_t_DeepTau", "CMS_fake_t_DeepTau", "CMS_scale_t_DeepTau")):
            continue
        for period, partners in rules.items():
            for partner in partners:
                groups.add((base, *sorted((period, partner))))
    return groups


def expected_groups():
    run2 = {
        (f"CMS_fake_t_DeepTau2017v2p1_VSe_{eta}", "2016", "2016APV")
        for eta in RUN2_VSE_ETA
    } | {
        (f"CMS_fake_t_DeepTau2017v2p1_VSmu_{eta}", "2016", "2016APV")
        for eta in RUN2_VSMU_ETA
    } | {
        (f"CMS_scale_t_DeepTau2017v2p1_DM{dm}_genTau", "2016", "2016APV")
        for dm in DMS
    } | {
        (f"CMS_scale_t_DeepTau2017v2p1_DM{dm}_genElectron", "2016", "2016APV")
        for dm in (0, 1)
    }
    run3_vsmu = {
        (f"CMS_fake_t_DeepTau2018v2p5_VSmu_{eta}", *pair)
        for eta in RUN3_VSMU_ETA
        for pair in (("2022", "2022EE"), ("2023", "2023BPix"))
    }
    run3_efes = {
        (f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron", *pair)
        for dm in DMS
        for pair in (("2022", "2022EE"), ("2023", "2023BPix"))
    }
    return run2, run3_vsmu, run3_efes


def final_identity(source, registry):
    for period in sorted((*RUN2_PERIODS, *RUN3_PERIODS), key=len, reverse=True):
        if source.endswith(f"_{period}"):
            base = source.removesuffix(f"_{period}")
            rules = registry.get(base, {})
            if period in rules:
                return f"{base}_{min((period, *rules[period]))}"
            return source
    return source


def observed_family_identities(registry, periods):
    families = {name: set() for name in ("vsjet", "vse", "vsmu", "tes", "efes", "mues")}
    for period in periods:
        for source in get_tau_vsjet_variation_specs(period):
            families["vsjet"].add(final_identity(source, registry))
        for source, spec in get_tau_eta_variation_specs(period).items():
            families[spec["source"].lower()].add(final_identity(source, registry))
        for source, (kind, _) in get_tau_energy_variation_specs(period).items():
            family = {"genuine": "tes", "electron": "efes", "muon": "mues"}[kind]
            families[family].add(final_identity(source, registry))
    return families


def expected_family_identities(run):
    periods = RUN2_PERIODS if run == "run2" else RUN3_PERIODS
    tagger = "DeepTau2017v2p1" if run == "run2" else "DeepTau2018v2p5"
    run_label = "Run2" if run == "run2" else "Run3"
    vsjet = {
        f"CMS_eff_t_{tagger}_VSjet_dm_{part}_DM{dm}_{period}"
        for period in periods for part in ("stat1", "stat2") for dm in DMS
    } | {
        f"CMS_eff_t_{tagger}_VSjet_dm_syst_{period}" for period in periods
    } | {
        f"CMS_eff_t_{tagger}_VSjet_dm_syst_DM{dm}_{period}"
        for period in periods for dm in DMS
    } | {f"CMS_eff_t_{tagger}_VSjet_dm_syst_alleras_{run_label}"}
    if run == "run2":
        years = ("2016", "2017", "2018")
        vse = {f"CMS_fake_t_{tagger}_VSe_{eta}_{year}" for eta in RUN2_VSE_ETA for year in years}
        vsmu = {f"CMS_fake_t_{tagger}_VSmu_{eta}_{year}" for eta in RUN2_VSMU_ETA for year in years}
        tes = {f"CMS_scale_t_{tagger}_DM{dm}_genTau_{year}" for dm in DMS for year in years}
        efes = {f"CMS_scale_t_{tagger}_DM{dm}_genElectron_{year}" for dm in (0, 1) for year in years}
    else:
        vse = {f"CMS_fake_t_{tagger}_VSe_DM{dm}_{eta}_{period}" for dm in DMS for eta in RUN3_VSE_ETA for period in periods}
        vsmu = {f"CMS_fake_t_{tagger}_VSmu_{eta}_{year}" for eta in RUN3_VSMU_ETA for year in ("2022", "2023")}
        tes = {f"CMS_scale_t_{tagger}_DM{dm}_genTau_{period}" for dm in DMS for period in periods}
        efes = {f"CMS_scale_t_{tagger}_DM{dm}_genElectron_{year}" for dm in DMS for year in ("2022", "2023")}
    mues = {f"CMS_scale_t_{tagger}_DM{dm}_genMuon_{run_label}" for dm in DMS}
    return dict(vsjet=vsjet, vse=vse, vsmu=vsmu, tes=tes, efes=efes, mues=mues)


def inventory_entry(observed, expected):
    observed = set(observed)
    expected = set(expected)
    return {
        "observed": sorted(observed), "expected": sorted(expected),
        "missing": sorted(expected - observed), "extra": sorted(observed - expected),
        "observed_count": len(observed), "expected_count": len(expected),
    }


def test_exact_tau_schema_inventory():
    registry = maker().syst_year_corr
    for base, rules in registry.items():
        for period, partners in rules.items():
            for partner in partners:
                assert period in rules[partner], (base, period, partner)
    groups = semantic_groups(registry)
    run2, run3_vsmu, run3_efes = expected_groups()
    run2_vsj = {group for group in groups if "DeepTau2017v2p1_VSjet" in group[0]}
    run3_vse = {group for group in groups if "DeepTau2018v2p5_VSe" in group[0]}
    run3_tes = {group for group in groups if "DeepTau2018v2p5" in group[0] and "_genTau" in group[0]}
    removed_vsj = {
        (f"CMS_eff_t_DeepTau2017v2p1_VSjet_dm_{part}_DM{dm}", "2016", "2016APV")
        for part in ("stat1", "stat2") for dm in DMS
    } | {("CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst", "2016", "2016APV")} | {
        (f"CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM{dm}", "2016", "2016APV")
        for dm in DMS
    }
    source_vsmu = {
        source for period in RUN3_PERIODS
        for source, spec in get_tau_eta_variation_specs(period).items()
        if spec["source"] == "VSmu"
    }
    expected_source_vsmu = {
        f"CMS_fake_t_DeepTau2018v2p5_VSmu_{eta}_{period}"
        for eta in RUN3_VSMU_ETA for period in RUN3_PERIODS
    }
    entries = {
        "run2_correlation_groups": inventory_entry(groups & run2, run2),
        "run2_vsj_forbidden_groups": inventory_entry(run2_vsj, set()),
        "run2_vsj_removed_group_identities": {
            "removed_expected": sorted(removed_vsj),
            "observed_present": sorted(groups & removed_vsj),
        },
        "standardizer_stale_apv_bases": inventory_entry(
            stale_run2_tau_apv_bases, {group[0] for group in run2}
        ),
        "run3_vsmu_source_identities": inventory_entry(source_vsmu, expected_source_vsmu),
        "run3_vsmu_final_groups": inventory_entry(groups & run3_vsmu, run3_vsmu),
        "run3_efes_final_groups": inventory_entry(groups & run3_efes, run3_efes),
        "run3_vse_forbidden_groups": inventory_entry(run3_vse, set()),
        "run3_tes_forbidden_groups": inventory_entry(run3_tes, set()),
        "all_tau_correlation_groups": inventory_entry(groups, run2 | run3_vsmu | run3_efes),
    }
    for run, periods in (("run2", RUN2_PERIODS), ("run3", RUN3_PERIODS)):
        observed = observed_family_identities(registry, periods)
        expected = expected_family_identities(run)
        for family in expected:
            entries[f"{run}_{family}_final_identities"] = inventory_entry(observed[family], expected[family])
        entries[f"{run}_all_final_identities"] = inventory_entry(
            set().union(*observed.values()), set().union(*expected.values())
        )
    entries["combined_final_identities"] = inventory_entry(
        set().union(*(set(entries[f"{run}_all_final_identities"]["observed"]) for run in ("run2", "run3"))),
        set().union(*(set(entries[f"{run}_all_final_identities"]["expected"]) for run in ("run2", "run3"))),
    )
    output_path = os.environ.get("TAU_SCHEMA_INVENTORY_PATH")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(entries, output, indent=2, sort_keys=True)
            output.write("\n")
    assert not entries["run2_vsj_removed_group_identities"]["observed_present"]
    assert len(entries["run2_vsj_removed_group_identities"]["removed_expected"]) == 13
    for name, entry in entries.items():
        if name == "run2_vsj_removed_group_identities":
            continue
        assert not entry["missing"] and not entry["extra"], (name, entry)
    assert [entries[f"run2_{family}_final_identities"]["observed_count"] for family in ("vsjet", "vse", "vsmu", "tes", "efes", "mues")] == [53, 6, 15, 12, 6, 4]
    assert [entries[f"run3_{family}_final_identities"]["observed_count"] for family in ("vsjet", "vse", "vsmu", "tes", "efes", "mues")] == [53, 48, 10, 16, 8, 4]
    assert [entries[f"{run}_all_final_identities"]["observed_count"] for run in ("run2", "run3")] == [96, 139]
    assert entries["combined_final_identities"]["observed_count"] == 235


def make_eft_histogram(cells, *, tracked=False):
    output = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0, 1, name="x"),
        wc_names=["ctG"], storage="Double", track_raw_counts=tracked,
    )
    for process, channel, systematic, value in cells:
        output.fill(
            process=process, channel=channel, systematic=systematic,
            x=np.asarray([0.5]), eft_coeff=np.asarray([[value, value / 2, value / 4]]),
            record_raw_count=True if tracked else None,
        )
    return output


def payload_by_key(histogram):
    return {tuple(key): np.array(value, copy=True) for key, value in histogram.view(flow=True).items()}


def test_route_d_transport_preserves_full_eft_payload(monkeypatch):
    base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    cells = [
        (f"sample{period}", "channel", f"{base}_{period[:4]}{side}", value)
        for period, value in (("2022", 11), ("2022EE", 13), ("2023", 17), ("2023BPix", 19))
        for side in ("Up", "Down")
    ]
    cells += [(f"sample{period}", "channel", "nominal", 10) for period in RUN3_PERIODS]
    source = make_eft_histogram(cells, tracked=True)
    before = payload_by_key(source)
    card = maker(enable_legacy_run3_vsmu_recovery=True)
    monkeypatch.setattr(card, "correlate_years", lambda histogram, **_: histogram)
    card.read(hists={"x": source})
    result = card.hists["x"]
    after = payload_by_key(result)
    expected = {
        (process, channel,
         f"{base}_{process.removeprefix('sample')}{'Up' if systematic.endswith('Up') else 'Down'}"
         if systematic.startswith(base) else systematic): payload
        for (process, channel, systematic), payload in before.items()
    }
    assert set(after) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(after[key], expected[key])
    assert result.wc_names == source.wc_names == ["ctG"]
    assert result.dense_axes == source.dense_axes
    raw_before = {tuple(key): value for key, value in source.raw_counts(flow=True).items()}
    raw_after = {tuple(key): value for key, value in result.raw_counts(flow=True).items()}
    assert set(raw_after) == set(expected)
    for process, channel, systematic in raw_before:
        target = (
            f"{base}_{process.removeprefix('sample')}{'Up' if systematic.endswith('Up') else 'Down'}"
            if systematic.startswith(base) else systematic
        )
        np.testing.assert_array_equal(
            raw_after[(process, channel, target)],
            raw_before[(process, channel, systematic)],
        )


def test_route_d_native_passthrough_and_fail_closed(monkeypatch):
    base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    card = maker()
    monkeypatch.setattr(card, "correlate_years", lambda histogram, **_: histogram)
    native = make_eft_histogram([
        ("sample2022EE", "channel", f"{base}_2022EE{side}", 4)
        for side in ("Up", "Down")
    ])
    card.read(hists={"x": native})
    actual = payload_by_key(card.hists["x"])
    expected = payload_by_key(native)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    bad_cases = [
        ("sample", f"{base}_2022Up", "period"),
        ("sample2022EE", f"{base}_2023Up", "mismatch"),
        ("sample2022EE", f"{base}_2022XUp", "Unsupported"),
        ("sample2022EE", "CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022Up", "pre-collapsed"),
        ("sample2022EE", "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022Up", "pre-collapsed"),
        ("sample2022EE", "CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022Up", "pre-collapsed"),
        ("sample2022EE", "CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2022Up", "pre-collapsed"),
    ]
    for process, systematic, message in bad_cases:
        broken = make_eft_histogram([(process, "channel", systematic, 4)])
        with pytest.raises(ValueError, match=message):
            card.read(hists={"x": broken})
    collision = make_eft_histogram([
        ("sample2022EE", "channel", f"{base}_{period}{side}", 4)
        for period in ("2022", "2022EE") for side in ("Up", "Down")
    ])
    with pytest.raises(ValueError, match="collision"):
        card.read(hists={"x": collision})
    incomplete = make_eft_histogram([
        ("sample2022EE", "channel", f"{base}_2022Up", 4),
    ])
    with pytest.raises(ValueError, match="Incomplete"):
        card.read(hists={"x": incomplete})
    mixed = make_eft_histogram([
        (f"sample{period}", "channel", f"{base}_{source_period}{side}", 4)
        for period, source_period in (("2022EE", "2022"), ("2023BPix", "2023BPix"))
        for side in ("Up", "Down")
    ])
    with pytest.raises(ValueError, match="Mixed"):
        card.read(hists={"x": mixed})
    global_nuisances = make_eft_histogram([
        ("sample2022EE", "channel", f"{base_name}{side}", 4)
        for base_name in (
            "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_Run3",
            "CMS_scale_t_DeepTau2018v2p5_DM0_genMuon_Run3",
        ) for side in ("Up", "Down")
    ])
    card.read(hists={"x": global_nuisances})
    assert set(payload_by_key(card.hists["x"])) == set(payload_by_key(global_nuisances))


@pytest.mark.parametrize("enable_recovery", (False, True))
def test_route_d_cross_eta_mixed_schema_fails_closed(monkeypatch, enable_recovery):
    bases = (
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4",
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8",
    )
    mixed = make_eft_histogram([
        ("sample2022EE", "channel", f"{base}_{period}{side}", 4)
        for base, period in zip(bases, ("2022", "2022EE"))
        for side in ("Up", "Down")
    ])
    card = maker(enable_legacy_run3_vsmu_recovery=enable_recovery)
    monkeypatch.setattr(card, "correlate_years", lambda histogram, **_: histogram)
    with pytest.raises(ValueError, match="Mixed native and legacy"):
        card.read(hists={"x": mixed})


def test_route_d_multi_eta_legacy_requires_opt_in(monkeypatch):
    bases = (
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4",
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8",
    )
    legacy = make_eft_histogram([
        ("sample2022EE", "channel", f"{base}_2022{side}", 4)
        for base in bases for side in ("Up", "Down")
    ])
    closed = maker()
    monkeypatch.setattr(
        closed, "correlate_years",
        lambda histogram, **_: pytest.fail("correlation reached before opt-in"),
    )
    with pytest.raises(ValueError, match="--enable-legacy-run3-vsmu-recovery"):
        closed.read(hists={"x": legacy})
    enabled = maker(enable_legacy_run3_vsmu_recovery=True)
    monkeypatch.setattr(enabled, "correlate_years", lambda histogram, **_: histogram)
    enabled.read(hists={"x": legacy})
    names = {key.systematic for key in enabled.hists["x"].categorical_keys}
    assert names == {f"{base}_2022EE{side}" for base in bases for side in ("Up", "Down")}


@pytest.mark.parametrize("periods", (("2022EE", "2023BPix"), ("2022", "2023"), ("2022EE",)))
def test_route_d_native_and_sparse_supported_schema_passes(periods, monkeypatch):
    bases = (
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4",
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8",
    )
    native = make_eft_histogram([
        (f"sample{period}", "channel", f"{base}_{period}{side}", 4)
        for period in periods for base in bases for side in ("Up", "Down")
    ])
    card = maker()
    monkeypatch.setattr(card, "correlate_years", lambda histogram, **_: histogram)
    card.read(hists={"x": native})
    assert set(payload_by_key(card.hists["x"])) == set(payload_by_key(native))


@pytest.mark.parametrize("systematic, message", (
    ("CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2023Up", "mismatch"),
    ("CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022XUp", "Unsupported"),
))
def test_route_d_opt_in_does_not_bypass_invalid_schema(systematic, message):
    bad = make_eft_histogram([("sample2022EE", "channel", systematic, 4)])
    card = maker(enable_legacy_run3_vsmu_recovery=True)
    with pytest.raises(ValueError, match=message):
        card.read(hists={"x": bad})


def test_route_d_opt_in_does_not_bypass_mixed_or_incomplete(monkeypatch):
    base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    card = maker(enable_legacy_run3_vsmu_recovery=True)
    monkeypatch.setattr(card, "correlate_years", lambda histogram, **_: histogram)
    mixed = make_eft_histogram([
        (f"sample{period}", "channel", f"{base}_{source}{side}", 4)
        for period, source in (("2022EE", "2022"), ("2023BPix", "2023BPix"))
        for side in ("Up", "Down")
    ])
    with pytest.raises(ValueError, match="Mixed"):
        card.read(hists={"x": mixed})
    incomplete = make_eft_histogram([("sample2022EE", "channel", f"{base}_2022Up", 4)])
    with pytest.raises(ValueError, match="Incomplete"):
        card.read(hists={"x": incomplete})
    collision = make_eft_histogram([
        ("sample2022EE", "channel", f"{base}_{period}{side}", 4)
        for period in ("2022", "2022EE") for side in ("Up", "Down")
    ])
    with pytest.raises(ValueError, match="collision"):
        card.read(hists={"x": collision})


def test_route_d_preserves_sumw2_sidecar(monkeypatch):
    base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    cells = [
        ("sample2022EE", "channel", f"{base}_2022{side}", value)
        for side, value in (("Up", 9), ("Down", 25))
    ]
    source = make_eft_histogram(cells)
    before = payload_by_key(source)
    card = maker(enable_legacy_run3_vsmu_recovery=True)
    monkeypatch.setattr(card, "correlate_years", lambda histogram, **_: histogram)
    card.read(hists={"x_sumw2": source})
    after = payload_by_key(card.hists["x_sumw2"])
    for side in ("Up", "Down"):
        np.testing.assert_array_equal(
            after[("sample2022EE", "channel", f"{base}_2022EE{side}")],
            before[("sample2022EE", "channel", f"{base}_2022{side}")],
        )


def test_route_d_legacy_read_reaches_year_pair_correlation_once():
    base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    cells = [
        (f"sample{period}", "channel", systematic, value)
        for period, nominal in (("2022", 10), ("2022EE", 20))
        for systematic, value in (
            ("nominal", nominal),
            (f"{base}_2022Up", nominal + 2),
            (f"{base}_2022Down", nominal - 2),
        )
    ]
    card = maker(
        {"x_sumw2": make_eft_histogram(cells)}, years=["2022", "2022EE"],
        enable_legacy_run3_vsmu_recovery=True,
    )
    values = {
        key.systematic: float(array[1, 1])
        for key, array in card.hists["x_sumw2"].view(flow=True).items()
    }
    assert values["nominal"] == 30
    assert values[f"{base}_2022Up"] == 34
    assert values[f"{base}_2022Down"] == 26
    assert not any("2022EE" in name for name in values)


def test_source_period_seam_stops_before_final_correlation(monkeypatch):
    base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    cells = [
        (f"sample{period}", "channel", systematic, value)
        for period, nominal in (("2022", 10), ("2022EE", 20))
        for systematic, value in (
            ("nominal", nominal),
            (f"{base}_2022Up", nominal + 2),
            (f"{base}_2022Down", nominal - 2),
        )
    ]
    card = maker(enable_legacy_run3_vsmu_recovery=True)
    calls = []
    original = card.correlate_years

    def record_correlation(histogram, **kwargs):
        calls.append(kwargs)
        return original(histogram, **kwargs)

    monkeypatch.setattr(card, "correlate_years", record_correlation)
    card.read_source_periods(hists={"x": make_eft_histogram(cells)})
    assert calls == []
    source_keys = {key.systematic for key in card.hists["x"].categorical_keys}
    assert f"{base}_2022EEUp" in source_keys
    assert f"{base}_2022EEDown" in source_keys
    card.read(hists={"x": make_eft_histogram(cells)})
    assert len(calls) == 1
    final_keys = {key.systematic for key in card.hists["x"].categorical_keys}
    assert f"{base}_2022Up" in final_keys
    assert f"{base}_2022EEUp" not in final_keys


@pytest.mark.parametrize("family", ("vsmu", "efes"))
def test_run3_year_pair_algebra(family):
    base = (
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4" if family == "vsmu"
        else "CMS_scale_t_DeepTau2018v2p5_DM0_genElectron"
    )
    cells = [
        (f"sample{period}", "channel", systematic, value)
        for period, nominal in zip(RUN3_PERIODS, (10, 20, 30, 40))
        for systematic, value in (
            ("nominal", nominal), (f"{base}_{period}Up", nominal + 2),
            (f"{base}_{period}Down", nominal - 2),
        )
    ]
    card = maker()
    merged = card.correlate_years(make_eft_histogram(cells), add_luminosity_shapes=False)
    values = {key.systematic: float(array[1, 1]) for key, array in merged.view(flow=True).items()}
    assert values["nominal"] == 100
    assert {key: values[key] for key in (f"{base}_2022Up", f"{base}_2022Down", f"{base}_2023Up", f"{base}_2023Down")} == {
        f"{base}_2022Up": 104, f"{base}_2022Down": 96,
        f"{base}_2023Up": 104, f"{base}_2023Down": 96,
    }
    assert len(values) == 5


def test_run3_vse_tes_independence_and_muon_runwide_identity():
    bases = (
        "CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46",
        "CMS_scale_t_DeepTau2018v2p5_DM0_genTau",
    )
    cells = [
        (f"sample{period}", "channel", systematic, value)
        for period, nominal in (("2022", 10), ("2022EE", 20))
        for systematic, value in (
            ("nominal", nominal),
            *((f"{base}_{period}{side}", nominal + (2 if side == "Up" else -2))
              for base in bases for side in ("Up", "Down")),
            *((f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genMuon_Run3{side}", nominal + 1)
              for dm in (0, 1) for side in ("Up", "Down")),
        )
    ]
    merged = maker().correlate_years(make_eft_histogram(cells), add_luminosity_shapes=False)
    names = {key.systematic for key in merged.categorical_keys}
    for base in bases:
        for period in ("2022", "2022EE"):
            for side in ("Up", "Down"):
                assert f"{base}_{period}{side}" in names
    for dm in (0, 1):
        for side in ("Up", "Down"):
            assert f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genMuon_Run3{side}" in names
