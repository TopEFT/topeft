"""Synthetic source-period to serialized-card correlation checks."""

import gzip
import json
import pickle

import hist
import numpy as np
import pytest
import uproot

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.datacard_tools import DatacardMaker
from topeft.modules.missing_parton_contract import load_missing_parton_channel_contract
from analysis.topeft_run2.validate_datacard_correlation import (
    compare_arrays,
    main,
    validate_correlation,
)


CHANNEL = "3l_1tau_1b_2j"
VARIABLE = "lj0pt"
PREFIX = "CMS_fake_t_DeepTau2018v2p5_VSmu_"
BASE = PREFIX + "abseta0to0p4"
PERIODS = ("2022", "2022EE", "2023", "2023BPix")


def _source(periods, bases=(BASE,), *, missing=None, negative=False,
            fractional=False, with_appl=False):
    axes = [
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
    ]
    if with_appl:
        axes.append(hist.axis.StrCategory([], name="appl", growth=True))
    axes.append(hist.axis.Regular(12, 0, 600, name=VARIABLE))
    source = HistEFT(*axes, wc_names=[])
    for index, period in enumerate(periods):
        nominal = (0.1 if fractional else 10.0) * (index + 1)
        shift = (0.07 if fractional else index + 2)
        systematics = [("nominal", nominal)]
        for base in bases:
            systematics += [(f"{base}_{period}Up", nominal + shift),
                            (f"{base}_{period}Down", nominal - shift)]
        for systematic, value in systematics:
            if missing == (period, systematic):
                continue
            first = -50.0 if negative and index == 0 else value
            if fractional:
                positions = [25.0, 75.0, 175.0]
                weights = [[first / 3.0], [value / 7.0], [value / 2.0]]
            else:
                positions = [25.0, 175.0]
                weights = [[first], [value / 2.0]]
            appl_labels = (
                (load_missing_parton_channel_contract().expected_sr_appl(CHANNEL),
                 "isAR_unselected") if with_appl else (None,)
            )
            for appl_label in appl_labels:
                sparse = {} if appl_label is None else {"appl": appl_label}
                multiplier = 1.0 if appl_label != "isAR_unselected" else 100.0
                source.fill(process=f"WZTo3LNu_{period}", channel=CHANNEL,
                            systematic=systematic, lj0pt=np.asarray(positions),
                            eft_coeff=np.asarray(weights) * multiplier,
                            **sparse)
    return source


def _artifacts(tmp_path, periods, groups, *, bases=(BASE,), negative_bins="crop",
               negative=False, missing=None, fractional=False, with_appl=False):
    source = _source(periods, bases, negative=negative, fractional=fractional,
                     with_appl=with_appl)
    source_path = tmp_path / "source.pkl.gz"
    with gzip.open(source_path, "wb") as stream:
        pickle.dump({VARIABLE: source}, stream)
    maker = DatacardMaker(
        pkl_path=str(source_path), year_lst=list(periods), var_lst=[VARIABLE],
        do_nuisance=True, skip_missing_parton_rate_syst=True,
        out_dir=str(tmp_path), verbose=False,
    )
    maker.analyze(VARIABLE, CHANNEL, {"Diboson": []},
                  negative_bins == "crop", {})
    if missing is not None:
        with gzip.open(source_path, "wb") as stream:
            pickle.dump({VARIABLE: _source(periods, bases, missing=missing,
                                           negative=negative,
                                           fractional=fractional,
                                           with_appl=with_appl)}, stream)
    stem = maker.FNAME_TEMPLATE.format(cat=CHANNEL, kmvar=VARIABLE, ext="")
    return source_path, tmp_path / f"{stem}txt", tmp_path / f"{stem}root"


def _validate(paths, periods, groups, *, bases=(BASE,), negative_bins="crop"):
    return validate_correlation(
        *map(str, paths), VARIABLE, CHANNEL, periods, groups, PREFIX, "Diboson",
        required_bases=bases, negative_bins=negative_bins,
    )


def test_two_source_periods_preserve_direction_and_count_once(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    paths = _artifacts(tmp_path, periods, groups)
    result = _validate(paths, periods, groups)
    assert result["passed"], result
    up = result["comparisons"][f"Diboson_sm_{BASE}_2022Up"]
    down = result["comparisons"][f"Diboson_sm_{BASE}_2022Down"]
    nominal = result["comparisons"]["Diboson_sm"]
    assert np.sum(up["expected"]) > np.sum(nominal["expected"])
    assert np.sum(down["expected"]) < np.sum(nominal["expected"])
    assert up["expected"] == up["observed"]
    assert down["expected"] == down["observed"]
    assert len(result["source_support"]) == 2
    assert result["source_processes"] == {
        "2022": ["Diboson_2022"], "2022EE": ["Diboson_2022EE"]
    }
    assert len(nominal["expected"]) == 4  # maintained fitting bins after flow removal


def test_disjoint_groups_and_five_eta_identities(tmp_path):
    bases = tuple(PREFIX + eta for eta in (
        "abseta0to0p4", "abseta0p4to0p8", "abseta0p8to1p2",
        "abseta1p2to1p7", "abseta1p7to2p4",
    ))
    groups = {"2022": ["2022", "2022EE"],
              "2023": ["2023", "2023BPix"]}
    paths = _artifacts(tmp_path, PERIODS, groups, bases=bases)
    result = _validate(paths, PERIODS, groups, bases=bases)
    assert result["passed"], result
    assert len(result["expected_card_identities"]) == 10
    assert len(result["root_identities"]) == 20
    for identity, comparison in result["comparisons"].items():
        assert comparison["passed"], identity


def test_metadata_sr_selection_matches_writer(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    result = _validate(_artifacts(tmp_path, periods, groups, with_appl=True),
                       periods, groups)
    assert result["passed"], result
    assert result["comparisons"]["Diboson_sm"]["expected"][0] < 100.0


@pytest.mark.parametrize("missing", (
    ("2022", "nominal"),
    ("2022EE", f"{BASE}_2022EEUp"),
    ("2022", f"{BASE}_2022Down"),
))
def test_missing_source_support_identifies_coordinate(tmp_path, missing):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    result = _validate(_artifacts(tmp_path, periods, groups, missing=missing),
                       periods, groups)
    assert not result["passed"]
    assert any(missing[0] in error and missing[1] in error
               for error in result["errors"])


def test_forbidden_alias_and_duplicate_root_cycle_fail(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    paths = _artifacts(tmp_path, periods, groups)
    root_path = paths[2]
    with uproot.open(root_path) as root:
        template = root[f"Diboson_sm_{BASE}_2022Up"].to_hist()
    with uproot.update(root_path) as root:
        root[f"Diboson_sm_{BASE}_2022EEUp"] = template
        root[f"Diboson_sm_{BASE}_2022Up"] = template
    result = _validate(paths, periods, groups)
    assert not result["passed"]
    assert f"Diboson_sm_{BASE}_2022EEUp" in result["forbidden"]
    assert f"Diboson_sm_{BASE}_2022Up" in result["duplicate"]


def test_missing_card_identity_and_root_template_fail(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    paths = _artifacts(tmp_path, periods, groups)
    card_path, root_path = paths[1:]
    card_path.write_text("\n".join(
        line for line in card_path.read_text().splitlines()
        if not line.startswith(f"{BASE}_2022 ")
    ) + "\n")
    with uproot.open(root_path) as root:
        templates = {key: root[key].to_hist() for key in root.keys(cycle=False)
                     if key != f"Diboson_sm_{BASE}_2022Down"}
    with uproot.recreate(root_path) as root:
        for key, template in templates.items():
            root[key] = template
    result = _validate(paths, periods, groups)
    assert not result["passed"]
    assert f"{BASE}_2022" in result["missing"]
    assert f"Diboson_sm_{BASE}_2022Down" in result["missing"]


def test_clear_serialized_physics_mismatch_fails(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    paths = _artifacts(tmp_path, periods, groups)
    root_path = paths[2]
    with uproot.open(root_path) as root:
        templates = {key: root[key].to_hist() for key in root.keys(cycle=False)}
    name = f"Diboson_sm_{BASE}_2022Up"
    templates[name].values(flow=True)[1] += 1.0
    with uproot.recreate(root_path) as root:
        for key, template in templates.items():
            root[key] = template
    result = _validate(paths, periods, groups)
    assert not result["passed"]
    assert not result["comparisons"][name]["passed"]


@pytest.mark.parametrize("negative_bins", ("crop", "keep"))
def test_negative_bin_modes_match_writer_and_detect_mismatch(tmp_path, negative_bins):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    paths = _artifacts(tmp_path, periods, groups, negative_bins=negative_bins,
                       negative=True)
    matching = _validate(paths, periods, groups, negative_bins=negative_bins)
    assert matching["passed"], matching
    other_mode = "keep" if negative_bins == "crop" else "crop"
    mismatched = _validate(paths, periods, groups, negative_bins=other_mode)
    assert not mismatched["passed"]
    assert any(not row["passed"] for row in mismatched["comparisons"].values())


def test_float64_comparison_contract():
    expected = np.asarray([0.1 + 0.2, 1.0], dtype=np.float64)
    assert compare_arrays(expected, expected.copy())
    assert compare_arrays(expected, np.asarray([0.3, 1.0], dtype=np.float64))
    assert not compare_arrays(expected, np.asarray([0.31, 1.0], dtype=np.float64))
    assert not compare_arrays(expected, expected.astype(np.float32))


def test_fractional_source_summation_fitting_rebin_and_root_roundtrip(tmp_path):
    periods = PERIODS
    groups = {"2022": ["2022", "2022EE"],
              "2023": ["2023", "2023BPix"]}
    result = _validate(_artifacts(tmp_path, periods, groups, fractional=True),
                       periods, groups)
    assert result["passed"], result
    assert all(row["observed_dtype"] == "float64"
               for row in result["comparisons"].values())
    assert any(not np.array_equal(row["expected"], row["observed"])
               for row in result["comparisons"].values())


def test_statistical_companion_is_outside_correlation_decision(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    paths = _artifacts(tmp_path, periods, groups)
    source_path = paths[0]
    with gzip.open(source_path, "rb") as stream:
        source = pickle.load(stream)
    source[f"{VARIABLE}_sumw2"] = "unused statistical companion"
    with gzip.open(source_path, "wb") as stream:
        pickle.dump(source, stream)
    result = _validate(paths, periods, groups)
    assert result["passed"], result


def test_cli_writes_report_and_returns_nonzero_on_failure(tmp_path):
    periods = PERIODS[:2]
    groups = {"2022": list(periods)}
    source_path, card_path, root_path = _artifacts(tmp_path, periods, groups)
    output_path = tmp_path / "validation.json"
    argv = ["--source-pkl", str(source_path), "--card", str(card_path),
            "--root", str(root_path), "--variable", VARIABLE,
            "--channel", CHANNEL, "--prefix", PREFIX, "--process", "Diboson",
            "--period", "2022", "--period", "2022EE",
            "--group", "2022=2022,2022EE", "--output", str(output_path)]
    assert main(argv) == 0
    assert output_path.exists()
    assert json.loads(output_path.read_text())["passed"]
    argv += ["--required-base", PREFIX + "missing_eta"]
    assert main(argv) == 1
    assert not json.loads(output_path.read_text())["passed"]
