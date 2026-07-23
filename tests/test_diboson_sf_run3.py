import base64
import gzip
import importlib.util
import json
import pickle
from pathlib import Path

import hist
import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analysis/diboson_njets/diboson_sf_run3.py"
CONFIG_PATH = REPO_ROOT / "analysis/diboson_njets/diboson_sf_run3_config.yml"
FIXTURE_PATH = Path(__file__).resolve().parent / "data/run3_histogram.pkl.gz.base64"

spec = importlib.util.spec_from_file_location("diboson_sf_run3", MODULE_PATH)
diboson_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(diboson_module)


def _materialize_histogram_fixture(tmp_path: Path) -> Path:
    fixture_path = tmp_path / "run3_histogram.pkl.gz"
    fixture_path.write_bytes(base64.b64decode(FIXTURE_PATH.read_text().strip()))
    return fixture_path


def _roles():
    config = diboson_module.load_diboson_config(CONFIG_PATH)
    return diboson_module._normalize_process_roles(config)


def _run_fixture(tmp_path, *, enabled=True, cache=None):
    fixture = _materialize_histogram_fixture(tmp_path)
    return diboson_module.process_year(
        str(fixture),
        "2022",
        "njets",
        "3l_CR",
        [0, 1, 2, 3, 4, 5, 6],
        process_roles=_roles(),
        propagation_enabled=enabled,
        configuration_source="config",
        cache=cache,
    )


def _write_payload(path, payload):
    with gzip.open(path, "wb") as stream:
        pickle.dump(payload, stream, protocol=5)


def _companion_with_axes(payload, mutation):
    nominal = payload["njets"]
    processes = [str(value) for value in nominal.axes["process"]]
    channels = [str(value) for value in nominal.axes["channel"]]
    years = [str(value) for value in nominal.axes["year"]]
    edges = nominal.axes["njets"].edges.tolist()
    underflow = False
    overflow = False
    if mutation == "category":
        processes = processes[:-1]
    elif mutation == "edge":
        edges[1] = 0.4
    elif mutation == "flow":
        overflow = True
    axes = [
        hist.axis.StrCategory(processes, name="process"),
        hist.axis.StrCategory(channels, name="channel"),
        hist.axis.StrCategory(years, name="year"),
        hist.axis.Variable(
            edges,
            name="njets_sumw2",
            underflow=underflow,
            overflow=overflow,
        ),
    ]
    if mutation == "axis":
        axes[0], axes[1] = axes[1], axes[0]
    return hist.Hist(*axes, storage=hist.storage.Double())


def _histogram_for_year(source_histogram, year):
    dense_axis = source_histogram.axes[-1]
    processes = [
        str(process).replace("2022", year)
        for process in source_histogram.axes["process"]
    ]
    transformed = hist.Hist(
        hist.axis.StrCategory(processes, name="process"),
        hist.axis.StrCategory(
            [str(channel) for channel in source_histogram.axes["channel"]],
            name="channel",
        ),
        hist.axis.StrCategory([year], name="year"),
        hist.axis.Variable(
            dense_axis.edges,
            name=dense_axis.name,
            underflow=dense_axis.traits.underflow,
            overflow=dense_axis.traits.overflow,
        ),
        storage=hist.storage.Double(),
    )
    transformed.view(flow=False)[...] = source_histogram.view(flow=False)
    return transformed


def _combined_histogram(year_payloads, histogram_key):
    years = list(year_payloads)
    example = year_payloads[years[0]][histogram_key]
    dense_axis = example.axes[-1]
    processes = [
        str(process)
        for year in years
        for process in year_payloads[year][histogram_key].axes["process"]
    ]
    combined = hist.Hist(
        hist.axis.StrCategory(processes, name="process"),
        hist.axis.StrCategory(
            [str(channel) for channel in example.axes["channel"]],
            name="channel",
        ),
        hist.axis.StrCategory(years, name="year"),
        hist.axis.Variable(
            dense_axis.edges,
            name=dense_axis.name,
            underflow=dense_axis.traits.underflow,
            overflow=dense_axis.traits.overflow,
        ),
        storage=hist.storage.Double(),
    )
    combined_processes = list(map(str, combined.axes["process"]))
    for year_index, year in enumerate(years):
        source = year_payloads[year][histogram_key]
        source_values = source.view(flow=False)
        for process_index, process in enumerate(map(str, source.axes["process"])):
            combined_index = combined_processes.index(process)
            combined.view(flow=False)[combined_index, :, year_index, :] = (
                source_values[process_index, :, 0, :]
            )
    return combined


def _write_year_config(path, year, *, propagation_enabled=True):
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    payload["diboson"]["propagate_statistical_uncertainties"] = (
        propagation_enabled
    )
    roles = payload["diboson"]["process_roles"]
    for role, processes in roles.items():
        roles[role] = [process.replace("2022", year) for process in processes]
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _build_multiyear_campaign(tmp_path):
    fixture = _materialize_histogram_fixture(tmp_path)
    source_payload = diboson_module.load_pkl_file(str(fixture))
    years = ["2022", "2022EE"]
    year_payloads = {
        year: {
            histogram_key: _histogram_for_year(histogram, year)
            for histogram_key, histogram in source_payload.items()
        }
        for year in years
    }
    input_paths = {}
    config_paths = {}
    for year in years:
        input_path = tmp_path / f"input_{year}.pkl.gz"
        config_path = tmp_path / f"roles_{year}.yml"
        _write_payload(input_path, year_payloads[year])
        _write_year_config(config_path, year)
        input_paths[year] = input_path
        config_paths[year] = config_path

    shared_path = tmp_path / "combined.pkl.gz"
    _write_payload(
        shared_path,
        {
            histogram_key: _combined_histogram(year_payloads, histogram_key)
            for histogram_key in source_payload
        },
    )
    shared_config = yaml.safe_load(CONFIG_PATH.read_text())
    shared_roles = shared_config["diboson"]["process_roles"]
    for role, processes in list(shared_roles.items()):
        shared_roles[role] = [
            process.replace("2022", year)
            for year in years
            for process in processes
        ]
    shared_config_path = tmp_path / "roles_combined.yml"
    shared_config_path.write_text(yaml.safe_dump(shared_config, sort_keys=False))
    return {
        "years": years,
        "input_paths": input_paths,
        "config_paths": config_paths,
        "input_template": str(tmp_path / "input_{year}.pkl.gz"),
        "config_template": str(tmp_path / "roles_{year}.yml"),
        "shared_path": shared_path,
        "shared_config_path": shared_config_path,
    }


def _cli_args(pkl_paths, config_paths, years, output_path, *extra):
    return [
        "--pkl",
        *map(str, pkl_paths),
        "--config",
        *map(str, config_paths),
        "--channel",
        "3l_CR",
        "--year",
        *years,
        "--output-dir",
        str(output_path),
        *extra,
    ]


def _read_output(output_path, year):
    return json.loads(
        (output_path / year / f"diboson_sf_{year}.json").read_text()
    )


def _statistical_arrays(payload):
    central_key = next(key for key in payload if key.startswith("dibosonSF_njets_"))
    return (
        payload[central_key],
        payload["scale_factor_statistical_variances"],
        payload["scale_factor_statistical_uncertainties"],
    )


def test_current_fixture_reopens_and_cache_extractor_completes(tmp_path):
    fixture = _materialize_histogram_fixture(tmp_path)
    histograms = diboson_module.load_pkl_file(str(fixture))
    assert set(histograms) == {"njets", "njets_sumw2"}
    assert [axis.name for axis in histograms["njets"].axes] == [
        "process",
        "channel",
        "year",
        "njets",
    ]
    result = _run_fixture(tmp_path)
    assert result["scale_factors"][0] > 0
    membership = result["provenance"]["source_to_final_bin_membership"]
    assert membership["final_bin_source_indices"][0] == [0, 1]


def test_get_yields_preserves_sparse_hist_support_when_disabled(tmp_path):
    from topcoffea.modules.sparseHist import SparseHist

    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    dense = payload["njets"]
    sparse = SparseHist(*dense.axes)
    values = dense.values(flow=False)
    for process_index, process in enumerate(dense.axes["process"]):
        for source_index, center in enumerate(dense.axes["njets"].centers):
            weight = float(values[process_index, 0, 0, source_index])
            if weight:
                sparse.fill(
                    process=process,
                    channel="3l_CR",
                    year="2022",
                    njets=center,
                    weight=weight,
                )
    sparse_payload = {"njets": sparse}
    processes = ["data_a_2022", "WZTo3LNu_2022"]
    dense_yields = diboson_module.get_yields_in_bins(
        payload,
        processes,
        [0, 1, 2, 3, 4, 5, 6],
        "njets",
        "3l_CR",
        extra_slices={"year": "2022"},
    )
    sparse_yields = diboson_module.get_yields_in_bins(
        sparse_payload,
        processes,
        [0, 1, 2, 3, 4, 5, 6],
        "njets",
        "3l_CR",
        extra_slices={"year": "2022"},
    )
    assert sparse_yields == dense_yields


def test_analytic_statistics_use_all_roles_and_aggregate_before_ratio(tmp_path):
    result = _run_fixture(tmp_path)

    # Independent primitive arithmetic for the first two 0.5-wide source bins.
    data = (30 + 34) + (5 + 6)
    background = (5 + 6) + (1 + 2)
    diboson = (8 + 9) + (2 + 3)
    var_data = (50 + 52) + (12 + 13)
    var_background = (9 + 10) + (3 + 4)
    var_diboson = (14 + 15) + (5 + 6)
    expected_central = (data - background) / diboson
    expected_variance = (var_data + var_background) / diboson**2 + (
        (data - background) ** 2 / diboson**4
    ) * var_diboson

    assert data != var_data  # weighted/non-Poisson data is observable
    assert result["data"][0] == data
    assert result["other"][0] == background
    assert result["diboson"][0] == diboson
    assert result["scale_factors"][0] == pytest.approx(expected_central)
    assert result["scale_factor_statistical_variances"][0] == pytest.approx(
        expected_variance
    )
    assert result["scale_factor_statistical_uncertainties"][0] == pytest.approx(
        expected_variance**0.5
    )

    # These deliberately wrong alternatives distinguish the frozen formula.
    source_ratio_average = np.mean([(35 - 6) / 10, (40 - 8) / 12])
    poisson_data_variance = (data + var_background) / diboson**2 + (
        (data - background) ** 2 / diboson**4
    ) * var_diboson
    no_background_variance = var_data / diboson**2 + (
        (data - background) ** 2 / diboson**4
    ) * var_diboson
    no_diboson_variance = (var_data + var_background) / diboson**2
    assert result["scale_factors"][0] != pytest.approx(source_ratio_average)
    assert expected_variance != pytest.approx(poisson_data_variance)
    assert expected_variance != pytest.approx(no_background_variance)
    assert expected_variance != pytest.approx(no_diboson_variance)


@pytest.mark.parametrize(
    ("data", "background", "diboson", "expected_central", "positive_variance"),
    [
        ([7.0], [7.0], [2.0], 0.0, True),
        ([2.0], [6.0], [2.0], -2.0, True),
    ],
)
def test_cancelled_and_negative_numerators(
    data, background, diboson, expected_central, positive_variance
):
    components = {
        "data": data,
        "background": background,
        "diboson": diboson,
        "var_data": [3.0],
        "var_background": [5.0],
        "var_diboson": [7.0],
    }
    central, variances, uncertainties = (
        diboson_module.compute_scale_factor_statistics(
            components,
            [0, 1],
            input_path="primitive",
            year="2022",
            channel="3l_CR",
            propagation_enabled=True,
        )
    )
    numerator = data[0] - background[0]
    expected_variance = (3 + 5) / diboson[0] ** 2 + (
        numerator**2 / diboson[0] ** 4
    ) * 7
    assert central == pytest.approx([expected_central])
    assert variances == pytest.approx([expected_variance])
    assert uncertainties == pytest.approx([expected_variance**0.5])
    assert (variances[0] > 0) is positive_variance


@pytest.mark.parametrize("denominator", [0.0, -1.0, float("nan")])
def test_invalid_denominator_is_structured(denominator):
    components = {
        "data": [4.0],
        "background": [1.0],
        "diboson": [denominator],
        "var_data": [2.0],
        "var_background": [1.0],
        "var_diboson": [1.0],
    }
    with pytest.raises(
        diboson_module.DibosonContractError,
        match=r"input='fixture'.*year='2022'.*channel='3l_CR'.*final_bin=\[0.0, 1.0\]",
    ):
        diboson_module.compute_scale_factor_statistics(
            components,
            [0, 1],
            input_path="fixture",
            year="2022",
            channel="3l_CR",
            propagation_enabled=True,
        )


@pytest.mark.parametrize("missing_key", ["njets", "njets_sumw2"])
def test_enabled_mode_requires_nominal_and_companion(tmp_path, missing_key):
    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    del payload[missing_key]
    with pytest.raises(diboson_module.DibosonContractError, match="[Oo]rphan|Missing"):
        diboson_module.process_year(
            str(fixture),
            "2022",
            "njets",
            "3l_CR",
            [0, 1, 2, 3, 4, 5, 6],
            process_roles=_roles(),
            propagation_enabled=True,
            configuration_source="config",
            cache={str(fixture): payload},
        )


@pytest.mark.parametrize("mutation", ["axis", "category", "edge", "flow"])
def test_enabled_mode_rejects_companion_structure_mismatch(tmp_path, mutation):
    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    payload["njets_sumw2"] = _companion_with_axes(payload, mutation)
    with pytest.raises(
        diboson_module.DibosonContractError,
        match="axes/categories/edges/flow differ",
    ):
        diboson_module.process_year(
            str(fixture),
            "2022",
            "njets",
            "3l_CR",
            [0, 1, 2, 3, 4, 5, 6],
            process_roles=_roles(),
            propagation_enabled=True,
            configuration_source="config",
            cache={str(fixture): payload},
        )


@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_enabled_mode_rejects_invalid_second_moment(tmp_path, bad_value):
    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    payload["njets_sumw2"].view(flow=False)[0, 0, 0, 0] = bad_value
    with pytest.raises(diboson_module.DibosonContractError, match="Invalid second moments"):
        diboson_module.process_year(
            str(fixture),
            "2022",
            "njets",
            "3l_CR",
            [0, 1, 2, 3, 4, 5, 6],
            process_roles=_roles(),
            propagation_enabled=True,
            configuration_source="config",
            cache={str(fixture): payload},
        )


def test_nonfinite_nominal_is_rejected(tmp_path):
    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    payload["njets"].view(flow=False)[0, 0, 0, 0] = np.nan
    with pytest.raises(diboson_module.DibosonContractError, match="Nonfinite nominal"):
        diboson_module.process_year(
            str(fixture),
            "2022",
            "njets",
            "3l_CR",
            [0, 1, 2, 3, 4, 5, 6],
            process_roles=_roles(),
            propagation_enabled=True,
            configuration_source="config",
            cache={str(fixture): payload},
        )


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        (
            {
                "data": ["data_a_2022", "data_a_2022"],
                "background": ["ttbar_2022"],
                "diboson": ["WZTo3LNu_2022"],
                "ignored": [],
            },
            "duplicate",
        ),
        (
            {
                "data": ["data_a_2022"],
                "background": ["data_a_2022"],
                "diboson": ["WZTo3LNu_2022"],
                "ignored": [],
            },
            "pairwise disjoint",
        ),
        (
            {
                "data": ["data_a_2022", "data_b_2022"],
                "background": ["ttbar_2022", "zjets_2022"],
                "diboson": ["WZTo3LNu_2022", "ZZTo2L2Nu_2022"],
                "ignored": [],
            },
            "unclassified",
        ),
    ],
)
def test_process_roles_reject_duplicates_overlap_and_unclassified(
    tmp_path, roles, message
):
    fixture = _materialize_histogram_fixture(tmp_path)
    with pytest.raises(diboson_module.DibosonContractError, match=message):
        diboson_module.process_year(
            str(fixture),
            "2022",
            "njets",
            "3l_CR",
            [0, 1, 2, 3, 4, 5, 6],
            process_roles=roles,
            propagation_enabled=True,
            configuration_source="config",
        )


def test_disabled_mode_never_accesses_companion(tmp_path):
    class PoisonCompanion:
        def __getattribute__(self, name):
            raise AssertionError(f"disabled mode accessed companion attribute {name}")

    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    payload["njets_sumw2"] = PoisonCompanion()
    result = _run_fixture(
        tmp_path, enabled=False, cache={str(fixture): payload}
    )
    assert result["scale_factor_statistical_variances"] is None
    assert result["scale_factor_statistical_uncertainties"] is None
    assert result["provenance"]["statistical_inputs_consumed"] is False


def test_propagation_resolution_precedence():
    assert diboson_module.resolve_propagation_state({}, None) == (True, "default")
    assert diboson_module.resolve_propagation_state(
        {"propagate_statistical_uncertainties": False}, None
    ) == (False, "config")
    assert diboson_module.resolve_propagation_state(
        {"propagate_statistical_uncertainties": False}, True
    ) == (True, "cli")
    assert diboson_module.resolve_propagation_state(
        {"propagate_statistical_uncertainties": True}, False
    ) == (False, "cli")


def test_enabled_json_and_plot_align_with_analytic_arrays(tmp_path):
    result = _run_fixture(tmp_path)
    output = tmp_path / "enabled"
    json_path = diboson_module.make_diboson_sf_json(
        [0, 1, 2, 3, 4, 5, 6], result, "2022", str(output)
    )
    plot = diboson_module.save_scale_factor_plot(
        "2022",
        "3l_CR",
        result["bin_centers"],
        result["scale_factors"],
        result["fitted_values"],
        result["scale_factor_statistical_uncertainties"],
        propagation_enabled=True,
        output_dir=str(output),
    )
    payload = json.loads(Path(json_path).read_text())
    propagation = payload["statistical_uncertainty_propagation"]
    assert propagation["enabled"] is True
    assert propagation["formula"] == (
        "independent_data_minus_background_over_diboson_v1"
    )
    assert propagation["configuration_source"] == "config"
    assert payload["scale_factor_statistical_variances"] == pytest.approx(
        result["scale_factor_statistical_variances"]
    )
    assert payload["scale_factor_statistical_uncertainties"] == pytest.approx(
        result["scale_factor_statistical_uncertainties"]
    )
    assert plot["statistical_error_bars"] is True
    assert plot["y_errors"] == pytest.approx(
        result["scale_factor_statistical_uncertainties"]
    )
    assert any(value > 0 for value in plot["y_errors"])
    assert Path(plot["path"]).is_file()


def test_disabled_json_and_plot_contract(tmp_path):
    result = _run_fixture(tmp_path, enabled=False)
    output = tmp_path / "disabled"
    json_path = diboson_module.make_diboson_sf_json(
        [0, 1, 2, 3, 4, 5, 6], result, "2022", str(output)
    )
    plot = diboson_module.save_scale_factor_plot(
        "2022",
        "3l_CR",
        result["bin_centers"],
        result["scale_factors"],
        result["fitted_values"],
        result["scale_factor_statistical_uncertainties"],
        propagation_enabled=False,
        output_dir=str(output),
    )
    payload = json.loads(Path(json_path).read_text())
    propagation = payload["statistical_uncertainty_propagation"]
    assert propagation["enabled"] is False
    assert propagation["formula"] is None
    assert propagation["statistical_inputs_consumed"] is False
    assert payload["scale_factor_statistical_variances"] is None
    assert payload["scale_factor_statistical_uncertainties"] is None
    assert plot["statistical_error_bars"] is False
    assert plot["y_errors"] is None
    assert plot["annotation"] == "statistical uncertainties disabled"
    assert Path(plot["path"]).is_file()


@pytest.mark.parametrize("denominator", [0.0, -1.0])
def test_cli_blocking_error_writes_no_partial_output(tmp_path, denominator):
    fixture = _materialize_histogram_fixture(tmp_path)
    payload = diboson_module.load_pkl_file(str(fixture))
    diboson_indices = [4, 5]
    payload["njets"].view(flow=False)[diboson_indices, 0, 0, 0:2] = (
        denominator / 4
    )
    bad_path = tmp_path / "bad.pkl.gz"
    _write_payload(bad_path, payload)
    output = tmp_path / "out"
    with pytest.raises(diboson_module.DibosonContractError, match="denominator"):
        diboson_module.main(
            [
                "--pkl",
                str(bad_path),
                "--config",
                str(CONFIG_PATH),
                "--channel",
                "3l_CR",
                "--year",
                "2022",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def test_cli_override_records_source_and_writes_outputs(tmp_path):
    fixture = _materialize_histogram_fixture(tmp_path)
    output = tmp_path / "cli"
    result = diboson_module.main(
        [
            "--pkl",
            str(fixture),
            "--config",
            str(CONFIG_PATH),
            "--channel",
            "3l_CR",
            "--year",
            "2022",
            "--output-dir",
            str(output),
            "--no-propagate-statistical-uncertainties",
        ]
    )["2022"]
    payload = json.loads((output / "2022/diboson_sf_2022.json").read_text())
    assert result["configuration_source"] == "cli"
    assert payload["statistical_uncertainty_propagation"][
        "configuration_source"
    ] == "cli"
    assert payload["scale_factor_statistical_variances"] is None
    assert (output / "2022/diboson_sf_2022.png").is_file()


def test_parser_preserves_single_config_behavior():
    args = diboson_module.build_parser().parse_args(
        [
            "--pkl",
            "input_2022.pkl.gz",
            "--config",
            "roles_2022.yml",
            "--year",
            "2022",
        ]
    )
    assert args.pkl == ["input_2022.pkl.gz"]
    assert args.config == ["roles_2022.yml"]
    assert args.year == ["2022"]


def test_shared_explicit_years_and_year_all_remain_equivalent(tmp_path):
    campaign = _build_multiyear_campaign(tmp_path)
    explicit_output = tmp_path / "shared_explicit"
    all_output = tmp_path / "shared_all"
    diboson_module.main(
        _cli_args(
            [campaign["shared_path"]],
            [campaign["shared_config_path"]],
            campaign["years"],
            explicit_output,
        )
    )
    diboson_module.main(
        _cli_args(
            [campaign["shared_path"]],
            [campaign["shared_config_path"]],
            ["all"],
            all_output,
        )
    )
    for year in campaign["years"]:
        assert _statistical_arrays(_read_output(explicit_output, year)) == (
            _statistical_arrays(_read_output(all_output, year))
        )
    assert (all_output / "all/diboson_sf_all.json").is_file()


@pytest.mark.parametrize(
    ("input_form", "config_form"),
    [
        ("explicit", "explicit"),
        ("template", "template"),
        ("template", "explicit"),
        ("explicit", "template"),
    ],
)
def test_independent_multiyear_mapping_forms_complete_and_match_shared(
    tmp_path,
    input_form,
    config_form,
):
    campaign = _build_multiyear_campaign(tmp_path)
    shared_output = tmp_path / "shared"
    mapped_output = tmp_path / f"mapped_{input_form}_{config_form}"
    diboson_module.main(
        _cli_args(
            [campaign["shared_path"]],
            [campaign["shared_config_path"]],
            campaign["years"],
            shared_output,
        )
    )
    input_arguments = (
        [campaign["input_template"]]
        if input_form == "template"
        else [campaign["input_paths"][year] for year in campaign["years"]]
    )
    config_arguments = (
        [campaign["config_template"]]
        if config_form == "template"
        else [campaign["config_paths"][year] for year in campaign["years"]]
    )
    arguments = _cli_args(
        input_arguments,
        config_arguments,
        campaign["years"],
        mapped_output,
    )
    parser = diboson_module.build_parser()
    records, _, _, _ = diboson_module._resolve_cli_inputs(
        parser.parse_args(arguments),
        parser,
    )
    assert [
        (record.year, record.pkl_path, record.config_path, record.shared_input)
        for record in records
    ] == [
        (
            year,
            str(campaign["input_paths"][year]),
            str(campaign["config_paths"][year].resolve()),
            False,
        )
        for year in campaign["years"]
    ]

    diboson_module.main(arguments)
    for year in campaign["years"]:
        shared_payload = _read_output(shared_output, year)
        mapped_payload = _read_output(mapped_output, year)
        shared_arrays = _statistical_arrays(shared_payload)
        mapped_arrays = _statistical_arrays(mapped_payload)
        assert shared_arrays[0].keys() == mapped_arrays[0].keys()
        np.testing.assert_allclose(
            list(shared_arrays[0].values()),
            list(mapped_arrays[0].values()),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shared_arrays[1], mapped_arrays[1], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            shared_arrays[2], mapped_arrays[2], rtol=1e-12, atol=1e-12
        )
        assert mapped_payload["statistical_uncertainty_propagation"]["year"] == year
        assert (mapped_output / year).is_dir()


def test_shared_config_is_loaded_once(tmp_path, monkeypatch):
    campaign = _build_multiyear_campaign(tmp_path)
    shared_config = str(campaign["shared_config_path"].resolve())
    records = [
        diboson_module.resolved_diboson_input(
            year=year,
            pkl_path=str(campaign["shared_path"]),
            config_path=shared_config,
            shared_input=True,
        )
        for year in campaign["years"]
    ]
    calls = []
    original = diboson_module.load_diboson_config

    def counted_load(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(diboson_module, "load_diboson_config", counted_load)
    diboson_module._load_resolved_configs(records, None)
    assert calls == [shared_config]


def test_config_cardinality_mismatch_fails_before_output(tmp_path, capsys):
    campaign = _build_multiyear_campaign(tmp_path)
    output = tmp_path / "cardinality_output"
    with pytest.raises(SystemExit):
        diboson_module.main(
            _cli_args(
                [campaign["input_paths"][year] for year in campaign["years"]],
                [
                    campaign["config_paths"]["2022"],
                    campaign["config_paths"]["2022EE"],
                    campaign["config_paths"]["2022"],
                ],
                campaign["years"],
                output,
            )
        )
    assert "Number of --config paths must match" in capsys.readouterr().err
    assert not output.exists()


def test_independent_inputs_reject_one_non_template_config_early(tmp_path, capsys):
    campaign = _build_multiyear_campaign(tmp_path)
    output = tmp_path / "one_config_output"
    with pytest.raises(SystemExit):
        diboson_module.main(
            _cli_args(
                [campaign["input_paths"][year] for year in campaign["years"]],
                [campaign["shared_config_path"]],
                campaign["years"],
                output,
            )
        )
    assert (
        "Multiple independent input files require one matching --config per input, "
        "or a --config path containing {year}."
        in capsys.readouterr().err
    )
    assert not output.exists()


def test_missing_expanded_config_paths_are_reported_together(tmp_path, capsys):
    campaign = _build_multiyear_campaign(tmp_path)
    missing_template = str(tmp_path / "missing_roles_{year}.yml")
    output = tmp_path / "missing_config_output"
    with pytest.raises(SystemExit):
        diboson_module.main(
            _cli_args(
                [campaign["input_template"]],
                [missing_template],
                campaign["years"],
                output,
            )
        )
    error = capsys.readouterr().err
    assert "Resolved config path(s) do not exist" in error
    assert "missing_roles_2022.yml" in error
    assert "missing_roles_2022EE.yml" in error
    assert not output.exists()


@pytest.mark.parametrize(
    "bad_template",
    ["roles_{period}.yml", "roles_{year}_{period}.yml", "roles_{year.yml"],
)
def test_unsupported_config_template_fields_fail_clearly(
    tmp_path,
    capsys,
    bad_template,
):
    campaign = _build_multiyear_campaign(tmp_path)
    with pytest.raises(SystemExit):
        diboson_module.main(
            _cli_args(
                [campaign["input_template"]],
                [str(tmp_path / bad_template)],
                campaign["years"],
                tmp_path / "bad_template_output",
            )
        )
    assert "supports only the literal '{year}' placeholder" in capsys.readouterr().err


def test_year_all_rejects_config_template_before_discovery(tmp_path, capsys):
    campaign = _build_multiyear_campaign(tmp_path)
    with pytest.raises(SystemExit):
        diboson_module.main(
            _cli_args(
                [campaign["shared_path"]],
                [campaign["config_template"]],
                ["all"],
                tmp_path / "all_template_output",
            )
        )
    assert "--year all cannot be used with a template --config path" in (
        capsys.readouterr().err
    )


def test_wrong_second_config_preserves_strict_roles_and_writes_no_output(tmp_path):
    campaign = _build_multiyear_campaign(tmp_path)
    output = tmp_path / "wrong_second_config_output"
    with pytest.raises(
        diboson_module.DibosonContractError,
        match="Configured process-role labels are absent.*2022",
    ) as error:
        diboson_module.main(
            _cli_args(
                [campaign["input_paths"][year] for year in campaign["years"]],
                [
                    campaign["config_paths"]["2022"],
                    campaign["config_paths"]["2022"],
                ],
                campaign["years"],
                output,
            )
        )
    assert "year='2022EE'" in str(error.value)
    assert str(campaign["input_paths"]["2022EE"]) in str(error.value)
    assert str(campaign["config_paths"]["2022"].resolve()) in str(error.value)
    assert not output.exists()


def test_differing_config_propagation_states_require_cli_override(
    tmp_path,
    monkeypatch,
):
    campaign = _build_multiyear_campaign(tmp_path)
    _write_year_config(
        campaign["config_paths"]["2022EE"],
        "2022EE",
        propagation_enabled=False,
    )
    no_override_output = tmp_path / "mixed_without_override"
    arguments = _cli_args(
        [campaign["input_paths"][year] for year in campaign["years"]],
        [campaign["config_paths"][year] for year in campaign["years"]],
        campaign["years"],
        no_override_output,
    )
    processed_years = []
    original_process_year = diboson_module.process_year

    def counted_process_year(*args, **kwargs):
        processed_years.append(args[1])
        return original_process_year(*args, **kwargs)

    monkeypatch.setattr(diboson_module, "process_year", counted_process_year)
    with pytest.raises(
        diboson_module.DibosonContractError,
        match="inconsistent statistical propagation states",
    ) as error:
        diboson_module.main(arguments)
    assert str(campaign["config_paths"]["2022"].resolve()) in str(error.value)
    assert str(campaign["config_paths"]["2022EE"].resolve()) in str(error.value)
    assert processed_years == []
    assert not no_override_output.exists()

    override_output = tmp_path / "mixed_with_override"
    diboson_module.main(
        _cli_args(
            [campaign["input_paths"][year] for year in campaign["years"]],
            [campaign["config_paths"][year] for year in campaign["years"]],
            campaign["years"],
            override_output,
            "--no-propagate-statistical-uncertainties",
        )
    )
    for year in campaign["years"]:
        payload = _read_output(override_output, year)
        propagation = payload["statistical_uncertainty_propagation"]
        assert propagation["enabled"] is False
        assert propagation["configuration_source"] == "cli"
        assert payload["scale_factor_statistical_variances"] is None
        assert payload["scale_factor_statistical_uncertainties"] is None
    assert processed_years == campaign["years"]
