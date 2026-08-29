from pathlib import Path

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info, info_2d
from topeft.modules.axis_binning import (
    aggregate_array,
    build_aggregation_map,
    histogram_dense_edges,
    processing_edges,
    rebin_histogram,
    resolve_and_rebin_histogram,
    resolve_axis_edges,
    resolve_common_axis_edges,
    validate_axis_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sparse_histogram(axis):
    return SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        axis,
        storage="Double",
    )


def test_typed_processing_definitions_parse_and_legacy_schema_fails():
    assert np.array_equal(
        processing_edges(
            {"processing": {"kind": "uniform", "bins": 3, "start": 0, "stop": 6}}
        ),
        [0, 2, 4, 6],
    )
    assert np.array_equal(
        processing_edges(
            {"processing": {"kind": "edges", "edges": [0, 1, 3, 8]}}
        ),
        [0, 1, 3, 8],
    )
    with pytest.raises(ValueError, match="Unknown processing kind"):
        processing_edges({"processing": {"kind": "mystery"}})
    for legacy_key in ("regular", "variable", "variable_multi"):
        with pytest.raises(ValueError, match="Legacy axis schema"):
            processing_edges(
                {
                    legacy_key: [0, 1],
                    "processing": {"kind": "edges", "edges": [0, 1]},
                }
            )


def test_full_registry_is_canonical_and_processing_migration_is_complete():
    assert len(info) == 54
    assert sum(len(entry["axes"]) for entry in info_2d.values()) == 14
    for axis_config in info.values():
        assert not {"regular", "variable", "variable_multi"}.intersection(axis_config)
        validate_axis_config(axis_config)
    for family in info_2d.values():
        for axis_config in family["axes"]:
            assert not {"regular", "variable", "variable_multi"}.intersection(axis_config)
            validate_axis_config(axis_config)

    common_grid = np.arange(0, 601, 50)
    for family in ("lj0pt", "lt", "ptll", "ptz", "ptz_wtau"):
        assert np.array_equal(processing_edges(info[family]), common_grid)
    assert np.array_equal(processing_edges(info["njets"]), np.arange(8))
    assert np.array_equal(processing_edges(info["ptbl"]), [0, 100, 200, 400])
    assert np.array_equal(
        processing_edges(info["l1conept"]), [10, 20, 30, 40, 50, 60, 80, 100]
    )


def test_live_consumers_do_not_reintroduce_legacy_axis_schema():
    consumer_paths = (
        "analysis/topeft_run2/analysis_processor.py",
        "analysis/topeft_run2/analysis_processor_diboson.py",
        "analysis/topeft_run2/comp.py",
        "topeft/modules/datacard_tools.py",
    )
    legacy_accesses = tuple(
        f"[{quote}{key}{quote}]"
        for quote in ('"', "'")
        for key in ("regular", "variable", "variable_multi")
    )
    sources = {
        path: (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        for path in consumer_paths
    }
    for source in sources.values():
        assert not any(access in source for access in legacy_accesses)
    assert "make_processing_axis" in sources["analysis/topeft_run2/analysis_processor.py"]
    assert "make_processing_axis" in sources[
        "analysis/topeft_run2/analysis_processor_diboson.py"
    ]
    assert "BINNING =" not in sources["analysis/topeft_run2/comp.py"]
    assert "resolve_axis_edges" in sources["analysis/topeft_run2/comp.py"]
    run_analysis = (
        REPOSITORY_ROOT / "analysis/topeft_run2/run_analysis.py"
    ).read_text(encoding="utf-8")
    assert "rebin=" not in run_analysis


def test_fitting_resolution_is_exact_and_representable():
    registry = {
        "x": {
            "processing": {"kind": "uniform", "bins": 4, "start": 0, "stop": 4},
            "fitting": {
                "default": [0, 2, 4],
                "channels": {"exact_channel": [0, 1, 4]},
            },
        }
    }
    assert np.array_equal(
        resolve_axis_edges("x", mode="processing", channel="exact_channel", registry=registry),
        [0, 1, 2, 3, 4],
    )
    assert np.array_equal(
        resolve_axis_edges("x", mode="fitting", channel="ordinary", registry=registry),
        [0, 2, 4],
    )
    assert np.array_equal(
        resolve_axis_edges("x", mode="fitting", channel="exact_channel", registry=registry),
        [0, 1, 4],
    )
    # Prefixes and regex-like text are ordinary nonmatching strings.
    for nonmatch in ("exact", "exact_.*", "channel"):
        assert np.array_equal(
            resolve_axis_edges("x", mode="fitting", channel=nonmatch, registry=registry),
            [0, 2, 4],
        )

    no_fitting = {
        "x": {"processing": {"kind": "edges", "edges": [0, 2, 4]}}
    }
    assert np.array_equal(
        resolve_axis_edges("x", mode="fitting", channel="anything", registry=no_fitting),
        [0, 2, 4],
    )
    with pytest.raises(ValueError, match="not exactly representable"):
        validate_axis_config(
            {
                "processing": {"kind": "edges", "edges": [0, 1, 2]},
                "fitting": {"default": [0, 1.5, 2]},
            }
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_axis_config(
            {
                "processing": {"kind": "edges", "edges": [0, 1, 2]},
                "fitting": {"default": [0, 2, 1]},
            }
        )
    with pytest.raises(ValueError, match="Unknown exact fitting channel"):
        validate_axis_config(registry["x"], known_channels={"ordinary"})


def test_frozen_target_memberships_and_final_bin_count_recompute():
    lj0pt_channels = set(info["lj0pt"]["fitting"]["channels"])
    ptll_channels = set(info["ptll"]["fitting"]["channels"])
    ptz_channels = set(info["ptz"]["fitting"].get("channels", {}))
    lt_channels = set(info["lt"]["fitting"]["channels"])
    ptz_wtau_channels = {
        f"2lss_{charge}_1tau_onZ_{njets}j"
        for charge in ("m", "p")
        for njets in (3, 4, 5, 6)
    }
    assert (len(lj0pt_channels), len(ptll_channels), len(ptz_channels), len(ptz_wtau_channels), len(lt_channels)) == (
        32,
        32,
        0,
        8,
        6,
    )
    changed = (
        {("lj0pt", channel) for channel in lj0pt_channels}
        | {("ptll", channel) for channel in ptll_channels}
        | {("ptz_wtau", channel) for channel in ptz_wtau_channels}
        | {("lt", channel) for channel in lt_channels}
    )
    assert len(changed) == 78
    category_counts = {
        "lj0pt": 53,
        "lt": 29,
        "ptll": 32,
        "ptz": 7,
        "ptz_wtau": 8,
    }
    assert sum(category_counts.values()) == 129
    assert sum(category_counts.values()) - len(changed) == 51
    overrides = {
        "lj0pt": lj0pt_channels,
        "lt": lt_channels,
        "ptll": ptll_channels,
        "ptz": ptz_channels,
        "ptz_wtau": set(),
    }
    final_bins = 0
    for family, count in category_counts.items():
        family_overrides = overrides[family]
        final_bins += (count - len(family_overrides)) * len(
            resolve_axis_edges(family, mode="fitting", channel="ordinary")
        )
        final_bins += sum(
            len(resolve_axis_edges(family, mode="fitting", channel=channel))
            for channel in family_overrides
        )
    assert final_bins == 555

    assert ptll_channels == {
        f"3l_{charge}_offZ_{pt_region}_{nbtags}b_{njets}j"
        for charge in ("m", "p")
        for pt_region in ("low", "high")
        for nbtags in (1, 2)
        for njets in (2, 3, 4, 5)
    }
    for channel in ptll_channels:
        assert np.array_equal(
            resolve_axis_edges("ptll", mode="fitting", channel=channel),
            [0, 50, 100, 200, 300],
        )

    for channel in ("3l_onZ_2b_2j", "3l_onZ_2b_3j"):
        assert channel not in lj0pt_channels
        assert np.array_equal(
            resolve_axis_edges("lj0pt", mode="fitting", channel=channel),
            [0, 150, 250, 500],
        )
    assert lt_channels == {
        f"3l_{charge}_offZ_2b_fwd_{njets}j"
        for charge in ("m", "p")
        for njets in (2, 3, 4)
    }
    for charge in ("m", "p"):
        assert f"3l_{charge}_offZ_2b_fwd_1j" not in lt_channels
    for channel in ptz_wtau_channels:
        assert np.array_equal(
            resolve_axis_edges("ptz_wtau", mode="fitting", channel=channel),
            [0, 50, 100, 150],
        )


def test_njets_fitting_view_is_processing_view():
    histogram = _sparse_histogram(hist.axis.Regular(1, 0, 1, name="njets"))
    assert "fitting" not in info["njets"]
    assert np.array_equal(
        resolve_axis_edges("njets", mode="processing", channel="2l"),
        resolve_axis_edges("njets", mode="fitting", channel="2l"),
    )
    assert resolve_and_rebin_histogram(histogram, "njets", channel="2l") is histogram


def test_exact_aggregation_preserves_nominal_sumw2_eft_and_flow():
    source_axis = hist.axis.Regular(12, 0, 600, name="lj0pt")
    values = np.array([-10, 25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 650])
    weights = np.arange(1, values.size + 1, dtype=float)

    nominal = _sparse_histogram(source_axis)
    nominal.fill(
        process="ttH",
        channel="3l_1tau_1b_2j",
        lj0pt=values,
        weight=weights,
    )
    sumw2 = _sparse_histogram(hist.axis.Regular(12, 0, 600, name="lj0pt_sumw2"))
    sumw2.fill(
        process="ttH",
        channel="3l_1tau_1b_2j",
        lj0pt_sumw2=values,
        weight=weights**2,
    )

    eft = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        source_axis,
        wc_names=["ctG"],
    )
    coefficients = np.arange(values.size * 3, dtype=float).reshape(values.size, 3) + 1
    eft.fill(
        process="ttH",
        channel="3l_1tau_1b_2j",
        lj0pt=values,
        eft_coeff=coefficients,
    )

    target_edges = [0, 150, 250, 350]
    mapping = build_aggregation_map(np.arange(0, 601, 50), target_edges)
    rebinned_nominal = rebin_histogram(nominal, target_edges)
    rebinned_sumw2 = rebin_histogram(sumw2, target_edges)
    rebinned_eft = rebin_histogram(eft, target_edges)

    nominal_source = next(iter(nominal.view(flow=True).values()))
    sumw2_source = next(iter(sumw2.view(flow=True).values()))
    eft_source = next(iter(eft.view(flow=True).values()))
    assert np.array_equal(
        next(iter(rebinned_nominal.view(flow=True).values())),
        aggregate_array(nominal_source, mapping),
    )
    assert np.array_equal(
        next(iter(rebinned_sumw2.view(flow=True).values())),
        aggregate_array(sumw2_source, mapping),
    )
    assert np.array_equal(
        next(iter(rebinned_eft.view(flow=True).values())),
        aggregate_array(eft_source, mapping),
    )
    assert np.array_equal(histogram_dense_edges(rebinned_eft), target_edges)

    rebinned_values = next(iter(rebinned_nominal.view(flow=True).values()))
    assert rebinned_values[0] == weights[0]
    assert rebinned_values[-1] == weights[8:].sum()
    assert rebinned_values.sum() == weights.sum()


def test_nonprompt_linear_algebra_commutes_with_exact_aggregation():
    mapping = build_aggregation_map([0, 50, 100, 150, 200], [0, 100, 200])
    data = np.array([1.0, 10.0, 20.0, 30.0, 40.0, 2.0])
    prompt = np.array([0.5, 3.0, 4.0, 5.0, 6.0, 0.25])
    data_sumw2 = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 1.0])
    prompt_sumw2 = np.array([0.25, 1.0, 1.0, 4.0, 4.0, 0.25])
    assert np.array_equal(
        aggregate_array(data - prompt, mapping),
        aggregate_array(data, mapping) - aggregate_array(prompt, mapping),
    )
    assert np.array_equal(
        aggregate_array(data_sumw2 + prompt_sumw2, mapping),
        aggregate_array(data_sumw2, mapping) + aggregate_array(prompt_sumw2, mapping),
    )


def test_incompatible_flow_axis_is_rejected():
    histogram = _sparse_histogram(
        hist.axis.Regular(4, 0, 4, name="lj0pt", underflow=False, overflow=True)
    )
    with pytest.raises(ValueError, match="underflow-and-overflow axis contract"):
        rebin_histogram(histogram, [0, 2, 4])


def test_grouped_resolution_rejects_incompatible_fitting_axes():
    with pytest.raises(ValueError, match="incompatible fitting axes"):
        resolve_common_axis_edges(
            "lt",
            mode="fitting",
            channels=["3l_m_offZ_2b_fwd_1j", "3l_m_offZ_2b_fwd_2j"],
        )
