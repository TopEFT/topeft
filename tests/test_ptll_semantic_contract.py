import ast
from collections import Counter
import json
from pathlib import Path
import re
import shlex

import hist
import numpy as np
import pytest

from analysis.topeft_run2 import analysis_processor
from analysis.topeft_run2 import make_cr_and_sr_plots
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info
from topeft.modules.axis_binning import (
    histogram_dense_edges,
    processing_edges,
    resolve_axis_edges,
)
from topeft.modules.datacard_tools import DatacardMaker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_PATH = REPOSITORY_ROOT / "analysis/topeft_run2/analysis_processor.py"
POSTPROCESSOR_PATH = (
    REPOSITORY_ROOT / "analysis/topeft_run2/datacards_post_processing.py"
)
RUN_CR_PATH = REPOSITORY_ROOT / "analysis/topeft_run2/run_cr.sh"
CARD_MATRIX_PATH = REPOSITORY_ROOT / "run_make_cards_run3_yawen_matrix.sh"
CHANNEL_REGISTRY_PATH = REPOSITORY_ROOT / "topeft/channels/ch_lst.json"


EXPECTED_PTLL_CATEGORIES = {
    f"3l_{charge}_offZ_{pt_region}_{nbtags}b_{njets}j"
    for charge in ("m", "p")
    for pt_region in ("low", "high")
    for nbtags in (1, 2)
    for njets in (2, 3, 4, 5)
}
EXPECTED_PTZ_CATEGORIES = {
    "2los_onZ_1tau_3j",
    "3l_onZ_1b_2j",
    "3l_onZ_1b_3j",
    "3l_onZ_1b_4j",
    "3l_onZ_1b_5j",
    "3l_onZ_2b_4j",
    "3l_onZ_2b_5j",
}
EXPECTED_OFFZ_NONE_CATEGORIES = {
    f"3l_{charge}_offZ_none_{nbtags}b_{njets}j"
    for charge in ("m", "p")
    for nbtags in (1, 2)
    for njets in (2, 3, 4, 5)
}


def _card_matrix_mapping():
    mapping = {}
    for raw_line in CARD_MATRIX_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("run_job "):
            continue
        fields = shlex.split(line)
        distribution = fields[5]
        expected_count = int(fields[6])
        patterns = fields[7:]
        assert len(patterns) == expected_count
        for pattern in patterns:
            category = pattern.removeprefix("^")
            if category.endswith("\\$"):
                category = category[:-2]
            else:
                category = category.removesuffix("$")
            assert category not in mapping
            mapping[category] = distribution
    return mapping


def _bash_array(script, name):
    match = re.search(rf"^{re.escape(name)}=\(\n(?P<body>.*?)^\)$", script, re.M | re.S)
    assert match, name
    return shlex.split(match.group("body"))


def _lep_channel(category):
    return category.rsplit("_", 1)[0]


def _sparse_histogram(family, channel):
    histogram = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.Regular(12, 0, 600, name=family),
        storage="Double",
    )
    histogram.fill(
        process="ttH",
        channel=channel,
        **{family: np.asarray([25.0, 75.0, 225.0])},
        weight=np.asarray([1.0, 2.0, 3.0]),
    )
    return histogram


def test_ptll_axis_is_canonical_and_ptz_has_no_offz_overrides():
    source_edges = np.arange(0, 601, 50)
    target_edges = [0, 50, 100, 200, 300]

    assert np.array_equal(processing_edges(info["ptll"]), source_edges)
    assert info["ptll"]["label"] == r"$p_{T}^{\ell\ell}$ (GeV) "
    assert "Z" in info["ptz"]["label"]
    assert "channels" not in info["ptz"]["fitting"]
    assert set(info["ptll"]["fitting"]["channels"]) == EXPECTED_PTLL_CATEGORIES
    assert info["ptll"]["fitting"]["default"] == target_edges
    for category in EXPECTED_PTLL_CATEGORIES:
        assert np.array_equal(
            resolve_axis_edges("ptll", mode="fitting", channel=category),
            target_edges,
        )


def test_processor_binds_distinct_physical_expressions_without_aliasing():
    tree = ast.parse(PROCESSOR_PATH.read_text(encoding="utf-8"))
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"ptz", "ptll"}:
            assignments.append((target.id, ast.unparse(node.value)))

    assert ("ptz", "te_es.get_Z_pt(l_fo_conept_sorted_padded[:, 0:3], 10.0)") in assignments
    assert ("ptll", "te_es.get_ll_pt(l_fo_conept_sorted_padded[:, 0:3], 10.0)") in assignments
    assert not any(name == "ptz" and "get_ll_pt" in value for name, value in assignments)


@pytest.mark.parametrize("all_analysis", [False, True])
def test_processor_routes_exact_offz_set_to_ptll_only(all_analysis):
    processor = analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        offZ_split=not all_analysis,
        all_analysis=all_analysis,
    )

    for category in EXPECTED_PTLL_CATEGORIES:
        lep_channel = _lep_channel(category)
        assert processor._should_skip_histogram_fill("ptll", category, lep_channel) is False
        assert processor._should_skip_histogram_fill("ptz", category, lep_channel) is True
    for category in EXPECTED_OFFZ_NONE_CATEGORIES:
        lep_channel = _lep_channel(category)
        assert processor._should_skip_histogram_fill("ptll", category, lep_channel) is True


def test_processor_keeps_onz_under_ptz_and_allocates_ptll_sumw2():
    processor = analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=["ptz", "ptll"],
        all_analysis=True,
    )

    for category in EXPECTED_PTZ_CATEGORIES:
        lep_channel = _lep_channel(category)
        assert processor._should_skip_histogram_fill("ptz", category, lep_channel) is False
        assert processor._should_skip_histogram_fill("ptll", category, lep_channel) is True

    assert {
        "ptz__scalar_nominal",
        "ptz_sumw2",
        "ptll__scalar_nominal",
        "ptll_sumw2",
    } == set(processor.accumulator)
    assert processor._hist_sumw2_axis_mapping["ptz"] == {"ptz_sumw2": "ptz"}
    assert processor._hist_sumw2_axis_mapping["ptll"] == {"ptll_sumw2": "ptll"}
    assert np.array_equal(
        processor.accumulator["ptll__scalar_nominal"].dense_axes[0].edges,
        processor.accumulator["ptll_sumw2"].dense_axes[0].edges,
    )


def test_card_matrix_freezes_exact_family_membership_and_555_bins():
    mapping = _card_matrix_mapping()
    counts = Counter(mapping.values())

    assert len(mapping) == 129
    assert counts == {
        "lj0pt": 53,
        "lt": 29,
        "ptll": 32,
        "ptz": 7,
        "ptz_wtau": 8,
    }
    assert {category for category, family in mapping.items() if family == "ptll"} == EXPECTED_PTLL_CATEGORIES
    assert {category for category, family in mapping.items() if family == "ptz"} == EXPECTED_PTZ_CATEGORIES
    assert {category for category in mapping if "offZ_none" in category} == EXPECTED_OFFZ_NONE_CATEGORIES
    assert {mapping[category] for category in EXPECTED_OFFZ_NONE_CATEGORIES} == {"lj0pt"}
    assert set(info["ptll"]["fitting"]["channels"]) == EXPECTED_PTLL_CATEGORIES

    fitting_bin_count = sum(
        len(resolve_axis_edges(family, mode="fitting", channel=category))
        for category, family in mapping.items()
    )
    assert fitting_bin_count == 555

    channel_registry = json.loads(CHANNEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert "ALL_CH_LST_SR" in channel_registry
    postprocessor = POSTPROCESSOR_PATH.read_text(encoding="utf-8")
    assert 'channelname = lep_ch_name + "_" + jet + "j_ptll"' in postprocessor


def test_plotting_and_datacardmaker_use_ptll_generically():
    category = "3l_m_offZ_low_1b_2j"
    histogram = _sparse_histogram("ptll", category)
    datacard_maker = DatacardMaker.__new__(DatacardMaker)

    datacard_maker.binning_mode = "processing"
    assert datacard_maker.binning_view(histogram, "ptll", category) is histogram
    datacard_maker.binning_mode = "fitting"
    rebinned = datacard_maker.binning_view(histogram, "ptll", category)
    assert np.array_equal(histogram_dense_edges(rebinned), [0, 50, 100, 200, 300])

    assert make_cr_and_sr_plots._resolve_plot_axis_label("ptll") == r"$p_{T}^{\ell\ell}$ (GeV) "
    assert make_cr_and_sr_plots._resolve_plot_axis_label("ptz") == r"$p_{T}$ Z (GeV) "

    datacard_maker.wc_ranges = {}
    scalings = datacard_maker.make_scalings_json(
        [], category, "ptll", "ttH", [], np.asarray([[1.0], [2.0]])
    )
    assert scalings[0]["channel"] == f"{category}_ptll"
    assert "ptll" in DatacardMaker.FNAME_TEMPLATE.format(
        cat=category, kmvar="ptll", ext="root"
    )


def test_historical_ptz_does_not_satisfy_new_ptll_card_lookup():
    category = "3l_m_offZ_low_1b_2j"
    datacard_maker = DatacardMaker.__new__(DatacardMaker)
    datacard_maker.hists = {"ptz": _sparse_histogram("ptz", category)}

    with pytest.raises(KeyError):
        datacard_maker.channels("ptll")


def test_run_cr_profiles_request_canonical_families():
    source = RUN_CR_PATH.read_text(encoding="utf-8")

    assert _bash_array(source, "rebin_fine_category_sets") == [
        "2lss_1tau 3l_m_offZ",
        "3l_p_offZ 3l_onZ_tau",
        "3l_fwd",
    ]
    assert _bash_array(source, "rebin_fine_2lss_1tau_3l_m_offz_var_sets") == [
        "lj0pt ptll ptz_wtau"
    ]
    assert _bash_array(source, "rebin_fine_3l_p_offz_3l_onZ_tau_var_sets") == [
        "lj0pt ptz ptll"
    ]
    assert _bash_array(source, "rebin_fine_3l_fwd_var_sets") == ["lt"]

    assert _bash_array(source, "sr_offz_var_sets") == ["njets lj0pt ptll lt"]
    assert _bash_array(source, "sr_onz_tau_var_sets") == ["njets lj0pt ptz lt"]
    assert _bash_array(source, "sr_fwd_var_sets") == ["njets lj0pt ptz lt"]
    assert _bash_array(source, "sr_run2_category_var_set_names") == [
        "sr_with_ptz_wtau_var_sets",
        "sr_offz_var_sets",
        "sr_onz_tau_var_sets",
        "sr_fwd_var_sets",
    ]
    assert _bash_array(source, "sr_run3_category_var_set_names") == [
        "sr_with_ptz_wtau_var_sets",
        "sr_offz_var_sets",
        "sr_offz_var_sets",
        "sr_onz_tau_var_sets",
        "sr_fwd_var_sets",
    ]
