from __future__ import annotations

import gzip

import cloudpickle
import hist
import numpy as np
import pytest

from analysis.topeft_run2 import run_data_driven
from topcoffea.modules.sparseHist import SparseHist
from topcoffea.modules.histEFT import HistEFT
from topeft.modules import dataDrivenEstimation
from topeft.modules.dataDrivenEstimation import (
    DataDrivenProducer,
    derive_data_driven_applicability,
)
from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.nominal_schema import eft_nominal_key, scalar_nominal_key


@pytest.fixture
def sparse_hist_axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="pt"),
    )


def _fill_histogram(entries, axes):
    histogram = SparseHist(*axes)
    for entry in entries:
        histogram.fill(
            process=entry["process"],
            appl=entry["appl"],
            systematic=entry.get("systematic", "nominal"),
            pt=entry.get("pt", 0.5),
            weight=entry["weight"],
        )
    return histogram


def _build_hist_dict(axes):
    entries = [
        {"process": "dataUL16", "appl": "isAR_3l", "weight": 10.0},
        {"process": "TTTo2L2Nu_centralUL16", "appl": "isAR_3l", "weight": 3.0},
        {"process": "dataUL16", "appl": "isAR_2lSS_OS", "weight": 4.0},
        {"process": "TTTo2L2Nu_centralUL16", "appl": "isSR_3l", "weight": 1.0},
    ]
    for systematic, factor in (
        ("renormUp", 1.1),
        ("renormDown", 0.9),
        ("factUp", 1.2),
        ("factDown", 0.8),
    ):
        entries.append(
            {
                "process": "TTTo2L2Nu_centralUL16",
                "appl": "isAR_3l",
                "systematic": systematic,
                "weight": 3.0 * factor,
            }
        )
    main_hist = _fill_histogram(entries, axes)

    sumw2_entries = [dict(entry, weight=entry["weight"] ** 2) for entry in entries]
    sumw2_hist = _fill_histogram(sumw2_entries, axes)

    return {"nominal": main_hist, "nominal_sumw2": sumw2_hist}


def test_data_driven_producer_streams_histograms(monkeypatch, tmp_path):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"placeholder")

    calls = []

    def fake_iterate(path, *, allow_empty, materialize=False):
        calls.append((path, allow_empty, materialize))
        return iter(())

    monkeypatch.setattr(dataDrivenEstimation, "iterate_hist_from_pkl", fake_iterate)
    monkeypatch.setattr(
        dataDrivenEstimation,
        "validate_histogram_artifact",
        lambda _path: {"schema": "split_sibling_v1", "metadata": {}},
    )

    producer = DataDrivenProducer(str(input_path), "")
    assert producer.getDataDrivenHistogram() == {}
    assert calls == [
        (str(input_path), True, False),
        (str(input_path), True, False),
    ]


def test_run_data_driven_matches_inline_output(tmp_path, sparse_hist_axes):
    expected_histograms = DataDrivenProducer(_build_hist_dict(sparse_hist_axes), "").getDataDrivenHistogram()

    input_histograms = _build_hist_dict(sparse_hist_axes)
    input_path = tmp_path / "input.pkl.gz"
    with gzip.open(input_path, "wb") as stream:
        cloudpickle.dump(input_histograms, stream)

    output_path = tmp_path / "output_np.pkl.gz"
    with pytest.warns(UserWarning, match="legacy uniform") as warning_records:
        run_data_driven._finalize_histograms(
            str(input_path),
            str(output_path),
            only_flips=False,
            apply_envelope=False,
            heartbeat_seconds=0.0,
            quiet=True,
        )
    assert len(warning_records) == 1

    streamed_histograms = get_hist_from_pkl(str(output_path))

    assert set(streamed_histograms) == set(expected_histograms)
    for key, expected_hist in expected_histograms.items():
        streamed_hist = streamed_histograms[key]
        expected_view = expected_hist.view(flow=True, as_dict=True)
        streamed_view = streamed_hist.view(flow=True, as_dict=True)
        assert set(streamed_view) == set(expected_view)
        for sparse_key, expected_payload in expected_view.items():
            np.testing.assert_allclose(
                np.asarray(streamed_view[sparse_key]),
                np.asarray(expected_payload),
            )
        assert list(streamed_hist.axes["process"]) == list(expected_hist.axes["process"])
        assert {"renormUp", "renormDown", "factUp", "factDown"}.issubset(
            set(streamed_hist.axes["systematic"])
        )


def _split_axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=dense_name),
    )


def _total_for_process(histogram, process):
    selected = histogram.integrate("process", process)
    if isinstance(selected, HistEFT):
        values = selected.eval({})
    else:
        values = selected.view(flow=True, as_dict=True)
    return sum(float(np.asarray(value).sum()) for value in values.values())


def test_split_nonprompt_uses_scalar_subtraction_sumw2_addition_and_eft_passthrough():
    scalar = SparseHist(*_split_axes("njets"), storage="Double")
    companion = SparseHist(*_split_axes("njets_sumw2"), storage="Double")
    for process, nominal, second_moment in (
        ("dataUL18", 10.0, 100.0),
        ("TTTo2L2Nu_centralUL18", 3.0, 9.0),
    ):
        scalar.fill(
            process=process,
            appl="isAR_3l",
            systematic="nominal",
            njets=np.asarray([0.5]),
            weight=np.asarray([nominal]),
        )
        companion.fill(
            process=process,
            appl="isAR_3l",
            systematic="nominal",
            njets_sumw2=np.asarray([0.5]),
            weight=np.asarray([second_moment]),
        )
    eft = HistEFT(*_split_axes("njets"), wc_names=["ctG"], label="Events")
    for appl, weight in (("isAR_3l", 9.0), ("isSR_3l", 5.0)):
        eft.fill(
            process="signal_centralUL18",
            appl=appl,
            systematic="nominal",
            njets=np.asarray([0.5]),
            weight=np.asarray([weight]),
            eft_coeff=np.asarray([[1.0, 2.0, 3.0]]),
        )

    output = DataDrivenProducer(
        {
            scalar_nominal_key("njets"): scalar,
            eft_nominal_key("njets"): eft,
            "njets_sumw2": companion,
        },
        "",
    ).getDataDrivenHistogram()
    assert _total_for_process(output[scalar_nominal_key("njets")], "nonpromptUL18") == pytest.approx(7.0)
    assert _total_for_process(output["njets_sumw2"], "nonpromptUL18") == pytest.approx(109.0)
    assert _total_for_process(output[eft_nominal_key("njets")], "signal_centralUL18") == pytest.approx(5.0)
    assert "appl" not in [axis.name for axis in output[eft_nominal_key("njets")].axes]


def test_split_nonprompt_rejects_missing_selected_scalar_companion():
    scalar = SparseHist(*_split_axes("njets"), storage="Double")
    scalar.fill(
        process="dataUL18",
        appl="isAR_3l",
        systematic="nominal",
        njets=np.asarray([0.5]),
        weight=np.asarray([1.0]),
    )
    producer = DataDrivenProducer(
        {scalar_nominal_key("njets"): scalar}, "", iterator_mode=True
    )
    with pytest.raises(RuntimeError, match="requires scalar statistical companions"):
        producer.getDataDrivenHistogram()


@pytest.mark.parametrize(
    "regions,expected_generated",
    [
        (["isAR_1l", "isSR_1l"], {"nonpromptUL18"}),
        (["isAR_2lSS_OS"], {"flipsUL18"}),
        (["isSR_1l"], set()),
        (["isAR_future", "isSR_1l"], set()),
    ],
    ids=[
        "legacy_nonprompt",
        "legacy_flips",
        "legacy_no_ar",
        "unknown_ar_is_not_nonprompt",
    ],
)
def test_data_driven_application_regions_have_explicit_physical_semantics(
    sparse_hist_axes,
    regions,
    expected_generated,
):
    entries = []
    for region in regions:
        if region == "isAR_2lSS_OS":
            entries.append(
                {"process": "dataUL18", "appl": region, "weight": 4.0}
            )
        elif region.startswith("isAR"):
            entries.extend(
                [
                    {"process": "dataUL18", "appl": region, "weight": 10.0},
                    {
                        "process": "TTTo2L2Nu_centralUL18",
                        "appl": region,
                        "weight": 3.0,
                    },
                ]
            )
        else:
            entries.append(
                {
                    "process": "TTTo2L2Nu_centralUL18",
                    "appl": region,
                    "weight": 2.0,
                }
            )
    histograms = {
        "nominal": _fill_histogram(entries, sparse_hist_axes),
        "nominal_sumw2": _fill_histogram(
            [dict(entry, weight=entry["weight"] ** 2) for entry in entries],
            sparse_hist_axes,
        ),
    }
    output = DataDrivenProducer(histograms, "").getDataDrivenHistogram()
    generated_names = {"nonpromptUL18", "flipsUL18"}
    observed = generated_names & {
        str(process) for process in output["nominal"].axes["process"]
    }
    assert observed == expected_generated
    assert derive_data_driven_applicability(regions) == {
        "nonprompt": "nonpromptUL18" in expected_generated,
        "flips": "flipsUL18" in expected_generated,
    }
