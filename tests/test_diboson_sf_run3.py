import base64
import importlib.util
from pathlib import Path

import sys
import types

import pytest


try:  # pragma: no cover - exercised indirectly in tests
    import awkward as _awkward_module  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on env
    _awkward_module = None
    sys.modules["awkward"] = types.ModuleType("awkward")


pytest.importorskip("hist")
pytest.importorskip("boost_histogram")


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "diboson_njets"
    / "diboson_sf_run3.py"
)
spec = importlib.util.spec_from_file_location("diboson_sf_run3", MODULE_PATH)
diboson_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader  # for type checkers
spec.loader.exec_module(diboson_module)

get_yields_in_bins = diboson_module.get_yields_in_bins
load_pkl_file = diboson_module.load_pkl_file


def _materialize_histogram_fixture(tmp_path: Path) -> Path:
    encoded_path = (
        Path(__file__).resolve().parent / "data" / "run3_histogram.pkl.gz.base64"
    )
    encoded_data = encoded_path.read_text().strip()
    data = base64.b64decode(encoded_data)
    fixture_path = tmp_path / "run3_histogram.pkl.gz"
    fixture_path.write_bytes(data)
    return fixture_path


def test_run3_histogram_yields_are_non_zero(tmp_path):
    data_path = _materialize_histogram_fixture(tmp_path)
    histograms = load_pkl_file(str(data_path))

    bins = [0, 1, 2, 3, 4, 5, 6]
    proc_list = ["ZZTo2L2Nu_2022", "WZTo3LNu_2022"]

    yields = get_yields_in_bins(
        histograms,
        proc_list=proc_list,
        bins=bins,
        hist_name="njets",
        channel_name="mu_mu",
        extra_slices={"year": "2022"},
    )

    assert yields["ZZTo2L2Nu_2022"][1][0] > 0.0
    assert yields["ZZTo2L2Nu_2022"][2][0] > 0.0
    assert yields["WZTo3LNu_2022"][3][0] > 0.0


def test_get_yields_in_bins_raises_on_slice_failure(tmp_path):
    data_path = _materialize_histogram_fixture(tmp_path)
    histograms = load_pkl_file(str(data_path))

    bins = [0, 1, 2, 3, 4, 5, 6]
    proc_list = ["ZZTo2L2Nu_2022"]

    with pytest.raises(RuntimeError, match="Failed to compute yields"):
        get_yields_in_bins(
            histograms,
            proc_list=proc_list,
            bins=bins,
            hist_name="njets",
            channel_name="non_existent_channel",
            extra_slices={"year": "2022"},
        )


def test_get_yields_in_bins_supports_sparse_hist(tmp_path):
    if _awkward_module is None or not hasattr(_awkward_module, "Array"):
        pytest.skip("SparseHist conversion requires the real awkward library")

    try:
        from topcoffea.modules.sparseHist import SparseHist
    except ModuleNotFoundError:
        pytest.skip("SparseHist regression test requires topcoffea")

    data_path = _materialize_histogram_fixture(tmp_path)
    dense_histograms = load_pkl_file(str(data_path))

    bins = [0, 1, 2, 3, 4, 5, 6]
    proc_list = ["ZZTo2L2Nu_2022", "WZTo3LNu_2022"]

    dense_yields = get_yields_in_bins(
        dense_histograms,
        proc_list=proc_list,
        bins=bins,
        hist_name="njets",
        channel_name="mu_mu",
        extra_slices={"year": "2022"},
    )

    dense_hist = dense_histograms["njets"]
    sparse_hist = SparseHist(*dense_hist.axes)

    values = dense_hist.values(flow=False)
    process_axis = dense_hist.axes["process"]
    channel_axis = dense_hist.axes["channel"]
    year_axis = dense_hist.axes["year"]
    njets_axis = dense_hist.axes["njets"]

    for iproc, proc in enumerate(process_axis):
        for ichan, channel in enumerate(channel_axis):
            for iyear, year in enumerate(year_axis):
                for inj, center in enumerate(njets_axis.centers):
                    weight = float(values[iproc, ichan, iyear, inj])
                    if weight:
                        sparse_hist.fill(
                            process=proc,
                            channel=channel,
                            year=year,
                            njets=center,
                            weight=weight,
                        )

    sparse_histograms = dict(dense_histograms)
    sparse_histograms["njets"] = sparse_hist

    sparse_yields = get_yields_in_bins(
        sparse_histograms,
        proc_list=proc_list,
        bins=bins,
        hist_name="njets",
        channel_name="mu_mu",
        extra_slices={"year": "2022"},
    )

    for proc in proc_list:
        dense_vals = [val for val, _ in dense_yields[proc]]
        sparse_vals = [val for val, _ in sparse_yields[proc]]
        assert sparse_vals == pytest.approx(dense_vals)
