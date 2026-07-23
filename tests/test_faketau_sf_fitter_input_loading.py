import gzip
import pickle

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.datacard_tools import load_and_merge_histogram_pkls
from topeft.modules.histogram_artifact import write_histogram_artifact
from topeft.modules.nominal_schema import eft_nominal_key, scalar_nominal_key
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy

from analysis.topeft_run2 import faketau_sf_fitter as fitter
from analysis.topeft_run2 import tauFitter as legacy_fitter
from tests.sumw2_profile_test_helpers import certify_test_profile


def _make_tau_hist(axis_name, value):
    tau_edges = [20.0, 30.0]
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Variable(tau_edges, name=axis_name),
        wc_names=[],
        label="Events",
    )
    histogram.metadata = {}
    histogram.fill(
        process="ttbar",
        channel=(
            "2los_1tau_Ftau_2j"
            if "Fpt" in axis_name
            else "2los_1tau_Ttau_2j"
        ),
        systematic="nominal",
        appl="isSR_2lOS",
        **{axis_name: np.array([25.0])},
        weight=np.array([float(value)]),
    )
    return histogram


def _make_tau_sparse(axis_name, process, value):
    histogram = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Variable([20.0, 30.0], name=axis_name),
        storage="Double",
    )
    histogram.fill(
        process=process,
        channel="2los_1tau_Ftau_2j" if "Fpt" in axis_name else "2los_1tau_Ttau_2j",
        systematic="nominal",
        appl="isSR_2lOS",
        **{axis_name: np.asarray([25.0])},
        weight=np.asarray([value]),
    )
    return histogram


def _make_payload(
    *,
    fake_value=2.0,
    tight_value=3.0,
    fake_sumw2=4.0,
    tight_sumw2=9.0,
    include_sumw2=True,
):
    payload = {
        "tau0Fpt": _make_tau_hist("tau0Fpt", fake_value),
        "tau0Tpt": _make_tau_hist("tau0Tpt", tight_value),
    }
    if include_sumw2:
        payload["tau0Fpt_sumw2"] = _make_tau_hist("tau0Fpt_sumw2", fake_sumw2)
        payload["tau0Tpt_sumw2"] = _make_tau_hist("tau0Tpt_sumw2", tight_sumw2)
    return payload


def _write_payload(path, payload):
    with gzip.open(path, "wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def _hist_total(histogram):
    return float(np.asarray(histogram.values(flow=False), dtype=float).sum())


def test_normalize_input_pkl_paths_single_multi_and_default():
    assert fitter.normalize_input_pkl_paths("one.pkl.gz") == ["one.pkl.gz"]
    assert fitter.normalize_input_pkl_paths(["a.pkl.gz", "b.pkl.gz"]) == [
        "a.pkl.gz",
        "b.pkl.gz",
    ]
    assert fitter.normalize_input_pkl_paths([]) == [fitter.DEFAULT_INPUT_PKL_PATH]


def test_build_arg_parser_accepts_single_multiple_and_default_paths():
    parser = fitter.build_arg_parser()

    assert parser.parse_args([]).pkl_file_path == [fitter.DEFAULT_INPUT_PKL_PATH]
    assert parser.parse_args(["-f", "one.pkl.gz"]).pkl_file_path == ["one.pkl.gz"]
    assert parser.parse_args(["-f", "a.pkl.gz", "b.pkl.gz"]).pkl_file_path == [
        "a.pkl.gz",
        "b.pkl.gz",
    ]
    args = parser.parse_args(["-f", "a.pkl.gz", "b.pkl.gz", "--dump-channels", "-"])
    assert args.pkl_file_path == ["a.pkl.gz", "b.pkl.gz"]
    assert args.dump_channels == "-"


def test_combine_faketau_histogram_pkls_adds_nominal_and_sumw2_contents(tmp_path):
    path_a = tmp_path / "a.pkl.gz"
    path_b = tmp_path / "b.pkl.gz"
    _write_payload(
        path_a,
        _make_payload(fake_value=2.0, tight_value=3.0, fake_sumw2=4.0, tight_sumw2=9.0),
    )
    _write_payload(
        path_b,
        _make_payload(
            fake_value=5.0,
            tight_value=7.0,
            fake_sumw2=25.0,
            tight_sumw2=49.0,
        ),
    )

    combined, summary = fitter.combine_faketau_histogram_pkls(
        [str(path_a), str(path_b)]
    )

    assert tuple(combined) == (
        "tau0Fpt",
        "tau0Tpt",
        "tau0Fpt_sumw2",
        "tau0Tpt_sumw2",
    )
    assert isinstance(combined["tau0Fpt"], HistEFT)
    assert isinstance(combined["tau0Tpt"], HistEFT)
    assert _hist_total(combined["tau0Fpt"]) == pytest.approx(7.0)
    assert _hist_total(combined["tau0Tpt"]) == pytest.approx(10.0)
    assert _hist_total(combined["tau0Fpt_sumw2"]) == pytest.approx(29.0)
    assert _hist_total(combined["tau0Tpt_sumw2"]) == pytest.approx(58.0)
    assert summary["num_inputs"] == 2
    assert summary["sumw2_status"]["tau0Fpt_sumw2"] == "present in all input files"
    assert summary["sumw2_status"]["tau0Tpt_sumw2"] == "present in all input files"


def test_combine_faketau_histogram_pkls_rejects_all_absent_sumw2(tmp_path):
    path_a = tmp_path / "a.pkl.gz"
    path_b = tmp_path / "b.pkl.gz"
    _write_payload(path_a, _make_payload(include_sumw2=False))
    _write_payload(
        path_b,
        _make_payload(fake_value=5.0, tight_value=7.0, include_sumw2=False),
    )

    with pytest.raises(RuntimeError, match=r"missing required \*_sumw2 companions"):
        fitter.combine_faketau_histogram_pkls([str(path_a), str(path_b)])


def test_combine_faketau_histogram_pkls_reports_missing_required_histogram(tmp_path):
    path_a = tmp_path / "a.pkl.gz"
    path_b = tmp_path / "missing_tight.pkl.gz"
    _write_payload(path_a, _make_payload())
    broken_payload = _make_payload()
    broken_payload.pop("tau0Tpt")
    _write_payload(path_b, broken_payload)

    with pytest.raises(RuntimeError) as exc_info:
        fitter.combine_faketau_histogram_pkls([str(path_a), str(path_b)])

    message = str(exc_info.value)
    assert "tau0Tpt" in message
    assert str(path_b) in message


def test_combine_faketau_histogram_pkls_rejects_mixed_sumw2_availability(tmp_path):
    path_a = tmp_path / "with_sumw2.pkl.gz"
    path_b = tmp_path / "without_sumw2.pkl.gz"
    _write_payload(path_a, _make_payload())
    payload_without_fake_sumw2 = _make_payload()
    payload_without_fake_sumw2.pop("tau0Fpt_sumw2")
    _write_payload(path_b, payload_without_fake_sumw2)

    with pytest.raises(RuntimeError) as exc_info:
        fitter.combine_faketau_histogram_pkls([str(path_a), str(path_b)])

    message = str(exc_info.value)
    assert "tau0Fpt" in message
    assert "missing required *_sumw2 companions" in message


def test_merge_faketau_histogram_dicts_does_not_mutate_loaded_inputs():
    payload_a = _make_payload(fake_value=2.0, tight_value=3.0)
    payload_b = _make_payload(fake_value=5.0, tight_value=7.0)
    original_a_fake_total = _hist_total(payload_a["tau0Fpt"])
    original_b_fake_total = _hist_total(payload_b["tau0Fpt"])

    combined, _summary = fitter.merge_faketau_histogram_dicts(
        [payload_a, payload_b],
        input_paths=["a.pkl.gz", "b.pkl.gz"],
    )

    assert combined["tau0Fpt"] is not payload_a["tau0Fpt"]
    assert combined["tau0Fpt"] is not payload_b["tau0Fpt"]
    assert _hist_total(payload_a["tau0Fpt"]) == pytest.approx(original_a_fake_total)
    assert _hist_total(payload_b["tau0Fpt"]) == pytest.approx(original_b_fake_total)
    assert _hist_total(combined["tau0Fpt"]) == pytest.approx(7.0)


def test_split_faketau_boundary_uses_wc_zero_scalar_view_and_strict_companions():
    split = {}
    for family in fitter.FAKETAU_REQUIRED_HISTOGRAMS:
        split[scalar_nominal_key(family)] = _make_tau_sparse(
            family, "data2018", 4.0
        )
        split[eft_nominal_key(family)] = _make_tau_hist(family, 3.0)
        split[f"{family}_sumw2"] = _make_tau_sparse(
            f"{family}_sumw2", "data2018", 16.0
        )
    scalar_view = fitter._materialize_faketau_scalar_view(split)
    assert tuple(scalar_view) == (
        "tau0Fpt",
        "tau0Fpt_sumw2",
        "tau0Tpt",
        "tau0Tpt_sumw2",
    )
    assert all(isinstance(value, SparseHist) for value in scalar_view.values())
    assert _hist_total(scalar_view["tau0Fpt"]) == pytest.approx(7.0)
    assert _hist_total(scalar_view["tau0Fpt_sumw2"]) == pytest.approx(16.0)

    missing = dict(split)
    missing.pop("tau0Fpt_sumw2")
    with pytest.raises(RuntimeError, match="requires selected companion"):
        fitter._materialize_faketau_scalar_view(missing)

    legacy_view = legacy_fitter.prepare_taufitter_histograms(split)
    assert tuple(legacy_view) == (
        "tau0Fpt",
        "tau0Fpt_sumw2",
        "tau0Tpt",
        "tau0Tpt_sumw2",
    )
    assert all(isinstance(value, SparseHist) for value in legacy_view.values())
    with pytest.raises(RuntimeError, match="requires selected companion"):
        legacy_fitter.prepare_taufitter_histograms(missing)


def test_schema_v2_tau_consumers_discover_sidecar_from_pkl_only(tmp_path):
    samples = {
        "ttbar_dataset": {
            "histAxisName": "ttbar",
            "isData": False,
            "WCnames": [],
        }
    }
    policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=samples,
        runtime_families=fitter.FAKETAU_REQUIRED_HISTOGRAMS,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    split = {}
    for family in fitter.FAKETAU_REQUIRED_HISTOGRAMS:
        split[scalar_nominal_key(family)] = _make_tau_sparse(
            family, "ttbar", 3.0
        )
        split[f"{family}_sumw2"] = _make_tau_sparse(
            f"{family}_sumw2", "ttbar", 9.0
        )
    path = tmp_path / "tau.pkl.gz"
    write_histogram_artifact(
        path,
        histograms=split,
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=certify_test_profile(policy, samples),
    )

    fake_tau_view, summary = fitter.combine_faketau_histogram_pkls([str(path)])
    assert summary["schema"] == "split_sibling_v1"
    assert tuple(fake_tau_view) == (
        "tau0Fpt",
        "tau0Fpt_sumw2",
        "tau0Tpt",
        "tau0Tpt_sumw2",
    )
    loaded, report = load_and_merge_histogram_pkls([str(path)])
    assert report["artifact_kind"] == "processor_output"
    legacy_tau_view = legacy_fitter.prepare_taufitter_histograms(loaded)
    assert tuple(legacy_tau_view) == tuple(fake_tau_view)
