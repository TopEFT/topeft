"""Small real v3 source/DD/consumer fixtures; no production inputs."""
import copy
import gzip
import json

import cloudpickle
import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topeft.modules import nominal_schema as ns
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.data_driven_products import (
    resolve_data_driven_products, certify_embedded_data_driven_preflight,
    validate_embedded_product_input,
)
from topeft.modules.histogram_artifact import (
    write_histogram_artifact, validate_histogram_artifact,
    histogram_artifact_error, metadata_sidecar_path,
)
from topeft.modules.production_sample_profile import (
    active_sample_universe, certify_embedded_sample_contract,
)
from topeft.modules.sumw2_policy import resolve_embedded_sumw2_mode


def source_fixture(tmp_path, *, store=True, diboson=False, systematics=False):
    samples = {
        "data": {"histAxisName": "dataUL18", "isData": True, "WCnames": []},
        "prompt": {"histAxisName": "TTTo2L2Nu_centralUL18", "isData": False, "WCnames": []},
        "eft": {"histAxisName": "ttHJet_privateUL18", "isData": False, "WCnames": ["ctG"]},
    }
    if diboson:
        samples["diboson"] = {"histAxisName": "WWTo2L2Nu_centralUL18", "isData": False, "WCnames": []}
    products = resolve_data_driven_products(
        None, data_driven_products_present=False, legacy_do_np=True,
        samples=samples, runtime_families=("njets",), metadata_path="unit_fixture",
    )
    requested, contract = certify_embedded_data_driven_preflight(
        products, sumw2_storage="embedded" if store else "none")
    universe = active_sample_universe("unit_fixture", (), tuple(sorted(samples)),
        tuple(sorted(s["histAxisName"] for s in samples.values())))
    sample_contract = certify_embedded_sample_contract(universe,
        resolve_embedded_sumw2_mode({"mode": "embedded"}, sumw2_storage_present=True))
    def make(wcs):
        return HistEFT(
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True),
            hist.axis.StrCategory([], name="appl", growth=True),
            hist.axis.Regular(10, 0, 10, name="njets"),
            wc_names=wcs, use_multicell=True, store_sumw2=store,
        )
    scalar, eft = make([]), make(["ctG"])
    for appl in ("isSR_3l", "isAR_3l", "isAR_2lSS_OS"):
        for process, weight in (("dataUL18", 3.), ("TTTo2L2Nu_centralUL18", 2.)):
            scalar.fill(process=process, channel="3l_p_offZ_1b", systematic="nominal",
                        appl=appl, njets=np.array([2.5]), weight=np.array([weight]))
        if diboson:
            scalar.fill(process="WWTo2L2Nu_centralUL18", channel="3l_p_offZ_1b", systematic="nominal",
                        appl=appl, njets=np.array([2.5]), weight=np.array([.25]))
        eft.fill(process="ttHJet_privateUL18", channel="3l_p_offZ_1b", systematic="nominal",
                 appl=appl, njets=np.array([2.5]), weight=np.array([1.]),
                 eft_coeff=np.array([[1., 2., 3.]]))
        if systematics:
            for variation, factor in (("renormUp", 1.1), ("renormDown", .9), ("factUp", 1.2), ("factDown", .8)):
                for process, weight in (("dataUL18", 3.), ("TTTo2L2Nu_centralUL18", 2.)):
                    scalar.fill(process=process, channel="3l_p_offZ_1b", systematic=variation,
                        appl=appl, njets=np.array([2.5]), weight=np.array([weight * factor]), fill_sumw2=False)
                eft.fill(process="ttHJet_privateUL18", channel="3l_p_offZ_1b", systematic=variation,
                    appl=appl, njets=np.array([2.5]), weight=np.array([factor]),
                    eft_coeff=np.array([[1., 2., 3.]]), fill_sumw2=False)
    payload = {ns.scalar_nominal_key("njets"): scalar, ns.eft_nominal_key("njets"): eft}
    source = tmp_path / "source.pkl.gz"
    from analysis.topeft_run2.analysis_processor import AnalysisProcessor
    applicability = AnalysisProcessor.build_histogram_applicability(
        analysis_mode="default", runtime_families=("njets",),
        selected_category_dicts=({"fixture": {
            "jet_lst": ["exactly_2j"], "lep_flav_lst": [],
            "appl_lst": ["isSR_3l", "isAR_3l", "isAR_2lSS_OS"],
            "lep_chan_lst": [["3l_p_offZ_1b"]],
        }},), is_run3_values=(False,))
    write_histogram_artifact(source, histograms=payload, artifact_kind="processor_output",
        histogram_applicability=applicability,
        sumw2_storage="embedded" if store else "none", runtime_families=("njets",),
        production_sample_contract=sample_contract,
        requested_data_driven_products=requested, resolved_data_driven_contract=contract,
        no_clobber=True)
    return source, payload


def test_v3_source_dd_roundtrip_and_actual_moments(tmp_path):
    source, payload = source_fixture(tmp_path)
    source_record = validate_histogram_artifact(source)
    producer = DataDrivenProducer(str(source), str(tmp_path / "derived"))
    result = producer.getDataDrivenHistogram()
    scalar = result[ns.scalar_nominal_key("njets")]
    key = ("nonpromptUL18", "3l_p_offZ_1b", "nominal")
    assert scalar.eval({})[key].sum() == 0.
    assert ns.get_nominal_sumw2(scalar, flow=True)[key].sum() == 14.
    assert scalar.eval({})[("flipsUL18", key[1], "nominal")].sum() == 3.
    producer.dumpToPickle()
    record = validate_histogram_artifact(tmp_path / "derived.pkl.gz")
    assert record["metadata"]["artifact"]["artifact_kind"] == "nonprompt_output"
    assert record["metadata"]["lineage"]["inputs"][0]["pkl_sha256"] == source_record["metadata"]["artifact"]["pkl_sha256"]
    view = ns.materialize_legacy_histogram_dict(result, require_companions=("njets",))
    assert view["njets"].eval({})[key].sum() == 0.
    assert view["njets_sumw2"].eval({})[key].sum() == 14.
    signal = ("ttHJet_privateUL18", key[1], "nominal")
    assert view["njets"].eval({"ctG": 1.})[signal].sum() == 6.


def test_v3_dd_streaming_and_materialized_match(tmp_path):
    source, _ = source_fixture(tmp_path)
    materialized = DataDrivenProducer(str(source), "").getDataDrivenHistogram()
    streamed = dict(DataDrivenProducer(str(source), "", iterator_mode=True).iter_data_driven_histograms())
    assert set(materialized) == set(streamed)
    for name in materialized:
        for key, value in materialized[name].view(flow=True).items():
            np.testing.assert_array_equal(value, streamed[name].view(flow=True)[key])
        assert materialized[name]._sumw2_provenance == streamed[name]._sumw2_provenance


def test_v3_false_year_contract_rejected_before_output(tmp_path):
    source, payload = source_fixture(tmp_path)
    record = validate_histogram_artifact(source)["metadata"]
    bad = copy.deepcopy(record)
    bad["resolved_data_driven_contract"]["products"]["nonprompt"]["generated_outputs"]["nonpromptUL18"]["year"] = "UL17"
    with pytest.raises(ValueError, match="label/year mismatch"):
        validate_embedded_product_input(bad, artifact_kind="nonprompt_output")


def test_v3_none_yield_view_and_explicit_uncertainty_failure():
    h = HistEFT(hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0, 1, name="njets"), wc_names=[], use_multicell=True)
    h.fill(process="dataUL18", systematic="nominal", njets=np.array([.5]), weight=2.)
    payload = {ns.scalar_nominal_key("njets"): h}
    view = ns.materialize_legacy_histogram_dict(payload)
    assert view["njets"].eval({})[("dataUL18", "nominal")].sum() == 2.
    with pytest.raises(ns.Sumw2NotStoredError, match="recorded nominal sumw2"):
        ns.materialize_legacy_histogram_dict(payload, require_companions=("njets",))


@pytest.mark.parametrize("storage", [{"mode": "full_custom"}, {"mode": "selective"}, {"mode": "production", "rules": []}])
def test_v3_selective_options_rejected(storage):
    with pytest.raises(ValueError, match="global embedded/none"):
        resolve_embedded_sumw2_mode(storage, sumw2_storage_present=True)


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("flips_only", [False, True])
def test_v3_actual_dd_entrypoint(tmp_path, materialized, flips_only):
    from analysis.topeft_run2 import run_data_driven
    source, _ = source_fixture(tmp_path)
    output = tmp_path / "cli_dd.pkl.gz"
    args = ["--input-pkl", str(source), "--output-pkl", str(output), "--quiet"]
    if materialized:
        args.append("--legacy-dict-mode")
    if flips_only:
        args.append("--only-flips")
    assert run_data_driven.main(args) == 0
    validation = validate_histogram_artifact(output)
    assert validation["metadata"]["artifact"]["artifact_kind"] == ("flips_output" if flips_only else "nonprompt_output")


def test_v3_actual_yield_consumer(tmp_path):
    from topeft.modules.yield_tools import YieldTools
    source, _ = source_fixture(tmp_path)
    output = DataDrivenProducer(str(source), "").getDataDrivenHistogram()
    value, error = YieldTools().get_normalized_yield(output, "nonpromptUL18",
        {"channel": "3l_p_offZ_1b"}, h_name="njets")
    assert value == 0.
    assert error == pytest.approx(np.sqrt(14.))


def test_v3_actual_make_cards_output(tmp_path, monkeypatch):
    import sys
    import uproot
    from analysis.topeft_run2 import make_cards
    source, _ = source_fixture(tmp_path)
    dd = DataDrivenProducer(str(source), str(tmp_path / "card_input"))
    dd.dumpToPickle()
    out = tmp_path / "cards"
    monkeypatch.setattr(sys, "argv", ["make_cards.py", str(tmp_path / "card_input.pkl.gz"),
        "--out-dir", str(out), "--var-lst", "njets", "--ch-lst", "3l_p_offZ_1b",
        "--year", "UL18", "--binning", "processing", "--skip-selected-wcs-check", "--use-AAC"])
    make_cards.main()
    roots = list(out.glob("*.root"))
    assert roots and list(out.glob("*.txt"))
    with uproot.open(roots[0]) as templates:
        assert "data_obs" in templates
        assert np.isfinite(templates["data_obs"].values()).all()
    assert json.loads((out / "scalings-preselect.json").read_text())


@pytest.mark.parametrize("uncertainty", ["stat", "total"])
def test_v3_actual_plot_renderer_output(tmp_path, uncertainty):
    from analysis.topeft_run2 import make_cr_and_sr_plots as plotter
    source, _ = source_fixture(tmp_path, diboson=True)
    producer = DataDrivenProducer(str(source), str(tmp_path / "plot_input"))
    producer.dumpToPickle()
    parser = plotter.build_arg_parser()
    args = parser.parse_args(["-f", str(tmp_path / "plot_input.pkl.gz"), "-o", str(tmp_path / "plots"),
        "-y", "2018", "--sr", "--variables", "njets", "--workers", "1"] + (["--skip-syst"] if uncertainty == "stat" else []))
    plotter.run_with_args(args, parser)
    figures = list((tmp_path / "plots").rglob("*.png"))
    assert figures and all(p.stat().st_size > 1000 for p in figures)


@pytest.mark.parametrize("materialized", [False, True])
def test_v3_two_dimensional_dd_recording_and_scaling(tmp_path, materialized):
    from topcoffea.modules.sparseHist import SparseHist
    from analysis.topeft_run2 import run_data_driven
    source, _ = source_fixture(tmp_path)
    meta = validate_histogram_artifact(source)["metadata"]
    family = "lepton_pt_vs_eta"
    h = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(2, 0, 2, name="lepton_pt_vs_eta_pt"),
        hist.axis.Regular(2, 0, 2, name="lepton_pt_vs_eta_abseta"), storage="Weight")
    for appl in ("isSR_3l", "isAR_3l", "isAR_2lSS_OS"):
        for proc, weight in (("dataUL18", 3.), ("TTTo2L2Nu_centralUL18", 2.), ("ttHJet_privateUL18", 1.)):
            for systematic in ("nominal", "FFUp", "FFDown"):
                h.fill(process=proc, channel="3l_p_offZ_1b", systematic=systematic,
                    appl=appl, lepton_pt_vs_eta_pt=np.array([.5]),
                    lepton_pt_vs_eta_abseta=np.array([.5]), weight=np.array([weight]),
                    _record_sumw2=systematic == "nominal")
    applicability = copy.deepcopy(meta["histogram_applicability"])
    applicability["families"] = {family: applicability["families"]["njets"]}
    source2 = tmp_path / "source2d.pkl.gz"
    write_histogram_artifact(source2, histograms={family: h}, artifact_kind="processor_output",
        sumw2_storage="embedded", runtime_families=(family,),
        histogram_applicability=applicability,
        production_sample_contract=meta["production_sample_contract"],
        requested_data_driven_products=meta["requested_data_driven_products"],
        resolved_data_driven_contract=meta["resolved_data_driven_contract"])
    output = tmp_path / "dd2d.pkl.gz"
    args = ["--input-pkl", str(source2), "--output-pkl", str(output), "--quiet"]
    if materialized:
        args.append("--legacy-dict-mode")
    assert run_data_driven.main(args) == 0
    validate_histogram_artifact(output)
    with gzip.open(output, "rb") as stream:
        result = cloudpickle.load(stream)[family]
    key = ("nonpromptUL18", "3l_p_offZ_1b", "nominal")
    assert result.view(flow=True)[key].value.sum() == 0.
    assert ns.get_nominal_sumw2(result, flow=True)[key].sum() == 14.
    scaled = copy.deepcopy(result)
    scaled.scale(-2.)
    assert ns.get_nominal_sumw2(scaled, flow=True)[key].sum() == 56.
    assert all(coordinate[-1] == "nominal" for coordinate in ns.get_nominal_sumw2(result))
    varied = result[{"systematic": "FFUp"}]
    assert ns.get_nominal_sumw2(varied) == {}


def test_v3_homogeneous_merge_then_dd(tmp_path):
    from topeft.modules.datacard_tools import load_and_merge_histogram_pkls
    source, payload = source_fixture(tmp_path)
    meta = validate_histogram_artifact(source)["metadata"]
    fragments = []
    for number, keep in enumerate(({"dataUL18"}, {"TTTo2L2Nu_centralUL18", "ttHJet_privateUL18"})):
        fragment = {}
        for key, histogram in payload.items():
            processes = [p for p in histogram.axes["process"] if p in keep]
            if processes:
                fragment[key] = histogram.prune("process", processes)
        path = tmp_path / f"fragment_{number}.pkl.gz"
        write_histogram_artifact(path, histograms=fragment, artifact_kind="processor_output",
            sumw2_storage="embedded", runtime_families=("njets",),
            histogram_applicability=meta["histogram_applicability"],
            production_sample_contract=meta["production_sample_contract"],
            requested_data_driven_products=meta["requested_data_driven_products"],
            resolved_data_driven_contract=meta["resolved_data_driven_contract"])
        fragments.append(str(path))
    merged, report = load_and_merge_histogram_pkls(fragments, require_sumw2=True)
    assert report["schema"] == ns.EMBEDDED_SUMW2_LAYOUT
    assert len(report["lineage_inputs"]) == 2
    combined = tmp_path / "merged.pkl.gz"
    options = {k: v for k, v in report.items() if k in {
        "artifact_kind", "sumw2_storage", "runtime_families", "lineage_inputs",
        "production_sample_contract", "requested_data_driven_products",
        "resolved_data_driven_contract", "histogram_applicability", "merged"}}
    write_histogram_artifact(combined, histograms=merged, **options)
    producer = DataDrivenProducer(str(combined), str(tmp_path / "merged_dd"))
    producer.dumpToPickle()
    validate_histogram_artifact(tmp_path / "merged_dd.pkl.gz")
    with pytest.raises(histogram_artifact_error, match="Duplicate histogram contribution"):
        load_and_merge_histogram_pkls([fragments[0], fragments[0]], require_sumw2=True)


def test_v3_unrecorded_source_rejected_by_dd(tmp_path):
    source, payload = source_fixture(tmp_path)
    meta = validate_histogram_artifact(source)["metadata"]
    scalar = payload[ns.scalar_nominal_key("njets")]
    for index in scalar._dense_hists:
        recorded, origin = scalar.sumw2_recording_state(index)
        scalar._sumw2_provenance[index] = (False, origin)
    path = tmp_path / "unrecorded.pkl.gz"
    write_histogram_artifact(path, histograms=payload, artifact_kind="processor_output",
        sumw2_storage="embedded", runtime_families=("njets",),
        production_sample_contract=meta["production_sample_contract"],
        requested_data_driven_products=meta["requested_data_driven_products"],
        resolved_data_driven_contract=meta["resolved_data_driven_contract"])
    with pytest.raises(ValueError, match="recorded source sumw2"):
        DataDrivenProducer(str(path), str(tmp_path / "forbidden"))
    assert not (tmp_path / "forbidden.pkl.gz").exists()


def test_v3_recorded_2d_invalid_q_rejected_before_publication(tmp_path):
    from topcoffea.modules.sparseHist import SparseHist
    h = SparseHist(hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0, 1, name="x"), hist.axis.Regular(1, 0, 1, name="y"), storage="Weight")
    h.fill(process="dataUL18", systematic="nominal", x=.5, y=.5, weight=1.)
    h.view(flow=True)[("dataUL18", "nominal")].variance[1, 1] = -1.
    target = tmp_path / "invalid.pkl.gz"
    with pytest.raises(histogram_artifact_error, match="Invalid recorded v3 statistical moment"):
        write_histogram_artifact(target, histograms={"lepton_pt_vs_eta": h},
            artifact_kind="processor_output", sumw2_storage="embedded",
            runtime_families=("lepton_pt_vs_eta",))
    assert not target.exists()


def test_v3_yield_only_keeps_nonzero_wc_coefficients(tmp_path):
    from topeft.modules.yield_tools import YieldTools
    source, payload = source_fixture(tmp_path)
    result = YieldTools().get_normalized_yield(payload, "ttHJet_privateUL18",
        {"channel": "3l_p_offZ_1b", "appl": "isSR_3l"}, rwgt_pt={"ctG": 1.},
        h_name="njets", include_uncertainty=False)
    assert result == [6., None]
    with pytest.raises(ValueError, match="only at the SM point"):
        YieldTools().get_normalized_yield(payload, "ttHJet_privateUL18",
            {"channel": "3l_p_offZ_1b", "appl": "isSR_3l"}, rwgt_pt={"ctG": 1.}, h_name="njets")


def test_v3_card_nuisances_and_eft_decomposition(tmp_path, monkeypatch):
    import sys
    import uproot
    from analysis.topeft_run2 import make_cards
    source, _ = source_fixture(tmp_path, systematics=True)
    dd = DataDrivenProducer(str(source), str(tmp_path / "nuisance_input"))
    dd.dumpToPickle()
    out = tmp_path / "cards"
    monkeypatch.setattr(sys, "argv", ["make_cards.py", str(tmp_path / "nuisance_input.pkl.gz"),
        "--out-dir", str(out), "--var-lst", "njets", "--ch-lst", "3l_p_offZ_1b",
        "--year", "UL18", "--binning", "processing", "--skip-selected-wcs-check", "--use-AAC",
        "--do-nuisance", "--skip-missing-parton-rate-syst"])
    make_cards.main()
    with uproot.open(next(out.glob("*.root"))) as templates:
        for name, expected in (("ttH_sm", 1.), ("ttH_lin_ctG", 6.), ("ttH_quad_ctG", 3.),
                ("ttH_sm_renorm_ttHUp", 1.1), ("ttH_sm_renorm_ttHDown", .9),
                ("ttH_sm_fact_ttHUp", 1.2), ("ttH_sm_fact_ttHDown", .8)):
            assert float(templates[name].values().sum()) == pytest.approx(expected)
        assert not any("renormfact" in name for name in templates.keys())
    card = next(out.glob("ttx_multileptons-*.txt")).read_text()
    assert "renorm" in card and "fact" in card and "renormfact" not in card
