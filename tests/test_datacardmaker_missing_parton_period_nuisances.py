from __future__ import annotations

import copy

import numpy as np
import pytest
import hist

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

from analysis.topeft_run2 import eft_event_diagnostics as origin_reference
from topeft.modules import datacard_tools
from topeft.modules.axis_binning import resolve_axis_edges
from topeft.modules.datacard_tools import DatacardMaker
from topeft.modules.missing_parton_contract import DEFAULT_SR_REGISTRY


RUN2_DEFAULT_PAYLOAD = "data/missing_parton/missing_parton_run2.root"
RUN3_DEFAULT_PAYLOAD = "data/missing_parton/missing_parton_run3.root"


def _systematics_loader(
    *,
    years,
    do_nuisance=True,
    skip=False,
):
    maker = object.__new__(DatacardMaker)
    maker.do_nuisance = do_nuisance
    maker.skip_missing_parton_rate_syst = skip
    maker.year_lst = list(years)
    return maker


def _mock_payload_validator(monkeypatch, values_by_path):
    calls = []

    def validate(path, *, sr_registry):
        path_name = str(path).rsplit("/", 1)[-1]
        calls.append((str(path), sr_registry))
        values = values_by_path[path_name]
        return {"3l_onZ_1b": np.asarray(values, dtype=float)}

    monkeypatch.setattr(
        datacard_tools,
        "validate_legacy_missing_parton_payload",
        validate,
    )
    monkeypatch.setattr(
        datacard_tools, "infer_legacy_missing_parton_layout",
        lambda _path: type("layout", (), {"registry": DEFAULT_SR_REGISTRY})(),
    )
    return calls


@pytest.mark.parametrize("year", ("UL16", "UL16APV", "UL17", "UL18"))
def test_all_supported_run2_card_years_resolve_to_one_nuisance_and_default_payload(year):
    assert DatacardMaker.missing_parton_run_era(year) == "run2"
    assert DatacardMaker.missing_parton_nuisance_name(year) == "missing_parton"
    assert DatacardMaker.resolve_missing_parton_payload_path((year,)) == RUN2_DEFAULT_PAYLOAD


@pytest.mark.parametrize("year", ("2022", "2022EE", "2023", "2023BPix"))
def test_all_supported_run3_card_years_resolve_to_one_nuisance_and_default_payload(year):
    assert DatacardMaker.missing_parton_run_era(year) == "run3"
    assert DatacardMaker.missing_parton_nuisance_name(year) == "missing_parton"
    assert DatacardMaker.resolve_missing_parton_payload_path((year,)) == RUN3_DEFAULT_PAYLOAD


@pytest.mark.parametrize("year", ("", "2018", "UL18_extra"))
def test_malformed_or_unsupported_years_are_rejected(year):
    with pytest.raises(ValueError, match="canonical year or period|Unsupported canonical"):
        DatacardMaker.missing_parton_nuisance_name(year)


@pytest.mark.parametrize("year", ("UL18", "2023"))
@pytest.mark.parametrize(
    "payload_path",
    ("custom/root_payload.root", "data/missing_parton/missing_parton.root"),
)
def test_explicit_payload_path_is_preserved_without_filename_inference(year, payload_path):
    assert (
        DatacardMaker.resolve_missing_parton_payload_path((year,), payload_path)
        == payload_path
    )


def test_empty_explicit_payload_path_is_rejected():
    with pytest.raises(ValueError, match="must be non-empty"):
        DatacardMaker.resolve_missing_parton_payload_path(("UL18",), "")


def test_direct_constructor_needs_no_registry(monkeypatch):
    monkeypatch.setattr(DatacardMaker, "load_systematics", lambda *args: {})

    maker = DatacardMaker(hists={}, do_nuisance=False, verbose=False)

    assert not hasattr(maker, "sr_registry")


def test_explicit_payload_preserves_override():
    arbitrary_path = "arbitrary/not-inferred-from-name.root"
    assert DatacardMaker.resolve_missing_parton_payload_path(
        ("UL18",),
        arbitrary_path,
    ) == arbitrary_path


def test_constructor_preserves_exact_explicit_payload(monkeypatch):
    captured_paths = []

    def capture_systematics(self, rate_syst_path, missing_parton_path):
        captured_paths.append(missing_parton_path)
        return {}

    monkeypatch.setattr(DatacardMaker, "load_systematics", capture_systematics)
    maker = DatacardMaker(
        hists={},
        do_nuisance=True,
        year_lst=["UL18"],
        missing_parton_path="names/do-not-select-the-registry.root",
        verbose=False,
    )

    assert not hasattr(maker, "sr_registry")
    assert maker.missing_parton_payload_path == (
        "names/do-not-select-the-registry.root"
    )
    assert captured_paths == ["names/do-not-select-the-registry.root"]


def test_mixed_era_years_fail_with_original_labels_and_resolved_eras():
    with pytest.raises(ValueError, match="one explicit missing-parton payload source") as exc_info:
        DatacardMaker.missing_parton_nuisance_name_for_years(
            ("UL18", "2022"),
            payload_path="run2.root",
        )

    message = str(exc_info.value)
    assert "UL18" in message
    assert "2022" in message
    assert "run2" in message
    assert "run3" in message
    assert "run2.root" in message
    assert "different nuisance names" not in message.lower()

    with pytest.raises(ValueError, match="one explicit missing-parton payload source"):
        DatacardMaker.resolve_missing_parton_payload_path(("UL18", "2022"))
    with pytest.raises(ValueError, match="one explicit missing-parton payload source"):
        DatacardMaker.resolve_missing_parton_payload_path(
            ("UL18", "2022"),
            "data/missing_parton/missing_parton.root",
        )


def test_mixed_era_loader_fails_before_opening_the_payload(monkeypatch):
    maker = _systematics_loader(years=("UL18", "2022"))

    def fail_if_opened(_):
        raise AssertionError("mixed-era resolution should happen before payload loading")

    monkeypatch.setattr(
        datacard_tools,
        "validate_legacy_missing_parton_payload",
        fail_if_opened,
    )

    with pytest.raises(ValueError, match="one explicit missing-parton payload source"):
        maker.load_systematics("params/rate_systs_run3.json", "synthetic.root")


@pytest.mark.parametrize(
    ("year", "expected_payload"),
    (("UL18", RUN2_DEFAULT_PAYLOAD), ("2023", RUN3_DEFAULT_PAYLOAD)),
)
def test_direct_constructor_uses_the_same_era_default_payload(
    monkeypatch,
    year,
    expected_payload,
):
    captured_payload_paths = []

    def capture_systematics(self, rate_syst_path, missing_parton_path):
        captured_payload_paths.append(missing_parton_path)
        return {}

    monkeypatch.setattr(DatacardMaker, "load_systematics", capture_systematics)
    maker = DatacardMaker(
        hists={},
        do_nuisance=True,
        year_lst=[year],
        verbose=False,
    )

    assert maker.missing_parton_payload_path == expected_payload
    assert captured_payload_paths == [expected_payload]


@pytest.mark.parametrize(
    ("year", "expected_payload"),
    (("UL18", RUN2_DEFAULT_PAYLOAD), ("2023", RUN3_DEFAULT_PAYLOAD)),
)
def test_default_payload_path_is_validated_for_the_resolved_era(
    monkeypatch,
    year,
    expected_payload,
):
    validation_calls = _mock_payload_validator(
        monkeypatch,
        {expected_payload.rsplit("/", 1)[-1]: (0.2, 0.3)},
    )
    maker = _systematics_loader(years=(year,))
    resolved_payload = DatacardMaker.resolve_missing_parton_payload_path((year,))

    systematics = maker.load_systematics("params/rate_systs_run3.json", resolved_payload)

    assert "missing_parton" in systematics
    assert resolved_payload == expected_payload
    assert validation_calls[0][0].endswith(expected_payload)
    assert validation_calls[0][1] == DEFAULT_SR_REGISTRY


@pytest.mark.parametrize(
    ("year", "payload_path"),
    (("UL18", RUN2_DEFAULT_PAYLOAD), ("2023", RUN3_DEFAULT_PAYLOAD)),
)
def test_consumer_boundary_accepts_packaged_current_payload(year, payload_path):
    maker = _systematics_loader(years=(year,))

    systematics = maker.load_systematics(
        "params/rate_systs_run3.json",
        payload_path,
    )

    assert systematics["missing_parton"].get_process("tllq")


@pytest.mark.parametrize("do_nuisance, skip", ((False, False), (True, True)))
def test_constructor_skip_and_disabled_modes_bypass_default_resolution(
    monkeypatch,
    do_nuisance,
    skip,
):
    def fail_if_resolved(*args, **kwargs):
        raise AssertionError("missing-parton default was resolved despite suppression")

    monkeypatch.setattr(
        DatacardMaker,
        "resolve_missing_parton_payload_path",
        fail_if_resolved,
    )
    monkeypatch.setattr(DatacardMaker, "load_systematics", lambda *args: {})

    maker = DatacardMaker(
        hists={},
        do_nuisance=do_nuisance,
        skip_missing_parton_rate_syst=skip,
        year_lst=["UL18", "2022"],
        verbose=False,
    )

    assert maker.missing_parton_payload_path is None


@pytest.mark.parametrize(
    ("years", "payload_values"),
    (
        (("UL16", "UL18"), (0.2, 0.3)),
        (("2022", "2023BPix"), (0.4, 0.5)),
    ),
)
def test_loader_uses_era_specific_name_and_preserves_process_scope(
    monkeypatch,
    years,
    payload_values,
):
    _mock_payload_validator(monkeypatch, {"synthetic.root": payload_values})
    maker = _systematics_loader(years=years)

    systematics = maker.load_systematics(
        "params/rate_systs_run3.json",
        "synthetic.root",
    )

    assert set(systematics).isdisjoint({"missing_parton_run2", "missing_parton_run3"})
    missing_parton = systematics["missing_parton"]
    assert missing_parton.name == "missing_parton"
    assert missing_parton.get_process("tllq") == {
        "3l_onZ_1b": pytest.approx(np.asarray(payload_values) + 1.0)
    }
    assert missing_parton.get_process("tHq") == missing_parton.get_process("tllq")
    for excluded_process in ("tZq", "ttll", "ttH", "unrelated"):
        assert missing_parton.get_process(excluded_process) == "-"


def test_shared_identity_keeps_explicit_payload_factors_distinct(monkeypatch):
    payload_values_by_path = {
        "run2.root": (0.2, 0.3),
        "run3.root": (0.4, 0.5),
    }
    _mock_payload_validator(monkeypatch, payload_values_by_path)

    run2 = _systematics_loader(years=("UL18",)).load_systematics(
        "params/rate_systs_run2.json",
        "run2.root",
    )["missing_parton"]
    run3 = _systematics_loader(years=("2023",)).load_systematics(
        "params/rate_systs_run3.json",
        "run3.root",
    )["missing_parton"]

    assert run2.name == run3.name == "missing_parton"
    assert run2.get_process("tllq")["3l_onZ_1b"][0] == pytest.approx(1.2)
    assert run3.get_process("tllq")["3l_onZ_1b"][0] == pytest.approx(1.4)
    assert run2.get_process("tHq") == run2.get_process("tllq")
    assert run3.get_process("tHq") == run3.get_process("tllq")


def test_skip_bypasses_payload_loading_and_mixed_era_resolution(monkeypatch):
    maker = _systematics_loader(years=("UL18", "2022"), skip=True)

    def fail_if_opened(_):
        raise AssertionError("missing-parton payload was opened despite suppression")

    monkeypatch.setattr(
        datacard_tools,
        "validate_legacy_missing_parton_payload",
        fail_if_opened,
    )
    systematics = maker.load_systematics(
        "params/rate_systs_run3.json",
        "does-not-exist.root",
    )

    assert "diboson_njets" in systematics
    assert "missing_parton" not in systematics


def test_disabled_nuisances_bypass_payload_loading_and_era_resolution(monkeypatch):
    maker = _systematics_loader(years=("UL18", "2022"), do_nuisance=False)

    def fail_if_opened(_):
        raise AssertionError("missing-parton payload was opened while nuisances were disabled")

    monkeypatch.setattr(
        datacard_tools,
        "validate_legacy_missing_parton_payload",
        fail_if_opened,
    )

    assert maker.load_systematics("params/rate_systs_run3.json", "does-not-exist.root") == {}


def test_consumer_boundary_uses_inferred_layout(monkeypatch):
    calls = _mock_payload_validator(monkeypatch, {"selected.root": (0.2, 0.3)})
    maker = _systematics_loader(years=("UL18",))

    systematics = maker.load_systematics(
        "params/rate_systs_run2.json",
        "selected.root",
    )

    assert calls[0][1] == DEFAULT_SR_REGISTRY
    assert systematics["missing_parton"].get_process("tllq") == {
        "3l_onZ_1b": pytest.approx(np.asarray((1.2, 1.3)))
    }


def test_consumer_boundary_rejects_obsolete_payload_without_fallback(monkeypatch):
    def reject_obsolete(path, *, sr_registry):
        raise ValueError(
            f"obsolete missing-parton schema for {sr_registry!r}: {path}"
        )

    monkeypatch.setattr(
        datacard_tools,
        "validate_legacy_missing_parton_payload",
        reject_obsolete,
    )
    monkeypatch.setattr(
        datacard_tools, "infer_legacy_missing_parton_layout",
        lambda _path: type("layout", (), {"registry": DEFAULT_SR_REGISTRY})(),
    )
    maker = _systematics_loader(years=("UL18",))

    with pytest.raises(ValueError, match="obsolete missing-parton schema"):
        maker.load_systematics("params/rate_systs_run2.json", "legacy.root")


@pytest.mark.parametrize("year", ("UL18", "2023"))
def test_writer_reuses_only_target_fivejet_missing_parton_response(
    monkeypatch, tmp_path, year
):
    channels = (
        "2los_onZ_1tau_5j",
        "2los_onZ_1tau_3j",
        "3l_onZ_1b_2j",
    )
    axes = (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Variable(
            resolve_axis_edges("ptz", mode="fitting", channel=channels[0]),
            name="ptz",
        ),
    )
    scalar_source = SparseHist(*axes)
    eft_source = HistEFT(
        *axes,
        wc_names=[],
        label="Events",
    )
    nominal_rates = dict(zip(channels, (5.0, 3.0, 2.0)))
    for channel in channels:
        for process in ("tllq", "tHq", "ttll"):
            source = scalar_source if process == "ttll" else eft_source
            source.fill(
                process=f"{process}{year}",
                channel=channel,
                systematic="nominal",
                ptz=np.asarray([0.5]),
                weight=np.asarray([nominal_rates[channel]]),
            )
    components = {
        "ptz__scalar_nominal": scalar_source,
        "ptz__eft_nominal": eft_source,
    }
    # The qualified reference keeps the combined process/category universe
    # through the native transformations, including each absent origin cell.
    bundle = origin_reference.combined_producer_views(
        copy.deepcopy(components), "ptz", [year], list(channels),
    )

    def validate(_path, *, sr_registry):
        assert sr_registry == DEFAULT_SR_REGISTRY
        # The target has no index 5; index 3 and unrelated index 2 differ.
        return {
            "2los_onZ_1tau": np.asarray((0.0, 0.0, 0.0, 0.2)),
            "3l_onZ_1b": np.asarray((0.0, 0.0, 0.4, 0.8)),
        }

    monkeypatch.setattr(datacard_tools, "validate_legacy_missing_parton_payload", validate)
    monkeypatch.setattr(
        datacard_tools, "infer_legacy_missing_parton_layout",
        lambda _path: type("layout", (), {"registry": DEFAULT_SR_REGISTRY})(),
    )
    maker = DatacardMaker(
        hists=copy.deepcopy(components),
        out_dir=str(tmp_path),
        binning_mode="processing",
        do_nuisance=False,
        year_lst=[year],
        verbose=False,
    )
    assert maker.hists["ptz"].wc_names == []
    assert set(maker.nominal_origins["ptz"]) == {"scalar", "eft"}
    assert "nominal_origin" not in maker.hists["ptz"].axes.name
    for channel in channels:
        for process in ("tllq", "tHq", "ttll"):
            reference = origin_reference.combined_bin_coefficients(
                bundle, channel, process,
            )
            actual = maker.nominal_origin_coefficients("ptz", channel, process)
            for origin in ("combined", "scalar", "eft"):
                # The reference excludes physical underflow; both retain overflow.
                np.testing.assert_allclose(actual[origin][1:], reference[origin])
                assert actual[origin].shape[-1] == 1  # constant only
            np.testing.assert_allclose(
                actual["scalar"] + actual["eft"], actual["combined"],
            )
            assert actual["combined"][1, 0] == nominal_rates[channel]
            scalar_rate = nominal_rates[channel] if process == "ttll" else 0.0
            assert actual["scalar"][1, 0] == scalar_rate
            assert actual["eft"][1, 0] == nominal_rates[channel] - scalar_rate
    # Use the maintained loader so process entries and nuisance identity are
    # source-derived; only its response payload is synthetic.
    maker.do_nuisance = True
    missing_parton = maker.load_systematics(
        "params/rate_systs_run2.json" if year == "UL18" else "params/rate_systs_run3.json",
        "synthetic.root",
    )["missing_parton"]
    maker.rate_systs = {"missing_parton": missing_parton}

    assert missing_parton.name == "missing_parton"
    assert set(missing_parton.corrs) == {"tllq", "tHq"}
    assert missing_parton.get_process("ttll") == "-"
    assert maker.missing_parton_nuisance_name(year) == "missing_parton"

    for channel, expected in zip(channels, ("0.800000/1.200000", "0.800000/1.200000", "0.600000/1.400000")):
        maker.analyze("ptz", channel, {process: [] for process in ("tllq", "tHq", "ttll")}, True, {})
        card = (tmp_path / f"ttx_multileptons-{channel}_ptz.txt").read_text(
            encoding="utf-8"
        )
        lines = [line.split() for line in card.splitlines()]
        process_row = next(
            fields for fields in lines
            if fields[0] == "process" and "tllq_sm" in fields
        )
        nuisance_row = next(fields for fields in lines if fields[:2] == ["missing_parton", "lnN"])
        values = dict(zip(process_row[1:], nuisance_row[2:]))
        assert values == {"tllq_sm": expected, "tHq_sm": expected, "ttll_sm": "-"}
        rate_row = next(fields for fields in lines if fields[0] == "rate")
        assert dict(zip(process_row[1:], map(float, rate_row[1:]))) == {
            f"{process}_sm": nominal_rates[channel]
            for process in ("tllq", "tHq", "ttll")
        }
        assert sum(fields[:2] == ["missing_parton", "lnN"] for fields in lines) == 1
