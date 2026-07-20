from __future__ import annotations

import numpy as np
import pytest

from topeft.modules import datacard_tools
from topeft.modules.datacard_tools import DatacardMaker
from topeft.modules.missing_parton_contract import (
    DEFAULT_SR_REGISTRY,
    SUPPORTED_SR_REGISTRIES,
)


RUN2_DEFAULT_PAYLOAD = "data/missing_parton/missing_parton_run2.root"
RUN3_DEFAULT_PAYLOAD = "data/missing_parton/missing_parton_run3.root"


class _fake_branch:
    def __init__(self, values):
        self.values = values

    def array(self):
        return np.asarray(self.values, dtype=float)


class _fake_missing_parton_file:
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return ["3l_onZ_1b"]

    def __getitem__(self, key):
        assert key == "3l_onZ_1b/tllq"
        return _fake_branch(self.values)


def _systematics_loader(*, years, do_nuisance=True, skip=False):
    maker = object.__new__(DatacardMaker)
    maker.do_nuisance = do_nuisance
    maker.skip_missing_parton_rate_syst = skip
    maker.year_lst = list(years)
    return maker


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


@pytest.mark.parametrize("sr_registry", SUPPORTED_SR_REGISTRIES)
def test_direct_constructor_accepts_and_stores_canonical_registry(
    monkeypatch,
    sr_registry,
):
    monkeypatch.setattr(DatacardMaker, "load_systematics", lambda *args: {})

    maker = DatacardMaker(
        hists={},
        do_nuisance=False,
        sr_registry=sr_registry,
        verbose=False,
    )

    assert maker.sr_registry == sr_registry
    assert maker.missing_parton_payload_path is None


def test_direct_constructor_defaults_to_all_registry(monkeypatch):
    monkeypatch.setattr(DatacardMaker, "load_systematics", lambda *args: {})

    maker = DatacardMaker(hists={}, do_nuisance=False, verbose=False)

    assert maker.sr_registry == DEFAULT_SR_REGISTRY


def test_direct_constructor_rejects_unknown_registry(monkeypatch):
    monkeypatch.setattr(DatacardMaker, "load_systematics", lambda *args: {})

    with pytest.raises(ValueError, match="Unsupported SR registry"):
        DatacardMaker(
            hists={},
            do_nuisance=False,
            sr_registry="UNKNOWN_SR",
            verbose=False,
        )


def test_direct_constructor_validates_registry_against_channel_config(monkeypatch):
    observed = []

    def validate(sr_registry):
        observed.append(sr_registry)
        return sr_registry, {"selected": True}

    monkeypatch.setattr(datacard_tools, "load_or_validate_selected_registry", validate)
    monkeypatch.setattr(DatacardMaker, "load_systematics", lambda *args: {})

    maker = DatacardMaker(
        hists={},
        do_nuisance=False,
        sr_registry="TAU_CH_LST_SR",
        verbose=False,
    )

    assert maker.sr_registry == "TAU_CH_LST_SR"
    assert observed == ["TAU_CH_LST_SR"]


def test_nondefault_registry_requires_explicit_payload_and_preserves_override():
    with pytest.raises(ValueError, match="no canonical implicit"):
        DatacardMaker.resolve_missing_parton_payload_path(
            ("UL18",),
            sr_registry="FWD_CH_LST_SR",
        )

    arbitrary_path = "arbitrary/not-inferred-from-name.root"
    assert DatacardMaker.resolve_missing_parton_payload_path(
        ("UL18",),
        arbitrary_path,
        sr_registry="FWD_CH_LST_SR",
    ) == arbitrary_path


def test_nondefault_constructor_preserves_exact_explicit_payload(monkeypatch):
    captured_paths = []

    def capture_systematics(self, rate_syst_path, missing_parton_path):
        captured_paths.append(missing_parton_path)
        return {}

    monkeypatch.setattr(DatacardMaker, "load_systematics", capture_systematics)
    maker = DatacardMaker(
        hists={},
        do_nuisance=True,
        year_lst=["UL18"],
        sr_registry="FWD_CH_LST_SR",
        missing_parton_path="names/do-not-select-the-registry.root",
        verbose=False,
    )

    assert maker.sr_registry == "FWD_CH_LST_SR"
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

    monkeypatch.setattr(datacard_tools.uproot, "open", fail_if_opened)

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
def test_default_payload_path_is_opened_for_the_resolved_era(
    monkeypatch,
    year,
    expected_payload,
):
    opened_paths = []

    def open_payload(path):
        opened_paths.append(str(path))
        return _fake_missing_parton_file((0.2, 0.3))

    monkeypatch.setattr(datacard_tools.uproot, "open", open_payload)
    maker = _systematics_loader(years=(year,))
    resolved_payload = DatacardMaker.resolve_missing_parton_payload_path((year,))

    systematics = maker.load_systematics("params/rate_systs_run3.json", resolved_payload)

    assert "missing_parton" in systematics
    assert resolved_payload == expected_payload
    assert opened_paths[0].endswith(expected_payload)


@pytest.mark.parametrize("do_nuisance, skip", ((False, False), (True, True)))
@pytest.mark.parametrize("sr_registry", (DEFAULT_SR_REGISTRY, "FWD_CH_LST_SR"))
def test_constructor_skip_and_disabled_modes_bypass_default_resolution(
    monkeypatch,
    do_nuisance,
    skip,
    sr_registry,
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
        sr_registry=sr_registry,
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
    monkeypatch.setattr(
        datacard_tools.uproot,
        "open",
        lambda _: _fake_missing_parton_file(payload_values),
    )
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
    monkeypatch.setattr(
        datacard_tools.uproot,
        "open",
        lambda path: _fake_missing_parton_file(
            payload_values_by_path[str(path).rsplit("/", 1)[-1]]
        ),
    )

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

    monkeypatch.setattr(datacard_tools.uproot, "open", fail_if_opened)
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

    monkeypatch.setattr(datacard_tools.uproot, "open", fail_if_opened)

    assert maker.load_systematics("params/rate_systs_run3.json", "does-not-exist.root") == {}
