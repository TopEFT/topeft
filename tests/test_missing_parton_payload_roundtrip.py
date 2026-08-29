from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

hist = pytest.importorskip("hist")

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.datacard_tools import DatacardMaker, RateSystematic
from topeft.modules.missing_parton_contract import (
    DEFAULT_SR_REGISTRY,
    LEGACY_MISSING_PARTON_BASE_CHANNELS,
    legacy_missing_parton_payload_lengths,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "missing_parton.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "missing_parton_payload_roundtrip_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_payload():
    lengths = legacy_missing_parton_payload_lengths()
    payload = {
        category: np.zeros(lengths[category], dtype=np.float64)
        for category in LEGACY_MISSING_PARTON_BASE_CHANNELS
    }
    payload["3l_onZ_1b"][2] = 0.2
    return payload


def make_card_hist(axis_name, channel):
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=axis_name),
        wc_names=[],
        label="Events",
    )
    histogram.metadata = {}
    for process in ("tllqUL18", "tHqUL18", "tZqUL18", "ttHUL18"):
        histogram.fill(
            process=process,
            channel=channel,
            systematic="nominal",
            **{axis_name: np.asarray([0.5])},
            weight=np.asarray([10.0]),
        )
    return histogram


def load_missing_parton_systematic(payload_path, year_lst=("UL18",)):
    loader = object.__new__(DatacardMaker)
    loader.do_nuisance = True
    loader.skip_missing_parton_rate_syst = False
    loader.year_lst = list(year_lst)
    loader.sr_registry = DEFAULT_SR_REGISTRY
    return loader.load_systematics(
        "params/rate_systs_run3.json",
        str(payload_path),
    )[DatacardMaker.missing_parton_nuisance_name_for_years(year_lst)]


def test_writer_loader_and_datacardmaker_round_trip(tmp_path):
    module = load_module()
    payload_path = tmp_path / "missing_parton.root"
    module.write_legacy_payload_atomic(payload_path, synthetic_payload())
    missing_parton = load_missing_parton_systematic(payload_path)

    assert missing_parton.name == "missing_parton"
    assert missing_parton.get_process("tllq")["3l_onZ_1b"][2] == pytest.approx(
        1.2
    )
    assert missing_parton.get_process("tHq")["3l_onZ_1b"][2] == pytest.approx(
        1.2
    )
    assert missing_parton.get_process("tZq") == "-"

    channel = "3l_onZ_1b_2j"
    hists = {
        "ht": make_card_hist("ht", channel),
        "ht_sumw2": make_card_hist("ht_sumw2", channel),
    }
    maker = DatacardMaker(
        hists=hists,
        out_dir=str(tmp_path),
        var_lst=["ht"],
        do_nuisance=False,
        verbose=False,
    )
    maker.rate_systs = {"missing_parton": missing_parton}
    maker.analyze(
        "ht",
        channel,
        {"tllq": [], "tHq": [], "tZq": [], "ttH": []},
        True,
        {},
    )

    card_path = tmp_path / f"ttx_multileptons-{channel}_ht.txt"
    process_names = None
    nuisance_values = None
    for line in card_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["process"] and any(
            field.endswith("_sm") for field in fields[1:]
        ):
            process_names = fields[1:]
        if fields[:2] == ["missing_parton", "lnN"]:
            nuisance_values = fields[2:]

    assert dict(zip(process_names, nuisance_values)) == {
        "tllq_sm": "0.800000/1.200000",
        "tHq_sm": "0.800000/1.200000",
        "tZq_sm": "-",
        "ttH_sm": "-",
    }


def test_missing_base_category_entry_still_formats_as_dash(tmp_path):
    channel = "3l_onZ_1b_2j"
    hists = {
        "ht": make_card_hist("ht", channel),
        "ht_sumw2": make_card_hist("ht_sumw2", channel),
    }
    maker = DatacardMaker(
        hists=hists,
        out_dir=str(tmp_path),
        var_lst=["ht"],
        do_nuisance=False,
        verbose=False,
    )
    missing_parton = RateSystematic("missing_parton")
    missing_parton.add_process(
        "tllq",
        {"different_base_category": np.asarray([1.2])},
    )
    maker.rate_systs = {"missing_parton": missing_parton}

    maker.analyze(
        "ht",
        channel,
        {"tllq": []},
        True,
        {},
    )

    card_path = tmp_path / f"ttx_multileptons-{channel}_ht.txt"
    nuisance_line = next(
        line
        for line in card_path.read_text(encoding="utf-8").splitlines()
        if line.split()[:2] == ["missing_parton", "lnN"]
    )
    assert nuisance_line.split()[2:] == ["-"]


def test_root_cycle_suffix_is_not_exposed_to_consumer_lookup(tmp_path):
    module = load_module()
    payload_path = tmp_path / "missing_parton.root"
    module.write_legacy_payload_atomic(payload_path, synthetic_payload())

    missing_parton = load_missing_parton_systematic(payload_path)

    assert "3l_onZ_1b" in missing_parton.get_process("tllq")
    assert "3l_onZ_1b;1" not in missing_parton.get_process("tllq")
