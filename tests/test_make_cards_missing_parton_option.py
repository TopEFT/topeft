from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")

from analysis.topeft_run2 import make_cards
from topcoffea.modules.histEFT import HistEFT
from topeft.modules import datacard_tools
from topeft.modules.datacard_tools import DatacardMaker, RateSystematic
from topeft.modules.missing_parton_contract import (
    DEFAULT_SR_REGISTRY,
    SUPPORTED_SR_REGISTRIES,
)


def _condor_namespace(
    skip_missing_parton_rate_syst,
    *,
    year_lst=(),
    missing_parton_payload_path=None,
    sr_registry=DEFAULT_SR_REGISTRY,
):
    return SimpleNamespace(
        do_mc_stat=False,
        verbose=False,
        use_real_data=False,
        do_nuisance=True,
        year_lst=list(year_lst),
        drop_syst=[],
        skip_missing_parton_rate_syst=skip_missing_parton_rate_syst,
        missing_parton_payload_path=missing_parton_payload_path,
        sr_registry=sr_registry,
    )


class _fake_branch:
    def array(self):
        return np.asarray([0.2, 0.3], dtype=float)


class _fake_missing_parton_file:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return ["3l_onZ_1b"]

    def __getitem__(self, key):
        assert key == "3l_onZ_1b/tllq"
        return _fake_branch()


def _systematics_loader(skip_missing_parton_rate_syst, year_lst=("UL18",)):
    maker = object.__new__(DatacardMaker)
    maker.do_nuisance = True
    maker.skip_missing_parton_rate_syst = skip_missing_parton_rate_syst
    maker.year_lst = list(year_lst)
    return maker


def _make_card_hist(axis_name, channel):
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=axis_name),
        wc_names=[],
        label="Events",
    )
    histogram.metadata = {}
    for process in ("tllqUL18", "tHqUL18", "ttHUL18"):
        histogram.fill(
            process=process,
            channel=channel,
            systematic="nominal",
            **{axis_name: np.asarray([0.5])},
            weight=np.asarray([10.0]),
        )
    return histogram


def test_parser_default_preserves_missing_parton_nuisance():
    args = make_cards.build_arg_parser().parse_args(["input.pkl.gz"])

    assert args.skip_missing_parton_rate_syst is False
    assert args.miss_parton_file is None
    assert args.sr_registry == DEFAULT_SR_REGISTRY


def test_parser_accepts_every_canonical_sr_registry():
    for sr_registry in SUPPORTED_SR_REGISTRIES:
        args = make_cards.build_arg_parser().parse_args(
            ["input.pkl.gz", "--sr-registry", sr_registry]
        )
        assert args.sr_registry == sr_registry

    with pytest.raises(SystemExit):
        make_cards.build_arg_parser().parse_args(
            ["input.pkl.gz", "--sr-registry", "UNKNOWN_SR_REGISTRY"]
        )


def test_parser_help_records_registry_default_and_payload_override():
    help_text = make_cards.build_arg_parser().format_help()

    assert "default: ALL_CH_LST_SR" in " ".join(help_text.split())
    assert "--miss-parton-file" in help_text


def test_parser_preserves_explicit_missing_parton_payload_override():
    args = make_cards.build_arg_parser().parse_args(
        ["input.pkl.gz", "--miss-parton-file", "custom/payload.root"]
    )

    assert args.miss_parton_file == "custom/payload.root"


def test_parser_accepts_targeted_missing_parton_suppression():
    args = make_cards.build_arg_parser().parse_args(
        ["input.pkl.gz", "--skip-missing-parton-rate-syst"]
    )

    assert args.skip_missing_parton_rate_syst is True


def test_local_main_propagates_targeted_suppression(monkeypatch):
    class captured_kwargs(RuntimeError):
        pass

    def capture_datacard_maker(*, hists, **kwargs):
        assert hists == {}
        raise captured_kwargs(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_cards.py",
            "input.pkl.gz",
            "--skip-missing-parton-rate-syst",
        ],
    )
    monkeypatch.setattr(
        make_cards,
        "load_and_merge_histogram_pkls",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    monkeypatch.setattr(make_cards, "DatacardMaker", capture_datacard_maker)

    with pytest.raises(captured_kwargs) as exc_info:
        make_cards.main()

    assert exc_info.value.args[0]["skip_missing_parton_rate_syst"] is True


def test_local_cli_omission_passes_direct_default_resolution_inputs(monkeypatch):
    class captured_kwargs(RuntimeError):
        pass

    def capture_datacard_maker(*, hists, **kwargs):
        assert hists == {}
        raise captured_kwargs(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        ["make_cards.py", "input.pkl.gz", "--do-nuisance", "--year", "UL18"],
    )
    monkeypatch.setattr(
        make_cards,
        "load_and_merge_histogram_pkls",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    monkeypatch.setattr(make_cards, "DatacardMaker", capture_datacard_maker)

    with pytest.raises(captured_kwargs) as exc_info:
        make_cards.main()

    assert exc_info.value.args[0]["missing_parton_path"] is None
    assert exc_info.value.args[0]["year_lst"] == ["UL18"]
    assert exc_info.value.args[0]["sr_registry"] == DEFAULT_SR_REGISTRY


def test_local_cli_propagates_registry_and_exact_payload_override(monkeypatch):
    class captured_kwargs(RuntimeError):
        pass

    def capture_datacard_maker(*, hists, **kwargs):
        assert hists == {}
        raise captured_kwargs(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_cards.py",
            "input.pkl.gz",
            "--sr-registry",
            "FWD_CH_LST_SR",
            "--miss-parton-file",
            "arbitrary/payload.root",
        ],
    )
    monkeypatch.setattr(
        make_cards,
        "load_and_merge_histogram_pkls",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    monkeypatch.setattr(make_cards, "DatacardMaker", capture_datacard_maker)

    with pytest.raises(captured_kwargs) as exc_info:
        make_cards.main()

    assert exc_info.value.args[0]["sr_registry"] == "FWD_CH_LST_SR"
    assert exc_info.value.args[0]["missing_parton_path"] == "arbitrary/payload.root"


def test_condor_option_propagation_is_additive_and_targeted():
    default_opts = make_cards._build_condor_base_other_opts(
        _condor_namespace(False),
        "error",
    )
    skip_opts = make_cards._build_condor_base_other_opts(
        _condor_namespace(True),
        "error",
    )

    assert "--skip-missing-parton-rate-syst" not in default_opts
    assert skip_opts == default_opts[:-2] + [
        "--skip-missing-parton-rate-syst",
        "--on-process-collision",
        "error",
    ]


@pytest.mark.parametrize(
    "payload_path",
    (
        "data/missing_parton/missing_parton_run2.root",
        "data/missing_parton/missing_parton_run3.root",
        "custom/payload.root",
    ),
)
def test_condor_option_reconstruction_materializes_resolved_payload_path(payload_path):
    opts = make_cards._build_condor_base_other_opts(
        _condor_namespace(
            False,
            year_lst=("UL18",),
            missing_parton_payload_path=payload_path,
        ),
        "error",
    )

    payload_option_index = opts.index("--miss-parton-file")
    assert opts[payload_option_index + 1] == payload_path


@pytest.mark.parametrize("sr_registry", SUPPORTED_SR_REGISTRIES)
def test_condor_option_reconstruction_preserves_one_registry(sr_registry):
    opts = make_cards._build_condor_base_other_opts(
        _condor_namespace(False, sr_registry=sr_registry),
        "error",
    )

    assert opts.count("--sr-registry") == 1
    registry_option_index = opts.index("--sr-registry")
    assert opts[registry_option_index + 1] == sr_registry


def test_skip_option_suppresses_only_missing_parton(monkeypatch):
    maker = _systematics_loader(True)

    def fail_if_opened(_):
        raise AssertionError("missing-parton payload was opened despite suppression")

    monkeypatch.setattr(datacard_tools.uproot, "open", fail_if_opened)
    systematics = maker.load_systematics(
        "params/rate_systs_run3.json",
        "does-not-exist.root",
    )

    assert "missing_parton" not in systematics
    assert "diboson_njets" in systematics
    assert len(systematics) > 1


def test_default_missing_parton_contract_remains_tllq_and_thq(monkeypatch):
    maker = _systematics_loader(False)
    monkeypatch.setattr(
        datacard_tools.uproot,
        "open",
        lambda _: _fake_missing_parton_file(),
    )

    systematics = maker.load_systematics(
        "params/rate_systs_run3.json",
        "synthetic.root",
    )
    missing_parton = systematics["missing_parton"]

    assert missing_parton.name == "missing_parton"
    assert missing_parton.get_process("tllq") == {
        "3l_onZ_1b": pytest.approx(np.asarray([1.2, 1.3]))
    }
    assert missing_parton.get_process("tHq") == {
        "3l_onZ_1b": pytest.approx(np.asarray([1.2, 1.3]))
    }
    assert missing_parton.get_process("ttH") == "-"


def test_missing_parton_card_formatting_and_missing_entry_remain_public(
    tmp_path,
):
    channel = "3l_onZ_1b_2j"
    hists = {
        "ht": _make_card_hist("ht", channel),
        "ht_sumw2": _make_card_hist("ht_sumw2", channel),
    }
    maker = DatacardMaker(
        hists=hists,
        out_dir=str(tmp_path),
        var_lst=["ht"],
        do_nuisance=False,
        verbose=False,
    )
    nuisance_name = "missing_parton"
    missing_parton = RateSystematic(nuisance_name)
    payload = {"3l_onZ_1b": np.asarray([1.0, 1.1, 1.2])}
    missing_parton.add_process("tllq", payload)
    missing_parton.add_process("tHq", payload)
    maker.rate_systs = {nuisance_name: missing_parton}

    maker.analyze(
        "ht",
        channel,
        {"tllq": [], "tHq": [], "ttH": []},
        True,
        {},
    )

    card_path = tmp_path / f"ttx_multileptons-{channel}_ht.txt"
    process_names = None
    missing_parton_values = None
    for line in card_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["process"] and any(
            field.endswith("_sm") for field in fields[1:]
        ):
            process_names = fields[1:]
        if fields[:2] == [nuisance_name, "lnN"]:
            missing_parton_values = fields[2:]

    assert dict(zip(process_names, missing_parton_values)) == {
        "tllq_sm": "0.800000/1.200000",
        "tHq_sm": "0.800000/1.200000",
        "ttH_sm": "-",
    }
