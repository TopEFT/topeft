"""Synthetic tests for the native observed-data producer path."""

import sys

import hist
import pytest

from analysis.topeft_run2 import make_cards
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules import datacard_tools


@pytest.mark.parametrize("cli_args, expected", (([], False), (["--unblind"], True)))
def test_make_cards_forwards_unblind_to_datacard_maker(monkeypatch, cli_args, expected):
    observed = {}

    class captured_kwargs(RuntimeError):
        pass

    def capture_datacard_maker(*, hists, **kwargs):
        observed.update(kwargs)
        raise captured_kwargs()

    monkeypatch.setattr(sys, "argv", ["make_cards.py", "synthetic.pkl.gz", *cli_args])
    monkeypatch.setattr(
        make_cards.datacard_tools,
        "load_and_merge_histogram_pkls",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    monkeypatch.setattr(make_cards.datacard_tools, "DatacardMaker", capture_datacard_maker)
    with pytest.raises(captured_kwargs):
        make_cards.main()
    assert observed["unblind"] is expected


@pytest.mark.parametrize("unblind, include_data, missing_data_error", (
    (False, False, False),
    (True, False, True),
    (True, True, False),
))
def test_native_missing_data_guard_precedes_output(
    tmp_path, monkeypatch, unblind, include_data, missing_data_error,
):
    sparse_hist = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="njets"),
        storage="Double",
    )
    for process in (["ttH", "data"] if include_data else ["ttH"]):
        sparse_hist.fill(
            process=process, channel="2lss_p_4j", systematic="nominal",
            njets=0.5, weight=1.0,
        )
    maker = object.__new__(datacard_tools.DatacardMaker)
    maker.hists = {"njets": sparse_hist}
    maker.use_real_data = unblind
    maker.out_dir = str(tmp_path)
    maker._resolve_supported_sr_appl = lambda *args: None
    maker.get_lep_mult = lambda channel: 2

    class post_guard_boundary(RuntimeError):
        pass

    post_guard_hits = []

    class guard_view:
        def __init__(self, selected_histogram):
            self.categorical_keys = selected_histogram.categorical_keys

        @property
        def dense_axis(self):
            post_guard_hits.append("dense_axis")
            raise post_guard_boundary()

    maker.binning_view = lambda value, *args: guard_view(value)

    class output_boundary(RuntimeError):
        pass

    monkeypatch.setattr(
        datacard_tools.uproot, "recreate",
        lambda *args, **kwargs: (_ for _ in ()).throw(output_boundary()),
    )
    if missing_data_error:
        with pytest.raises(RuntimeError, match="Unblinded datacard requires the data process"):
            maker.analyze("njets", "2lss_p_4j", {"ttH": set()}, True, {})
        assert post_guard_hits == []
    else:
        with pytest.raises(post_guard_boundary):
            maker.analyze("njets", "2lss_p_4j", {"ttH": set()}, True, {})
        assert post_guard_hits == ["dense_axis"]
    assert list(tmp_path.iterdir()) == []
