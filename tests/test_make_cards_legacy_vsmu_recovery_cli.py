"""Explicit CLI authorization for structurally supported legacy Run-3 VSmu transport."""

import sys

import pytest

from analysis.topeft_run2 import make_cards


@pytest.mark.parametrize("cli_args, expected", (
    ([], False),
    (["--enable-legacy-run3-vsmu-recovery"], True),
))
def test_make_cards_passes_legacy_recovery_opt_in_to_datacard_maker(
    monkeypatch, cli_args, expected,
):
    observed = {}

    class captured_kwargs(RuntimeError):
        pass

    def capture_datacard_maker(*, hists, **kwargs):
        observed.update(kwargs)
        raise captured_kwargs()

    monkeypatch.setattr(sys, "argv", ["make_cards.py", "input.pkl.gz", *cli_args])
    monkeypatch.setattr(
        make_cards.datacard_tools, "load_and_merge_histogram_pkls",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    monkeypatch.setattr(
        make_cards.datacard_tools, "DatacardMaker", capture_datacard_maker,
    )
    with pytest.raises(captured_kwargs):
        make_cards.main()
    assert observed["enable_legacy_run3_vsmu_recovery"] is expected
