import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from analysis.topeft_run2 import datacards_post_processing


script_path = Path("analysis/topeft_run2/datacards_post_processing.py")


def test_forward_selection_reports_root_count_and_requires_card_template_symmetry(
    tmp_path,
    capsys,
):
    (tmp_path / "scalings-preselect.json").write_text("[]", encoding="utf-8")
    (tmp_path / "selectedWCs.txt").write_text("{}", encoding="utf-8")
    for filename in (
        "ttx_multileptons-2lss_fwd_p_4j_lt.txt",
        "ttx_multileptons-2lss_fwd_p_4j_lt.root",
        "ttx_multileptons-2lss_fwd_p_5j_lt.root",
    ):
        (tmp_path / filename).write_text("fixture", encoding="utf-8")

    with mock.patch.object(
        sys,
        "argv",
        [str(script_path), str(tmp_path), "--fwd-flag"],
    ):
        with pytest.raises(
            Exception,
            match="expected one text card per ROOT template",
        ):
            runpy.run_path(str(script_path), run_name="__main__")

    output = capsys.readouterr().out
    assert "Number of text templates copied: 1" in output
    assert "Number of root templates copied: 2" in output
    assert json.loads(
        (tmp_path / "selectedWCs.txt").read_text(encoding="utf-8")
    ) == {}


def test_selector_error_names_all_analysis(tmp_path):
    with mock.patch.object(sys, "argv", [str(script_path), str(tmp_path)]):
        with pytest.raises(ValueError, match="--all-analysis"):
            runpy.run_path(str(script_path), run_name="__main__")


def test_all_analysis_requires_exact_text_and_root_counts():
    args = SimpleNamespace(
        set_up_top22006=False,
        set_up_offZdivision=False,
        tau_flag=False,
        fwd_flag=False,
        all_analysis=True,
    )

    datacards_post_processing._validate_copied_template_counts(args, 129, 129)
    with pytest.raises(Exception, match="one text card per ROOT template"):
        datacards_post_processing._validate_copied_template_counts(args, 128, 129)
    with pytest.raises(Exception, match="unexpected number"):
        datacards_post_processing._validate_copied_template_counts(args, 128, 128)


def test_forward_selection_requires_symmetry_without_an_exact_total():
    args = SimpleNamespace(
        set_up_top22006=False,
        set_up_offZdivision=False,
        tau_flag=False,
        fwd_flag=True,
        all_analysis=False,
    )

    datacards_post_processing._validate_copied_template_counts(args, 7, 7)
    with pytest.raises(Exception, match="one text card per ROOT template"):
        datacards_post_processing._validate_copied_template_counts(args, 7, 8)
