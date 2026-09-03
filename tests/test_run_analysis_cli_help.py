import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest

_SCRIPT_PATH = Path("analysis/topeft_run2/run_analysis.py")


def test_cli_help_shows_canonical_analysis_mode_flags(capsys):
    argv = ["run_analysis.py", "--help"]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())

    assert "--offZ-3l-split" in help_text
    assert "--tau-h-analysis" in help_text
    assert "--fwd-analysis" in help_text
    assert "--all-analysis" in help_text
    assert "--category-groups" in help_text
    assert (
        "Names are validated in run_analysis.py against the active SR/CR block selection"
        in normalized_help
    )
    assert "Accepts multiple group names and preserves user order after deduplication." in normalized_help
    assert "When omitted, all groups in each resolved block are used." in normalized_help

    assert "--offZ-split" not in help_text
    assert "--tau_h_analysis" not in help_text
    assert "--use-legacy-histeft" not in help_text


def test_removed_legacy_backend_option_is_rejected(capsys):
    argv = ["run_analysis.py", "--use-legacy-histeft"]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments: --use-legacy-histeft" in stderr


def test_legacy_backend_yaml_option_fails_clearly(tmp_path):
    options_file = tmp_path / "legacy-backend.yaml"
    options_file.write_text("use_legacy_histeft: false\n", encoding="utf-8")
    argv = ["run_analysis.py", "--options", str(options_file)]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    message = str(excinfo.value)
    assert "use_legacy_histeft" in message
    assert "MultiCell is now the only supported HistEFT backend" in message
