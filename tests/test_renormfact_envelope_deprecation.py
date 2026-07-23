import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest

from topeft.modules.get_renormfact_envelope import (
    apply_renormfact_envelope_to_histogram,
    get_renormfact_envelope,
    unsupported_renormfact_envelope_message,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_ANALYSIS = _REPO_ROOT / "analysis" / "topeft_run2" / "run_analysis.py"
_RUN_DIBOSON = _REPO_ROOT / "analysis" / "topeft_run2" / "run_analysis_diboson.py"
_HELPER = _REPO_ROOT / "topeft" / "modules" / "get_renormfact_envelope.py"


class _UnreadableHistogram:
    def __getattribute__(self, name):
        raise AssertionError(f"histogram must not be accessed: {name}")


def test_public_envelope_helpers_fail_without_accessing_input():
    histogram = _UnreadableHistogram()
    with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
        apply_renormfact_envelope_to_histogram(histogram)
    payload = {"met": histogram}
    with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
        get_renormfact_envelope(payload)
    assert payload == {"met": histogram}
    assert "No histogram or output was modified" in unsupported_renormfact_envelope_message


def test_standalone_helper_fails_before_opening_input_or_creating_output(tmp_path):
    input_path = tmp_path / "must_not_be_opened.pkl.gz"
    output_path = tmp_path / "must_not_be_created.pkl.gz"
    with mock.patch.object(sys, "argv", [str(_HELPER), str(input_path), "-n", str(output_path)]):
        with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
            runpy.run_path(str(_HELPER), run_name="__main__")
    assert not input_path.exists()
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("script_path", "flag"),
    [(_RUN_ANALYSIS, "--do-renormfact-envelope"), (_RUN_DIBOSON, "--do-renormfact-envelope")],
)
def test_analysis_clis_fail_before_loading_missing_sample(script_path, flag, tmp_path):
    missing_sample = tmp_path / "must_not_be_loaded.json"
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(script_path.parent))
    try:
        with mock.patch.object(sys, "argv", [str(script_path), str(missing_sample), flag]):
            with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
                runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.path = original_sys_path
    assert not missing_sample.exists()


def test_run_analysis_options_envelope_request_fails_before_loading_missing_sample(tmp_path):
    missing_sample = tmp_path / "must_not_be_loaded.json"
    options_path = tmp_path / "deprecated_envelope.yaml"
    options_path.write_text("do_renormfact_envelope: true\n", encoding="utf-8")
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_RUN_ANALYSIS.parent))
    try:
        with mock.patch.object(
            sys,
            "argv",
            [str(_RUN_ANALYSIS), str(missing_sample), "--options", str(options_path)],
        ):
            with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
                runpy.run_path(str(_RUN_ANALYSIS), run_name="__main__")
    finally:
        sys.path = original_sys_path
    assert not missing_sample.exists()


def test_four_supported_sow_normalizations_remain_exact_and_no_combined_template_is_produced():
    source = _RUN_ANALYSIS.with_name("analysis_processor.py").read_text(encoding="utf-8")
    expected = {
        "events.renormUp * (sow / sow_renormUp)",
        "events.renormDown * (sow / sow_renormDown)",
        "events.factUp * (sow / sow_factUp)",
        "events.factDown * (sow / sow_factDown)",
    }
    normalized_source = source.replace(" ", "")
    for expression in expected:
        assert expression.replace(" ", "") in normalized_source
    assert "weights_obj_base.add('renormfact'" not in source
    assert '"renormfactUp"' not in source
    assert '"renormfactDown"' not in source
    for consumer_path in (
        _RUN_ANALYSIS,
        _RUN_ANALYSIS.with_name("run_data_driven.py"),
        _RUN_ANALYSIS.with_name("make_cr_and_sr_plots.py"),
        _REPO_ROOT / "topeft" / "modules" / "datacard_tools.py",
    ):
        consumer_source = consumer_path.read_text(encoding="utf-8")
        for sow_name in (
            "sow_renormUp",
            "sow_renormDown",
            "sow_factUp",
            "sow_factDown",
        ):
            assert sow_name not in consumer_source
