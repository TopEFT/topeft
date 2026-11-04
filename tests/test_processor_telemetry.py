import ast
import inspect
import json
from pathlib import Path

import pytest

from analysis.topeft_run2 import analysis_processor as module


def test_phase_reporter_retains_only_fixed_scalar_context(capsys):
    reporter = module._PhaseTelemetry(sample="sample", year="2023BPix",
                                      filename="input.root", entry_start=2,
                                      entry_stop=5, event_count=3)
    before = dict(reporter._context)
    for _ in range(3):
        reporter.mark("processor_start", rss_mb=12.5, peak_rss_mb=20.)
    records = [json.loads(line.split("] ", 1)[1])
               for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 3
    assert reporter._context == before
    assert not hasattr(reporter, "__dict__")
    assert type(reporter._wall_start_ns) is int
    assert type(reporter._cpu_start_ns) is int
    for record in records:
        assert all(type(value) in (str, int, float, bool, type(None))
                   for value in record.values())
        assert record["rss_mib"] == 12.5
        assert record["peak_rss_mib"] == 20.
        assert record["wall_elapsed_ns"] >= 0
        assert record["cpu_elapsed_ns"] >= 0
    assert records[-1]["wall_ns"] >= records[0]["wall_ns"]


def test_reporter_rejects_scientific_objects():
    with pytest.raises(TypeError, match="only scalars"):
        module._PhaseTelemetry(event_count=[1, 2, 3])


def test_processor_default_and_fixed_phase_sites():
    signature = inspect.signature(module.AnalysisProcessor)
    assert signature.parameters["phase_telemetry"].default is False
    tree = ast.parse(Path(module.__file__).read_text())
    process = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "process")
    marks = [node for node in ast.walk(process)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "telemetry" and node.func.attr == "mark"]
    phases = {node.args[0].value for node in marks}
    assert phases == {
        "processor_start", "processor_inputs_prepared",
        "processor_base_weights_and_met_ready", "processor_nominal_objects_ready",
        "processor_nominal_weights_ready", "processor_nominal_selections_ready",
        "processor_nominal_variables_ready", "processor_nominal_fills_complete",
        "processor_complete",
    }
    assert len(marks) == len(phases)
    # Every loop-local record is guarded by the single nominal variation.
    for loop in (node for node in ast.walk(process) if isinstance(node, (ast.For, ast.While))):
        loop_marks = [mark for mark in marks if mark in list(ast.walk(loop))]
        if loop_marks:
            assert isinstance(loop, ast.For)
            assert isinstance(loop.target, ast.Name) and loop.target.id == "syst_var"
            nominal_guards = [node for node in ast.walk(loop)
                              if isinstance(node, ast.If)
                              and 'syst_var == \'nominal\'' in ast.unparse(node.test)]
            for mark in loop_marks:
                assert any(mark in list(ast.walk(guard)) for guard in nominal_guards)


def test_run_cr_flag_is_forwarded_without_production_execution():
    root = Path(module.__file__).resolve().parents[2]
    source = (root / "analysis/topeft_run2/run_cr.sh").read_text()
    assert "profile_phase_telemetry=false" in source
    assert "--dry-run|--resume|--phase-telemetry)" in source
    assert "cmd_ref+=(--phase-telemetry)" in source
    assert "data_driven_cmd+=(--phase-telemetry)" in source
    wrapper = (root / "analysis/topeft_run2/fullR3_run.sh").read_text()
    assert 'EXTRA_ARGS+=("$1")' in wrapper
