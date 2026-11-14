import ast
import copy
import re
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.topeft_run2 import analysis_processor as module


def test_linux_memory_uses_one_status_observation(monkeypatch, capsys):
    reads = []
    def status(path):
        assert path == "/proc/self/status"
        reads.append(path)
        return io.StringIO("VmHWM:\t32768 kB\nName:\tfixture\nVmRSS:\t16384 kB\n")
    monkeypatch.setattr(module, "open", status, raising=False)
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda *_: pytest.fail("rusage used despite VmHWM"))
    module._PhaseTelemetry().mark("processor_start")
    record = json.loads(capsys.readouterr().out.split("] ", 1)[1])
    assert (record["rss_mib"], record["peak_rss_mib"]) == (16., 32.)
    assert len(reads) == 1


@pytest.mark.parametrize("status,current,peak", [
    ("VmRSS: 8192 kB\n", 8., 8.),  # rusage can lag current accounting
    ("VmRSS: 1024 kB\n", 1., 4.),
    ("VmHWM: 8192 kB\n", None, 8.),
    ("Name: fixture\n", None, 4.),
    ("VmRSS: unavailable\nVmHWM: unavailable\n", None, 4.),
])
def test_missing_memory_fields_are_honest_and_fallback_is_bounded(
    monkeypatch, status, current, peak
):
    monkeypatch.setattr(module, "open", lambda *_: io.StringIO(status), raising=False)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda *_: SimpleNamespace(ru_maxrss=4096))
    assert module._phase_memory_mib() == (current, peak)


@pytest.mark.parametrize("platform,raw", [("linux", 4096), ("darwin", 4194304)])
def test_no_proc_has_no_current_rss(monkeypatch, platform, raw):
    def unavailable(*_):
        raise OSError("no proc")
    monkeypatch.setattr(module, "open", unavailable, raising=False)
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda *_: SimpleNamespace(ru_maxrss=raw))
    assert module._phase_memory_mib() == (None, 4.)
    monkeypatch.setattr(module.resource, "getrusage", unavailable)
    assert module._phase_memory_mib() == (None, None)


def test_runtime_phase_memory_invariant(capsys):
    reporter = module._PhaseTelemetry()
    for _ in range(3):
        reporter.mark("processor_complete")
    for line in capsys.readouterr().out.splitlines():
        record = json.loads(line.split("] ", 1)[1])
        if record["rss_mib"] is not None and record["peak_rss_mib"] is not None:
            assert record["peak_rss_mib"] >= record["rss_mib"]


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


def processor_process_node():
    tree = ast.parse(Path(module.__file__).read_text())
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "process")


def assert_processor_phase_sites(process):
    """Check explicit telemetry sites without executing scientific work."""
    def same(node, expression):
        return ast.dump(node) == ast.dump(ast.parse(expression, mode="eval").body)

    parents = {child: parent for parent in ast.walk(process)
               for child in ast.iter_child_nodes(parent)}
    marks = [node for node in ast.walk(process)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "telemetry" and node.func.attr == "mark"]
    fixed = {"processor_start", "processor_inputs_prepared",
             "processor_base_weights_and_met_ready", "processor_iteration_start",
             "processor_complete"}
    paired = {"objects_ready", "weights_ready", "selections_ready",
              "variables_ready", "fills_complete"}
    loops = [node for node in ast.walk(process) if isinstance(node, ast.For)
             and same(node.iter, "enumerate(syst_var_list)")]
    assert len(loops) == 1
    loop = loops[0]
    expected_target = ast.parse("for iteration_index, syst_var in []: pass").body[0].target
    assert ast.dump(loop.target) == ast.dump(expected_target)
    constants, pairs = [], []
    for mark in marks:
        assert len(mark.args) == 1
        phase = mark.args[0]
        if isinstance(phase, ast.Constant):
            assert isinstance(phase.value, str) and phase.value in fixed
            constants.append(phase.value)
            in_iteration = phase.value == "processor_iteration_start"
        else:
            assert isinstance(phase, ast.IfExp)
            assert same(phase.test, 'syst_var == "nominal"')
            assert isinstance(phase.body, ast.Constant)
            assert isinstance(phase.orelse, ast.Constant)
            pair = (phase.body.value, phase.orelse.value)
            assert pair in {("processor_nominal_" + name, "processor_" + name)
                            for name in paired}
            pairs.append(pair)
            in_iteration = True
        statement = parents[mark]
        assert isinstance(statement, ast.Expr)
        guard = parents[statement]
        assert isinstance(guard, ast.If) and same(guard.test, "telemetry is not None")
        assert statement in guard.body and not guard.orelse
        # Direct placement rejects an additional nominal-only guard as well.
        assert parents[guard] is (loop if in_iteration else process)
        if in_iteration:
            keywords = {keyword.arg: keyword.value for keyword in mark.keywords}
            assert same(keywords["syst_var"], "syst_var")
            assert same(keywords["iteration_index"], "iteration_index")
    assert sorted(constants) == sorted(fixed)
    assert sorted(pairs) == sorted(("processor_nominal_" + name, "processor_" + name)
                                  for name in paired)
    # Ten call sites expand to fifteen names.
    assert len(marks) == len(fixed) + len(paired)


def test_processor_default_and_fixed_phase_sites():
    signature = inspect.signature(module.AnalysisProcessor)
    assert signature.parameters["phase_telemetry"].default is False
    assert_processor_phase_sites(processor_process_node())


@pytest.mark.parametrize("defect", [
    "missing_phase", "extra_phase", "swapped_arms", "condition", "loop_target", "unguarded",
])
def test_phase_site_checks_reject_contract_regressions(defect):
    process = processor_process_node()
    loop = next(node for node in ast.walk(process) if isinstance(node, ast.For)
                and ast.unparse(node.iter) == "enumerate(syst_var_list)")
    guard = next(node for node in loop.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "telemetry is not None"
                 and any(isinstance(child, ast.IfExp) for child in ast.walk(node)))
    phase = next(node for node in ast.walk(guard) if isinstance(node, ast.IfExp))
    if defect == "missing_phase":
        loop.body.remove(guard)
    elif defect == "extra_phase":
        loop.body.append(copy.deepcopy(guard))
    elif defect == "swapped_arms":
        phase.body, phase.orelse = phase.orelse, phase.body
    elif defect == "condition":
        phase.test.ops = [ast.NotEq()]
    elif defect == "loop_target":
        loop.target = ast.Name(id="syst_var", ctx=ast.Store())
    elif defect == "unguarded":
        guard.test = ast.Constant(value=True)
    with pytest.raises(AssertionError):
        assert_processor_phase_sites(process)


def assert_run_cr_telemetry_contract(source):
    assert re.search(r"^profile_phase_telemetry=false$", source, re.M)
    parser = re.search(r'^\s*--phase-telemetry\)\s*\n(.*?)^\s*;;',
                       source, re.M | re.S)
    assert parser is not None
    assert parser.group(1).split() == ["profile_phase_telemetry=true", "shift"]
    combined = source.split('case "${remaining_args[0]}" in', 1)[1].split("esac", 1)[0]
    arms = re.findall(r"^\s*(--[^\n)]+)\)\s*\n(.*?)^\s*;;",
                      combined, re.M | re.S)
    telemetry_arms = [body for patterns, body in arms
                      if "--phase-telemetry" in patterns.split("|")]
    assert len(telemetry_arms) == 1
    assert [line.strip() for line in telemetry_arms[0].splitlines()] == [
        'component_args+=("${remaining_args[0]}")',
        'remaining_args=("${remaining_args[@]:1}")',
    ]
    # Other single flags, including environment migration, can share this arm.
    assert re.search(r"^\s*--resume\)", combined, re.M)
    for command in ("cmd_ref", "data_driven_cmd"):
        addition = command + "+=(--phase-telemetry)"
        assert source.count(addition) == 1
        assert re.search(
            r'if \[\[ "\$\{profile_phase_telemetry\}" == "true" \]\]; then\s*\n\s*'
            + re.escape(addition) + r"\s*\n\s*fi", source)


def test_run_cr_flag_is_forwarded_without_production_execution():
    root = Path(module.__file__).resolve().parents[2]
    source = (root / "analysis/topeft_run2/run_cr.sh").read_text()
    assert_run_cr_telemetry_contract(source)
    wrapper = (root / "analysis/topeft_run2/fullR3_run.sh").read_text()
    assert re.search(r'\*\)\s*\n\s*EXTRA_ARGS\+=\("\$1"\)\s*\n\s*shift\s*\n\s*;;',
                     wrapper)
    assert 'RUN_CMD+=("${EXTRA_ARGS[@]}")' in wrapper


@pytest.mark.parametrize("old,new", [
    ("|--phase-telemetry|", "|"),
    ('component_args+=("${remaining_args[0]}")', 'component_args+=("lost")'),
    ('cmd_ref+=(--phase-telemetry)', 'cmd_ref+=()'),
    ('data_driven_cmd+=(--phase-telemetry)', 'data_driven_cmd+=()'),
])
def test_flag_checks_reject_lost_forwarding(old, new):
    root = Path(module.__file__).resolve().parents[2]
    source = (root / "analysis/topeft_run2/run_cr.sh").read_text()
    assert source.count(old) == 1
    with pytest.raises(AssertionError):
        assert_run_cr_telemetry_contract(source.replace(old, new, 1))
