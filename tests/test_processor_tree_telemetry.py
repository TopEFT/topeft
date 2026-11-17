"""Small in-process fixtures for maintained telemetry, with no WQ or campaign IO."""

import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import resource
import re
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"


def load_nodes(filename, names, namespace):
    tree = ast.parse((SOURCE / filename).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.fixture
def telemetry():
    ns = dict(json=json, os=os, time=time, sys=sys, resource=resource, uuid=uuid)
    load_nodes("analysis_processor.py", ["_phase_memory_mib", "_PhaseTelemetry",
                                       "_IterationTelemetry"], ns)
    ns["analysis_processor"] = SimpleNamespace(**ns)
    load_nodes("run_analysis.py", ["_persist_task_phase_telemetry",
                                   "_install_phase_telemetry_persistence"], ns)
    return ns


def persist_stdout(ns, capsys, tmp_path):
    stdout = tmp_path / "task.stdout"
    stdout.write_text(capsys.readouterr().out)
    sink = tmp_path / "phase_telemetry.jsonl"
    ns["_persist_task_phase_telemetry"](SimpleNamespace(outfile_stdout=stdout, id=719), sink)
    return [json.loads(line) for line in sink.read_text().splitlines()]


def process_node():
    tree = ast.parse((SOURCE / "analysis_processor.py").read_text())
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "process")


def test_event_count_and_existing_fields_survive(telemetry, capsys, tmp_path):
    reporter = telemetry["_PhaseTelemetry"](event_count=17, sample="private_sample")
    reporter.mark("processor_start")
    reporter.mark("processor_inputs_prepared", syst_var="nominal", iteration_index=0)
    rows = persist_stdout(telemetry, capsys, tmp_path)
    assert rows[0]["event_count"] == 17
    assert "event_count" not in rows[1]
    assert (rows[1]["syst_var"], rows[1]["iteration_index"]) == ("nominal", 0)
    for row in rows:
        assert row["task_id"] == 719
        assert {"schema_version", "phase", "wall_ns", "cpu_ns", "rss_mib",
                "peak_rss_mib", "wall_elapsed_ns", "cpu_elapsed_ns"} <= row.keys()
        assert not ({"sample", "filename", "entry_start", "entry_stop"} & row.keys())
    source = ast.unparse(process_node())
    assert "event_count=len(events)" in source


def test_dd_inherits_context_and_extended_observations(telemetry, capsys, tmp_path):
    ns = dict(telemetry, contextlib=contextlib, io=io)
    load_nodes("run_data_driven.py", ["_DurablePhaseTelemetry"], ns)
    output = tmp_path / "dd.pkl"
    reporter = ns["_DurablePhaseTelemetry"](filename="input.pkl", output_path=output)
    reporter.mark("dd_start", rss_mb=12.5, peak_rss_mb=20.)
    reporter.mark("dd_finalized", iteration_completed_count=2)
    rows = [json.loads(line.split("] ", 1)[1])
            for line in capsys.readouterr().out.splitlines()]
    assert [row["phase"] for row in rows] == ["dd_start", "dd_finalized"]
    context = dict(sample="", year="", filename="input.pkl", entry_start=None,
                   entry_stop=None, event_count=None)
    for row in rows:
        assert {key: row[key] for key in context} == context
    assert rows[0]["rss_mib"] == 12.5
    assert rows[1]["iteration_completed_count"] == 2
    base_fields = {"schema_version", "phase", "wall_ns", "cpu_ns", "wall_elapsed_ns",
                   "cpu_elapsed_ns", "rss_mib", "peak_rss_mib"}
    persisted = [json.loads(line) for line in
                 Path(str(output) + ".phase_telemetry.jsonl").read_text().splitlines()]
    assert persisted == [{key: row[key] for key in base_fields} for row in rows]
    assert not output.exists()


def test_disabled_telemetry_preserves_processor_and_dd_silence(capsys, tmp_path):
    def unexpected_reporter(**kwargs):
        pytest.fail("disabled telemetry constructed a reporter")

    processor_body = process_node().body
    dd_tree = ast.parse((SOURCE / "run_data_driven.py").read_text())
    dd_body = next(node for node in dd_tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "main").body
    for body, reporter_name in ((processor_body, "_PhaseTelemetry"),
                                (dd_body, "_DurablePhaseTelemetry")):
        index = next(index for index, node in enumerate(body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "telemetry"
                             for target in node.targets)
                     and reporter_name in ast.unparse(node.value))
        ns = dict(self=SimpleNamespace(_phase_telemetry=False),
                  args=SimpleNamespace(phase_telemetry=False),
                  **{reporter_name: unexpected_reporter})
        exec(compile(ast.Module(body=body[index:index + 2], type_ignores=[]),
                     "disabled_telemetry", "exec"), ns)
        assert ns["telemetry"] is None
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())


def test_malformed_and_nontelemetry_stdout_are_not_persisted(telemetry, capsys, tmp_path):
    telemetry["_PhaseTelemetry"](event_count=3).mark("processor_start")
    valid = capsys.readouterr().out
    record = json.loads(valid.split("] ", 1)[1])
    stdout = tmp_path / "mixed.stdout"
    stdout.write_text("ordinary stdout\n[phase_telemetry] broken json\n"
                      "[phase_telemetry] {}\n" + valid +
                      "[phase_telemetry] " + json.dumps(dict(record, phase="dd_start")) + "\n" +
                      "[phase_telemetry] " + json.dumps(dict(record, iteration_index=[])) + "\n")
    sink = tmp_path / "phase_telemetry.jsonl"
    telemetry["_persist_task_phase_telemetry"](
        SimpleNamespace(outfile_stdout=stdout, id=719), sink)
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert len(rows) == 1
    assert (rows[0]["phase"], rows[0]["event_count"]) == ("processor_start", 3)
    assert capsys.readouterr().err.count("skipped malformed phase telemetry record") == 4


def test_executed_iteration_marks_and_counts(telemetry, capsys, tmp_path):
    # Execute the maintained telemetry statements in their actual process order;
    # substitute only the scientific work, which this test does not qualify.
    process = process_node()

    def select(body):
        result = []
        for node in body:
            text = ast.unparse(node)
            if isinstance(node, ast.For) and "enumerate(syst_var_list)" in ast.unparse(node.iter):
                node = copy.deepcopy(node)
                node.body = select(node.body)
                result.append(node)
            elif isinstance(node, ast.If) and ast.unparse(node.test) == "telemetry is not None":
                result.append(copy.deepcopy(node))
            elif isinstance(node, (ast.Assign, ast.AugAssign)) and any(
                    text.startswith(name) for name in (
                        "iteration_entered_count", "iteration_completed_count", "iteration_telemetry")):
                result.append(copy.deepcopy(node))
        return result

    ns = dict(telemetry, telemetry=telemetry["_PhaseTelemetry"](event_count=3),
              syst_var_list=["nominal", "JESUp"])
    exec(compile(ast.Module(body=select(process.body), type_ignores=[]), "iterations", "exec"), ns)
    rows = persist_stdout(telemetry, capsys, tmp_path)
    old = ["processor_start", "processor_inputs_prepared", "processor_base_weights_and_met_ready",
           *["processor_nominal_" + phase for phase in (
               "objects_ready", "weights_ready", "selections_ready", "variables_ready", "fills_complete")],
           "processor_complete"]
    assert [row["phase"] for row in rows if row["phase"] in old] == old
    nonnominal = [row for row in rows if row.get("syst_var") == "JESUp"]
    assert [row["phase"] for row in nonnominal] == [
        "processor_" + phase for phase in ("iteration_start", "objects_ready", "weights_ready",
                                            "selections_ready", "variables_ready", "fills_complete")]
    assert all(row["iteration_index"] == 1 for row in nonnominal)
    assert rows[-1]["iteration_entered_count"] == rows[-1]["iteration_completed_count"] == 2
    assert [row["syst_var_list_len"] for row in rows if "syst_var_list_len" in row] == [2]
    outer = next(node for node in ast.walk(process) if isinstance(node, ast.For)
                 and ast.unparse(node.target) == "(iteration_index, syst_var)")
    assert ast.unparse(outer).index("processor_iteration_start") < ast.unparse(outer).index("copy.deepcopy")


class Histogram:
    def __init__(self):
        self.calls = []

    def fill(self, **kwargs):
        self.calls.append(kwargs)


def execute_fill_nest(telemetry, *, domain="scalar", split=False, applicable=True,
                      drop_axis_entry=False, empty=False):
    """Execute the real fill loop with tiny arrays and deterministic selections."""
    family = "jvm" if domain == "jvm" else "x"
    variables = {family: np.array([1., 2., 3., 4.])}
    category = {"2l": {"exactly_2j": dict(appl_lst=["isSR_x", "isAR_x"],
                                         lep_chan_lst=["2l_c"], lep_flav_lst=["ee", "mm"])}}
    histograms = {family: Histogram(), family + "_sumw2": Histogram()}
    weights = SimpleNamespace(variations={"fake"}, weight=lambda label: np.ones(4))
    masks = SimpleNamespace(all=lambda *args: np.ones(4, dtype=bool),
                            any=lambda *args: np.ones(4, dtype=bool))
    self = SimpleNamespace(
        _base_hist_name_set={family}, _hist_sumw2_axis_mapping={family: {family + "_sumw2": family}},
        _sumw2_policy=None, _fill_sumw2_hist=domain != "jvm", _do_systematics=True,
        _split_by_lepton_flavor=split, _ecut_threshold=1,
        _hist_axis_map={family: [family], family + "_sumw2": [family + "_sumw2"]},
        _hist_requires_eft={family: domain == "eft"},
        histogram_fill_is_applicable=lambda *args, **kwargs: applicable,
        _should_fill_sumw2_histogram=lambda selected, wgt_fluct: selected and wgt_fluct == "nominal",
        _raw_count_fill_classification=lambda *args, **kwargs: None,
    )
    reporter = telemetry["_PhaseTelemetry"](event_count=4)
    iteration = telemetry["_IterationTelemetry"](reporter, "nominal", 0)
    ns = dict(telemetry, np=np, ak=SimpleNamespace(values_astype=np.asarray,
              is_none=lambda value: (np.array([False, False, False, True])
                                    if drop_axis_entry else np.zeros(len(value), dtype=bool)), sum=np.sum),
              self=self, varnames=variables, cat_dict=category, weights_dict={"2l": weights},
              iteration_telemetry=iteration, syst_var="nominal", isData=False, isEFT=domain == "eft",
              dataset_key="test", histAxisName="test", hout=histograms, axes_info_2d={},
              wgt_correction_syst_lst=[], data_syst_lst=["fake"], obj_correction_syst_lst=[],
              select_histogram_weight_variations=lambda *args: ["nominal", "missing", "fake"],
              scalar_nominal_key=lambda name: name, eft_nominal_key=lambda name: name,
              selections=masks, sr_category_names={"2l"}, is_run3=True, year="2022",
              events={"ttgamma_photon_history_pass_conversion_overlap_removal":
                      np.array([True, False, True, True]) & (not empty)},
              ecut_mask=np.array([True, True, False, True]),
              eft_coeffs=np.ones((4, 2)) if domain == "eft" else None,
              JVM_ETA_PHI_DIAGNOSTIC_HISTOGRAMS={"jvm"},
              should_include_jet_veto_in_histogram_selection=lambda name: True,
              construct_cat_name=lambda *args, **kwargs: "2l_c",
              should_fill_jvm_eta_phi_diagnostic=lambda run3, syst, wgt: wgt == "nominal",
              get_jvm_eta_phi_event_mask=lambda mask, veto, name: mask,
              veto_map_mask=np.ones(4, dtype=bool),
              flatten_jagged_jet_eta_phi_weights=lambda values, mask, weight: (
                  values[mask], values[mask], weight[mask]),
              calculate_sm_sumw2_weights=lambda weight, eft: weight ** 2)
    if domain == "jvm":
        self._hist_axis_map[family] = ["eta", "phi"]
    prepare = next(node for node in ast.walk(process_node())
                   if isinstance(node, ast.FunctionDef) and node.name == "_prepare_axis_values")
    fill_loop = next(node for node in ast.walk(process_node())
                     if isinstance(node, ast.For) and ast.unparse(node.target) == "(dense_axis_name, dense_axis_vals)")
    exec(compile(ast.Module(body=[prepare, fill_loop], type_ignores=[]), "fill_nest", "exec"), ns)
    iteration.flush()
    return iteration.rows, histograms


@pytest.mark.parametrize("domain", ["scalar", "eft", "jvm"])
def test_reached_tuple_counters_and_fill_sites(telemetry, capsys, tmp_path, domain):
    counters, histograms = execute_fill_nest(telemetry, domain=domain)
    rows = persist_stdout(telemetry, capsys, tmp_path)
    family = "jvm" if domain == "jvm" else "x"
    nominal, missing, fake = [counters[(family, label, "2l")] for label in ("nominal", "missing", "fake")]
    for row in rows:
        for name in telemetry["_IterationTelemetry"].domains:
            assert row[name + "_attempted"] >= row[name + "_applicable"] >= row[name + "_nonempty"]
    assert missing["missing_weight_skips"] == 1
    assert sum(missing[name + "_attempted"] for name in telemetry["_IterationTelemetry"].domains) == 0
    assert fake["sr_fake_skips"] == 1
    assert nominal[domain + "_calls"] == len(histograms[family].calls) - fake[domain + "_calls"]
    assert nominal[domain + "_calls"] == (4 if domain == "jvm" else 2)
    assert nominal[domain + "_entries"] == 2 * nominal[domain + "_calls"]
    assert nominal["sumw2_calls"] == (0 if domain == "jvm" else 2)
    assert fake["sumw2_calls"] == 0
    assert nominal["pre_overlap_sr_memberships"] == 2 * nominal["selected_sr_memberships"]
    assert nominal["pre_overlap_ar_memberships"] == 2 * nominal["selected_ar_memberships"]
    assert nominal["selected_sr_memberships"] == nominal["selected_ar_memberships"]
    assert all(row["syst_var"] == "nominal" and row["iteration_index"] == 0 for row in rows)


def test_inapplicable_tuples_never_fill(telemetry):
    counters, histograms = execute_fill_nest(telemetry, applicable=False)
    assert not histograms["x"].calls
    assert counters[("x", "nominal", "2l")]["scalar_attempted"] > 0
    assert counters[("x", "nominal", "2l")]["scalar_applicable"] == 0


def test_axis_entries_are_distinct_from_selected_events(telemetry):
    counters, histograms = execute_fill_nest(telemetry, drop_axis_entry=True)
    nominal = counters[("x", "nominal", "2l")]
    assert nominal["selected_sr_memberships"] == nominal["selected_ar_memberships"] == 2
    assert nominal["scalar_entries"] == nominal["sumw2_entries"] == 2
    assert all(len(call["weight"]) == 1 for call in histograms["x"].calls)


def test_flavor_break_and_empty_fill_calls(telemetry):
    split, _ = execute_fill_nest(telemetry, split=True)
    assert split[("x", "nominal", "2l")]["scalar_calls"] == 4
    empty, histograms = execute_fill_nest(telemetry, empty=True)
    nominal = empty[("x", "nominal", "2l")]
    assert nominal["scalar_calls"] == nominal["scalar_applicable"] == 2
    assert nominal["scalar_nonempty"] == nominal["scalar_entries"] == 0
    assert len(histograms["x"].calls) == 3


def test_empty_arguments_and_distinct_domains(telemetry):
    cls = telemetry["_IterationTelemetry"]
    row = dict.fromkeys(cls.fields, 0)
    for domain in cls.domains:
        row[domain + "_attempted"] += 2
        cls.filled(row, domain, (np.array([]),), np.array([]))
        cls.filled(row, domain, (np.array([1]),), np.array([1]))
        assert row[domain + "_applicable"] == row[domain + "_calls"] == 2
        assert row[domain + "_nonempty"] == row[domain + "_entries"] == 1


@pytest.fixture
def coordinator():
    ns = dict(time=time, math=__import__("math"))
    load_nodes("wq_processing_admission.py", ["_coordinator"], ns)
    return ns["_coordinator"].__new__(ns["_coordinator"])


def manager_queue(telemetry, tmp_path):
    queue = SimpleNamespace(executor=SimpleNamespace(tasks_accum_log=str(tmp_path / "tasks.log")),
                            wait=lambda timeout: None, tasks_to_accumulate=[])
    telemetry["_install_phase_telemetry_persistence"](queue)
    return queue


def test_reducer_edges_durable_exact_ids(telemetry, coordinator, tmp_path, monkeypatch):
    queue = manager_queue(telemetry, tmp_path)
    logged = []
    children = [SimpleNamespace(id=identity, fout_size=1., task_accum_log=lambda *args: logged.append(args))
                for identity in (991, 47)]
    queue.tasks_to_accumulate = children[:]
    queue.executor.treereduction = 2
    queue.stats_coffea = dict(events_processed=2, events_total=2, accumulations_submitted=0)
    queue.submit = lambda task: setattr(task, "id", 8301)
    runtime = SimpleNamespace(early_terminate=False,
                              AccumCoffeaWQTask=lambda *args: SimpleNamespace(id=None))
    monkeypatch.setitem(sys.modules, "coffea", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "coffea.processor", SimpleNamespace(work_queue_tools=runtime))
    monkeypatch.setitem(sys.modules, "coffea.processor.work_queue_tools", runtime)
    coordinator.queue = queue
    coordinator.config = SimpleNamespace(pass_maximum=2, pass_seconds=1)
    coordinator.finalizing = True
    coordinator._native_states = lambda: {}
    coordinator._measurement = lambda: True
    coordinator.output_estimate_bytes = 1
    coordinator.free_inodes = 10000
    coordinator.available_bytes = 10**9
    coordinator._predicted_additional = lambda counts: 0
    coordinator.min_free_bytes = 0
    coordinator.submit_accumulation(None, "accumulator.py")
    rows = [json.loads(line) for line in (tmp_path / "phase_telemetry.jsonl").read_text().splitlines()]
    assert [(row["reducer_task_id"], row["child_task_id"]) for row in rows] == [(8301, 991), (8301, 47)]
    assert all("parent" not in row for row in rows)
    assert logged == [(queue.executor.tasks_accum_log, "done", 991),
                      (queue.executor.tasks_accum_log, "done", 47)]


@pytest.mark.parametrize("raised", [False, True])
def test_manager_final_and_runner_return(telemetry, coordinator, tmp_path, raised):
    queue = manager_queue(telemetry, tmp_path)
    queue.tasks_to_accumulate = [SimpleNamespace(id=8301)]
    coordinator.queue = queue

    def original(accumulator):
        if raised:
            raise ValueError("synthetic")
        return accumulator

    if raised:
        with pytest.raises(ValueError, match="synthetic"):
            coordinator._local_final_accumulation(original, 17)
    else:
        assert coordinator._local_final_accumulation(original, 17) == 17
        queue._phase_telemetry_record("runner_return", outcome="returned")
    rows = [json.loads(line) for line in (tmp_path / "phase_telemetry.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["manager_final_accumulation_before",
        "manager_final_accumulation_after"] + ([] if raised else ["runner_return"])
    assert rows[1]["outcome"] == ("raised" if raised else "returned")
    assert all(row["root_task_id"] == 8301 and row["clock_domain"] == "manager_local" for row in rows)
    assert len({row["manager_attempt_id"] for row in rows}) == 1
    assert all(row["manager_pid"] == os.getpid() for row in rows)
    assert [row["wall_ns"] for row in rows] == sorted(row["wall_ns"] for row in rows)
    source = (SOURCE / "run_analysis.py").read_text()
    boundary = source.index('output = runner(flist, treename, processor_instance)')
    assert boundary < source.index('("runner_return", outcome="returned")', boundary) < source.index('diagnostic_phase = "runner_result_inspection"', boundary)


def test_archive_surface_preserved():
    source = (SOURCE / "run_cr.sh").read_text()
    archive = source.split("archive_native_wq_logs() {", 1)[1].split("\n}\n", 1)[0]
    regular, optional = archive.split(
        'source_path="${native_wq_log_dir}/phase_telemetry.jsonl"', 1)
    # Regular native logs still require a checked copy before source removal.
    assert 'for name in "${native_wq_log_names[@]}"; do' in regular
    assert 'if ! cp -- "${source_path}" "${destination_path}"; then' in regular
    assert '"${source_size}" != "${destination_size}"' in regular
    assert '[[ ! -f "${destination_path}"' in regular
    # Behavioral coverage stays in test_telemetry_persistence.archive_shell:
    # attempt separation, inode/readability, failure retention and archive_logs.
    assert "os.rename(source, target)" in optional
    assert "os.symlink(os.path.relpath(target, archive), link)" in optional
    assert 'with link.open("rb"):' in optional
    assert "WARNING: optional phase telemetry archive failed" in optional
    assert not re.search(r"\b(?:cp|cmp|sha256sum)\b", optional)
    assert 'production_state_tool archive ' in optional
    assert 'rm -- "${source_path}"' in optional
