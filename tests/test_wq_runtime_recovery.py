"""Bounded loopback qualification of processing, reduction, and monitor retry."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from coffea.processor.executor import WorkItem, WorkQueueExecutor, _compress, _compression_wrapper
from coffea.processor.work_queue_tools import CoffeaWQ, ProcCoffeaWQTask

import pytest

from ndcctools import work_queue as wq
from ndcctools.work_queue import Task, WorkQueue
from analysis.topeft_run2.wq_processing_admission import install_processing_admission
from analysis.topeft_run2 import wq_processing_admission as admission


def test_real_coffea_exhaustion_drains_deferred_groups(tmp_path, monkeypatch):
    """Use the installed Coffea event loop, queue and reducer task classes."""
    def tiny_coffea_result(item):
        return {"value": item.entrystart + 1, "processed": {item}}
    executor = WorkQueueExecutor(
        manager_name="loopback-coffea-drain", port=0, filepath=str(tmp_path),
        stats_log=str(tmp_path / "stats.log"),
        tasks_accum_log=str(tmp_path / "accum.log"),
        events_total=1000, treereduction=2, chunksize=1,
        status=False, verbose=False, resource_monitor="off",
        resources_mode="fixed", cores=1, memory=64, disk=64,
        x509_proxy=False)
    queue = CoffeaWQ(executor)
    release_at = [None]
    pressure_observations = []
    final_pause_observations = []
    coordinator_ref = []

    def space(*args):
        now = time.monotonic()
        finalizing = bool(coordinator_ref and coordinator_ref[0].finalizing)
        if finalizing and release_at[0] is None:
            release_at[0] = now + 1.5
        pressure = bool(queue.stats_coffea["events_processed"]) and (
            release_at[0] is None or now < release_at[0])
        pressure_observations.append((queue.stats_coffea["events_processed"], pressure))
        if coordinator_ref:
            final_pause_observations.append((
                coordinator_ref[0].finalizing, queue.empty(), pressure,
                queue.stats_coffea["events_processed"]))
        return (0 if pressure else 1_000_000_000), 10**7

    monkeypatch.setattr(admission, "_space", space)
    coordinator = install_processing_admission(
        queue, base=str(tmp_path), quota_root=str(tmp_path),
        warning_bytes=0, min_free_bytes=0,
        config=admission.admission_config(
            measurement_seconds=0.05, measurement_max_age_seconds=1,
            output_bootstrap_bytes=1024))
    coordinator_ref.append(coordinator)
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    with (tmp_path / "worker.stdout").open("w") as out, \
            (tmp_path / "worker.stderr").open("w") as err:
        worker = subprocess.Popen([
            os.environ.get("WQ_TEST_WORKER_ELF",
                           "/users/apiccine/work/miniconda3/envs/clib-env/bin/work_queue_worker"),
            "--cores=1", "--memory=512", "--disk=512",
            "--connect-timeout=10", "--idle-timeout=30",
            "--workdir", str(worker_dir), "127.0.0.1", str(queue.port),
        ], stdout=out, stderr=err)
        try:
            items = [WorkItem("tiny", "/dev/null", "tree", index, index + 1,
                              "tiny-uuid") for index in range(4)]
            result = queue._processing(items, tiny_coffea_result,
                                       {"value": 0, "processed": set()})
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
    assert result["value"] == 10
    assert result["processed"] == set(items)
    assert queue._items_empty
    assert coordinator.reductions_done == 3
    assert queue.stats_coffea["accumulations_submitted"] == 3
    assert any(pressure for _, pressure in pressure_observations)
    assert any(not pressure for _, pressure in pressure_observations[1:])
    assert any(finalizing and empty and pressure and processed == 4
               for finalizing, empty, pressure, processed in final_pause_observations)


def test_real_proc_multi_child_split_bypasses_ready_admission(tmp_path, monkeypatch):
    """An exhausted ProcCoffeaWQTask creates native children with exact coverage."""
    def tiny_result(item):
        return {"value": sum(range(item.entrystart + 1, item.entrystop + 1)),
                "processed": {item}}

    monkeypatch.setattr(admission, "_space", lambda *args: (1_000_000_000, 10**7))
    executor = WorkQueueExecutor(
        manager_name="loopback-coffea-split", port=0, filepath=str(tmp_path),
        stats_log=str(tmp_path / "stats.log"),
        tasks_accum_log=str(tmp_path / "accum.log"),
        events_total=8, treereduction=10, chunksize=2,
        status=False, verbose=False, resource_monitor="off",
        resources_mode="fixed", cores=1, memory=64, disk=64,
        x509_proxy=False)
    queue = CoffeaWQ(executor)
    coordinator = install_processing_admission(
        queue, base=str(tmp_path), quota_root=str(tmp_path),
        warning_bytes=0, min_free_bytes=0)
    queue.chunksize_current = 2
    item = WorkItem("tiny", "/dev/null", "tree", 0, 8, "tiny-uuid")
    queue.known_workitems.add(item)
    infile_fn = queue.function_to_file(
        _compression_wrapper(executor.compression, tiny_result), "proc")
    exhausted = ProcCoffeaWQTask(queue, infile_fn, item)
    exhausted._task.result = wq.WORK_QUEUE_RESULT_RESOURCE_EXHAUSTION
    exhausted.resubmit(queue)
    children = list(queue._task_table.values())
    assert len(children) == 4
    assert coordinator.new_admitted == 0
    assert coordinator.recovery_submissions == 4
    assert [(task.item.entrystart, task.item.entrystop) for task in children] == [
        (0, 2), (2, 4), (4, 6), (6, 8)]
    assert queue.known_workitems == {task.item for task in children}
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    with (tmp_path / "worker.stdout").open("w") as out, \
            (tmp_path / "worker.stderr").open("w") as err:
        worker = subprocess.Popen([
            os.environ.get("WQ_TEST_WORKER_ELF",
                           "/users/apiccine/work/miniconda3/envs/clib-env/bin/work_queue_worker"),
            "--cores=1", "--memory=512", "--disk=512",
            "--connect-timeout=10", "--idle-timeout=30",
            "--workdir", str(worker_dir), "127.0.0.1", str(queue.port),
        ], stdout=out, stderr=err)
        try:
            completed = []
            deadline = time.monotonic() + 60
            while len(completed) < 4 and time.monotonic() < deadline:
                task = queue.wait(2)
                if task is not None:
                    assert task.successful(), (task.result, task.return_status)
                    completed.append(task)
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
    assert len(completed) == 4
    assert sum(task.output["value"] for task in completed) == 36
    assert set().union(*(task.output["processed"] for task in completed)) == \
        queue.known_workitems


def test_real_coffea_early_termination_uses_local_final_merge(
    tmp_path, monkeypatch
):
    from coffea.processor import work_queue_tools

    executor = WorkQueueExecutor(
        manager_name="loopback-coffea-early", port=0, filepath=str(tmp_path),
        stats_log=str(tmp_path / "stats.log"),
        tasks_accum_log=None,
        events_total=1000, treereduction=2, chunksize=1,
        status=False, verbose=False, resource_monitor="off",
        resources_mode="fixed", cores=1, memory=64, disk=64,
        x509_proxy=False)
    queue = CoffeaWQ(executor)
    coordinator = install_processing_admission(
        queue, base=str(tmp_path), quota_root=str(tmp_path),
        warning_bytes=0, min_free_bytes=0)
    infile_fn = queue.function_to_file(lambda item: item, "proc")
    items = [WorkItem("tiny", "/dev/null", "tree", index, index + 1,
                      "tiny-uuid") for index in range(2)]
    for index, item in enumerate(items):
        task = ProcCoffeaWQTask(queue, infile_fn, item)
        Path(task.outfile_output).write_bytes(_compress(
            {"value": index + 1, "processed": {item}}, executor.compression))
        queue.tasks_to_accumulate.append(task)
    queue.stats_coffea["events_processed"] = 2
    queue.stats_coffea["events_total"] = 1000
    monkeypatch.setattr(work_queue_tools, "early_terminate", True)
    result = queue._final_accumulation({"value": 0, "processed": set()})
    assert result["value"] == 3
    assert result["processed"] == set(items)
    assert not coordinator.finalizing
    assert queue.stats_coffea["accumulations_submitted"] == 0


@pytest.mark.parametrize("mode,count", [
    ("healthy", 4), ("stale", 1), ("once", 4), ("repeated", 1),
])
def test_real_coffea_patched_monitor_result_and_retry(
    tmp_path, monkeypatch, mode, count,
):
    """The loaded patched binding, real Coffea loop and built worker agree."""
    from coffea.processor import work_queue_tools
    from ndcctools import _cwork_queue

    binding = os.environ.get("WQ_TEST_BINDING_SO")
    worker_elf = os.environ.get("WQ_TEST_WORKER_ELF")
    monitors = {
        "healthy": os.environ.get("WQ_TEST_MONITOR_NORMAL_ELF"),
        "stale": os.environ.get("WQ_TEST_MONITOR_STALE_ELF"),
        "once": os.environ.get("WQ_TEST_MONITOR_ONCE_ELF"),
        "repeated": os.environ.get("WQ_TEST_MONITOR_STALE_ELF"),
    }
    if not binding or not worker_elf or not all(monitors.values()):
        pytest.skip("matching private manager binding, worker and monitors required")
    assert Path(_cwork_queue.__file__).resolve() == Path(binding).resolve()
    assert Path(worker_elf).is_file()
    monitor = monitors[mode]
    assert Path(monitor).is_file()
    monkeypatch.setattr(work_queue_tools, "early_terminate", False)
    monkeypatch.setattr(admission, "_space", lambda *args:
                        (1_000_000_000, 10**7))
    monkeypatch.setenv("CCTOOLS_RESOURCE_MONITOR", monitor)
    if mode == "once":
        monkeypatch.setenv("CCTOOLS_TEST_FATAL_ONCE_MARKER",
                           str(tmp_path / "fault_once.marker"))

    def tiny_result(item):
        return {"value": item.entrystart + 1, "processed": {item}}

    executor = WorkQueueExecutor(
        manager_name="private-coffea-%s" % mode, port=0,
        filepath=str(tmp_path), stats_log=str(tmp_path / "stats.log"),
        tasks_accum_log=str(tmp_path / "accum.log"),
        events_total=count, treereduction=2, chunksize=1,
        status=False, verbose=False, resource_monitor="measure",
        resources_mode="fixed", cores=1, memory=64, disk=64,
        retries=(0 if mode == "stale" else 1), x509_proxy=False)
    queue = CoffeaWQ(executor)
    monkeypatch.setattr(work_queue_tools, "_wq_queue", queue)
    coordinator = install_processing_admission(
        queue, base=str(tmp_path), quota_root=str(tmp_path),
        warning_bytes=0, min_free_bytes=0,
        config=admission.admission_config(output_bootstrap_bytes=1024))
    original_wait = queue.wait
    attempts = []

    def observed_wait(timeout=None):
        task = original_wait(timeout)
        if task is not None:
            attempts.append({
                "category": task.category,
                "result": task.result,
                "return_status": task.return_status,
                "successful": task.successful(),
                "summary_exit_status": (task.resources_measured.exit_status
                                        if task.resources_measured else None),
                "output_present": Path(task.outfile_output).is_file(),
                "stdout_present": Path(task.outfile_stdout).is_file(),
                "logical_range": ([task.item.entrystart, task.item.entrystop]
                                  if task.category == "processing" else None),
            })
        return task

    queue.wait = observed_wait
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    with (tmp_path / "worker.stdout").open("w") as out, \
            (tmp_path / "worker.stderr").open("w") as err:
        environment = os.environ.copy()
        if mode in ("stale", "repeated"):
            environment["LD_PRELOAD"] = os.environ["WQ_TEST_UNLINK_PRELOAD"]
        worker = subprocess.Popen([
            worker_elf, "--cores=1", "--memory=512", "--disk=512",
            "--connect-timeout=10", "--idle-timeout=30",
            "--workdir", str(worker_dir), "127.0.0.1", str(queue.port),
        ], stdout=out, stderr=err, env=environment)
        try:
            items = [WorkItem("tiny", "/dev/null", "tree", index, index + 1,
                              "tiny-uuid") for index in range(count)]
            result = queue._processing(items, tiny_result,
                                       {"value": 0, "processed": set()})
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
    processing_attempts = [item for item in attempts
                           if item["category"] == "processing"]
    failures = [item for item in processing_attempts if not item["successful"]]
    successes = [item for item in processing_attempts if item["successful"]]
    if mode in ("healthy", "once"):
        assert result["value"] == sum(range(1, count + 1))
        assert result["processed"] == set(items)
        assert "unprocessed" not in result or not result["unprocessed"]
        assert len(successes) == count
        assert len(failures) == (1 if mode == "once" else 0)
        assert coordinator.recovery_submissions == len(failures)
    else:
        assert result["value"] == 0
        assert result["processed"] == set()
        assert result["unprocessed"] == set(items)
        assert len(failures) == (1 if mode == "stale" else 2)
        assert len(successes) == 0
        assert all(item["result"] == wq.WORK_QUEUE_RESULT_RMONITOR_ERROR
                   and item["return_status"] != 0
                   and item["summary_exit_status"] == 0
                   and item["output_present"] and item["stdout_present"]
                   for item in failures)
    record = {"mode": mode, "binding": _cwork_queue.__file__,
              "worker": worker_elf, "monitor": monitor,
              "attempts": attempts, "value": result["value"],
              "processed": len(result["processed"]),
              "unprocessed": len(result.get("unprocessed", set())),
              "recovery_submissions": coordinator.recovery_submissions}
    (tmp_path / "combined_result.json").write_text(json.dumps(record, sort_keys=True))
    print(json.dumps(record, sort_keys=True))


@pytest.mark.parametrize("initial_backlog,replenish", [(161, False), (32, True)])
def test_sustained_bounded_backlog_advances_both_categories(
    tmp_path, monkeypatch, initial_backlog, replenish
):
    monkeypatch.setattr(admission, "_space", lambda *args: (1_000_000_000_000, 10**7))
    root = tmp_path / "backlog"
    root.mkdir()
    queue = synthetic_queue(root, "arbitrary-backlog-label")
    coordinator = install_processing_admission(
        queue, base=str(root), quota_root=str(root),
        warning_bytes=0, min_free_bytes=0)
    for index in range(initial_backlog):
        queue.submit(synthetic_task(
            "/bin/printf '0\\n' > value.txt", (), "accumulating",
            root / ("backlog_%03d.txt" % index)))
    queue._submit_processing_tasks(None, input_items())
    assert coordinator.new_admitted == 10
    worker_dir = root / "worker"
    worker_dir.mkdir()
    with (root / "worker.stdout").open("w") as out, \
            (root / "worker.stderr").open("w") as err:
        worker = subprocess.Popen([
            os.environ.get("WQ_TEST_WORKER_ELF",
                           "/users/apiccine/work/miniconda3/envs/clib-env/bin/work_queue_worker"),
            "--cores=1", "--memory=512", "--disk=512",
            "--connect-timeout=10", "--idle-timeout=30",
            "--workdir", str(worker_dir), "127.0.0.1", str(queue.port),
        ], stdout=out, stderr=err)
        try:
            processing_done = reductions_done = 0
            replenished = 0
            first_processing_after = None
            deadline = time.monotonic() + 60
            while (processing_done < 10 or reductions_done < 20) and \
                    time.monotonic() < deadline:
                task = queue.wait(1)
                if task is None:
                    continue
                assert task.successful(), (task.result, task.return_status)
                if task.category == "processing":
                    processing_done += 1
                    if first_processing_after is None:
                        first_processing_after = reductions_done
                else:
                    reductions_done += 1
                    if replenish and replenished < 64:
                        queue.submit(synthetic_task(
                            "/bin/printf '0\\n' > value.txt", (),
                            "accumulating",
                            root / ("refill_%03d.txt" % replenished)))
                        replenished += 1
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
    assert processing_done == 10
    assert reductions_done >= 20
    if replenish:
        assert replenished >= 20
    else:
        assert first_processing_after is not None
        assert first_processing_after < initial_backlog


def test_two_category_replenishment_beyond_priority_burst(tmp_path, monkeypatch):
    """Both backlogs progress with bounded priority use and two workers."""
    monkeypatch.setattr(admission, "_space", lambda *args:
                        (1_000_000_000_000, 10**7))
    worker_elf = os.environ.get(
        "WQ_TEST_WORKER_ELF",
        "/users/apiccine/work/miniconda3/envs/clib-env/bin/work_queue_worker")
    root = tmp_path / "both_categories"
    root.mkdir()
    queue = synthetic_queue(root, "arbitrary-two-category-label")
    queue.stats_coffea["events_total"] = 60
    queue.executor.memory = 64
    queue.executor.disk = 64
    queue.executor.cores = 1
    queue.specify_category_max_resources(
        "processing", {"cores": 1, "memory": 64, "disk": 64})
    queue.specify_category_min_resources(
        "accumulating", {"cores": 1, "memory": 128, "disk": 64})
    coordinator = install_processing_admission(
        queue, base=str(root), quota_root=str(root),
        warning_bytes=0, min_free_bytes=0,
        accumulating_memory_mb=128, accumulating_cores=1,
        config=admission.admission_config(priority_window_seconds=0.25))
    for index in range(20):
        queue.submit(synthetic_task(
            "/bin/printf '0\\n' > value.txt", (), "accumulating",
            root / ("reduction_%03d.txt" % index)))
    submitted_reducers = 20
    items = ((index,) for index in range(1, 61))
    workers = []
    try:
        for index, memory in enumerate((96, 256)):
            worker_dir = root / ("worker_%d" % index)
            worker_dir.mkdir()
            out = (root / ("worker_%d.stdout" % index)).open("w")
            err = (root / ("worker_%d.stderr" % index)).open("w")
            worker = subprocess.Popen([
                worker_elf,
                "--cores=1", "--memory=%d" % memory, "--disk=512",
                "--connect-timeout=10", "--idle-timeout=90",
                "--workdir", str(worker_dir), "127.0.0.1", str(queue.port),
            ], stdout=out, stderr=err)
            workers.append((worker, out, err))
        completed = {"processing": 0, "accumulating": 0}
        checkpoints = []
        first_completion = {}
        observed_capacity_pair = False
        windows_seen = {coordinator.priority_window_at}
        started = time.monotonic()
        while min(completed.values()) < 60 and time.monotonic() - started < 90:
            queue._submit_processing_tasks(None, items)
            windows_seen.add(coordinator.priority_window_at)
            task = queue.wait(1)
            summary = queue.workers_summary()
            if sum(row["workers"] for row in summary) == 2:
                assert coordinator._capacity()[0] == 2
                assert coordinator._reducer_capacity(summary) == 1
                observed_capacity_pair = True
            if task is None:
                continue
            assert task.successful(), (task.category, task.result,
                                       task.return_status)
            category = task.category
            completed[category] += 1
            first_completion.setdefault(category, time.monotonic() - started)
            if category == "accumulating" and submitted_reducers < 60:
                queue.submit(synthetic_task(
                    "/bin/printf '0\\n' > value.txt", (), "accumulating",
                    root / ("reduction_%03d.txt" % submitted_reducers)))
                submitted_reducers += 1
            if sum(completed.values()) % 20 == 0:
                checkpoints.append(tuple(completed.values()))
        assert completed == {"processing": 60, "accumulating": 60}
        assert observed_capacity_pair
        assert coordinator.new_admitted == 60
        assert coordinator.priority_used <= coordinator.config.priority_burst
        assert first_completion["processing"] < 30
        assert first_completion["accumulating"] < 30
        assert any(processing > 32 and accumulating > 0
                   for processing, accumulating in checkpoints)
        assert any(processing > 0 and accumulating > 32
                   for processing, accumulating in checkpoints)
        dispatch_record = {
            "completed": completed, "first_completion_seconds": first_completion,
            "checkpoints": checkpoints, "worker": worker_elf,
            "processing_capacity": 2, "reducer_capacity": 1,
            "scaled_priority_window_seconds": coordinator.config.priority_window_seconds,
            "default_priority_window_seconds": admission.admission_config().priority_window_seconds,
            "priority_windows_seen": len(windows_seen),
            "elapsed_seconds": time.monotonic() - started,
        }
        (root / "dispatch_result.json").write_text(
            json.dumps(dispatch_record, sort_keys=True))
        print(json.dumps(dispatch_record, sort_keys=True))
    finally:
        for worker, out, err in workers:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            out.close()
            err.close()


def test_native_proportional_profiles_match_connected_capacity(
    tmp_path, monkeypatch,
):
    """One-core scaled worker profiles both run work despite largest allocation."""
    from ndcctools import _cwork_queue

    binding = os.environ.get("WQ_TEST_BINDING_SO")
    worker_elf = os.environ.get("WQ_TEST_WORKER_ELF")
    if not binding or not worker_elf:
        pytest.skip("private native binding and worker required")
    assert Path(_cwork_queue.__file__).resolve() == Path(binding).resolve()
    monkeypatch.setattr(admission, "_space", lambda *args:
                        (1_000_000_000, 10**7))
    root = tmp_path / "proportional"
    root.mkdir()
    queue = synthetic_queue(root, "arbitrary-proportional-label")
    queue.executor.cores = 1
    queue.executor.memory = None
    queue.executor.disk = None
    queue.tune("force-proportional-resources", 1)
    queue.specify_category_max_resources("processing", {"cores": 1})
    coordinator = install_processing_admission(
        queue, base=str(root), quota_root=str(root),
        warning_bytes=0, min_free_bytes=0)
    workers = []
    try:
        for index, (memory, disk) in enumerate(((63, 150), (192, 250))):
            worker_dir = root / ("worker_%d" % index)
            worker_dir.mkdir()
            out = (root / ("worker_%d.stdout" % index)).open("w")
            err = (root / ("worker_%d.stderr" % index)).open("w")
            worker = subprocess.Popen([
                worker_elf, "--cores=1", "--memory=%d" % memory,
                "--disk=%d" % disk, "--connect-timeout=10",
                "--idle-timeout=30", "--workdir", str(worker_dir),
                "127.0.0.1", str(queue.port),
            ], stdout=out, stderr=err)
            workers.append((worker, out, err))
        for index in range(2):
            queue.submit(synthetic_task(
                "/bin/sh -c 'echo %d > value.txt; sleep 1'" % index,
                (index,), "processing", root / ("output_%d.txt" % index)))
        completed = []
        capacity_seen = False
        first_allocation = None
        deadline = time.monotonic() + 45
        while len(completed) < 2 and time.monotonic() < deadline:
            summary = queue.workers_summary()
            if sum(row["workers"] for row in summary) == 2:
                assert coordinator._capacity()[0] == 2
                categories = queue.status("categories")
                processing = next(row for row in categories
                                  if row["category"] == "processing")
                first_allocation = processing["first_allocation"]
                capacity_seen = True
            task = queue.wait(1)
            if task is not None:
                assert task.successful(), (task.result, task.return_status)
                completed.append(task)
        assert capacity_seen
        assert first_allocation["memory"] > 63
        assert len(completed) == 2
        assert len({task.host for task in completed}) == 2
        assert sorted(task.resources_allocated.memory for task in completed) == [63, 192]
        capacity_record = {
            "binding": _cwork_queue.__file__, "worker": worker_elf,
            "capacity": coordinator._capacity()[0],
            "quality": coordinator.last_capacity_quality,
            "largest_worker_first_allocation": first_allocation,
            "assigned_memory_mb": sorted(
                task.resources_allocated.memory for task in completed),
            "distinct_worker_addresses": len({task.host for task in completed}),
        }
        (root / "capacity_result.json").write_text(
            json.dumps(capacity_record, sort_keys=True))
        print(json.dumps(capacity_record, sort_keys=True))
    finally:
        for worker, out, err in workers:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            out.close()
            err.close()


class synthetic_task(Task):
    def __init__(self, command, logical_ids, category, path):
        super().__init__(command)
        self.logical_ids = tuple(logical_ids)
        self.specify_category(category)
        self.outfile_output = str(path)
        self.specify_output_file(str(path), "value.txt")
        self.fout_size = 0.000001
        self.fin_size = 0.000001
        self.tasks_to_accumulate = []

    def successful(self):
        return self.result == 0 and self.return_status == 0

    def task_accum_log(self, *args):
        pass


def synthetic_accum_task(queue, infile_fn, group):
    queue.serial += 1
    output = queue.root / ("value_%03d.txt" % queue.serial)
    task = synthetic_task(
        "/usr/bin/awk '{s+=$1} END{printf \"%d\\n\",s}' in0.txt in1.txt > value.txt",
        group[0].logical_ids + group[1].logical_ids,
        "accumulating", output)
    task.tasks_to_accumulate = group
    task.fin_size = sum(child.fout_size for child in group)
    for index, child in enumerate(group):
        task.specify_input_file(child.outfile_output, "in%d.txt" % index,
                                cache=False)
    return task


class synthetic_queue(WorkQueue):
    def __init__(self, root, name):
        super().__init__(port=0)
        self.root = root
        self.executor = SimpleNamespace(stats_log=str(root / "stats.log"),
                                        manager_name=name, desc="Processing",
                                        treereduction=2,
                                        tasks_accum_log=str(root / "accum.log"))
        self.stats_coffea = collections.defaultdict(int)
        self.stats_coffea["events_total"] = 20
        self.tasks_to_accumulate = []
        self.chunksize_current = 1
        self._items_empty = False
        self.serial = 0
        self.submitted = []

    def make_processing(self, logical_id, retry=False):
        self.serial += 1
        path = self.root / ("value_%03d.txt" % self.serial)
        task = synthetic_task("/bin/printf '%d\\n' > value.txt" % logical_id,
                              (logical_id,), "processing", path)
        self.submit(task)
        self.submitted.append((logical_id, retry, task.id))
        if not retry:
            self.stats_coffea["events_submitted"] += 1
        return task

    def _submit_processing_task(self, infile_procc_fn, item):
        return self.make_processing(item[0])

    def _submit_processing_tasks(self, infile_procc_fn, items):
        raise RuntimeError("adapter missing")

    def _submit_accum_tasks(self, infile_accum_fn):
        force = self.stats_coffea["events_processed"] >= self.stats_coffea["events_total"]
        if len(self.tasks_to_accumulate) < (2 if force else 3):
            return
        self.tasks_to_accumulate.sort(key=lambda task: task.fout_size)
        while len(self.tasks_to_accumulate) >= (2 if force else 3):
            group = self.tasks_to_accumulate[:2]
            self.serial += 1
            output = self.root / ("value_%03d.txt" % self.serial)
            task = synthetic_task("/usr/bin/awk '{s+=$1} END{printf \"%d\\n\",s}' in0.txt in1.txt > value.txt",
                                  group[0].logical_ids + group[1].logical_ids,
                                  "accumulating", output)
            task.tasks_to_accumulate = group
            for index, child in enumerate(group):
                task.specify_input_file(child.outfile_output, "in%d.txt" % index,
                                        cache=False)
            self.submit(task)
            del self.tasks_to_accumulate[:2]
            self.stats_coffea["accumulations_submitted"] += 1

    def _final_accumulation(self, accumulator):
        return accumulator

    def wait(self, timeout=None):
        task = super().wait(timeout)
        if task and task.successful():
            task.fout_size = os.path.getsize(task.outfile_output) / 1_000_000
        return task


def input_items():
    for logical_id in range(1, 21):
        yield (logical_id,)


def main():
    root = Path(sys.argv[1]).resolve()
    monitor = Path(sys.argv[2]).resolve()
    mode = sys.argv[3]
    root.mkdir(mode=0o700)
    os.environ["CCTOOLS_RESOURCE_MONITOR"] = str(monitor)
    if mode == "fault_once":
        os.environ["CCTOOLS_TEST_FATAL_ONCE_MARKER"] = str(root / "once.marker")
    queue = synthetic_queue(root, "arbitrary-identity-" + mode)
    assert queue.enable_monitoring(dirname=str(root), watchdog=False) == 1
    coordinator = install_processing_admission(
        queue, base=str(root), quota_root=None,
        warning_bytes=0, min_free_bytes=0, accumulating_memory_mb=1,
        accumulating_cores=1)
    workers = []
    try:
        for index in range(2):
            worker_dir = root / ("worker_%d" % index)
            worker_dir.mkdir(mode=0o700)
            out = (root / ("worker_%d.log" % index)).open("w")
            err = (root / ("worker_%d.err" % index)).open("w")
            worker = subprocess.Popen(
                [os.environ.get("WQ_TEST_WORKER_ELF",
                                "/users/apiccine/work/miniconda3/envs/clib-env/bin/work_queue_worker"),
                 "--cores=1", "--memory=512", "--disk=512", "--connect-timeout=10",
                 "--idle-timeout=60", "--workdir", str(worker_dir),
                 "127.0.0.1", str(queue.port)], stdout=out, stderr=err)
            workers.append((worker, out, err))
        items = input_items()
        completed = set()
        failed_attempts = 0
        reductions = 0
        final_value = None
        max_retained = 0
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            queue._submit_processing_tasks(None, items)
            task = queue.wait(1)
            if task:
                if task.category == "processing":
                    if task.successful():
                        assert task.logical_ids[0] not in completed
                        completed.add(task.logical_ids[0])
                        queue.stats_coffea["events_processed"] += 1
                        queue.tasks_to_accumulate.append(task)
                    else:
                        failed_attempts += 1
                        assert failed_attempts <= 2
                        queue.make_processing(task.logical_ids[0], retry=True)
                elif task.category == "accumulating":
                    assert task.successful(), (task.result_str, task.return_status)
                    reductions += 1
                    for child in task.tasks_to_accumulate:
                        Path(child.outfile_output).unlink()
                    queue.tasks_to_accumulate.append(task)
            queue._submit_accum_tasks(None)
            max_retained = max(max_retained, sum(path.stat().st_size for path in root.glob("value_*.txt")))
            if (len(completed) == 20 and len(queue.tasks_to_accumulate) == 1 and
                    queue.empty()):
                final_value = int(Path(queue.tasks_to_accumulate[0].outfile_output).read_text())
                break
        result = {"mode": mode, "completed_logical_ids": len(completed),
                  "aggregate": final_value, "expected": sum(range(1, 21)),
                  "failed_attempts": failed_attempts, "reductions": reductions,
                  "new_submissions": coordinator.new_admitted,
                  "recovery_submissions": coordinator.recovery_submissions,
                  "max_retained_bytes": max_retained,
                  "monitor_sha256": hashlib.sha256(monitor.read_bytes()).hexdigest(),
                  "telemetry_bytes": (root / "admission.jsonl").stat().st_size
                  if (root / "admission.jsonl").exists() else 0}
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, sort_keys=True))
        assert final_value == 210 and len(completed) == 20
        assert coordinator.new_admitted == 20 and reductions > 0
        assert max_retained < 128 * 1024 * 1024
        if mode == "fault_once":
            assert failed_attempts == 1 and coordinator.recovery_submissions == 1
        else:
            assert failed_attempts == 0
    finally:
        for worker, out, err in workers:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            out.close()
            err.close()


@pytest.mark.parametrize(
    "mode,monitor_key",
    [("normal", "WQ_TEST_MONITOR_ELF"),
     ("fault_once", "WQ_TEST_MONITOR_FAULT_ONCE_ELF")],
)
def test_mixed_work_and_monitor_recovery(tmp_path, monkeypatch, mode, monitor_key):
    monitor = os.environ.get(monitor_key)
    if not monitor:
        pytest.skip("an isolated monitor ELF is required")
    root = tmp_path / mode
    from coffea.processor import work_queue_tools
    monkeypatch.setattr(work_queue_tools, "AccumCoffeaWQTask", synthetic_accum_task)
    monkeypatch.setattr(admission, "_space", lambda *args:
                        (1_000_000_000_000, 10**7))
    monkeypatch.setenv("CCTOOLS_RESOURCE_MONITOR", monitor)
    if mode == "fault_once":
        monkeypatch.setenv("CCTOOLS_TEST_FATAL_ONCE_MARKER", str(root / "once.marker"))
    monkeypatch.setattr(sys, "argv", [sys.argv[0], str(root), monitor, mode])
    main()
