"""Focused fake-state contract checks for queue-local Work Queue admission."""

from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from ndcctools import work_queue as wq

from analysis.topeft_run2 import wq_processing_admission as admission
from analysis.topeft_run2 import wq_staging_guard as staging_guard


@pytest.mark.parametrize("visibility", ("unrelated_opaque", "bound_opaque", "manager_opaque",
                                        "observed_fd", "observed_cwd", "self_argv"))
def test_reader_scan_requires_invocation_relevance(tmp_path, monkeypatch, capsys, visibility):
    root = tmp_path / "exact_invocation"
    proc = tmp_path / (str(os.getpid()) if visibility == "self_argv" else "12345")
    proc.mkdir()
    (proc / "fd").mkdir()
    (proc / "cmdline").write_bytes(
        b"python\0--wq-invocation-id\0" + root.name.encode() + b"\0"
        if visibility in ("bound_opaque", "self_argv") else b"python\0--wq-manager-name\0exact-manager-a1\0"
        if visibility == "manager_opaque" else b"sshd\0session\0")
    (proc / "cwd").symlink_to(root if visibility == "observed_cwd" else tmp_path)
    (proc / "root").symlink_to(Path("/"))
    if visibility == "observed_fd":
        (proc / "fd" / "4").symlink_to(root / "nested" / "accum.p")
    original = Path.iterdir
    def iterdir(path):
        if path == Path("/proc"):
            return iter([proc])
        if path == proc / "fd" and visibility.endswith("opaque"):
            raise PermissionError("unreadable SSH/SFTP descriptors")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert staging_guard._local_readers(root, relevant_tokens=("exact-manager-a1",)) == (
        visibility not in ("unrelated_opaque", "self_argv"))
    if visibility == "unrelated_opaque":
        assert "WARNING" in capsys.readouterr().out


def test_root_backed_staging_uses_concrete_path_checks(monkeypatch):
    base = Path("/synthetic-root-backed/selected")
    fake_stat = SimpleNamespace(st_uid=os.getuid(), st_dev=1)
    with monkeypatch.context() as fixture:
        fixture.setattr(staging_guard.Path, "is_dir", lambda _: True)
        fixture.setattr(staging_guard.Path, "is_symlink", lambda _: False)
        fixture.setattr(staging_guard.Path, "resolve", lambda path, strict=True: path)
        fixture.setattr(staging_guard.os, "stat", lambda _: fake_stat)
        fixture.setattr(staging_guard.os, "access", lambda *_: True)
        assert staging_guard._validate_base(base) == base
        with pytest.raises(staging_guard.staging_guard_error, match="filesystem root"):
            staging_guard._validate_base(Path("/"))
        with pytest.raises(staging_guard.staging_guard_error, match="root temporary staging"):
            staging_guard._validate_base(Path("/tmp/selected"))
        with pytest.raises(staging_guard.staging_guard_error, match="root temporary staging"):
            staging_guard._validate_base(Path("/var/tmp/selected"))
        fixture.setattr(staging_guard.os, "stat", lambda _: SimpleNamespace(
            st_uid=os.getuid() + 1, st_dev=1))
        with pytest.raises(staging_guard.staging_guard_error, match="ownership"):
            staging_guard._validate_base(base)
        fixture.setattr(staging_guard.os, "stat", lambda _: fake_stat)
        fixture.setattr(staging_guard.Path, "is_symlink", lambda path: path == base)
        with pytest.raises(staging_guard.staging_guard_error, match="symlink"):
            staging_guard._validate_base(base)
        fixture.setattr(staging_guard.Path, "is_symlink", lambda _: False)
        fixture.setattr(staging_guard.Path, "resolve", lambda path, strict=True:
                        Path("/other-root-backed/selected") if path == base else path)
        with pytest.raises(staging_guard.staging_guard_error, match="canonical path identity"):
            staging_guard._validate_base(base)


def test_local_staging_space_uses_statvfs_without_quota(monkeypatch, tmp_path):
    calls = []

    def local_space(path):
        calls.append(path)
        return SimpleNamespace(f_bavail=12, f_frsize=1024, f_favail=345)

    monkeypatch.setattr(staging_guard.os, "statvfs", local_space)
    monkeypatch.setattr(staging_guard, "_quota_available", lambda _: (_ for _ in ()).throw(
        AssertionError("local staging must not query quota")))
    selected_base = tmp_path / "selected_staging"
    assert staging_guard._space(selected_base, None) == (12 * 1024, 345)
    assert calls == [selected_base]


@pytest.mark.parametrize("invocation", (".", "..", "../escape", "nested/escape"))
def test_staging_guard_rejects_nonisolated_invocation(invocation):
    with pytest.raises(staging_guard.staging_guard_error, match="one path component"):
        staging_guard.start_guard("/synthetic-root-backed/selected", invocation, None,
                                  20_000_000_000, 10_000_000_000, 1)


def test_quota_staging_space_uses_exact_supplied_root(monkeypatch, tmp_path):
    selected_base = tmp_path / "selected_staging"
    monkeypatch.setattr(staging_guard.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=12, f_frsize=1024, f_favail=345))
    roots = []
    monkeypatch.setattr(staging_guard, "_quota_available", lambda root:
                        roots.append(root) or 1000)
    assert staging_guard._space(selected_base, "/groups/selected") == (1000, 345)
    assert roots == ["/groups/selected"]


def test_quota_unavailable_falls_back_to_statvfs(monkeypatch, tmp_path):
    monkeypatch.setattr(staging_guard.os, "statvfs", lambda path: SimpleNamespace(
        f_bavail=12, f_frsize=1024, f_favail=345))
    monkeypatch.setattr(staging_guard, "_quota_available", lambda root:
                        (_ for _ in ()).throw(staging_guard.staging_guard_error(
                            "quota headroom could not be measured")))
    assert staging_guard._space(tmp_path, "/groups/selected") == (12 * 1024, 345)


class fake_task:
    def __init__(self, category="processing", size=1, fout_size=0):
        self.category = category
        self.size = size
        self.fout_size = fout_size
        self.fin_size = 0
        self.priority = 0
        self.id = 0
        self.tasks_to_accumulate = []

    def specify_priority(self, priority):
        self.priority = priority

    def successful(self):
        return True

    def __len__(self):
        return self.size

    def task_accum_log(self, *args):
        pass


class fake_accum_task(fake_task):
    def __init__(self, queue, infile_fn, group):
        super().__init__(category="accumulating", size=sum(len(t) for t in group))
        self.tasks_to_accumulate = group
        self.fin_size = sum(t.fout_size for t in group)


class fake_queue:
    def __init__(self, log_dir, capacity=0):
        self.executor = SimpleNamespace(
            stats_log=str(log_dir / "stats.log"), manager_name="arbitrary-manager",
            desc="Processing", treereduction=2,
            tasks_accum_log=str(log_dir / "accum.log"))
        self._task_table = {}
        self.states = {}
        self.stats_coffea = defaultdict(int, events_total=100000,
                                        chunksize_original=1)
        self.tasks_to_accumulate = []
        self.chunksize_current = 1
        self._items_empty = False
        self.capacity = capacity
        self.hungry_called = 0
        self.demand = []
        self.reductions = 0
        self.next_id = 1
        self.category_max = {}
        self.category_min = {}

    def specify_category_max_resources(self, category, resources):
        self.category_max[category] = resources

    def specify_category_min_resources(self, category, resources):
        self.category_min[category] = resources

    def workers_summary(self):
        if self.capacity is None:
            raise RuntimeError("summary unavailable")
        if not self.capacity:
            return []
        return [{"workers": self.capacity, "cores": 2, "memory": 63000,
                 "disk": 150000, "gpus": 0}]

    def task_state(self, task_id):
        return self.states[task_id]

    def specify_num_tasks_left(self, count):
        self.demand.append(count)

    def _submit_processing_tasks(self, infile_procc_fn, items):
        raise AssertionError("original global-hunger loop must be bypassed")

    def _submit_processing_task(self, infile_procc_fn, item):
        self.submit(fake_task(size=len(item)))
        self.stats_coffea["events_submitted"] += len(item)

    def _submit_accum_tasks(self, infile_accum_fn):
        self.reductions += 1
        self.tasks_to_accumulate.clear()

    def _final_accumulation(self, accumulator):
        return accumulator

    def submit(self, task):
        task_id = self.next_id
        self.next_id += 1
        task.id = task_id
        self._task_table[task_id] = task
        self.states[task_id] = wq.WORK_QUEUE_TASK_READY
        return task_id

    def wait(self, timeout=None):
        if getattr(self, "wait_results", None):
            task = self.wait_results.pop(0)
            self._task_table.pop(task.id, None)
            return task
        return None


def work_items(count):
    for _ in range(count):
        yield (0,)


@pytest.fixture
def setup_queue(tmp_path, monkeypatch):
    available = [10**12]
    monkeypatch.setattr(admission, "_space", lambda base, quota_root:
                        (available[0], 10**7))
    from coffea.processor import work_queue_tools
    monkeypatch.setattr(work_queue_tools, "AccumCoffeaWQTask", fake_accum_task)

    def make(capacity=0, config=None):
        queue = fake_queue(tmp_path, capacity)
        coordinator = admission.install_processing_admission(
            queue, base=str(tmp_path), quota_root="/groups/klannon",
            warning_bytes=10**9, min_free_bytes=10**8,
            config=config)
        return queue, coordinator

    return make, available


@pytest.mark.parametrize("capacity,target", [(0, 10), (8, 16),
                                              (53, 106), (500, 1000)])
def test_capacity_targets_and_remaining_input(setup_queue, capacity, target):
    make, _ = setup_queue
    queue, coordinator = make(capacity)
    queue.stats_coffea["events_total"] = 20
    decision = coordinator._decision(20, 1)
    assert decision[3] == target
    assert decision[0] == min(target, 20)
    queue._submit_processing_tasks(None, work_items(20))
    assert queue.stats_coffea["events_submitted"] == min(target, 20)
    assert len(queue._task_table) == min(target, 20)
    assert queue.demand[-1] == 20 - min(target, 20)


def test_ready_only_and_aggregate_backlog_does_not_veto(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(8)
    for _ in range(6):
        queue.submit(fake_task())
    for _ in range(161):
        queue.submit(fake_task(category="accumulating"))
    decision = coordinator._decision(100, 1)
    assert decision[0] == 10
    assert decision[2]["ready"] == 6
    queue._submit_processing_tasks(None, work_items(10))
    assert queue.stats_coffea["events_submitted"] == 10
    assert all(task.priority in (0, 1) for task in queue._task_table.values())


def test_retrieval_states_and_running_above_reserve(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(8)
    for state in (wq.WORK_QUEUE_TASK_READY, wq.WORK_QUEUE_TASK_RUNNING,
                  wq.WORK_QUEUE_TASK_WAITING_RETRIEVAL,
                  wq.WORK_QUEUE_TASK_RETRIEVED):
        task_id = queue.submit(fake_task())
        queue.states[task_id] = state
    counts = coordinator._native_states()
    assert [counts[key] for key in ("ready", "running", "waiting_retrieval",
                                    "retrieved")] == [1, 1, 1, 1]
    for _ in range(20):
        task_id = queue.submit(fake_task())
        queue.states[task_id] = wq.WORK_QUEUE_TASK_RUNNING
    assert coordinator._decision(100, 1)[0] == 15


def test_pressure_pause_recovery_and_deferred_reduction(setup_queue):
    make, available = setup_queue
    queue, coordinator = make(8)
    available[0] = 10**8 + 1
    assert coordinator._decision(100, 1)[1] == "storage_pressure"
    queue.tasks_to_accumulate = [fake_task(fout_size=1),
                                 fake_task(fout_size=1),
                                 fake_task(fout_size=1)]
    queue._submit_accum_tasks("fn")
    assert coordinator.deferred_accumulation
    assert len(queue.tasks_to_accumulate) == 3
    available[0] = 10**12
    coordinator.last_measurement_at = -float("inf")
    assert queue.wait(0) is None
    assert not coordinator.deferred_accumulation
    assert queue.stats_coffea["accumulations_submitted"] == 1
    assert len(queue.tasks_to_accumulate) == 1
    assert coordinator._decision(100, 1)[0] == 16


def test_incremental_reduction_preserves_unadmitted_inputs(setup_queue):
    make, available = setup_queue
    queue, coordinator = make(8)
    queue.stats_coffea["events_processed"] = queue.stats_coffea["events_total"]
    coordinator.warning_bytes = coordinator.min_free_bytes = 0
    coordinator.output_estimate_bytes = 64_000_000
    available[0] = 500_000_000
    children = [fake_task(fout_size=100) for _ in range(4)]
    queue.tasks_to_accumulate = children[:]
    queue._submit_accum_tasks("reducer")
    assert queue.stats_coffea["accumulations_submitted"] == 1
    assert queue.tasks_to_accumulate == children[2:]
    assert coordinator.deferred_accumulation
    assert list(queue._task_table.values())[0].tasks_to_accumulate == children[:2]
    available[0] = 1_000_000_000
    coordinator.last_measurement_at = -float("inf")
    queue.wait(0)
    assert queue.stats_coffea["accumulations_submitted"] == 2
    assert queue.tasks_to_accumulate == []
    second, second_coordinator = make(8)
    second.stats_coffea["events_processed"] = second.stats_coffea["events_total"]
    second_coordinator.warning_bytes = second_coordinator.min_free_bytes = 0
    second_coordinator.output_estimate_bytes = 64_000_000
    second.tasks_to_accumulate = [fake_task(fout_size=100) for _ in range(4)]
    second._submit_accum_tasks("reducer")
    assert second.stats_coffea["accumulations_submitted"] == 2
    assert second.tasks_to_accumulate == []


def test_arriving_outputs_invalidate_cached_headroom(setup_queue, monkeypatch):
    make, _ = setup_queue
    queue, coordinator = make(8)
    coordinator.warning_bytes = coordinator.min_free_bytes = 0
    available = 256 * 1024 * 1024
    output_size = int(31.875 * 1024 * 1024)
    consumed = [0]
    measurements = []

    def space(*args):
        measurements.append(consumed[0])
        return available - consumed[0], 10**7

    monkeypatch.setattr(admission, "_space", space)
    assert coordinator._decision(100, 1)[0] == 2
    for _ in range(8):
        task = fake_task(fout_size=output_size / 1_000_000)
        queue.submit(task)
        queue.wait_results = [task]
        consumed[0] += output_size
        assert queue.wait(0) is task
        coordinator._decision(100, 1)
    assert len(measurements) == 9
    assert coordinator.measurement_count == 9
    assert coordinator.measurement_invalidations == 8
    assert coordinator._decision(100, 1)[0] == 0
    assert available - consumed[0] == 1024 * 1024


def test_negative_priority_is_preserved(setup_queue):
    make, _ = setup_queue
    queue, _ = make(8)
    task = fake_task()
    task.priority = -3
    queue.submit(task)
    assert task.priority == -3


def test_exhausted_input_demand_stays_zero(setup_queue):
    make, _ = setup_queue
    queue, _ = make(8, config=admission.admission_config(pass_maximum=1))
    items = work_items(1)
    queue._submit_processing_tasks(None, items)
    assert queue.demand[-1] > 0
    queue._submit_processing_tasks(None, items)
    assert queue._items_empty
    assert queue.demand[-1] == 0
    queue._submit_processing_tasks(None, work_items(0))
    assert queue.demand[-1] == 0


def test_heterogeneous_rounded_capacity(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(0)
    queue.executor.cores = 1
    queue.executor.memory = 31552
    queue.executor.disk = 75000
    queue.executor.gpus = 0
    queue.workers_summary = lambda: [
        {"workers": 50, "cores": 1, "memory": 1024, "disk": 1024, "gpus": 0},
        {"workers": 8, "cores": 2, "memory": 63000, "disk": 150000, "gpus": 0},
        {"workers": 4, "cores": 2, "memory": 30720, "disk": 150000, "gpus": 0},
        {"workers": 2, "cores": 2, "memory": 63000, "disk": 70000, "gpus": 0},
    ]
    assert coordinator._capacity()[0] == 8
    assert coordinator.last_capacity_quality.endswith("with_unknown")
    assert coordinator._decision(100, 1)[3] == 16
    queue.executor.gpus = 1
    assert coordinator._capacity()[0] == 0
    queue.executor.gpus = 0
    queue.executor.memory = None
    queue.executor.disk = None
    queue.status = lambda request: [{
        "category": "processing",
        "first_allocation": {"cores": 1, "memory": 31552,
                             "disk": 75000, "gpus": 0}}]
    assert coordinator._capacity()[0] == 64


def test_proportional_allocation_on_largest_worker_is_not_a_minimum(
    setup_queue,
):
    make, _ = setup_queue
    queue, coordinator = make(0)
    queue.executor.cores = 1
    queue.executor.memory = None
    queue.executor.disk = None
    queue.workers_summary = lambda: [
        {"workers": 50, "cores": 2, "memory": 61440,
         "disk": 147456, "gpus": 0},
        {"workers": 1, "cores": 2, "memory": 192000,
         "disk": 250000, "gpus": 0},
    ]
    queue.status = lambda request: [{
        "category": "processing",
        "first_allocation": {"cores": 1, "memory": 96000,
                             "disk": 125000, "gpus": 0},
    }]
    assert coordinator._capacity()[0] == 51
    assert "auto_allocation_estimate" in coordinator.last_capacity_quality
    assert coordinator._decision(1000, 1)[3] == 102


def test_fixed_proportional_gpu_and_reducer_resource_profiles(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(0)
    queue.executor.cores = 1
    queue.executor.memory = None
    queue.executor.disk = None
    profiles = [
        {"workers": 1, "cores": 2, "memory": 63000,
         "disk": 150000, "gpus": 0},
        {"workers": 1, "cores": 2, "memory": 128000,
         "disk": 180000, "gpus": 0},
        {"workers": 1, "cores": 2, "memory": 192000,
         "disk": 250000, "gpus": 0},
        {"workers": 1, "cores": 4, "memory": 126000,
         "disk": 300000, "gpus": 0},
    ]
    queue.workers_summary = lambda: profiles
    assert coordinator._capacity()[0] == 4
    assert "auto_allocation_estimate" in coordinator.last_capacity_quality
    queue.executor.memory = 120000
    queue.executor.disk = 170000
    assert coordinator._capacity()[0] == 3
    coordinator.accumulating_memory_mb = 180000
    coordinator.accumulating_cores = 2
    assert coordinator._reducer_capacity(profiles) == 1
    queue.executor.gpus = 1
    assert coordinator._capacity()[0] == 0
    profiles[2]["gpus"] = 1
    assert coordinator._capacity()[0] == 1
    queue.workers_summary = lambda: []
    assert coordinator._capacity()[0] == 0


def test_final_drain_allows_long_running_healthy_reducer(
    setup_queue, monkeypatch,
):
    from coffea.processor import work_queue_tools

    make, _ = setup_queue
    clock = [0.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(work_queue_tools, "early_terminate", False)
    completed = []
    queue_ref = []

    def wait_for_reducer(self, timeout=None):
        clock[0] += 10
        if clock[0] < 120:
            return None
        queue = queue_ref[0]
        pending = next(iter(queue._task_table.values()))
        queue._task_table.pop(pending.id)
        completed.append(pending)
        return pending

    monkeypatch.setattr(fake_queue, "wait", wait_for_reducer)
    queue, coordinator = make(0)
    queue_ref.append(queue)
    queue.stats_coffea["events_processed"] = queue.stats_coffea["events_total"]
    queue.tasks_to_accumulate = [fake_task(fout_size=1),
                                 fake_task(fout_size=1)]
    marker = object()
    assert queue._final_accumulation(marker) is marker
    assert clock[0] >= 120
    assert len(completed) == 1


def test_final_drain_retries_reducer_and_honors_operator_interrupt(
    setup_queue, monkeypatch,
):
    from coffea.processor import work_queue_tools

    make, _ = setup_queue
    monkeypatch.setattr(work_queue_tools, "early_terminate", False)
    queue_ref = []
    attempts = []

    def wait_with_retry(self, timeout=None):
        queue = queue_ref[0]
        task = next(iter(queue._task_table.values()))
        queue._task_table.pop(task.id)
        attempts.append(task)
        return task

    monkeypatch.setattr(fake_queue, "wait", wait_with_retry)
    queue, coordinator = make(0)
    queue_ref.append(queue)
    queue.stats_coffea["events_processed"] = queue.stats_coffea["events_total"]
    queue.tasks_to_accumulate = [fake_task(fout_size=1),
                                 fake_task(fout_size=1)]
    original_submit = queue.submit
    failed_once = [False]

    def submit_reducer(task):
        if task.category == "accumulating":
            task.successful = lambda: failed_once[0]

            def resubmit(queue):
                failed_once[0] = True
                original_submit(task)

            task.resubmit = resubmit
        return original_submit(task)

    queue.submit = submit_reducer
    marker = object()
    assert queue._final_accumulation(marker) is marker
    assert len(attempts) == 2
    assert queue.stats_coffea["accumulations_done"] == 1
    assert coordinator.finalizing is False

    def interrupt_wait(self, timeout=None):
        monkeypatch.setattr(work_queue_tools, "early_terminate", True)
        return None

    # The installed adapter captured wait_with_retry, so replace the
    # underlying fake method before installing a fresh adapter below.
    monkeypatch.setattr(fake_queue, "wait", interrupt_wait)
    next_queue, next_coordinator = make(0)
    next_queue.stats_coffea["events_processed"] = next_queue.stats_coffea["events_total"]
    next_queue.tasks_to_accumulate = [fake_task(fout_size=1),
                                      fake_task(fout_size=1)]
    queue_ref[0] = next_queue
    assert next_queue._final_accumulation(marker) is marker
    assert next_coordinator.finalizing is False


def test_result_transitions_refresh_cached_space(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(8)
    for category, success in (("processing", True), ("accumulating", True),
                              ("processing", False), ("accumulating", False)):
        task = fake_task(category=category)
        task.successful = lambda outcome=success: outcome
        queue.submit(task)
        coordinator.last_measurement_at = 123.0
        queue.wait_results = [task]
        assert queue.wait(0) is task
        assert coordinator.last_measurement_at == -float("inf")


def test_telemetry_walk_and_rotation_are_bounded(setup_queue, monkeypatch):
    make, _ = setup_queue
    queue, coordinator = make(8, config=admission.admission_config(
        telemetry_limit_bytes=2500))
    current = fake_task()
    for _ in range(3000):
        parent = fake_task(category="accumulating")
        parent.tasks_to_accumulate = [current]
        current = parent
    queue.tasks_to_accumulate = [current]
    assert coordinator._retained_bytes()[1] is False
    queue.tasks_to_accumulate = [fake_task() for _ in range(3000)]
    assert coordinator._retained_bytes()[1] is False
    queue.tasks_to_accumulate = []
    decision = coordinator._decision(20, 1)
    clock = [admission.time.monotonic()]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    for index in range(100):
        clock[0] += 31
        coordinator._emit("refill" if index % 2 else "storage_pressure",
                          decision[2], decision[3], decision[4], decision[5],
                          decision[6])
    assert Path(coordinator.log_path).stat().st_size <= 2500
    assert Path(coordinator.log_path + ".previous").stat().st_size <= 2500
    import json
    for line in Path(coordinator.log_path).read_text().splitlines():
        row = json.loads(line)
        assert row["reason_duration_seconds"] == 0
        assert row["previous_reason_duration_seconds"] is not None
    monkeypatch.setattr(admission.json, "dumps", lambda *args, **kwargs:
                        (_ for _ in ()).throw(TypeError("telemetry down")))
    coordinator._emit("refill", decision[2], decision[3], decision[4],
                      decision[5], decision[6], force=True)
    assert coordinator.telemetry_disabled


def test_telemetry_collection_and_write_failures_do_not_change_decisions(
    setup_queue, monkeypatch, tmp_path
):
    make, _ = setup_queue
    queue, coordinator = make(8)
    decision = coordinator._decision(20, 1)
    monkeypatch.setattr(coordinator, "_retained_bytes", lambda: (_ for _ in ())
                        .throw(OSError("collection unavailable")))
    coordinator._emit("refill", decision[2], decision[3], decision[4],
                      decision[5], decision[6], force=True)
    import json
    row = json.loads(Path(coordinator.log_path).read_text().splitlines()[-1])
    assert row["retained_measurement_complete"] is False
    coordinator.log_path = str(tmp_path / "missing_parent" / "admission.jsonl")
    coordinator._emit("storage_pressure", decision[2], decision[3],
                      decision[4], decision[5], decision[6], force=True)
    assert coordinator.telemetry_disabled
    assert coordinator._decision(20, 1)[0] > 0


def test_operator_early_termination_does_not_enter_final_drain(
    setup_queue, monkeypatch
):
    from coffea.processor import work_queue_tools
    make, _ = setup_queue
    queue, coordinator = make(8)
    queue.tasks_to_accumulate = [fake_task(), fake_task()]
    monkeypatch.setattr(work_queue_tools, "early_terminate", True)
    marker = object()
    assert queue._final_accumulation(marker) is marker
    assert len(queue.tasks_to_accumulate) == 2
    assert not coordinator.finalizing


def test_unknown_measurement_and_unknown_capacity(setup_queue, monkeypatch):
    make, _ = setup_queue
    queue, coordinator = make(8)
    assert coordinator._decision(100, 1)[3] == 16
    queue.capacity = None
    assert coordinator._decision(100, 1)[3] == 16
    coordinator.capacity_at = -float("inf")
    assert coordinator._decision(100, 1)[3] == 10
    assert coordinator.last_capacity_quality == "unknown_bootstrap"
    monkeypatch.setattr(admission, "_space", lambda *args:
                        (_ for _ in ()).throw(RuntimeError("quota failed")))
    coordinator.last_measurement_at = -float("inf")
    assert coordinator._decision(100, 1)[1] == "headroom_unknown"


def test_slow_measurement_pauses_processing_then_recovers(
    setup_queue, monkeypatch,
):
    make, _ = setup_queue
    clock = [100.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    queue, coordinator = make(8)
    reads = ["slow", "quick"]

    def space(*args):
        if reads.pop(0) == "slow":
            clock[0] = 120.0
        return 10**12, 10**7

    monkeypatch.setattr(admission, "_space", space)
    items = work_items(4)
    queue._submit_processing_tasks(None, items)
    assert queue.stats_coffea["events_submitted"] == 0
    assert not queue._task_table
    assert coordinator.measurement_last_seconds == 20.0
    assert coordinator.last_measurement_at == 100.0
    import json
    row = json.loads(Path(coordinator.log_path).read_text().splitlines()[-1])
    assert row["reason"] == "headroom_unknown"
    assert row["measurement_quality"] == "unknown_or_stale"
    assert row["measurement_age_seconds"] == 20.0

    clock[0] = 121.0
    queue._submit_processing_tasks(None, items)
    assert queue.stats_coffea["events_submitted"] == 4
    assert len(queue._task_table) == 4
    assert coordinator.measurement_count == 2
    assert coordinator._decision(1, 1)[1] == "refill"


def test_slow_measurement_preserves_deferred_reducer_inputs(
    setup_queue, monkeypatch,
):
    make, _ = setup_queue
    clock = [100.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    queue, coordinator = make(8)
    queue.stats_coffea["events_processed"] = queue.stats_coffea["events_total"]
    children = [fake_task(fout_size=1), fake_task(fout_size=1)]
    queue.tasks_to_accumulate = children[:]
    reads = ["slow", "quick"]

    def space(*args):
        if reads.pop(0) == "slow":
            clock[0] = 120.0
        return 10**12, 10**7

    monkeypatch.setattr(admission, "_space", space)
    queue._submit_accum_tasks("fn")
    assert queue.stats_coffea["accumulations_submitted"] == 0
    assert queue.tasks_to_accumulate == children
    assert coordinator.deferred_accumulation
    clock[0] = 121.0
    queue._submit_accum_tasks("fn")
    assert queue.stats_coffea["accumulations_submitted"] == 1
    assert queue.tasks_to_accumulate == []
    assert list(queue._task_table.values())[0].tasks_to_accumulate == children
    assert coordinator.measurement_count == 2


def test_failed_measurement_pauses_until_valid_read(
    setup_queue, monkeypatch,
):
    make, _ = setup_queue
    clock = [100.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    queue, coordinator = make(8)
    calls = [0]

    def space(*args):
        calls[0] += 1
        if calls[0] == 1:
            raise OSError("quota unavailable")
        return 10**12, 10**7

    monkeypatch.setattr(admission, "_space", space)
    assert coordinator._decision(4, 1)[:2] == (0, "headroom_unknown")
    assert coordinator.measurement_error == "OSError"
    clock[0] = 106.0
    assert coordinator._decision(4, 1)[0] == 4
    assert coordinator.measurement_error is None


def test_recovery_submission_and_independent_queues(setup_queue):
    make, _ = setup_queue
    first, first_coordinator = make(8)
    second, second_coordinator = make(53)
    assert first_coordinator is not second_coordinator
    first.submit(fake_task())
    assert first_coordinator.recovery_submissions == 1
    assert second_coordinator.recovery_submissions == 0
    assert first._processing_admission_coordinator is first_coordinator
    assert admission.install_processing_admission(
        first, base=first_coordinator.base,
        quota_root=first_coordinator.quota_root,
        warning_bytes=first_coordinator.warning_bytes,
        min_free_bytes=first_coordinator.min_free_bytes) is first_coordinator


def test_no_compatible_reducer_has_bounded_intermediate_budget(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(8)
    coordinator.accumulating_memory_mb = 128000
    coordinator.accumulating_cores = 2
    assert coordinator._decision(100, 1)[0] == 16
    queue.submit(fake_task(category="accumulating"))
    assert coordinator._decision(100, 1)[0] > 0
    queue.tasks_to_accumulate = [fake_task() for _ in range(16)]
    assert coordinator._decision(100, 1)[1] == "reducer_backlog_budget"
    queue.capacity = 0
    coordinator.last_capacity = None
    assert coordinator._decision(100, 1)[1] == "reducer_backlog_budget"


def test_priority_preference_is_bounded_and_recovery_uses_same_path(setup_queue):
    make, _ = setup_queue
    queue, coordinator = make(53)
    tasks = [fake_task() for _ in range(40)]
    for task in tasks:
        queue.submit(task)
    assert sum(task.priority == 1 for task in tasks) == 32
    assert coordinator.recovery_submissions == 40
    higher = fake_task()
    higher.priority = 5
    queue.submit(higher)
    assert higher.priority == 5


def test_default_priority_burst_rolls_over_multiple_windows(
    setup_queue, monkeypatch,
):
    make, _ = setup_queue
    clock = [0.0]
    monkeypatch.setattr(admission.time, "monotonic", lambda: clock[0])
    queue, coordinator = make(0)
    assert coordinator.config.priority_burst == 32
    assert coordinator.config.priority_window_seconds == 30.0
    first = [fake_task() for _ in range(40)]
    for task in first:
        queue.submit(task)
    assert sum(task.priority == 1 for task in first) == 32
    clock[0] = 31.0
    second = [fake_task() for _ in range(40)]
    for task in second:
        queue.submit(task)
    assert sum(task.priority == 1 for task in second) == 32
    negative = fake_task()
    negative.priority = -4
    queue.submit(negative)
    assert negative.priority == -4


def test_retained_measurement_counts_only_existing_owned_outputs(setup_queue, tmp_path):
    make, _ = setup_queue
    queue, coordinator = make(8)
    output = tmp_path / "owned-output"
    output.write_bytes(b"12345")
    child = fake_task(fout_size=100)
    child.outfile_output = str(output)
    parent = fake_task(category="accumulating", fout_size=200)
    parent.tasks_to_accumulate = [child]
    queue.tasks_to_accumulate = [parent]
    assert coordinator._retained_bytes() == (5, True, 1)
    output.unlink()
    assert coordinator._retained_bytes() == (0, True, 1)


def test_rate_limited_telemetry(setup_queue):
    make, _ = setup_queue
    config = admission.admission_config(telemetry_limit_bytes=4096)
    queue, coordinator = make(8, config=config)
    decision = coordinator._decision(20, 1)
    for _ in range(100):
        coordinator._emit("refill", decision[2], decision[3], decision[4],
                          decision[5], decision[6])
    rows = Path(coordinator.log_path).read_text().splitlines()
    assert len(rows) == 1
    assert len(rows[0]) < 4096
    import json
    record = json.loads(rows[0])
    assert record["config"]["ready_minimum"] == 10
    assert record["processing"]["ready"] == 0
    assert record["effective_available_bytes"] == 10**12
    assert record["schema_version"] == 3
    assert record["measurement_count"] == coordinator.measurement_count
    assert record["measurement_last_seconds"] is not None


@pytest.mark.parametrize("launcher", ["run_analysis", "run_analysis_excluded_mc"])
def test_guarded_launcher_callback_is_composed(
    setup_queue, monkeypatch, launcher, tmp_path
):
    make, _ = setup_queue
    queue, coordinator = make(8)
    module_dir = str(Path(__file__).resolve().parents[1] / "analysis" /
                     "topeft_run2")
    monkeypatch.syspath_prepend(module_dir)
    module = importlib.import_module(f"analysis.topeft_run2.{launcher}")
    calls = []
    monkeypatch.setattr(module, "install_processing_admission",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    module._configure_wq_categories(
        queue, processing_cores=1, accumulating_memory_mb=128000,
        accumulating_cores=2, admission_base=str(tmp_path),
        admission_quota_root="/groups/klannon",
        admission_warning_bytes=10**9, admission_min_free_bytes=10**8)
    assert queue.category_max["processing"] == {"cores": 1}
    assert queue.category_min["accumulating"] == {
        "memory": 128000, "cores": 2}
    assert len(calls) == 1
    assert calls[0][1]["base"] == str(tmp_path)
    assert calls[0][1]["accumulating_memory_mb"] == 128000


def test_quota_timeout_uses_statvfs_for_admission(
    setup_queue, monkeypatch, tmp_path
):
    make, _ = setup_queue
    queue, coordinator = make(8)
    monkeypatch.setattr(admission, "_space", staging_guard._space)

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(["quota", "/groups/klannon"], 15)

    monkeypatch.setattr(staging_guard.subprocess, "run", timed_out)
    monkeypatch.setattr(staging_guard.os, "statvfs", lambda path: SimpleNamespace(
        f_bavail=10**12, f_frsize=1, f_favail=2000))
    existing = [fake_task() for _ in range(4)]
    for task in existing:
        queue.submit(task)
    for task, state in zip(existing, (
        wq.WORK_QUEUE_TASK_READY, wq.WORK_QUEUE_TASK_RUNNING,
        wq.WORK_QUEUE_TASK_WAITING_RETRIEVAL, wq.WORK_QUEUE_TASK_RETRIEVED,
    )):
        queue.states[task.id] = state
    queued_before = dict(queue._task_table)
    assert coordinator._decision(100, 1)[0] > 0
    assert queue._task_table == queued_before
    assert coordinator.measurement_error is None
    pending = [fake_task(fout_size=1) for _ in range(3)]
    queue.tasks_to_accumulate = pending[:]
    queue._submit_accum_tasks("reducer")
    assert queue.stats_coffea["accumulations_submitted"] > 0

    def recovered(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=(
            "Filesystem quota used use available\n"
            "/groups/klannon 5T 2T 40% 2T\n"))

    monkeypatch.setattr(staging_guard.subprocess, "run", recovered)
    coordinator.last_measurement_at = -float("inf")
    assert coordinator._decision(100, 1)[0] > 0


def _owned_invocation(base, name):
    owned = base / name
    owned.mkdir()
    (owned / "guard.json").write_text(json.dumps({
        "status": "completed_pending_receipt", "path": str(owned),
        "owner_uid": os.getuid(), "owner_pid": os.getpid(),
        "owner_start": staging_guard._process_start(os.getpid()),
        "owner_inode": owned.stat().st_ino,
    }))
    return owned


def test_invocation_closeout_preserves_only_pkl_and_matching_metadata(
    tmp_path, monkeypatch
):
    base = tmp_path / "workers"
    base.mkdir()
    owned = _owned_invocation(base, "mine")
    sibling = base / "other"
    sibling.mkdir()
    (sibling / "data.log").write_bytes(b"sibling")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    (owned / "escape").symlink_to(outside)
    nested = owned / "tmpabcwq-tmp-"
    nested.mkdir()
    (nested / "result.pkl.gz").write_bytes(b"pkl")
    (nested / "result.pkl.gz.metadata.json").write_bytes(b"metadata")
    for name in ("task.p", "debug.log", "failure.json", "note.txt",
                 "orphan.pkl.gz.metadata.json", "env.tar.gz"):
        (nested / name).write_bytes(b"transient")
    cache = nested / "cache"
    cache.mkdir()
    (cache / "copy.py").write_bytes(b"transient")
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    inode = owned.stat().st_ino
    staging_guard.closeout_invocation_staging(base, owned, inode)
    assert base.is_dir() and (sibling / "data.log").read_bytes() == b"sibling"
    assert outside.read_bytes() == b"outside"
    assert sorted(path.name for path in nested.iterdir()) == [
        "result.pkl.gz", "result.pkl.gz.metadata.json"]
    assert not (owned / "guard.json").exists()
    staging_guard.closeout_invocation_staging(base, owned, inode)
    assert (nested / "result.pkl.gz").read_bytes() == b"pkl"
    assert (nested / "result.pkl.gz.metadata.json").read_bytes() == b"metadata"


def test_invocation_closeout_rejects_base_sibling_and_symlink_path(
    tmp_path, monkeypatch
):
    base = tmp_path / "workers"
    base.mkdir()
    owned = _owned_invocation(base, "mine")
    sibling = _owned_invocation(base, "other")
    escape = base / "escape"
    escape.symlink_to(owned, target_is_directory=True)
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    for target in (base, sibling, escape, tmp_path):
        with pytest.raises(staging_guard.staging_guard_error):
            staging_guard.closeout_invocation_staging(base, target,
                                                       owned.stat().st_ino)
    assert (owned / "guard.json").exists()
    assert (sibling / "guard.json").exists()


def test_failed_invocation_closeout_removes_all_transients_and_is_idempotent(
    tmp_path, monkeypatch
):
    base = tmp_path / "workers"
    base.mkdir()
    owned = _owned_invocation(base, "failed")
    marker = owned / "guard.json"
    metadata = json.loads(marker.read_text())
    metadata.update(status="cleanup_needed", reason="manager_failed")
    marker.write_text(json.dumps(metadata))
    (owned / "diagnostic.json").write_bytes(b"transient")
    (owned / "orphan.pkl.gz.metadata.json").write_bytes(b"orphan")
    nested = owned / "cache"
    nested.mkdir()
    (nested / "env_copy.whl").write_bytes(b"transient")
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    inode = owned.stat().st_ino
    staging_guard.closeout_invocation_staging(base, owned, inode)
    assert base.is_dir() and not owned.exists()
    staging_guard.closeout_invocation_staging(base, owned, inode)


def test_run_analysis_closeout_degradation_keeps_primary_failure_and_output(
    tmp_path, monkeypatch, capsys
):
    module_dir = str(Path(__file__).resolve().parents[1] / "analysis" /
                     "topeft_run2")
    monkeypatch.syspath_prepend(module_dir)
    run_analysis = importlib.import_module("analysis.topeft_run2.run_analysis")
    pkl = tmp_path / "result.pkl.gz"
    sidecar = tmp_path / "result.pkl.gz.metadata.json"
    pkl.write_bytes(b"valid")
    sidecar.write_bytes(b"metadata")
    monkeypatch.setattr(run_analysis, "finish_guard", lambda *args, **kwargs: None)

    def cleanup_failure(*args):
        raise staging_guard.staging_guard_error("synthetic cleanup failure")

    monkeypatch.setattr(run_analysis, "closeout_invocation_staging", cleanup_failure)
    guard = (tmp_path / "owned", 1, object(), object())
    with pytest.raises(ValueError, match="primary failure"):
        try:
            raise ValueError("primary failure")
        finally:
            run_analysis._finalize_work_queue_staging(guard, tmp_path, False)
    run_analysis._finalize_work_queue_staging(guard, tmp_path, True)
    assert "closeout degraded" in capsys.readouterr().err
    assert pkl.read_bytes() == b"valid"
    assert sidecar.read_bytes() == b"metadata"
