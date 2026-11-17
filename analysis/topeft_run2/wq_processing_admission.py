"""Queue-local processing admission for the supported Coffea Work Queue interface.

The native queue remains the scheduler.  This adapter only decides when Coffea
may take another logical input item and when an existing reduction group may be
submitted.  It does not own a second task registry.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
import math
import os
import time
from types import MethodType
import uuid

from analysis.topeft_run2.wq_staging_guard import _space


@dataclass(frozen=True)
class admission_config:
    ready_minimum: int = 10
    ready_factor: float = 2.0
    pass_maximum: int = 512
    pass_seconds: float = 0.25
    measurement_seconds: float = 5.0
    measurement_max_age_seconds: float = 15.0
    output_bootstrap_bytes: int = 64 * 1024 * 1024
    telemetry_seconds: float = 30.0
    telemetry_limit_bytes: int = 2 * 1024 * 1024
    priority_burst: int = 32
    priority_window_seconds: float = 30.0

    def __post_init__(self):
        if (self.ready_minimum < 1 or self.ready_factor <= 0 or
                self.pass_maximum < 1 or self.pass_seconds <= 0 or
                self.measurement_seconds <= 0 or
                self.measurement_max_age_seconds < self.measurement_seconds or
                self.output_bootstrap_bytes < 1 or self.telemetry_seconds <= 0 or
                self.telemetry_limit_bytes < 1024 or self.priority_burst < 0 or
                self.priority_window_seconds <= 0):
            raise ValueError("invalid Work Queue admission configuration")


class _coordinator:
    def __init__(self, queue, config, base, quota_root, warning_bytes,
                 min_free_bytes):
        self.queue = queue
        self.config = config
        self.base = os.fspath(base)
        self.quota_root = quota_root
        self.warning_bytes = warning_bytes
        self.min_free_bytes = min_free_bytes
        self.created = time.monotonic()
        self.invocation_id = str(uuid.uuid4())
        self.last_measurement_at = -math.inf
        self.available_bytes = None
        self.free_inodes = None
        self.measurement_error = None
        self.measurement_count = 0
        self.measurement_seconds_total = 0.0
        self.measurement_last_seconds = None
        self.measurement_invalidations = 0
        self.capacity_at = -math.inf
        self.last_capacity = None
        self.last_capacity_quality = "unknown_bootstrap"
        self.last_reducer_capacity_quality = "unknown"
        self.output_estimate_bytes = config.output_bootstrap_bytes
        self.new_admitted = 0
        self.recovery_submissions = 0
        self.processing_done = 0
        self.reductions_done = 0
        self.last_reason = None
        self.reason_since = self.created
        self.last_emit_at = -math.inf
        self.telemetry_bytes = 0
        self.telemetry_disabled = False
        self.last_accum_fn = None
        self.deferred_accumulation = False
        self.ordinary_submit = False
        self.finalizing = False
        self.priority_window_at = self.created
        self.priority_used = 0
        self.processing_cores = None
        self.log_path = os.path.join(
            os.path.dirname(os.path.abspath(queue.executor.stats_log)),
            "admission.jsonl",
        )

    def begin_processing(self):
        self.last_accum_fn = None
        self.deferred_accumulation = False
        self.last_reason = None
        self.reason_since = time.monotonic()

    def _measurement(self):
        now = time.monotonic()
        if now - self.last_measurement_at >= self.config.measurement_seconds:
            self.last_measurement_at = now
            started = time.monotonic()
            try:
                self.available_bytes, self.free_inodes = _space(
                    self.base, self.quota_root)
                self.measurement_error = None
            except (OSError, ValueError, RuntimeError) as error:
                self.available_bytes = None
                self.free_inodes = None
                self.measurement_error = type(error).__name__
            finally:
                elapsed = max(0.0, time.monotonic() - started)
                self.measurement_count += 1
                self.measurement_seconds_total += elapsed
                self.measurement_last_seconds = elapsed
        # The sample began before _space read filesystem and quota state.
        # Judge its oldest possible age when the complete read has returned.
        now = time.monotonic()
        fresh = (self.available_bytes is not None and
                 now - self.last_measurement_at <=
                 self.config.measurement_max_age_seconds)
        return fresh

    def _capacity(self):
        now = time.monotonic()
        try:
            summary = self.queue.workers_summary()
            if not isinstance(summary, list):
                raise TypeError("worker summary is not a list")
            compatible = 0
            unknown = 0
            requirements = self._requirements("processing")
            for worker in summary:
                count = int(worker["workers"])
                resources = {name: int(worker[name]) for name in
                             ("cores", "memory", "disk", "gpus")}
                if count < 0 or min(resources.values()) < 0:
                    raise ValueError("negative worker resource")
                classification = self._compatible(resources, requirements)
                if classification is True:
                    compatible += count
                elif classification is None:
                    unknown += count
            self.last_capacity = compatible
            self.capacity_at = now
            if any(value is None for value in requirements.values()):
                self.last_capacity_quality = (
                    "connected_rounded_declared_fit_auto_allocation_estimate" +
                    ("_with_rounding_unknown" if unknown else ""))
            else:
                self.last_capacity_quality = (
                    "connected_rounded_declared_resource_bounds" +
                    ("_with_unknown" if unknown else "_conservative"))
            return compatible, self.last_capacity_quality
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            if (self.last_capacity is not None and
                    now - self.capacity_at <= 30):
                self.last_capacity_quality = "recent_valid_summary"
                return self.last_capacity, self.last_capacity_quality
            self.last_capacity_quality = "unknown_bootstrap"
            return 0, self.last_capacity_quality

    def _requirements(self, category):
        executor = self.queue.executor
        cores = (self.processing_cores if category == "processing" else
                 self.accumulating_cores)
        memory = (getattr(executor, "memory", None) if category == "processing"
                  else self.accumulating_memory_mb)
        if not memory:
            memory = getattr(executor, "memory", None)
        requirements = {
            "cores": max(1, int(cores or getattr(executor, "cores", None) or 1)),
            "memory": (max(1, int(memory)) if memory else None),
            "disk": (max(1, int(executor.disk)) if getattr(executor, "disk", None)
                     else None),
            "gpus": max(0, int(getattr(executor, "gpus", None) or 0)),
        }
        # The category first_allocation is computed on the largest connected
        # worker. With proportional allocation it is not a minimum for other
        # workers. These are the requirements declared by Coffea and the
        # guarded launchers; native per-worker allocation remains authoritative.
        return requirements

    @staticmethod
    def _compatible(resources, requirements):
        unknown = False
        for name, needed in requirements.items():
            observed = resources[name]
            if name in ("cores", "gpus"):
                if observed < needed:
                    return False
            elif observed < 1:
                return False
            elif needed is None:
                # Proportional allocation fills this dimension from the
                # worker itself; its value is not a cross-worker minimum.
                continue
            elif observed < needed:
                # Native workers_summary rounds memory/disk down to one of
                # eight subdivisions of the current power-of-two interval.
                lower_power = 2 ** math.floor(math.log2(observed))
                if observed + lower_power / 8 <= needed:
                    return False
                unknown = True
        return None if unknown else True

    def _native_states(self):
        from ndcctools import work_queue as wq

        counts = {"ready": 0, "running": 0, "waiting_retrieval": 0,
                  "retrieved": 0, "accum_pending": 0}
        for task_id, task in tuple(self.queue._task_table.items()):
            category = task.category
            if not (category.startswith("processing") or
                    category == "accumulating"):
                continue
            state = self.queue.task_state(task_id)
            if category == "accumulating":
                if state in (wq.WORK_QUEUE_TASK_READY,
                             wq.WORK_QUEUE_TASK_RUNNING,
                             wq.WORK_QUEUE_TASK_WAITING_RETRIEVAL,
                             wq.WORK_QUEUE_TASK_RETRIEVED):
                    counts["accum_pending"] += 1
                continue
            if state == wq.WORK_QUEUE_TASK_READY:
                counts["ready"] += 1
            elif state == wq.WORK_QUEUE_TASK_RUNNING:
                counts["running"] += 1
            elif state == wq.WORK_QUEUE_TASK_WAITING_RETRIEVAL:
                counts["waiting_retrieval"] += 1
            elif state == wq.WORK_QUEUE_TASK_RETRIEVED:
                counts["retrieved"] += 1
            elif state not in (wq.WORK_QUEUE_TASK_DONE,
                               wq.WORK_QUEUE_TASK_CANCELED):
                raise RuntimeError("unknown native Work Queue processing state")
        return counts

    def _retained_bytes(self):
        # Inspect only known Coffea output paths, at telemetry cadence.  A
        # bounded walk avoids making large task tables a periodic filesystem
        # storm.  Missing/deleted inputs are not counted as retained bytes.
        seen_tasks = set()
        seen_paths = set()
        total = 0
        complete = True

        roots = (task for task in self.queue._task_table.values()
                 if task.category == "accumulating")
        stack = [iter(self.queue.tasks_to_accumulate), roots]
        while stack:
            if len(seen_tasks) >= 2048:
                complete = False
                break
            try:
                task = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            marker = id(task)
            if marker in seen_tasks:
                continue
            seen_tasks.add(marker)
            path = getattr(task, "outfile_output", None)
            if path and path not in seen_paths:
                seen_paths.add(path)
                if len(seen_paths) > 2048:
                    complete = False
                    break
                try:
                    total += os.stat(path).st_size
                except FileNotFoundError:
                    pass
                except OSError:
                    complete = False
            stack.append(iter(getattr(task, "tasks_to_accumulate", ())))
        return (total if complete else None), complete, len(seen_paths)

    def _predicted_additional(self, counts):
        processing = (counts["ready"] + counts["running"] +
                      counts["waiting_retrieval"])
        reduction = 0
        for task in self.queue._task_table.values():
            if task.category == "accumulating":
                reduction += 2 * max(
                    self.output_estimate_bytes,
                    int(max(0, getattr(task, "fin_size", 0)) * 1_000_000),
                )
        return processing * self.output_estimate_bytes + reduction

    def _reducer_capacity(self, summary):
        try:
            requirements = self._requirements("accumulating")
            compatible = sum(int(row["workers"]) for row in summary
                       if self._compatible(
                           {name: int(row[name]) for name in requirements},
                           requirements) is True)
            self.last_reducer_capacity_quality = (
                "connected_rounded_declared_fit_auto_allocation_estimate"
                if any(value is None for value in requirements.values()) else
                "connected_rounded_declared_resource_bounds")
            return compatible
        except (KeyError, TypeError, ValueError):
            self.last_reducer_capacity_quality = "unknown"
            return None

    def _decision(self, remaining_events, chunksize):
        counts = self._native_states()
        capacity, capacity_quality = self._capacity()
        target = max(self.config.ready_minimum,
                     math.ceil(self.config.ready_factor * capacity))
        deficit = max(0, target - counts["ready"])
        estimate = max(0, math.ceil(remaining_events / max(1, chunksize)))
        reason = "refill"
        allowance = min(deficit, self.config.pass_maximum, estimate)
        if remaining_events <= 0:
            reason, allowance = "input_accounted", 0
        elif deficit == 0:
            reason, allowance = "ready_reserve_full", 0
        elif not self._measurement():
            reason, allowance = "headroom_unknown", 0
        else:
            predicted = self._predicted_additional(counts)
            reducer_transient = 2 * self.output_estimate_bytes
            spendable = (self.available_bytes - self.warning_bytes -
                         predicted - reducer_transient)
            allowance = min(allowance, max(0, spendable // self.output_estimate_bytes))
            if self.free_inodes is None or self.free_inodes < 1000:
                reason, allowance = "inodes_low", 0
            elif allowance == 0:
                reason = "storage_pressure"
            else:
                try:
                    reducer_slots = self._reducer_capacity(
                        self.queue.workers_summary())
                except (AttributeError, RuntimeError):
                    reducer_slots = None
                if reducer_slots == 0 and (
                        counts["accum_pending"] > 0 or self.deferred_accumulation):
                    # Permit a finite intermediate budget while real disk
                    # headroom remains. Re-evaluate on every pass.
                    # Queued reducers already consume transient reservations
                    # in _predicted_additional. Only Coffea-held outputs can
                    # grow without that native reservation; an unrelated
                    # native reducer backlog must not veto processing.
                    intermediate = len(self.queue.tasks_to_accumulate)
                    allowance = min(allowance,
                                    max(0, target - intermediate))
                    if allowance == 0:
                        reason = "reducer_backlog_budget"
        return allowance, reason, counts, target, capacity, capacity_quality, estimate

    def _emit(self, reason, counts, target, capacity, capacity_quality,
              remaining_estimate, admitted=0, force=False):
        now = time.monotonic()
        changed = reason != self.last_reason
        previous_reason = self.last_reason
        if changed:
            pause_duration = now - self.reason_since
            self.reason_since = now
            self.last_reason = reason
        else:
            pause_duration = now - self.reason_since
        if self.telemetry_disabled or (not force and
                now - self.last_emit_at < self.config.telemetry_seconds and
                not (changed and now - self.last_emit_at >= 5)):
            return
        try:
            retained_bytes, retained_complete, retained_paths = self._retained_bytes()
        except (OSError, ValueError, TypeError, RuntimeError, RecursionError):
            retained_bytes, retained_complete, retained_paths = None, False, 0
        try:
            predicted_bytes = self._predicted_additional(counts)
            processing_requirements = self._requirements("processing")
            reducer_requirements = self._requirements("accumulating")
        except (OSError, ValueError, TypeError, RuntimeError):
            predicted_bytes = None
            processing_requirements = reducer_requirements = None
        record = {
            "schema_version": 3,
            "policy_id": "coffea_0_7_22_wq_admission_repair_005",
            "invocation_id": self.invocation_id,
            "manager_label": self.queue.executor.manager_name,
            "utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(now - self.created, 3),
            "config": dict(vars(self.config)),
            "capacity": capacity,
            "capacity_quality": capacity_quality,
            "reducer_capacity_quality": self.last_reducer_capacity_quality,
            "processing_requirements": processing_requirements,
            "reducer_requirements": reducer_requirements,
            "resource_units": {"cores": "cores", "memory": "MB",
                               "disk": "MB", "gpus": "count"},
            "ready_target": target,
            "processing": counts,
            "held_for_grouping": len(self.queue.tasks_to_accumulate),
            "remaining_input_tasks_estimate": remaining_estimate,
            "admitted_in_pass": admitted,
            "admitted_total": self.new_admitted,
            "recovery_submissions_total": self.recovery_submissions,
            "processing_done_total": self.processing_done,
            "reductions_done_total": self.reductions_done,
            "retained_measured_bytes": retained_bytes,
            "retained_measurement_complete": retained_complete,
            "retained_paths_examined": retained_paths,
            "predicted_additional_bytes": predicted_bytes,
            "effective_available_bytes": self.available_bytes,
            "free_inodes": self.free_inodes,
            "measurement_age_seconds": (None if self.last_measurement_at == -math.inf
                                        else round(now - self.last_measurement_at, 3)),
            "measurement_error": self.measurement_error,
            "measurement_count": self.measurement_count,
            "measurement_last_seconds": self.measurement_last_seconds,
            "measurement_seconds_total": round(self.measurement_seconds_total, 6),
            "measurement_invalidations": self.measurement_invalidations,
            "measurement_quality": ("fresh" if self.available_bytes is not None and
                                    now - self.last_measurement_at <=
                                    self.config.measurement_max_age_seconds else
                                    "unknown_or_stale"),
            "warning_bytes": self.warning_bytes,
            "critical_bytes": self.min_free_bytes,
            "output_estimate_bytes": self.output_estimate_bytes,
            "reason": reason,
            "previous_reason": previous_reason if changed else None,
            "reason_duration_seconds": round(0 if changed else pause_duration, 3),
            "previous_reason_duration_seconds": (round(pause_duration, 3)
                                                 if changed else None),
        }
        try:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            if len(encoded) > self.config.telemetry_limit_bytes:
                return
            if self.telemetry_bytes + len(encoded) > self.config.telemetry_limit_bytes:
                os.replace(self.log_path, self.log_path + ".previous")
                self.telemetry_bytes = 0
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(encoded)
            self.telemetry_bytes += len(encoded)
            self.last_emit_at = now
        except (OSError, ValueError, TypeError, OverflowError):
            # Telemetry is explanatory only; it never certifies task success.
            self.telemetry_disabled = True

    def submit_processing(self, original, infile_procc_fn, items):
        from coffea.processor import work_queue_tools

        queue = self.queue
        if work_queue_tools.early_terminate or queue._items_empty:
            queue.specify_num_tasks_left(0)
            return
        stats = queue.stats_coffea
        remaining_events = max(0, stats["events_total"] - stats["events_submitted"])
        allowance, reason, counts, target, capacity, quality, estimated = (
            self._decision(remaining_events, queue.chunksize_current))
        queue.specify_num_tasks_left(estimated)
        admitted = 0
        started = time.monotonic()
        while admitted < allowance:
            if time.monotonic() - started >= self.config.pass_seconds:
                break
            if stats["events_submitted"] >= stats["events_total"]:
                break
            try:
                if stats["events_submitted"] > 0:
                    from coffea.processor.work_queue_tools import _sample_chunksize
                    item = items.send(_sample_chunksize(queue.chunksize_current))
                else:
                    item = next(items)
            except StopIteration:
                queue._items_empty = True
                queue.specify_num_tasks_left(0)
                reason = "input_exhausted"
                break
            self.ordinary_submit = True
            try:
                queue._submit_processing_task(infile_procc_fn, item)
            finally:
                self.ordinary_submit = False
            admitted += 1
            self.new_admitted += 1
            remaining_events = max(0, stats["events_total"] - stats["events_submitted"])
        remaining_estimate = (0 if queue._items_empty else math.ceil(
            remaining_events / max(1, queue.chunksize_current)))
        queue.specify_num_tasks_left(remaining_estimate)
        self._emit(reason, self._native_states(), target, capacity, quality,
                   remaining_estimate,
                   admitted=admitted)

    def submit_accumulation(self, original, infile_accum_fn):
        from coffea.processor import work_queue_tools

        self.last_accum_fn = infile_accum_fn
        queue = self.queue
        treereduction = queue.executor.treereduction
        stats = queue.stats_coffea
        force = (self.finalizing or work_queue_tools.early_terminate or
                 stats["events_processed"] >= stats["events_total"])
        minimum = 2 if force else treereduction
        if (len(queue.tasks_to_accumulate) < minimum or
                (not force and len(queue.tasks_to_accumulate) <
                 2 * treereduction - 1)):
            self.deferred_accumulation = False
            return
        counts = self._native_states()
        if not self._measurement():
            self.deferred_accumulation = True
            return
        from coffea.processor.work_queue_tools import AccumCoffeaWQTask

        queue.tasks_to_accumulate.sort(key=lambda task: task.fout_size)
        started = time.monotonic()
        submitted = 0
        while len(queue.tasks_to_accumulate) >= minimum:
            if (submitted >= self.config.pass_maximum or
                    (submitted and time.monotonic() - started >=
                     self.config.pass_seconds)):
                break
            group = queue.tasks_to_accumulate[:treereduction]
            cost = 2 * max(
                self.output_estimate_bytes,
                sum(max(0, int(task.fout_size * 1_000_000)) for task in group))
            counts = self._native_states()
            if (self.free_inodes is None or self.free_inodes < 1000 or
                    self.available_bytes - self._predicted_additional(counts) -
                    cost < self.min_free_bytes):
                break
            accum_task = AccumCoffeaWQTask(queue, infile_accum_fn, group)
            queue.submit(accum_task)
            record = getattr(queue, "_phase_telemetry_record", None)
            if record is not None:
                for child in group:
                    record("reducer_child_edge", reducer_task_id=accum_task.id,
                           child_task_id=child.id)
            del queue.tasks_to_accumulate[:len(group)]
            stats["accumulations_submitted"] += 1
            for task in group:
                task.task_accum_log(queue.executor.tasks_accum_log,
                                    "done", task.id)
            submitted += 1
        self.deferred_accumulation = len(queue.tasks_to_accumulate) >= minimum

    def _local_final_accumulation(self, original, accumulator):
        record = getattr(self.queue, "_phase_telemetry_record", None)
        roots = self.queue.tasks_to_accumulate
        root_task_id = roots[0].id if len(roots) == 1 else None
        if record is not None:
            record("manager_final_accumulation_before", root_task_id=root_task_id,
                   root_task_count=len(roots), outcome="entered")
        outcome = "returned"
        try:
            return original(accumulator)
        except BaseException:
            outcome = "raised"
            raise
        finally:
            if record is not None:
                record("manager_final_accumulation_after", root_task_id=root_task_id,
                       root_task_count=len(roots), outcome=outcome)

    def final_accumulation(self, original, accumulator):
        from coffea.processor import work_queue_tools

        if work_queue_tools.early_terminate:
            return self._local_final_accumulation(original, accumulator)
        queue = self.queue
        self.finalizing = True
        try:
            while True:
                if work_queue_tools.early_terminate:
                    break
                pending = self._native_states()["accum_pending"]
                if len(queue.tasks_to_accumulate) <= 1 and pending == 0:
                    break
                if len(queue.tasks_to_accumulate) >= 2:
                    queue._submit_accum_tasks(self.last_accum_fn)
                task = queue.wait(1)
                if task is None:
                    continue
                if not task.successful():
                    task.resubmit(queue)
                    continue
                queue.tasks_to_accumulate.append(task)
                if task.category == "accumulating":
                    queue.stats_coffea["accumulations_done"] += 1
                else:
                    raise RuntimeError("unexpected processing result during final drain")
            return self._local_final_accumulation(original, accumulator)
        finally:
            self.finalizing = False


def install_processing_admission(queue, *, base, quota_root, warning_bytes,
                                 min_free_bytes, processing_cores=None,
                                 accumulating_memory_mb=None,
                                 accumulating_cores=None, config=None):
    """Install once on one Coffea 0.7.22 queue; update phase-local settings."""
    import coffea
    from ndcctools import work_queue as wq

    if coffea.__version__ != "0.7.22":
        raise RuntimeError("processing admission requires Coffea 0.7.22")
    required = ("_submit_processing_tasks", "_submit_processing_task",
                "_submit_accum_tasks", "_final_accumulation", "submit", "wait", "workers_summary",
                "task_state", "specify_num_tasks_left", "_task_table",
                "stats_coffea", "tasks_to_accumulate", "executor")
    if any(not hasattr(queue, name) for name in required):
        raise RuntimeError("incompatible Coffea/Work Queue admission interface")
    if not isinstance(queue._task_table, dict):
        raise RuntimeError("native Work Queue task table contract changed")
    if (len(inspect.signature(queue._submit_processing_tasks).parameters) != 2 or
            len(inspect.signature(queue._submit_accum_tasks).parameters) != 1):
        raise RuntimeError("Coffea submission signatures changed")
    if not all(hasattr(wq, name) for name in (
            "WORK_QUEUE_TASK_READY", "WORK_QUEUE_TASK_RUNNING",
            "WORK_QUEUE_TASK_WAITING_RETRIEVAL", "WORK_QUEUE_TASK_RETRIEVED")):
        raise RuntimeError("native Work Queue task states unavailable")
    if config is None:
        config = admission_config()
    if not isinstance(config, admission_config):
        raise TypeError("config must be admission_config")
    existing = getattr(queue, "_processing_admission_coordinator", None)
    if existing is not None:
        if (existing.base, existing.quota_root, existing.warning_bytes,
                existing.min_free_bytes, existing.config) != (
                os.fspath(base), quota_root, warning_bytes,
                min_free_bytes, config):
            raise RuntimeError("same queue initialized with different admission policy")
        existing.accumulating_memory_mb = accumulating_memory_mb
        existing.accumulating_cores = accumulating_cores
        existing.processing_cores = processing_cores
        if queue.executor.desc != "Preprocessing":
            existing.begin_processing()
        return existing

    coordinator = _coordinator(queue, config, base, quota_root,
                               warning_bytes, min_free_bytes)
    coordinator.accumulating_memory_mb = accumulating_memory_mb
    coordinator.accumulating_cores = accumulating_cores
    coordinator.processing_cores = processing_cores
    original_processing = queue._submit_processing_tasks
    original_accumulation = queue._submit_accum_tasks
    original_submit = queue.submit
    original_wait = queue.wait
    original_final = queue._final_accumulation

    def processing_adapter(self, infile_procc_fn, items):
        return coordinator.submit_processing(original_processing,
                                             infile_procc_fn, items)

    def accumulation_adapter(self, infile_accum_fn):
        return coordinator.submit_accumulation(original_accumulation,
                                               infile_accum_fn)

    def submit_adapter(self, task):
        if task.category.startswith("processing"):
            if not coordinator.ordinary_submit:
                coordinator.recovery_submissions += 1
            now = time.monotonic()
            if now - coordinator.priority_window_at >= coordinator.config.priority_window_seconds:
                coordinator.priority_window_at = now
                coordinator.priority_used = 0
            if (coordinator.priority_used < coordinator.config.priority_burst and
                    task.priority == 0):
                task.specify_priority(1)
                coordinator.priority_used += 1
        return original_submit(task)

    def wait_adapter(self, timeout=None):
        task = original_wait(timeout)
        if task is not None:
            # Original wait has already cleaned successful reducer inputs.
            # A cached pre-result free-space observation cannot cover files
            # materialized or removed by this transition.
            coordinator.last_measurement_at = -math.inf
            coordinator.measurement_invalidations += 1
        if task is not None and task.successful():
            if task.category.startswith("processing"):
                coordinator.processing_done += 1
                size_bytes = max(0, int(task.fout_size * 1_000_000))
                coordinator.output_estimate_bytes = max(
                    coordinator.output_estimate_bytes, 2 * size_bytes)
            elif task.category == "accumulating":
                coordinator.reductions_done += 1
        if (task is None and coordinator.deferred_accumulation and
                coordinator.last_accum_fn is not None):
            self._submit_accum_tasks(coordinator.last_accum_fn)
        return task

    def final_adapter(self, accumulator):
        return coordinator.final_accumulation(original_final, accumulator)

    queue._submit_processing_tasks = MethodType(processing_adapter, queue)
    queue._submit_accum_tasks = MethodType(accumulation_adapter, queue)
    queue.submit = MethodType(submit_adapter, queue)
    queue.wait = MethodType(wait_adapter, queue)
    queue._final_accumulation = MethodType(final_adapter, queue)
    queue._processing_admission_coordinator = coordinator
    if queue.executor.desc != "Preprocessing":
        coordinator.begin_processing()
    return coordinator
