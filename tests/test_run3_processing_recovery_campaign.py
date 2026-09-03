import csv
from pathlib import Path
import subprocess
import time


WORKSPACE = Path("/users/apiccine/work/correction-lib")
WRAPPER = WORKSPACE / "topeft/analysis/topeft_run2/run3_processing_recovery_campaign.sh"
BLOCK1_MANIFEST = WORKSPACE / "reports/diagnostics/SRPLOT_010R2_reconcile_live_run3_processing_state_after_ambiguous_interruption/SRPLOT_010R2_processing_snapshot_2.tsv"
BLOCK1_MANIFEST_HEADER = [
    "snapshot_timestamp", "block", "path", "file_type", "size_bytes",
    "mtime_ns", "sha256", "snapshot_read_state",
]
INPUT_ROOT = Path("/groups/klannon/apiccine/run3_full_260819_v2_corrected_np_260822")
OUTPUT_ROOT = WORKSPACE / "topeft/histos/SR_preappr_Aug31_resilient_run3/merged_njets"
EXPECTED_TASKS = [f"run3_block_{index}_processing" for index in range(2, 6)]
EXPECTED_PKLS = [
    INPUT_ROOT / "2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_m_offZ_njets-lj0pt-ptll-lt_np.pkl.gz",
    INPUT_ROOT / "2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_p_offZ_njets-lj0pt-ptll-lt_np.pkl.gz",
    INPUT_ROOT / "2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_onZ_tau_njets-lj0pt-ptz-lt_np.pkl.gz",
    INPUT_ROOT / "2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_fwd_njets-lj0pt-ptz-lt_np.pkl.gz",
]
EXPECTED_OUTPUTS = [OUTPUT_ROOT / f"block_{index}" for index in range(2, 6)]


def run_wrapper(*args):
    return subprocess.run(
        [str(WRAPPER), *map(str, args)], text=True, capture_output=True,
        check=False, timeout=60,
    )


def run_stub(tmp_path, scenario):
    runtime = tmp_path / f"{scenario}_runtime"
    fixture = tmp_path / f"{scenario}_fixture"
    result = run_wrapper(
        "--stub-scenario", scenario, "--runtime-root", runtime,
        "--fixture-root", fixture,
    )
    return result, runtime, fixture


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def assert_uniform_tsv(path):
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    rows = list(csv.reader(raw.decode().splitlines(), delimiter="\t"))
    assert rows and len({len(row) for row in rows}) == 1


def terminal_rows(runtime):
    return [row for row in read_tsv(runtime / "attempt_events.tsv") if row["event"] == "terminal"]


def test_shell_syntax_and_exact_independent_print_plan():
    assert subprocess.run(["bash", "-n", str(WRAPPER)], check=False).returncode == 0
    result = run_wrapper("--print-plan")
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
    assert len(rows) == 4
    assert [row["logical_task"] for row in rows] == EXPECTED_TASKS
    assert [Path(row["pkl"]) for row in rows] == EXPECTED_PKLS
    assert [Path(row["output_directory"]) for row in rows] == EXPECTED_OUTPUTS
    assert all(row["channel_output"] == "merged-njets" for row in rows)
    assert all(row["binning"] == "processing" for row in rows)
    assert all(row["workers"] == "6" for row in rows)
    for block_number, row in zip(range(2, 6), rows):
        command = row["exact_command"]
        assert "--binning processing" in command
        assert "--channel-output merged-njets" in command
        assert "--workers 6" in command
        assert f"srplot008v7_run3_block{block_number}" in command
        assert "--binning fitting" not in command
        assert "srplot008v7_run3_block1" not in command
        assert "processing/merged_njets" not in command


def test_all_success_and_existing_parent_are_valid(tmp_path):
    for scenario in ("all_success", "parent_exists", "unrelated_tracked_mutation"):
        result, runtime, fixture = run_stub(tmp_path, scenario)
        assert result.returncode == 0, result.stderr
        terminals = terminal_rows(runtime)
        assert [row["logical_task"] for row in terminals] == EXPECTED_TASKS
        assert {row["terminal_state"] for row in terminals} == {"success_processing"}
        assert {row["exit_code"] for row in terminals} == {"0"}
        assert {row["output_validation_state"] for row in terminals} == {"passed"}
        assert {row["fitting_manifest_integrity_state"] for row in terminals} == {"passed"}
        assert {row["block1_acceptance_state"] for row in terminals} == {"passed"}
        assert {row["dependency_integrity_state"] for row in terminals} == {"passed"}
        assert len(read_tsv(runtime / "processing_output_inventory.tsv")) == 4
        assert len(list(fixture.glob("block_*/*_processing.png"))) == 5
        assert (fixture / "block_1/run3_block_1_processing.png").read_text() == "accepted block1 processing output\n"
        stub_manifest = read_tsv(fixture / "accepted_block1_manifest.tsv")
        assert list(stub_manifest[0]) == BLOCK1_MANIFEST_HEADER
        assert {row["snapshot_read_state"] for row in stub_manifest} == {"stable_during_read"}
        for path in runtime.glob("*.tsv"):
            assert_uniform_tsv(path)


def test_exact_real_block1_manifest_is_accepted_nonproducing(tmp_path):
    manifest_rows = read_tsv(BLOCK1_MANIFEST)
    assert len(manifest_rows) == 111
    assert list(manifest_rows[0]) == BLOCK1_MANIFEST_HEADER
    assert {row["block"] for row in manifest_rows} == {"block_1"}
    assert {row["snapshot_read_state"] for row in manifest_rows} == {"stable_during_read"}
    runtime = tmp_path / "existing_runtime"
    runtime.mkdir()
    result = run_wrapper("--execute", "--runtime-root", runtime)
    assert result.returncode == 3
    assert result.stderr == f"Preflight error: runtime root already exists and requires recovery classification: {runtime}.\n"
    assert not (runtime / "attempt_events.tsv").exists()


def test_known_failure_stops_later_tasks_and_never_retries(tmp_path):
    result, runtime, _ = run_stub(tmp_path, "known_failure")
    assert result.returncode == 1
    events = read_tsv(runtime / "attempt_events.tsv")
    terminals = [row for row in events if row["event"] == "terminal"]
    assert [row["logical_task"] for row in terminals] == EXPECTED_TASKS[:2]
    assert terminals[0]["terminal_state"] == "success_processing"
    assert terminals[1]["terminal_state"] == "failed_processing"
    assert terminals[1]["exit_code"] == "7"
    assert not any(row["logical_task"] in EXPECTED_TASKS[2:] for row in events)
    assert {row["attempt_id"] for row in events if row["logical_task"] == EXPECTED_TASKS[1]} == {
        "run3_block_3_processing_attempt_001"
    }


def test_ambiguous_attempt_has_durable_identity_and_is_not_retried(tmp_path):
    result, runtime, fixture = run_stub(tmp_path, "ambiguous")
    assert result.returncode == 75
    time.sleep(1.0)
    events = read_tsv(runtime / "attempt_events.tsv")
    block_three = [row for row in events if row["logical_task"] == EXPECTED_TASKS[1]]
    assert [row["event"] for row in block_three] == ["started", "process_started"]
    started = block_three[1]
    for field in ("pid", "ppid", "process_group_id", "session_id", "process_start_ticks", "process_start_time", "command_sha256"):
        assert started[field]
    assert len(started["command_sha256"]) == 64
    assert all(row["exit_code"] == "" for row in block_three)
    assert not any(row["logical_task"] in EXPECTED_TASKS[2:] for row in events)
    processing_before = sorted(fixture.glob("block_*/*_processing.png"))
    status = run_wrapper("--status", "--runtime-root", runtime)
    assert status.returncode == 0
    status_rows = list(csv.DictReader(status.stdout.splitlines(), delimiter="\t"))
    assert [row["status"] for row in status_rows] == [
        "externally_accepted_existing_success", "success_processing",
        "ambiguous_interruption", "not_started", "not_started",
    ]
    assert run_wrapper("--finalize", "--runtime-root", runtime).returncode == 75
    assert sorted(fixture.glob("block_*/*_processing.png")) == processing_before
    assert len([row for row in read_tsv(runtime / "attempt_events.tsv") if row["logical_task"] == EXPECTED_TASKS[1]]) == 2


def test_process_identity_setup_failure_fails_closed(tmp_path):
    result, runtime, fixture = run_stub(tmp_path, "process_identity_setup_failure")
    assert result.returncode == 75
    time.sleep(0.5)
    events = read_tsv(runtime / "attempt_events.tsv")
    block_two = [row for row in events if row["logical_task"] == EXPECTED_TASKS[0]]
    assert [row["event"] for row in block_two] == ["started", "identity_binding_failed"]
    failed_identity = block_two[-1]
    assert failed_identity["terminal_state"] == "ambiguous_interruption"
    assert failed_identity["exit_code"] == ""
    assert failed_identity["pid"]
    assert failed_identity["command_sha256"]
    assert not any(row["logical_task"] in EXPECTED_TASKS[1:] for row in events)
    assert len(list(fixture.glob("block_*/*_processing.png"))) == 1
    status = run_wrapper("--status", "--runtime-root", runtime)
    assert status.returncode == 0
    rows = list(csv.DictReader(status.stdout.splitlines(), delimiter="\t"))
    assert [row["status"] for row in rows] == [
        "externally_accepted_existing_success", "ambiguous_interruption",
        "not_started", "not_started", "not_started",
    ]


def test_prelaunch_integrity_and_collision_gates(tmp_path):
    cases = {
        "collision": "Processing-collision error",
        "block1_mutation": "Block1-acceptance error",
        "wrong_block1_snapshot_state": "Block1-acceptance error",
        "fitting_mutation": "Fitting-manifest error",
        "dependency_mutation": "Dependency-integrity error",
        "same_size_changed_mtime": "mtime_ns mismatch",
    }
    for scenario, message in cases.items():
        result, runtime, _ = run_stub(tmp_path, scenario)
        assert result.returncode == 3
        assert message in result.stderr
        assert not runtime.exists()


def test_status_and_finalize_do_not_launch_tasks(tmp_path):
    runtime = tmp_path / "read_only_runtime"
    status = run_wrapper("--status", "--runtime-root", runtime)
    assert status.returncode == 0
    rows = list(csv.DictReader(status.stdout.splitlines(), delimiter="\t"))
    assert rows[0] == {
        "logical_task": "run3_block_1_processing",
        "status": "externally_accepted_existing_success",
    }
    assert [row["status"] for row in rows[1:]] == ["not_started"] * 4
    assert not (runtime / "attempt_events.tsv").exists()
    finalized = run_wrapper("--finalize", "--runtime-root", runtime)
    assert finalized.returncode == 3
    assert not (runtime / "attempt_events.tsv").exists()
    assert not runtime.exists()
    assert "finalization_state\tno_runtime_state" in finalized.stdout
