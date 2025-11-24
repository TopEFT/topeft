import subprocess
from pathlib import Path


def _extract_block(output: str):
    marker_prefix = "--- "
    marker_suffix = " ---"
    lines = output.splitlines()
    marker_index = None

    for index, line in enumerate(lines):
        if line.startswith(marker_prefix) and line.endswith(marker_suffix):
            marker_index = index
            path = line[len(marker_prefix):-len(marker_suffix)]
            break
    else:
        return None, ""

    body_lines = []
    for line in lines[marker_index + 1:]:
        if line.startswith(marker_prefix) and line.endswith(marker_suffix):
            break
        body_lines.append(line)

    return path, "\n".join(body_lines)


def test_submit_plotter_condor_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]

    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "plots"

    submit_helper = repo_root / "analysis/topeft_run2/submit_plotter_condor.sh"

    result = subprocess.run(
        [
            str(submit_helper),
            "--dry-run",
            "--ceph-root",
            str(repo_root),
            "--log-dir",
            str(log_dir),
            "-f",
            "/cephfs/example/plotsCR_Run2.pkl.gz",
            "-o",
            str(output_dir),
            "-y",
            "run2",
            "--log-y",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert log_dir.exists(), "Log directory should be created during dry-run"

    sub_path, sub_body = _extract_block(result.stdout)
    assert sub_path is not None, "Dry-run output did not include a submission file"

    analysis_dir = repo_root / "analysis/topeft_run2"

    assert f"initialdir              = {analysis_dir}" in sub_body
    assert f"log                     = {log_dir}/plotter.$(Cluster).$(Process).log" in sub_body
    assert f"output                  = {log_dir}/plotter.$(Cluster).$(Process).out" in sub_body
    assert f"error                   = {log_dir}/plotter.$(Cluster).$(Process).err" in sub_body
    assert "queue 1" in sub_body
    assert "should_transfer_files   = NO" in sub_body
    assert "transfer_executable" not in sub_body

    ceph_entry = analysis_dir / "condor_plotter_entry.sh"
    assert f'executable              = "{ceph_entry}"' in sub_body

    staged_repo_root = analysis_dir.parent
    expected_environment = (
        "environment              = "
        f'"TOPEFT_REPO_ROOT={staged_repo_root};TOPEFT_ENTRY_DIR={analysis_dir}"'
    )
    assert expected_environment in sub_body

    assert "--log-y" in sub_body
    assert "--log-y" in result.stderr

    entry_script = (analysis_dir / "condor_plotter_entry.sh").read_text()
    assert "unset PYTHONPATH" in entry_script
    assert '"${ENTRY_DIR}/run_plotter.sh" "$@"' in entry_script
