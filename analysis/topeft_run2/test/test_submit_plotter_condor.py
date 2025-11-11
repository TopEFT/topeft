import subprocess
from pathlib import Path


def _parse_dry_run_sections(output: str):
    sections = {}
    current = None
    job_path = None
    sub_path = None

    for line in output.splitlines():
        if line.startswith("--- ") and line.endswith(" ---"):
            candidate = line[4:-4]
            if candidate.endswith("plotter_job.sh"):
                current = "job"
                job_path = candidate
                sections[current] = []
            elif candidate.endswith("plotter_job.sub"):
                current = "sub"
                sub_path = candidate
                sections[current] = []
            else:
                current = None
            continue

        if current is not None:
            sections[current].append(line)

    return job_path, sub_path, {key: "\n".join(lines) for key, lines in sections.items()}


def test_submit_plotter_condor_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]

    fake_conda_root = tmp_path / "fake_conda"
    conda_profile = fake_conda_root / "etc/profile.d"
    conda_profile.mkdir(parents=True)
    (fake_conda_root / "envs" / "clib-env").mkdir(parents=True)

    # A minimal conda.sh stub so the helper finds an activation script.
    (conda_profile / "conda.sh").write_text(
        "conda() {\n"
        "    if [ \"$1\" = \"activate\" ]; then\n"
        "        return 0\n"
        "    fi\n"
        "}\n"
    )

    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "plots"

    submit_helper = repo_root / "analysis/topeft_run2/submit_plotter_condor.sh"

    result = subprocess.run(
        [
            str(submit_helper),
            "--dry-run",
            "--ceph-root",
            str(repo_root),
            "--conda-prefix",
            str(fake_conda_root / "envs" / "clib-env"),
            "--log-dir",
            str(log_dir),
            "--",
            "-f",
            "/cephfs/example/plotsCR_Run2.pkl.gz",
            "-o",
            str(output_dir),
            "-y",
            "run2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    job_path, sub_path, sections = _parse_dry_run_sections(result.stdout)

    assert job_path is not None, "Dry-run did not emit a job wrapper"
    assert sub_path is not None, "Dry-run did not emit a submission file"

    job_body = sections["job"]
    sub_body = sections["sub"]

    assert str(repo_root) in job_body
    assert "--ceph-root \"${ceph_root}\"" in job_body
    assert "--conda-prefix \"${conda_prefix}\"" in job_body

    assert f'executable              = "{job_path}"' in sub_body
    assert f'initialdir              = "{repo_root}"' in sub_body
    assert "should_transfer_files   = YES" in sub_body

    entry_script = (repo_root / "analysis/topeft_run2/condor_plotter_entry.sh").read_text()
    assert "unset PYTHONPATH" in entry_script
    assert "conda activate clib-env" in entry_script
