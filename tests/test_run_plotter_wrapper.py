import os
import subprocess
from pathlib import Path


script_path = Path("analysis/topeft_run2/run_plotter.sh").resolve()


def _write_fake_python(tmp_path):
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_PYTHON_ARGS:?}\"\n"
        "exit \"${FAKE_PYTHON_EXIT_CODE:-0}\"\n"
    )
    fake_python.chmod(0o755)
    return fake_python


def test_dry_run_prints_resolved_command_without_creating_output_directory(tmp_path):
    input_path = tmp_path / "CR_input.pkl.gz"
    input_path.write_bytes(b"readable dry-run fixture")
    output_dir = tmp_path / "plots"

    result = subprocess.run(
        [
            str(script_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--year",
            "2018",
            "--cr",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Executing make_cr_and_sr_plots.py with command:" in result.stdout
    assert "Dry-run requested; skipping execution." in result.stdout
    assert str(output_dir) in result.stdout
    assert not output_dir.exists()


def test_sr_wrapper_forwards_parallel_contract_to_python_bin(tmp_path):
    first_input = tmp_path / "SR_first.pkl.gz"
    second_input = tmp_path / "SR_second.pkl.gz"
    first_input.write_bytes(b"first")
    second_input.write_bytes(b"second")
    output_dir = tmp_path / "plots"
    args_path = tmp_path / "fake_python_args.txt"
    fake_python = _write_fake_python(tmp_path)

    result = subprocess.run(
        [
            str(script_path),
            "-f",
            str(first_input),
            "-f",
            str(second_input),
            "-o",
            str(output_dir),
            "-y",
            "run3",
            "--variables",
            "met",
            "lt",
            "--workers",
            "3",
            "--channel-output",
            "split",
            "--verbose",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "FAKE_PYTHON_ARGS": str(args_path),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Resolved plotting region: SR" in result.stdout
    assert "Resolved blinding mode: blinded" in result.stdout
    assert output_dir.is_dir()
    assert args_path.read_text().splitlines() == [
        str(script_path.parent / "make_cr_and_sr_plots.py"),
        "-f",
        str(first_input),
        "-f",
        str(second_input),
        "-o",
        str(output_dir),
        "-y",
        "2022",
        "2022EE",
        "2023",
        "2023BPix",
        "--sr",
        "--blind",
        "--variables",
        "met",
        "lt",
        "--workers",
        "3",
        "--channel-output",
        "split",
        "--verbose",
    ]


def test_cr_wrapper_preserves_default_unblinding_with_python_bin_override(tmp_path):
    input_path = tmp_path / "CR_input.pkl.gz"
    input_path.write_bytes(b"control region")
    args_path = tmp_path / "fake_python_args.txt"
    fake_python = _write_fake_python(tmp_path)

    result = subprocess.run(
        [
            str(script_path),
            "-f",
            str(input_path),
            "-o",
            str(tmp_path / "plots"),
            "-y",
            "run2",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "FAKE_PYTHON_ARGS": str(args_path),
        },
    )

    assert result.returncode == 0, result.stderr
    forwarded_args = args_path.read_text().splitlines()
    assert "--cr" in forwarded_args
    assert "--unblind" in forwarded_args
    assert forwarded_args[forwarded_args.index("-y") + 1 : forwarded_args.index("--cr")] == [
        "2016",
        "2016APV",
        "2017",
        "2018",
    ]


def test_wrapper_propagates_single_child_failure_status(tmp_path):
    input_path = tmp_path / "SR_input.pkl.gz"
    input_path.write_bytes(b"signal region")
    fake_python = _write_fake_python(tmp_path)

    result = subprocess.run(
        [
            str(script_path),
            "-f",
            str(input_path),
            "-o",
            str(tmp_path / "plots"),
            "-y",
            "2022",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "FAKE_PYTHON_ARGS": str(tmp_path / "fake_python_args.txt"),
            "FAKE_PYTHON_EXIT_CODE": "7",
        },
    )

    assert result.returncode == 7
