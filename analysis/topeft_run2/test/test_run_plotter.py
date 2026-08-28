import shlex
import subprocess
from pathlib import Path


def _run_plotter(tmp_path, *args):
    repo_root = Path(__file__).resolve().parents[3]
    wrapper_path = repo_root / "analysis/topeft_run2/run_plotter.sh"
    return subprocess.run(
        [str(wrapper_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _dry_run_command(result):
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("  ")
    )
    return shlex.split(command_line)


def _input_path(tmp_path, name):
    path = tmp_path / name
    path.touch()
    return path


def _base_args(tmp_path, input_path, *extra_args):
    return (
        "-f",
        str(input_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--dry-run",
        *extra_args,
    )


def test_single_input_backward_compatibility(tmp_path):
    input_path = _input_path(tmp_path, "plots_CR_input.pkl.gz")

    result = _run_plotter(tmp_path, *_base_args(tmp_path, input_path))

    assert result.returncode == 0, result.stderr
    assert "Auto-detected region 'CR' from input filename." in result.stdout
    command = _dry_run_command(result)
    assert command.count("-f") == 1
    assert command[command.index("-f") + 1] == str(input_path)
    assert "--cr" in command
    assert "--unblind" in command


def test_two_input_command_forwards_two_flags_in_order(tmp_path):
    period_2022_path = _input_path(tmp_path, "first_CR_input.pkl.gz")
    period_2023_path = _input_path(tmp_path, "second_CR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        "-f",
        str(period_2022_path),
        "-f",
        str(period_2023_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    command = _dry_run_command(result)
    input_indexes = [index for index, token in enumerate(command) if token == "-f"]
    assert [command[index + 1] for index in input_indexes] == [
        str(period_2022_path),
        str(period_2023_path),
    ]


def test_missing_any_input_rejected(tmp_path):
    readable_path = _input_path(tmp_path, "first_CR_input.pkl.gz")
    missing_path = tmp_path / "missing_CR_input.pkl.gz"

    result = _run_plotter(
        tmp_path,
        "-f",
        str(readable_path),
        "-f",
        str(missing_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--dry-run",
    )

    assert result.returncode != 0
    assert f"Input pickle '{missing_path}' is missing or unreadable." in result.stderr


def test_two_readable_inputs_accepted_in_dry_run(tmp_path):
    first_path = _input_path(tmp_path, "first_CR_input.pkl.gz")
    second_path = _input_path(tmp_path, "second_CR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        "-f",
        str(first_path),
        "-f",
        str(second_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "Dry-run requested; skipping execution." in result.stdout


def test_conflicting_regions_fail_without_override(tmp_path):
    cr_path = _input_path(tmp_path, "first_CR_input.pkl.gz")
    sr_path = _input_path(tmp_path, "second_SR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        "-f",
        str(cr_path),
        "-f",
        str(sr_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "Input filenames resolve to conflicting regions" in result.stderr


def test_conflicting_regions_accept_with_explicit_cr(tmp_path):
    cr_path = _input_path(tmp_path, "first_CR_input.pkl.gz")
    sr_path = _input_path(tmp_path, "second_SR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        "-f",
        str(cr_path),
        "-f",
        str(sr_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--cr",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    command = _dry_run_command(result)
    assert "--cr" in command
    assert "--unblind" in command


def test_conflicting_regions_accept_with_explicit_sr(tmp_path):
    cr_path = _input_path(tmp_path, "first_CR_input.pkl.gz")
    sr_path = _input_path(tmp_path, "second_SR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        "-f",
        str(cr_path),
        "-f",
        str(sr_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "2022",
        "--sr",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    command = _dry_run_command(result)
    assert "--sr" in command
    assert "--blind" in command


def test_run3_alias_expansion_preserved(tmp_path):
    input_path = _input_path(tmp_path, "plots_CR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        "-f",
        str(input_path),
        "-o",
        str(tmp_path / "plots"),
        "-y",
        "run3",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    command = _dry_run_command(result)
    year_index = command.index("-y")
    assert command[year_index + 1:year_index + 5] == [
        "2022",
        "2022EE",
        "2023",
        "2023BPix",
    ]


def test_workers_forwarding_preserved(tmp_path):
    input_path = _input_path(tmp_path, "plots_CR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        *_base_args(tmp_path, input_path, "--workers", "4"),
    )

    assert result.returncode == 0, result.stderr
    command = _dry_run_command(result)
    assert command[command.index("--workers") + 1] == "4"


def test_channel_output_forwarding_preserved(tmp_path):
    input_path = _input_path(tmp_path, "plots_CR_input.pkl.gz")

    result = _run_plotter(
        tmp_path,
        *_base_args(tmp_path, input_path, "--channel-output", "merged"),
    )

    assert result.returncode == 0, result.stderr
    command = _dry_run_command(result)
    assert command[command.index("--channel-output") + 1] == "merged"
