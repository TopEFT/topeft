import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "topeft_run2"
FULL_R3_RUN = ANALYSIS_DIR / "fullR3_run.sh"


def run_dry_wrapper(mode, *extra_args):
    return subprocess.run(
        [
            str(FULL_R3_RUN),
            "-y",
            "2022",
            "-t",
            "wrapper_test",
            mode,
            "--dry-run",
            *extra_args,
        ],
        cwd=ANALYSIS_DIR,
        check=False,
        capture_output=True,
        text=True,
    )


def resolved_command(result):
    marker = "Running the following command:\n"
    assert marker in result.stdout
    return result.stdout.split(marker, maxsplit=1)[1].split("\n\n", maxsplit=1)[0]


def option_count(command, option):
    return command.split().count(option)


def test_cr_and_sr_do_not_enable_data_driven_flags_without_caller_options():
    for mode in ("--cr", "--sr"):
        result = run_dry_wrapper(mode)

        assert result.returncode == 0, result.stderr
        command = resolved_command(result)
        assert option_count(command, "--do-systs") == 0
        assert option_count(command, "--do-np") == 0


@pytest.mark.parametrize(
    ("extra_args", "expected_systs", "expected_np"),
    [
        (("--do-systs",), 1, 0),
        (("--do-np",), 0, 1),
        (("--do-systs", "--do-np"), 1, 1),
    ],
)
def test_sr_forwards_caller_controlled_data_driven_flags_once(
    extra_args, expected_systs, expected_np
):
    result = run_dry_wrapper("--sr", *extra_args)

    assert result.returncode == 0, result.stderr
    command = resolved_command(result)
    assert option_count(command, "--do-systs") == expected_systs
    assert option_count(command, "--do-np") == expected_np


@pytest.mark.parametrize(
    ("extra_args", "expected_wrapper"),
    [
        ((), "fullR3_run.sh"),
        (("--sample-universe-wrapper", "run_cr.sh -> fullR3_run.sh"), "run_cr.sh -> fullR3_run.sh"),
        (("--sample-universe-wrapper=run_cr.sh -> fullR3_run.sh",), "run_cr.sh -> fullR3_run.sh"),
    ],
)
def test_sample_universe_wrapper_is_forwarded_once(extra_args, expected_wrapper):
    result = run_dry_wrapper("--sr", *extra_args)

    assert result.returncode == 0, result.stderr
    command = resolved_command(result)
    assert option_count(command, "--sample-universe-wrapper") == 1
    assert f"--sample-universe-wrapper {expected_wrapper}" in command


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--sample-universe-wrapper",),
        ("--sample-universe-wrapper", ""),
        ("--sample-universe-wrapper=",),
        (
            "--sample-universe-wrapper",
            "first",
            "--sample-universe-wrapper=second",
        ),
    ],
)
def test_sample_universe_wrapper_rejects_missing_empty_or_duplicate_values(extra_args):
    result = run_dry_wrapper("--sr", *extra_args)

    assert result.returncode != 0
    assert "sample-universe-wrapper" in result.stderr
