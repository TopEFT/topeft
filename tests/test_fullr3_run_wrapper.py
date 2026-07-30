import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis" / "topeft_run2"
FULL_R3_RUN = ANALYSIS_DIR / "fullR3_run.sh"
RUN_CR = ANALYSIS_DIR / "run_cr.sh"


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


def _bash_string_array(script_text, variable_name):
    match = re.search(
        rf"^{re.escape(variable_name)}=\(\n(?P<body>.*?)^\)",
        script_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, variable_name
    return re.findall(r'^\s*"([^"]+)"', match.group("body"), flags=re.MULTILINE)


def _run_cr_command_matrix():
    script_text = RUN_CR.read_text()
    year_sets = _bash_string_array(script_text, "cr_year_sets")
    category_sets = _bash_string_array(script_text, "cr_category_sets")
    variable_set_names = _bash_string_array(
        script_text,
        "cr_category_var_set_names",
    )
    assert len(category_sets) == len(variable_set_names)

    rows = []
    for year_set in year_sets:
        for category_set, variable_set_name in zip(
            category_sets,
            variable_set_names,
        ):
            for histogram_chunk in _bash_string_array(
                script_text,
                variable_set_name,
            ):
                rows.append(
                    {
                        "year_group": year_set,
                        "categories": category_set,
                        "histogram_chunk": histogram_chunk,
                        "output_identity": "{}_{}_{}".format(
                            year_set.replace(" ", "-"),
                            category_set.replace(" ", "-"),
                            histogram_chunk.replace(" ", "-"),
                        ),
                    }
                )
    return script_text, rows


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


def test_run_cr_matrix_has_18_ordered_unique_outputs():
    _script_text, rows = _run_cr_command_matrix()

    assert len(rows) == 18
    assert len({row["output_identity"] for row in rows}) == 18
    assert [row["year_group"] for row in rows] == (
        ["2016APV 2016 2017 2018"] * 6
        + ["2022 2022EE"] * 6
        + ["2023 2023BPix"] * 6
    )
    assert [row["categories"] for row in rows[:6]] == (
        ["2l_CR 2l_CRflip 2los_CRZ 2los_CRtt 3l_CR"] * 3
        + ["1l_1tau_CRtt 1l_1tau_CRDY 2los_1tau"] * 3
    )


def test_run_cr_non_tau_commands_exclude_tau_only_histograms():
    _script_text, rows = _run_cr_command_matrix()
    tau_only_histograms = {"ptz_wtau", "tau0Fpt", "tau0Tpt"}
    non_tau_categories = "2l_CR 2l_CRflip 2los_CRZ 2los_CRtt 3l_CR"

    for command_index in (3, 9, 15):
        histogram_chunk = set(rows[command_index - 1]["histogram_chunk"].split())
        assert histogram_chunk.isdisjoint(tau_only_histograms)
    for row in rows:
        if row["categories"] == non_tau_categories:
            assert set(row["histogram_chunk"].split()).isdisjoint(
                tau_only_histograms
            )


def test_run_cr_tau_third_chunks_retain_tau_only_histograms():
    _script_text, rows = _run_cr_command_matrix()
    tau_only_histograms = {"ptz_wtau", "tau0Fpt", "tau0Tpt"}

    for command_index in (6, 12, 18):
        histogram_chunk = set(rows[command_index - 1]["histogram_chunk"].split())
        assert tau_only_histograms.issubset(histogram_chunk)


def test_run_cr_production_switches_flags_and_provenance_are_preserved():
    script_text, _rows = _run_cr_command_matrix()

    assert re.search(r"^run_sr=false$", script_text, flags=re.MULTILINE)
    assert re.search(r"^dry_run=false$", script_text, flags=re.MULTILINE)
    assert 'cmd_ref+=(--ttgamma-sample-role-policy "${ttgamma_sample_role_policy}")' in script_text
    assert (
        'cmd_ref+=(--sample-universe-wrapper "run_cr.sh -> fullR3_run.sh")'
        in script_text
    )
    assert 'cmd_ref+=(--do-systs)' in script_text
    assert 'cmd_ref+=(--do-np)' in script_text
