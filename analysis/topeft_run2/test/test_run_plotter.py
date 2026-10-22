import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


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


def _campaign_fixture(tmp_path):
    rows = []
    for unit, kind, variable in (
        ("first_other", "other", "lj0pt"),
        ("middle_njets", "njets", "njets"),
        ("last_other", "other", "ptz"),
    ):
        pkl = tmp_path / f"{unit} SR.pkl.gz"
        sidecar = tmp_path / f"{unit} SR.pkl.gz.metadata.json"
        pkl.touch()
        sidecar.touch()
        rows.append("\t".join((
            unit, kind, "run3", "combined", unit, "2022", variable, "1",
            str(pkl), "0" * 64, str(sidecar), "0" * 64,
        )))
    manifest = tmp_path / "campaign.tsv"
    manifest.write_text(
        "unit\tkind\trun\tview\towner\tyears\tvariables\texpected_figures\t"
        "pkl\thistorical_pkl_sha256\tsidecar\thistorical_sidecar_sha256\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "histos" / f"test_campaign_{tmp_path.name}"
    assert not root.exists() and not root.is_symlink()
    return manifest, root


def test_campaign_dry_run_orders_njets_before_other_without_output(tmp_path):
    manifest, root = _campaign_fixture(tmp_path)
    result = _run_plotter(
        tmp_path, "--campaign-manifest", str(manifest),
        "--campaign-phase", "all", "-o", str(root), "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    units = [line.split()[1] for line in result.stdout.splitlines()
             if line.startswith("Unit: ")]
    assert units == ["middle_njets", "first_other", "last_other"]
    assert "Rows: njets=1 other=2" in result.stdout
    assert not root.exists() and not root.is_symlink()
    commands = [shlex.split(line) for line in result.stdout.splitlines()
                if line.startswith("  ")]
    assert len(commands) == 3
    assert all(command.count("-f") == 1 and "--sr" in command
               and "--unblind" in command and "--workers" in command
               for command in commands)


def test_campaign_phase_selection_and_occupied_root_refusal(tmp_path):
    manifest, root = _campaign_fixture(tmp_path)
    selected = _run_plotter(
        tmp_path, "--campaign-manifest", str(manifest),
        "--campaign-phase", "njets", "-o", str(root), "--dry-run",
    )
    assert selected.returncode == 0, selected.stderr
    assert selected.stdout.count("Unit: ") == 1
    assert "Unit: middle_njets" in selected.stdout
    repo_root = Path(__file__).resolve().parents[3]
    occupied = _run_plotter(
        tmp_path, "--campaign-manifest", str(manifest),
        "--campaign-phase", "njets", "-o", str(repo_root / "histos"), "--dry-run",
    )
    assert occupied.returncode != 0
    assert "must be absent" in occupied.stderr


def _inert_campaign_repo(tmp_path, *, failing_child=False, failing_tee=False,
                         missing_guard=False):
    real_repo = Path(__file__).resolve().parents[3]
    fixture_repo = tmp_path / "repo"
    script_dir = fixture_repo / "analysis/topeft_run2"
    script_dir.mkdir(parents=True)
    shutil.copyfile(real_repo / "analysis/topeft_run2/run_plotter.sh",
                    script_dir / "run_plotter.sh")
    (script_dir / "run_plotter.sh").chmod(0o755)
    (script_dir / "make_cr_and_sr_plots.py").write_text(
        "import pathlib,sys\n"
        "args=sys.argv[1:]\n"
        "path=pathlib.Path(args[args.index('-o')+1])/args[args.index('-n')+1]\n"
        "path.mkdir(parents=True,exist_ok=False)\n"
        "print('inert child', path)\n"
        "sys.exit(17 if 'fail' in args[args.index('-f')+1] else 0)\n",
        encoding="utf-8",
    )
    (fixture_repo / "histos").mkdir()
    module_path = fixture_repo / "topeft/modules/node_memory_guard.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# inert source identity fixture\n", encoding="utf-8")
    metadata = fixture_repo / "topeft/params/cr_sr_plots_metadata.yml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("# inert metadata fixture\n", encoding="utf-8")
    bin_dir = fixture_repo / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/bin/sh\ncase \"$*\" in\n"
        "  *'branch --show-current'*) echo te/datacard_sr_reco;;\n"
        "  *'rev-parse HEAD'*) echo fixture_head;;\n"
        "  *) exit 2;;\nesac\n", encoding="utf-8",
    )
    fake_git.chmod(0o755)
    if failing_tee:
        fake_tee = bin_dir / "tee"
        fake_tee.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
        fake_tee.chmod(0o755)
    python_bin = "/users/apiccine/work/miniconda3/envs/clib-env/bin/python"
    if missing_guard:
        fake_python = bin_dir / "python_guard_fixture"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        python_bin = str(fake_python)
    digest = hashlib.sha256(b"").hexdigest()
    rows = []
    for unit in ("first", "second"):
        label = "fail" if failing_child and unit == "first" else unit
        pkl = fixture_repo / f"{label} SR.pkl.gz"
        sidecar = fixture_repo / f"{label} SR.pkl.gz.metadata.json"
        pkl.touch()
        sidecar.touch()
        rows.append("\t".join((unit, "njets", "run3", "combined", unit,
                               "2022", "njets", "1", str(pkl), digest,
                               str(sidecar), digest)))
    manifest = fixture_repo / "campaign.tsv"
    manifest.write_text(
        "unit\tkind\trun\tview\towner\tyears\tvariables\texpected_figures\t"
        "pkl\thistorical_pkl_sha256\tsidecar\thistorical_sidecar_sha256\n"
        + "\n".join(rows) + "\n", encoding="utf-8",
    )
    env = dict(os.environ, PYTHON_BIN=python_bin,
               PYTHONPATH=str(real_repo), PATH=f"{bin_dir}:{os.environ['PATH']}")
    return script_dir / "run_plotter.sh", manifest, fixture_repo / "histos/run", env


def _run_inert_campaign(wrapper, manifest, root, env, *extra):
    return subprocess.run(
        [str(wrapper), "--campaign-manifest", str(manifest),
         "--campaign-phase", "njets", "-o", str(root), *extra],
        cwd=wrapper.parents[2], env=env, capture_output=True, text=True,
        check=False,
    )


def test_campaign_inert_execute_records_sequential_guarded_children(tmp_path):
    wrapper, manifest, root, env = _inert_campaign_repo(tmp_path)
    result = _run_inert_campaign(wrapper, manifest, root, env, "--execute")
    assert result.returncode == 0, result.stderr
    assert (root / "first.guard.jsonl").is_file()
    assert (root / "second.guard.jsonl").is_file()
    assert (root / "first.log").is_file()
    assert (root / "second.log").is_file()
    ledger = (root / "units.tsv").read_text(encoding="utf-8")
    assert ledger.index("first\tstart") < ledger.index("first\tresult")
    assert ledger.index("first\tresult") < ledger.index("second\tstart")
    assert "campaign\tcomplete\tunits=2" in ledger


def test_campaign_execute_accepts_repo_and_script_directory(tmp_path):
    for location in ("repo", "script"):
        wrapper, manifest, root, env = _inert_campaign_repo(tmp_path / location)
        cwd = wrapper.parents[2] if location == "repo" else wrapper.parent
        result = subprocess.run(
            [str(wrapper), "--campaign-manifest", str(manifest),
             "--campaign-phase", "njets", "-o", str(root), "--execute"],
            cwd=cwd, env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert (root / "units.tsv").read_text(encoding="utf-8").count("\tresult\t") == 2


@pytest.mark.parametrize(
    ("fixture_options", "expected"),
    [({"failing_child": True}, "guard=17"),
     ({"failing_tee": True}, "tee=7"),
     ({"missing_guard": True}, "terminal=1")],
)
def test_campaign_inert_failure_stops_before_next_child(tmp_path, fixture_options, expected):
    wrapper, manifest, root, env = _inert_campaign_repo(tmp_path, **fixture_options)
    result = _run_inert_campaign(wrapper, manifest, root, env, "--execute")
    assert result.returncode != 0
    assert expected in (root / "units.tsv").read_text(encoding="utf-8")
    assert not (root / "second.guard.jsonl").exists()
    assert "campaign\tstopped\tfirst" in (root / "units.tsv").read_text(encoding="utf-8")


def test_campaign_rejects_dangling_output_symlink(tmp_path):
    wrapper, manifest, root, env = _inert_campaign_repo(tmp_path)
    root.symlink_to(root.parent / "missing")
    result = _run_inert_campaign(wrapper, manifest, root, env, "--dry-run")
    assert result.returncode != 0
    assert "must be absent" in result.stderr


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
