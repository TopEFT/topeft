import subprocess
from pathlib import Path


script_path = Path("analysis/topeft_run2/run_plotter.sh").resolve()


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
