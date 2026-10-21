"""The maintained matrix must freeze the Tau option on every native unit."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

from analysis.topeft_run2 import datacard_matrix_runner as runner
from analysis.topeft_run2 import make_datacard_matrix_manifest as generator
from analysis.topeft_run2 import make_cards


def _manifest_args(tmp_path, era, *, suppress=False, mva="per-run"):
    profile = yaml.safe_load(generator._profile_path.read_text())
    roles = {row["input_roles"][era] for row in profile["rows"]}
    inputs = []
    for role in sorted(roles):
        pkl = tmp_path / f"{era}_{role}_np.pkl.gz"
        pkl.write_bytes(b"fixture")
        inputs.append(f"{role}={pkl}")
    missing_parton = tmp_path / f"missing_parton_{era}.root"
    missing_parton.write_bytes(b"fixture")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        era=era,
        lepton_mva_correlation=mva,
        enable_legacy_run3_vsmu_recovery=(era == "run3"),
        suppress_nominal_like_tau_shapes=suppress,
        unblind=False,
        manifest_output=manifest_dir / f"{era}.json",
        output_root=tmp_path / "raw" / era,
        control_root=tmp_path / "control" / era,
        working_directory=generator._repository_root,
        attempt_id="attempt_01",
        python_executable=Path(sys.executable),
        make_cards_path=Path(make_cards.__file__),
        missing_parton_file=missing_parton,
        runtime_contract_id=f"test_{era}",
        runtime_file=[],
        channel_set_key="ALL_CH_LST_SR",
        physical_target=[],
        input_pkl=inputs,
    )


def _assert_perrun_tau_rows(manifest, expected_rows):
    assert len(manifest["rows"]) == expected_rows
    targets = [f"{channel}_{row['distribution']}" for row in manifest["rows"]
               for channel in row["physical_channels"]]
    assert len(targets) == len(set(targets)) == 129
    for row in manifest["rows"]:
        args = row["producer_args"]
        assert args.count("--suppress-nominal-like-tau-shapes") == 1
        assert args.count("--lepton-mva-correlation") == 1
        assert args[args.index("--lepton-mva-correlation") + 1] == "per-run"
        assert "cross-run" not in args
        assert "--use-AAC" not in args
        for option in ("--do-nuisance", "--do-mc-stat", "--skip-selected-wcs-check"):
            assert option in args
        assert args[args.index("--binning") + 1] == "fitting"
        assert args[args.index("--year-coverage-policy") + 1] == "error"
        assert row["input_pkl"].endswith("_np.pkl.gz")
        assert ("--enable-legacy-run3-vsmu-recovery" in args) == (row["era"] == "run3")
        resolved = runner.resolved_argv(manifest, row)
        staged = runner.staged_argv(manifest, row, Path(row["output_root"]).parent / "attempt")
        assert "--suppress-nominal-like-tau-shapes" in resolved
        assert "--suppress-nominal-like-tau-shapes" in staged
        assert resolved[2] == row["input_pkl"]


@pytest.mark.parametrize("era,expected_rows", [("run2", 11), ("run3", 34)])
def test_perrun_tau_flag_reaches_every_native_execution_unit(tmp_path, era, expected_rows):
    args = _manifest_args(tmp_path, era, suppress=True)
    manifest = generator.make_manifest(args)
    _assert_perrun_tau_rows(manifest, expected_rows)

    off_args = _manifest_args(tmp_path, era)
    off_manifest = generator.make_manifest(off_args)
    assert len(manifest["rows"]) == len(off_manifest["rows"])
    for on_row, off_row in zip(manifest["rows"], off_manifest["rows"]):
        assert off_row["producer_args"] == [
            value for value in on_row["producer_args"]
            if value != "--suppress-nominal-like-tau-shapes"
        ]


def test_crossrun_selection_stays_independent_and_missing_row_flag_is_detected(tmp_path):
    args = _manifest_args(tmp_path, "run2", mva="cross-run")
    manifest = generator.make_manifest(args)
    assert all("--suppress-nominal-like-tau-shapes" not in row["producer_args"]
               for row in manifest["rows"])
    assert all(row["producer_args"][row["producer_args"].index("--lepton-mva-correlation") + 1]
               == "cross-run" for row in manifest["rows"])

    args.suppress_nominal_like_tau_shapes = True
    args.lepton_mva_correlation = "per-run"
    manifest = generator.make_manifest(args)
    manifest["rows"][1]["producer_args"].remove("--suppress-nominal-like-tau-shapes")
    with pytest.raises(AssertionError):
        _assert_perrun_tau_rows(manifest, 11)
    manifest["rows"][1]["producer_args"].append("--suppress-nominal-like-tau-shapes")
    index = manifest["rows"][1]["producer_args"].index("--lepton-mva-correlation")
    manifest["rows"][1]["producer_args"][index + 1] = "cross-run"
    with pytest.raises(AssertionError):
        _assert_perrun_tau_rows(manifest, 11)
