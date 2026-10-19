"""Focused tests for input-bound standard datacard matrix manifests."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from analysis.topeft_run2 import datacard_matrix_runner as runner
from analysis.topeft_run2 import make_datacard_matrix_manifest as generator


# Digests come from the accepted input matrix's ordered row records.
_matrix_row_digests = {
    "run2": "d8c350bba65bb87d51644f4b927346f0abce5cfa32fee0603602efd5ade59bd1",
    "run3": "f2225975aec519aae450a3a76d6c1096bba32d22bf49e2156fb2ce64f3164f54",
}
_execution_crosswalk_digests = {
    "run2": "f898db1dd73fdfc99d3fe04dbfe188666b2ed431465a121039bab807a77f2f2c",
    "run3": "a87ac5a26703dadcec3d1f4fa9ead35e30864bb69608598ceb1e995c0123c0ef",
}
_role_order = {
    "run2": ["block1", "block2", "block3", "block4", "block5"],
    "run3": ["2l_mixed", "3l_m_offz", "3l_p_offz", "3l_onz_tau", "3l_fwd"],
}
_row_role_positions = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4]


def _args(tmp_path, era, roles=None):
    roles = _role_order[era] if roles is None else roles
    inputs = {}
    for role in roles:
        path = tmp_path / f"{role}.pkl.gz"
        path.write_bytes(b"synthetic input")
        inputs[role] = path
    missing = tmp_path / "missing.root"
    missing.write_bytes(b"synthetic payload")
    make_cards = tmp_path / "make_cards.py"
    make_cards.write_text("# synthetic producer\n")
    argv = [
        "--era", era, "--manifest-output", str(tmp_path / "matrix.json"),
        "--output-root", str(tmp_path / "cards"), "--control-root", str(tmp_path / "control"),
        "--python-executable", sys.executable, "--make-cards-path", str(make_cards),
        "--missing-parton-file", str(missing), "--runtime-contract-id", f"{era}-synthetic",
    ]
    for role, path in inputs.items():
        argv.extend(["--input-pkl", f"{role}={path}"])
    return argv, inputs, make_cards


@pytest.mark.parametrize("era,years", [
    ("run2", ["UL16APV", "UL16", "UL17", "UL18"]),
    ("run3", ["2022", "2022EE", "2023", "2023BPix"]),
])
def test_standard_manifest_matches_accepted_input_matrix(tmp_path, era, years):
    argv, inputs, make_cards = _args(tmp_path, era)
    assert generator.main(argv) == 0
    path = tmp_path / "matrix.json"
    observed = json.loads(path.read_text())
    parsed, digest = runner.load_manifest(path)
    assert parsed == observed
    assert len(digest) == 64
    assert observed["schema"] == "topeft_datacard_matrix_v3"
    assert observed["runtime_contract"]["make_cards_path"] == str(make_cards)
    assert observed["runtime_contract"]["python_executable"] == sys.executable
    assert observed["runtime_contract"]["fingerprints"] == [
        {"path": str(make_cards), "sha256": runner.hash_file(make_cards)}]
    rows = observed["rows"]
    assert len(rows) == {"run2": 11, "run3": 34}[era]
    counts = {}
    for row in rows:
        parent = row["logical_row_id"]
        counts[parent] = counts.get(parent, 0) + 1
        assert row["row_id"] == f"{parent}_exec_{counts[parent]:02d}"
        role = _role_order[era][_row_role_positions[int(parent[-2:]) - 1]]
        assert row["input_pkl"] == str(inputs[role])
        assert "sr_registry" not in row and "--sr-registry" not in row["producer_args"]
        assert row["producer_args"].count("--enable-legacy-run3-vsmu-recovery") == 0
        assert "--unblind" not in row["producer_args"]
        assert runner.option_values(row["producer_args"], "--ch-lst") == row["physical_channels"]
        assert runner.option_values(row["producer_args"], "--year-coverage-policy") == ["error"]
        assert runner.option_values(row["producer_args"], "--binning") == ["fitting"]
        assert runner.option_values(row["producer_args"], "--lepton-mva-correlation") == ["per-run"]
    crosswalk_lines = "".join(
        f"{era}|{row['row_id']}|{row['logical_row_id']}|"
        f"{_role_order[era][_row_role_positions[int(row['logical_row_id'][-2:]) - 1]]}|"
        f"{row['distribution']}|{','.join(row['physical_channels'])}\n" for row in rows
    )
    assert hashlib.sha256(crosswalk_lines.encode()).hexdigest() == _execution_crosswalk_digests[era]
    logical_lines = ""
    for index in range(1, 12):
        parent = f"{era}_{index:02d}"
        group = [row for row in rows if row["logical_row_id"] == parent]
        channels = [channel for row in group for channel in row["physical_channels"]]
        assert len(channels) == len(set(channels))
        logical_lines += f"{era}|{parent}|{group[0]['distribution']}|{','.join(channels)}\n"
    assert hashlib.sha256(logical_lines.encode()).hexdigest() == _matrix_row_digests[era]
    assert len({f"{channel}_{row['distribution']}" for row in rows for channel in row["physical_channels"]}) == 129
    assert all(row["years"] == years for row in rows)
    assert all(str(tmp_path) in row["output_root"] and str(tmp_path) in row["log_path"]
               for row in rows)
    profile_text = generator._profile_path.read_text()
    assert "/groups/" not in profile_text and "/users/" not in profile_text
    profile = yaml.safe_load(profile_text)
    assert all("execution_units" not in profile_row for profile_row in profile["rows"])
    for profile_row in profile["rows"]:
        groups = generator._execution_channels(profile_row, era)
        assert [channel for group in groups for channel in group] == profile_row["physical_channels"]
        assert all(groups)
        if len(groups) > 1:
            assert profile_row["execution_unit_sizes"][era] == [len(group) for group in groups]


def test_invalid_execution_partition_rejected():
    row = {"physical_channels": ["a", "b"], "execution_unit_sizes": {"run3": [1, 0, 1]}}
    with pytest.raises(ValueError, match="invalid execution unit sizes"):
        generator._execution_channels(row, "run3")
    row["execution_unit_sizes"]["run3"] = [1]
    with pytest.raises(ValueError, match="invalid execution unit sizes"):
        generator._execution_channels(row, "run3")


@pytest.mark.parametrize("era", ("run2", "run3"))
def test_cross_run_mode_is_only_a_producer_argument(tmp_path, era):
    argv, _, _ = _args(tmp_path, era)
    assert generator.main(argv + ["--lepton-mva-correlation", "cross-run"]) == 0
    manifest, _ = runner.load_manifest(tmp_path / "matrix.json")
    assert manifest["schema"] == runner.MANIFEST_SCHEMA
    for row in manifest["rows"]:
        assert runner.option_values(row["producer_args"], "--lepton-mva-correlation") == ["cross-run"]
        assert "lepton_mva_correlation" not in row
        assert "--lepton-mva-correlation" in runner.resolved_argv(manifest, row)


def test_invalid_mva_mode_leaves_manifest_absent(tmp_path):
    argv, _, _ = _args(tmp_path, "run3")
    with pytest.raises(SystemExit) as error:
        generator.main(argv + ["--lepton-mva-correlation", "shared"])
    assert error.value.code == 2
    assert not (tmp_path / "matrix.json").exists()


@pytest.mark.parametrize("era", ("run2", "run3"))
def test_unblind_opt_in_preserves_matrix_and_other_producer_options(tmp_path, era):
    default_root = tmp_path / "default"
    enabled_root = tmp_path / "enabled"
    default_root.mkdir()
    enabled_root.mkdir()
    default_argv, _, _ = _args(default_root, era)
    enabled_argv, _, _ = _args(enabled_root, era)
    other_options = [
        "--lepton-mva-correlation", "cross-run",
        "--enable-legacy-run3-vsmu-recovery",
        "--suppress-nominal-like-tau-shapes",
    ]
    assert generator.main(default_argv + other_options) == 0
    assert generator.main(enabled_argv + other_options + ["--unblind"]) == 0
    default_manifest, _ = runner.load_manifest(default_root / "matrix.json")
    enabled_manifest, _ = runner.load_manifest(enabled_root / "matrix.json")
    default_rows = {row["row_id"]: row for row in default_manifest["rows"]}
    enabled_rows = {row["row_id"]: row for row in enabled_manifest["rows"]}
    assert default_rows.keys() == enabled_rows.keys()
    assert len(enabled_rows) == {"run2": 11, "run3": 34}[era]
    for row_id, row in enabled_rows.items():
        default_row = default_rows[row_id]
        assert row["physical_channels"] == default_row["physical_channels"]
        assert row["years"] == default_row["years"]
        assert row["distribution"] == default_row["distribution"]
        assert row["producer_args"].count("--unblind") == 1
        assert "--unblind" not in default_row["producer_args"]
        assert row["producer_args"].count("--suppress-nominal-like-tau-shapes") == 1
        assert row["producer_args"].count("--enable-legacy-run3-vsmu-recovery") == (era == "run3")
        assert runner.option_values(row["producer_args"], "--lepton-mva-correlation") == ["cross-run"]
        assert [arg for arg in row["producer_args"] if arg != "--unblind"] == [
            arg.replace(str(default_root), str(enabled_root))
            for arg in default_row["producer_args"]
        ]
        assert runner.resolved_argv(enabled_manifest, row).count("--unblind") == 1


def test_v3_row_without_new_option_retains_structural_and_argv_contract(tmp_path):
    argv, _, _ = _args(tmp_path, "run2")
    assert generator.main(argv) == 0
    manifest, _ = runner.load_manifest(tmp_path / "matrix.json")
    row = json.loads(json.dumps(manifest["rows"][0]))
    index = row["producer_args"].index("--lepton-mva-correlation")
    del row["producer_args"][index:index + 2]
    assert runner.validate_row(row, 0, Path(manifest["control_root"])) == row
    assert "--lepton-mva-correlation" not in runner.resolved_argv(manifest, row)
    assert manifest["schema"] == runner.MANIFEST_SCHEMA
    assert runner.RECEIPT_SCHEMA == "topeft_datacard_execution_receipt_v2"


def test_advanced_subset_needs_only_its_input_role_and_does_not_overwrite(tmp_path):
    argv, inputs, _ = _args(tmp_path, "run2", roles=["block1"])
    argv.extend(["--physical-target", "2lss_p_4j_lj0pt"])
    assert generator.main(argv) == 0
    path = tmp_path / "matrix.json"
    parsed, _ = runner.load_manifest(path)
    assert [(row["row_id"], row["physical_channels"], row["input_pkl"])
            for row in parsed["rows"]] == [("run2_02_exec_01", ["2lss_p_4j"], str(inputs["block1"]))]
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="be absent"):
        generator.main(argv)
    assert path.read_bytes() == previous


def test_maintained_channel_set_selects_restricted_manifest(tmp_path):
    argv, _, _ = _args(tmp_path, "run3")
    argv.extend(["--channel-set-key", "OFFZ_SPLIT_CH_LST_SR"])
    assert generator.main(argv) == 0
    parsed, _ = runner.load_manifest(tmp_path / "matrix.json")
    targets = sorted(f"{channel}_{row['distribution']}" for row in parsed["rows"]
                     for channel in row["physical_channels"])
    assert len(targets) == 75
    assert targets == generator.per_era._canonical_physical_names(
        generator._registry, "OFFZ_SPLIT_CH_LST_SR")
    assert all("sr_registry" not in row for row in parsed["rows"])


@pytest.mark.parametrize("era,expected_count", [("run2", 0), ("run3", 1)])
def test_legacy_run3_recovery_opt_in_applies_only_to_run3_rows(tmp_path, era, expected_count):
    default_root = tmp_path / "default"
    enabled_root = tmp_path / "enabled"
    default_root.mkdir()
    enabled_root.mkdir()
    default_argv, _, _ = _args(default_root, era)
    enabled_argv, _, _ = _args(enabled_root, era)
    assert generator.main(default_argv) == 0
    assert generator.main(enabled_argv + ["--enable-legacy-run3-vsmu-recovery"]) == 0
    default_manifest, _ = runner.load_manifest(default_root / "matrix.json")
    enabled_manifest, _ = runner.load_manifest(enabled_root / "matrix.json")
    assert default_manifest["schema"] == enabled_manifest["schema"] == runner.MANIFEST_SCHEMA
    default_rows = {row["row_id"]: row for row in default_manifest["rows"]}
    enabled_rows = {row["row_id"]: row for row in enabled_manifest["rows"]}
    assert default_rows.keys() == enabled_rows.keys()
    assert len(enabled_rows) == {"run2": 11, "run3": 34}[era]
    for row_id, row in enabled_rows.items():
        producer_args = row["producer_args"]
        assert row["era"] == era
        assert producer_args.count("--enable-legacy-run3-vsmu-recovery") == expected_count
        assert "--sr-registry" not in producer_args
        default_args = default_rows[row_id]["producer_args"]
        assert default_args.count("--enable-legacy-run3-vsmu-recovery") == 0
        assert [arg for arg in producer_args if arg != "--enable-legacy-run3-vsmu-recovery"] == [
            arg.replace(str(default_root), str(enabled_root)) for arg in default_args
        ]


@pytest.mark.parametrize("value", ["true", "false", "1", "0"])
def test_legacy_run3_recovery_cli_rejects_string_boolean(tmp_path, value):
    argv, _, _ = _args(tmp_path, "run3")
    with pytest.raises(SystemExit) as error:
        generator.main(argv + ["--enable-legacy-run3-vsmu-recovery", value])
    assert error.value.code == 2
    assert not (tmp_path / "matrix.json").exists()


def test_missing_role_unknown_target_and_duplicate_binding_leave_manifest_absent(tmp_path):
    argv, _, _ = _args(tmp_path, "run3")
    position = argv.index("--input-pkl")
    del argv[position:position + 2]
    with pytest.raises(ValueError, match="2l_mixed PKL is required"):
        generator.main(argv)
    assert not (tmp_path / "matrix.json").exists()
    argv.extend(["--physical-target", "unknown_lj0pt"])
    with pytest.raises(ValueError, match="outside the channel set"):
        generator.main(argv)
    assert not (tmp_path / "matrix.json").exists()
    argv[-1] = "2lss_p_4j_lj0pt"
    argv.extend(["--input-pkl", "3l_m_offz=duplicate.pkl"])
    with pytest.raises(ValueError, match="duplicate --input-pkl binding"):
        generator.main(argv)
    assert not (tmp_path / "matrix.json").exists()
