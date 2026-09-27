"""Focused tests for input-bound standard datacard matrix manifests."""

import hashlib
import json
import sys

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
        assert runner.option_values(row["producer_args"], "--ch-lst") == row["physical_channels"]
        assert runner.option_values(row["producer_args"], "--year-coverage-policy") == ["error"]
        assert runner.option_values(row["producer_args"], "--binning") == ["fitting"]
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
