"""Focused synthetic checks for the portable fit-input projection."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from analysis.topeft_run2 import build_fit_input_package as fit_input


_mapping_fields = {
    "era", "physical_name", "per_era_chN", "combined_chN",
    "combined_order_index", "destination_txt_name", "destination_root_name",
}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_from_manifest(manifest):
    datacards = manifest["datacards"]
    payload = {"files": datacards["files"], "channel_mapping": datacards["channel_mapping"]}
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _refresh_hashes(package):
    provenance_path = package / "package-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for name, field in (
        ("scalings.json", "scalings_sha256"),
        ("ordered_card_inputs.txt", "ordered_card_inputs_sha256"),
        ("combined_mapping_manifest.json", "manifest_sha256"),
    ):
        provenance[field] = _sha256(package / name)
    _write_json(provenance_path, provenance)


def _fixture(tmp_path):
    config = tmp_path / "plain-config.yml"
    config.write_text(
        "analysis: TOP_26_006\nwc_population:\n  scientific: [ctW]\n"
        "  fixed_zero: [ctu1]\nwc_ranges:\n  ctW: [-3.0, 3.0]\n",
        encoding="utf-8",
    )
    package = tmp_path / "combined"
    cards = package / "cards"
    cards.mkdir(parents=True)
    for name in ("one", "two"):
        (cards / f"{name}.txt").write_bytes(f"shapes * * {name}.root $PROCESS\n".encode())
        (cards / f"{name}.root").write_bytes(f"synthetic-{name}\x00".encode())
    (package / "ordered_card_inputs.txt").write_text(
        "cards/two.txt\ncards/one.txt\n", encoding="utf-8")
    _write_json(package / "scalings.json", [{"channel": "ch1"}, {"channel": "ch2"}])
    rows = []
    for index, name in enumerate(("two", "one"), start=1):
        rows.append({
            "era": "run2", "physical_name": name, "per_era_chN": f"ch{index}",
            "combined_chN": f"ch{index}", "combined_order_index": index,
            "destination_txt_name": f"{name}.txt",
            "destination_root_name": f"{name}.root",
            "source_txt_path": "/groups/historical/card.txt",
        })
    _write_json(package / "combined_mapping_manifest.json", {
        "schema": "TOP26006_v1", "artifact_type": "combined_mapping_manifest",
        "package_root": "/groups/historical/package", "rows": rows,
    })
    (package / "README.md").write_text(
        "Historical source at /groups/historical/package\n", encoding="utf-8")
    _write_json(package / "package-provenance.json", {
        "schema": "TOP26006_v1", "artifact_type": "package_provenance",
        "analysis": "TOP-26-006", "package_root": "/groups/historical/package",
        "package_version": "v1", "package_date": "260925",
        "assembler_commit": "a" * 40,
        "source_mapping_sha256": {"run2": "b" * 64},
    })
    _refresh_hashes(package)
    return config, package, tmp_path / "fit-input"


def test_valid_projection_copies_only_consumer_bytes(tmp_path):
    config, package, output = _fixture(tmp_path)
    source_bytes = {f"cards/{name}{suffix}": (package / "cards" / f"{name}{suffix}").read_bytes()
                    for name in ("one", "two") for suffix in (".txt", ".root")}
    source_bytes.update({name: (package / name).read_bytes()
                         for name in ("scalings.json", "ordered_card_inputs.txt")})
    result = fit_input.build_fit_input_package(
        package, config, output, package_creation_topeft_commit="c" * 40,
        input_production_topeft_commit="d" * 40,
        input_production_topcoffea_commit="e" * 40)
    observed = json.loads((output / "fit-input-manifest.json").read_text())
    assert result == observed
    assert {path.relative_to(output / "datacards").as_posix(): path.read_bytes()
            for path in (output / "datacards").rglob("*") if path.is_file()} == source_bytes
    assert (output / "config" / "TOP_26_006.yml").read_bytes() == config.read_bytes()
    assert observed["fit_config"] == {
        "path": "config/TOP_26_006.yml", "sha256": _sha256(config)}
    assert observed["datacards"]["files"] == [
        {"path": f"datacards/{relative}", "size_bytes": len(content),
         "sha256": hashlib.sha256(content).hexdigest()}
        for relative, content in sorted(source_bytes.items())
    ]
    assert observed["datacards"]["input_snapshot_sha256"] == _snapshot_from_manifest(observed)
    assert observed["datacards"]["files"] == sorted(
        observed["datacards"]["files"], key=lambda row: row["path"])
    rows = observed["datacards"]["channel_mapping"]["rows"]
    assert [set(row) for row in rows] == [_mapping_fields, _mapping_fields]
    assert [row["destination_txt_name"] for row in rows] == ["two.txt", "one.txt"]
    assert observed["provenance"]["package_creation"]["topeft_commit"] == "c" * 40
    assert observed["provenance"]["input_production"] == {
        "topeft_commit": "d" * 40, "topcoffea_commit": "e" * 40}
    assert observed["provenance"]["source_datacard_package"]["package_version"] == "v1"
    assert "/groups/historical" not in (output / "fit-input-manifest.json").read_text()
    readme = (output / "README.md").read_text()
    assert "/groups/historical" not in readme
    for phrase in ("sole card-order authority", "wildcard or glob", "scalings.json",
                   "cards/<name>.txt", "leading `cards/`", "writable cards directory",
                   'combineCards.py "${ordered_basenames[@]}" > combinedcard.txt',
                   "TOPEFT documents this command but does not execute it"):
        assert phrase in readme
    assert not (output / "combinedcard.txt").exists()
    assert not (output / "selectedWCs.txt").exists()
    inventory_path = os.environ.get("FIT_INPUT_SYNTHETIC_INVENTORY_PATH")
    if inventory_path:
        Path(inventory_path).write_text(json.dumps({
            "source_package": str(package), "source_config": str(config),
            "output_package": str(output),
        }, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("ordered", [
    "", "cards/two.txt\ncards/two.txt\n", "/tmp/two.txt\n",
    "cards/../two.txt\n", "../cards/two.txt\n", "cards/missing.txt\n",
    "cards/nested/two.txt\n", "cards/one.root\n",
])
def test_ordered_card_inputs_fail_closed(tmp_path, ordered):
    config, package, output = _fixture(tmp_path)
    (package / "ordered_card_inputs.txt").write_text(ordered, encoding="utf-8")
    _refresh_hashes(package)
    with pytest.raises(fit_input.FitInputError):
        fit_input.build_fit_input_package(package, config, output)
    assert not output.exists()


@pytest.mark.parametrize("member", ["cards/one.root", "scalings.json"])
def test_symlink_source_fails_closed(tmp_path, member):
    config, package, output = _fixture(tmp_path)
    path = package / member
    path.unlink()
    path.symlink_to(package / "cards" / "one.txt")
    with pytest.raises(fit_input.FitInputError):
        fit_input.build_fit_input_package(package, config, output)
    assert not output.exists()


@pytest.mark.parametrize("content", [None, "analysis: [bad\n", "analysis: wrong\n"])
def test_missing_or_malformed_config_has_no_fallback(tmp_path, content):
    config, package, output = _fixture(tmp_path)
    if content is None:
        config.unlink()
    else:
        config.write_text(content, encoding="utf-8")
    with pytest.raises(fit_input.FitInputError):
        fit_input.build_fit_input_package(package, config, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["output", "staging", "source_inside_output", "output_inside_source"])
def test_existing_or_overlapping_paths_fail_closed(tmp_path, kind):
    config, package, output = _fixture(tmp_path)
    if kind == "output":
        output.mkdir()
    elif kind == "staging":
        (tmp_path / ".fit-input.staging").mkdir()
    elif kind == "source_inside_output":
        output = tmp_path
    else:
        output = package / "nested-output"
    with pytest.raises(fit_input.FitInputError):
        fit_input.build_fit_input_package(package, config, output)


@pytest.mark.parametrize("name", ["combinedcard.txt", "selectedWCs.txt", "other.root"])
def test_unmapped_card_artifacts_are_ignored(tmp_path, name):
    config, package, output = _fixture(tmp_path)
    baseline = fit_input.build_fit_input_package(package, config, tmp_path / "baseline")
    (package / "cards" / name).write_bytes(b"not an input")
    observed = fit_input.build_fit_input_package(package, config, output)
    assert not (output / "datacards" / "cards" / name).exists()
    assert observed["datacards"]["input_snapshot_sha256"] == baseline["datacards"]["input_snapshot_sha256"]


def test_arbitrary_card_name_is_not_statistical_ownership_proxy(tmp_path):
    config, package, output = _fixture(tmp_path)
    (package / "cards" / "impacts.root").write_bytes(b"opaque template")
    (package / "unrelated.txt").write_bytes(b"unrelated top-level member")
    baseline_package = tmp_path / "baseline-combined"
    baseline_package.mkdir()
    for name in ("scalings.json", "ordered_card_inputs.txt", "combined_mapping_manifest.json",
                 "package-provenance.json"):
        (baseline_package / name).write_bytes((package / name).read_bytes())
    (baseline_package / "cards").mkdir()
    for name in ("one.txt", "one.root", "two.txt", "two.root"):
        (baseline_package / "cards" / name).write_bytes((package / "cards" / name).read_bytes())
    baseline = fit_input.build_fit_input_package(baseline_package, config, tmp_path / "baseline")
    observed = fit_input.build_fit_input_package(package, config, output)
    assert not (output / "datacards" / "cards" / "impacts.root").exists()
    assert not (output / "unrelated.txt").exists()
    assert observed["datacards"]["input_snapshot_sha256"] == baseline["datacards"]["input_snapshot_sha256"]


def test_provenance_variation_does_not_change_input_identities(tmp_path):
    config, package, first = _fixture(tmp_path)
    second = tmp_path / "second-fit-input"
    one = fit_input.build_fit_input_package(
        package, config, first, package_creation_topeft_commit="a" * 40,
        input_production_topeft_commit="b" * 40,
        input_production_topcoffea_commit="c" * 40)
    two = fit_input.build_fit_input_package(
        package, config, second, package_creation_topeft_commit="d" * 40,
        input_production_topeft_commit="e" * 40,
        input_production_topcoffea_commit="f" * 40)
    assert one["provenance"] != two["provenance"]
    assert one["datacards"]["input_snapshot_sha256"] == two["datacards"]["input_snapshot_sha256"]
    assert one["fit_config"]["sha256"] == two["fit_config"]["sha256"]


def test_source_location_disguised_as_provenance_is_omitted(tmp_path):
    config, package, output = _fixture(tmp_path)
    provenance_path = package / "package-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["package_version"] = "/groups/historical/package"
    provenance["source_scalings_sha256"] = {"run2": "/groups/historical/scalings.json"}
    _write_json(provenance_path, provenance)
    manifest = fit_input.build_fit_input_package(package, config, output)
    source_fields = manifest["provenance"]["source_datacard_package"]
    assert "package_version" not in source_fields
    assert "source_scalings_sha256" not in source_fields
    assert "/groups/historical" not in (output / "fit-input-manifest.json").read_text()


@pytest.mark.parametrize("config_location", ["external", "package_local", "package_local_alias"])
def test_cli_uses_plain_config_file_without_git(tmp_path, capsys, config_location):
    config, package, output = _fixture(tmp_path)
    if config_location != "external":
        (package / "config").mkdir()
        local_config = package / "config" / "TOP_26_006.yml"
        config.rename(local_config)
        config = local_config
        if config_location == "package_local_alias":
            alias = tmp_path / "source-alias"
            alias.symlink_to(package, target_is_directory=True)
            config = alias / "config" / "TOP_26_006.yml"
    source_hashes = {path.relative_to(package).as_posix(): _sha256(path)
                     for path in package.rglob("*") if path.is_file()}
    config_hash = _sha256(config)
    assert fit_input.main(["--combined-package", str(package), "--fit-config", str(config),
                           "--output", str(output)]) == 0
    response = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "fit-input-manifest.json").read_text())
    assert response["input_snapshot_sha256"] == manifest["datacards"]["input_snapshot_sha256"]
    assert manifest["datacards"]["input_snapshot_sha256"] == _snapshot_from_manifest(manifest)
    assert (output / "config" / "TOP_26_006.yml").read_bytes() == config.read_bytes()
    assert manifest["fit_config"] == {"path": "config/TOP_26_006.yml", "sha256": config_hash}
    for row in manifest["datacards"]["files"]:
        relative = row["path"].removeprefix("datacards/")
        destination = output / row["path"]
        assert destination.read_bytes() == (package / relative).read_bytes()
        assert row["sha256"] == _sha256(destination)
        assert row["size_bytes"] == destination.stat().st_size
    assert (output / "datacards" / "ordered_card_inputs.txt").read_bytes() == (
        package / "ordered_card_inputs.txt").read_bytes()
    assert (output / "datacards" / "scalings.json").read_bytes() == (
        package / "scalings.json").read_bytes()
    assert [row["destination_txt_name"] for row in manifest["datacards"]["channel_mapping"]["rows"]] == [
        "two.txt", "one.txt"]
    assert source_hashes == {path.relative_to(package).as_posix(): _sha256(path)
                             for path in package.rglob("*") if path.is_file()}
    assert _sha256(config) == config_hash


@pytest.mark.parametrize("kind", [
    "output_is_config", "output_contains_source", "output_inside_source",
    "output_via_symlink", "output_with_dotdot", "relative_source",
    "staging_exists", "staging_is_source",
])
def test_package_local_config_keeps_write_paths_isolated(tmp_path, monkeypatch, kind):
    external_config, package, output = _fixture(tmp_path)
    (package / "config").mkdir()
    config = package / "config" / "TOP_26_006.yml"
    config.write_bytes(external_config.read_bytes())
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep-existing-content")
    if kind == "output_is_config":
        output = config
    elif kind == "output_contains_source":
        output = tmp_path
    elif kind in {"output_inside_source", "relative_source"}:
        output = package / "nested-output"
    elif kind == "output_via_symlink":
        (package / "nested").mkdir()
        (tmp_path / "source-alias").symlink_to(package, target_is_directory=True)
        output = tmp_path / "source-alias" / "nested" / "fit-input"
    elif kind == "output_with_dotdot":
        (package / "nested").mkdir()
        output = package / "nested" / ".." / "fit-input"
    elif kind == "staging_exists":
        staging = tmp_path / ".fit-input.staging"
        staging.mkdir()
        (staging / "sentinel").write_bytes(b"keep-staging-content")
    else:
        staging = tmp_path / ".fit-input.staging"
        package.rename(staging)
        package = staging
        config = package / "config" / "TOP_26_006.yml"
    source_hashes = {path.relative_to(package).as_posix(): _sha256(path)
                     for path in package.rglob("*") if path.is_file()}
    selected_package, selected_config = package, config
    if kind == "relative_source":
        monkeypatch.chdir(tmp_path)
        selected_package = package.relative_to(tmp_path)
        selected_config = config.relative_to(tmp_path)
    with pytest.raises(fit_input.FitInputError):
        fit_input.main(["--combined-package", str(selected_package), "--fit-config", str(selected_config),
                        "--output", str(output)])
    assert source_hashes == {path.relative_to(package).as_posix(): _sha256(path)
                             for path in package.rglob("*") if path.is_file()}
    assert sentinel.read_bytes() == b"keep-existing-content"
    if kind == "staging_exists":
        assert (staging / "sentinel").read_bytes() == b"keep-staging-content"
    if kind not in {"output_is_config", "output_contains_source"}:
        assert not output.exists()
    if kind not in {"staging_exists", "staging_is_source", "output_is_config", "output_contains_source"}:
        assert not (output.parent / f".{output.name}.staging").exists()


@pytest.mark.parametrize("kind,expected", [
    ("missing", "missing or non-regular input"),
    ("symlink", "missing or non-regular input"),
    ("schema", "fit config lacks required"),
])
def test_invalid_package_local_config_still_fails(tmp_path, kind, expected):
    external_config, package, output = _fixture(tmp_path)
    (package / "config").mkdir()
    config = package / "config" / "TOP_26_006.yml"
    config.write_bytes(external_config.read_bytes())
    if kind == "missing":
        config.unlink()
    elif kind == "symlink":
        config.unlink()
        config.symlink_to(external_config)
    else:
        config.write_text("analysis: wrong\n", encoding="utf-8")
    with pytest.raises(fit_input.FitInputError, match=expected):
        fit_input.main(["--combined-package", str(package), "--fit-config", str(config),
                        "--output", str(output)])
    assert not output.exists()
    assert not (tmp_path / ".fit-input.staging").exists()


def test_non_regular_unmapped_card_member_is_ignored(tmp_path):
    config, package, output = _fixture(tmp_path)
    os.mkfifo(package / "cards" / "named_pipe")
    fit_input.build_fit_input_package(package, config, output)
    assert not (output / "datacards" / "cards" / "named_pipe").exists()


def test_mapping_order_or_required_field_fails_closed(tmp_path):
    config, package, output = _fixture(tmp_path)
    path = package / "combined_mapping_manifest.json"
    mapping = json.loads(path.read_text())
    mapping["rows"][0]["destination_txt_name"] = "one.txt"
    _write_json(path, mapping)
    _refresh_hashes(package)
    with pytest.raises(fit_input.FitInputError, match="disagrees"):
        fit_input.build_fit_input_package(package, config, output)
    assert not output.exists()


def test_mapping_only_change_changes_snapshot(tmp_path):
    config, package, output = _fixture(tmp_path)
    first = fit_input.build_fit_input_package(package, config, output)
    mapping_path = package / "combined_mapping_manifest.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["rows"][0]["physical_name"] = "renamed-channel"
    _write_json(mapping_path, mapping)
    _refresh_hashes(package)
    second = fit_input.build_fit_input_package(package, config, tmp_path / "second")
    assert first["datacards"]["files"] == second["datacards"]["files"]
    assert first["datacards"]["channel_mapping"] != second["datacards"]["channel_mapping"]
    assert first["datacards"]["input_snapshot_sha256"] != second["datacards"]["input_snapshot_sha256"]
    assert second["datacards"]["input_snapshot_sha256"] == _snapshot_from_manifest(second)


def test_fit_config_bytes_have_independent_identity(tmp_path):
    config, package, output = _fixture(tmp_path)
    first = fit_input.build_fit_input_package(package, config, output)
    config.write_text(config.read_text() + "# distinct config bytes\n", encoding="utf-8")
    second = fit_input.build_fit_input_package(package, config, tmp_path / "second")
    assert first["fit_config"]["sha256"] != second["fit_config"]["sha256"]
    assert first["datacards"]["input_snapshot_sha256"] == second["datacards"]["input_snapshot_sha256"]


@pytest.mark.parametrize("bad_value", ["a" * 39, "A" * 40, "g" * 40, "a" * 41, "", 123])
@pytest.mark.parametrize("field", [
    "package_creation_topeft_commit", "input_production_topeft_commit",
    "input_production_topcoffea_commit",
])
def test_invalid_provenance_commit_fails_syntax_only(tmp_path, monkeypatch, field, bad_value):
    config, package, output = _fixture(tmp_path)
    monkeypatch.setattr(os, "system", lambda *_: pytest.fail("Git lookup attempted"))
    monkeypatch.setattr(os, "popen", lambda *_: pytest.fail("Git lookup attempted"))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Git lookup attempted"))
    with pytest.raises(fit_input.FitInputError, match="invalid provenance commit syntax"):
        fit_input.build_fit_input_package(package, config, output, **{field: bad_value})
    assert not output.exists()


def test_null_provenance_values_and_cli_names(tmp_path, capsys):
    config, package, output = _fixture(tmp_path)
    fit_input.main(["--combined-package", str(package), "--fit-config", str(config),
                    "--output", str(output), "--package-creation-topeft-commit", "a" * 40,
                    "--input-production-topeft-commit", "b" * 40,
                    "--input-production-topcoffea-commit", "c" * 40])
    capsys.readouterr()
    observed = json.loads((output / "fit-input-manifest.json").read_text())
    assert observed["provenance"]["package_creation"] == {"topeft_commit": "a" * 40}
    assert observed["provenance"]["input_production"] == {
        "topeft_commit": "b" * 40, "topcoffea_commit": "c" * 40}
    null_output = tmp_path / "nulls"
    fit_input.build_fit_input_package(package, config, null_output)
    nulls = json.loads((null_output / "fit-input-manifest.json").read_text())
    assert nulls["provenance"]["package_creation"] == {"topeft_commit": None}
    assert nulls["provenance"]["input_production"] == {
        "topeft_commit": None, "topcoffea_commit": None}


@pytest.mark.parametrize("field,value", [
    ("destination_txt_name", "nested/two.txt"),
    ("destination_txt_name", "two.root"),
    ("destination_root_name", "nested\\two.root"),
    ("destination_root_name", "two.txt"),
    ("destination_root_name", "one.root"),
    ("destination_root_name", "absent.root"),
])
def test_mapped_name_validation_fails_closed(tmp_path, field, value):
    config, package, output = _fixture(tmp_path)
    mapping_path = package / "combined_mapping_manifest.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["rows"][0][field] = value
    _write_json(mapping_path, mapping)
    _refresh_hashes(package)
    with pytest.raises(fit_input.FitInputError):
        fit_input.build_fit_input_package(package, config, output)
    assert not output.exists()
