"""Fit configuration contract for new and historical combined packages."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from analysis.topeft_run2 import build_combined_datacard_package as combined
from topeft.modules import datacard_packaging


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source(tmp_path, era):
    root = tmp_path / era
    cards = root / "cards"
    cards.mkdir(parents=True)
    mapping = datacard_packaging.build_per_era_mapping(["signal", "control"])
    scalings = []
    for row in mapping:
        name = row["physical_name"]
        stem = f"ttx_multileptons-{name}"
        (cards / f"{stem}.root").write_bytes(b"ROOT\0" + era.encode() + name.encode())
        (cards / f"{stem}.txt").write_bytes(
            f"shapes * * {stem}.root $PROCESS $PROCESS_$SYSTEMATIC\n"
            f"bin {name}\nrate 1.0\n".encode())
        scalings.append({"channel": row["per_era_chN"], "process": "ttH_sm",
                         "parameters": ["cSM[1]"], "scaling": [[1.0, 2.0]]})
    _write_json(root / "physical_to_chN.json", mapping)
    _write_json(root / "scalings.json", scalings)
    (root / "selectedWCs.txt").write_text("ttH_sm cSM\n", encoding="utf-8")
    _write_json(root / "package-provenance.json", {
        "schema": "TOP26006_v1", "artifact_type": "package_provenance", "analysis": "TOP-26-006",
        "era": era, "package_root": str(root), "packaged_txt_count": len(mapping),
        "packaged_root_count": len(mapping), "source_manifest_sha256s": [], "source_unit_count": 1,
        "builder_git_head": "a" * 40, "builder_source_sha256": "b" * 64, "builder_git_dirty": False,
        "selected_wcs_sha256": datacard_packaging.sha256_file(root / "selectedWCs.txt"),
        "scalings_sha256": datacard_packaging.sha256_file(root / "scalings.json"),
        "physical_to_chN_sha256": datacard_packaging.sha256_file(root / "physical_to_chN.json"),
    })
    return root


@pytest.fixture
def sources(tmp_path):
    return _source(tmp_path, "run2"), _source(tmp_path, "run3")


def _build(sources, output, fit_config):
    return combined.build_combined_package(*sources, output, "TOP-26-006", "260925", "v1",
                                           fit_config=fit_config)


@pytest.mark.parametrize("output_name", ["portable_alpha", "other_beta"])
def test_arbitrary_package_names_copy_selected_config_and_certify(tmp_path, sources, output_name, capsys):
    selected = tmp_path / "selected.yml"
    content = b"fit: selected\n"
    selected.write_bytes(content)
    output = tmp_path / output_name
    assert combined.main([
        "build", "--run2-package", str(sources[0]), "--run3-package", str(sources[1]),
        "--output", str(output), "--analysis", "TOP-26-006", "--package-date", "260925",
        "--package-version", "v1", "--fit-config", str(selected)]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "pass"
    assert (output / "config/TOP_26_006.yml").read_bytes() == content
    assert combined.certify_combined_package(output, *sources)["result"] == "pass"
    provenance = json.loads((output / "package-provenance.json").read_text())["fit_config"]
    assert provenance["logical_path"] == "config/TOP_26_006.yml"
    assert provenance["sha256"] == hashlib.sha256(content).hexdigest()
    assert provenance["source_path"] == str(selected)


@pytest.mark.parametrize("content", [None, b""])
def test_unavailable_or_empty_selected_config_blocks_publication(tmp_path, sources, content):
    selected = tmp_path / "selected.yml"
    if content is not None:
        selected.write_bytes(content)
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError, match="selected fit config"):
        _build(sources, output, selected)
    assert not output.exists()
    assert not (tmp_path / ".unpublished.staging").exists()


def test_default_source_and_observable_repository_provenance(tmp_path, sources):
    output = tmp_path / "default_source"
    combined.build_combined_package(*sources, output, "TOP-26-006", "260925", "v1")
    source = combined._default_fit_config()
    assert (output / "config/TOP_26_006.yml").read_bytes() == source.read_bytes()
    provenance = json.loads((output / "package-provenance.json").read_text())["fit_config"]
    assert provenance["source_path"] == "topeft/params/fits/TOP_26_006.yml"
    assert provenance["source_repository"] == "topeft"
    assert provenance["source_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source.parents[3], check=True,
        capture_output=True, text=True).stdout.strip()


def test_unavailable_git_provenance_does_not_block_build(tmp_path, sources, monkeypatch):
    original_run = combined.subprocess.run

    def unavailable_repository(*args, **kwargs):
        if args[0][1:] == ["rev-parse", "--show-toplevel"]:
            raise subprocess.CalledProcessError(1, args[0])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(combined.subprocess, "run", unavailable_repository)
    output = tmp_path / "git_metadata_unavailable"
    combined.build_combined_package(*sources, output, "TOP-26-006", "260925", "v1")
    assert (output / "config/TOP_26_006.yml").read_bytes() == combined._default_fit_config().read_bytes()
    provenance = json.loads((output / "package-provenance.json").read_text())["fit_config"]
    assert provenance["source_repository"] is None
    assert provenance["source_commit"] is None
    assert combined.certify_combined_package(output, *sources)["result"] == "pass"


def test_fit_config_provenance_hash_mismatch_is_warning(tmp_path, sources):
    selected = tmp_path / "selected.yml"
    selected.write_bytes(b"fit: selected\n")
    output = tmp_path / "warning_case"
    _build(sources, output, selected)
    path = output / "package-provenance.json"
    provenance = json.loads(path.read_text())
    provenance["fit_config"]["sha256"] = "0" * 64
    _write_json(path, provenance)
    result = combined.certify_combined_package(output, *sources)
    assert result["result"] == "pass"
    assert "fit_config_provenance_sha256_mismatch" in result["checks"]["fit_config"]["evidence"]["warnings"]


def test_missing_config_fails_default_and_legacy_requires_old_layout(tmp_path, sources, capsys):
    selected = tmp_path / "selected.yml"
    selected.write_bytes(b"fit: selected\n")
    output = tmp_path / "legacy_layout"
    _build(sources, output, selected)
    current_result = combined.certify_combined_package(output, *sources)
    assert current_result["result"] == "pass"
    assert current_result["checks"]["fit_config"]["evidence"]["portability"] is True
    assert current_result["checks"]["fit_config"]["evidence"]["legacy_missing_fit_config"] is False
    (output / "config/TOP_26_006.yml").unlink()
    missing_file_result = combined.certify_combined_package(output, *sources)
    assert missing_file_result["result"] == "fail"
    assert missing_file_result["checks"]["fit_config"]["result"] == "fail"
    assert "certify --allow-legacy-missing-fit-config" in (
        missing_file_result["checks"]["fit_config"]["failures"][0])
    (output / "config").rmdir()
    provenance_path = output / "package-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.pop("fit_config")
    _write_json(provenance_path, provenance)
    readme_path = output / "README.md"
    readme_path.write_text(readme_path.read_text().replace(
        "Package-local fit configuration: `config/TOP_26_006.yml`.", ""), encoding="utf-8")
    assert combined.main([
        "certify", "--package-root", str(output), "--run2-package", str(sources[0]),
        "--run3-package", str(sources[1]),
    ]) == 1
    normal_output = capsys.readouterr().out
    assert "certify --allow-legacy-missing-fit-config" in normal_output
    assert "legacy/historical package certification only" in normal_output
    assert "does not make the package EFTFit-portable" in normal_output
    assert "does not bypass current/new package requirements" in normal_output
    result = combined.certify_combined_package(output, *sources, allow_legacy_missing_fit_config=True)
    assert result["result"] == "pass"
    assert result["checks"]["fit_config"]["evidence"]["legacy_missing_fit_config"] is True
    assert result["checks"]["fit_config"]["evidence"]["portability"] is False
    with pytest.raises(SystemExit):
        combined.main(["build", "--allow-legacy-missing-fit-config"])
    capsys.readouterr()


def test_payload_mapping_ordering_and_scalings_preserved(tmp_path, sources):
    selected = tmp_path / "selected.yml"
    selected.write_bytes(b"fit: selected\n")
    output = tmp_path / "payload_invariants"
    _build(sources, output, selected)
    manifest = json.loads((output / "combined_mapping_manifest.json").read_text())
    expected_mapping = datacard_packaging.build_combined_mapping(*(
        json.loads((source / "physical_to_chN.json").read_text()) for source in sources))
    assert manifest == combined._manifest(expected_mapping, output)
    expected_order = datacard_packaging.build_ordered_card_inputs(expected_mapping)
    assert (output / "ordered_card_inputs.txt").read_text() == "\n".join(expected_order) + "\n"
    for row in expected_mapping:
        source = sources[0] if row["era"] == "run2" else sources[1]
        stem = f"ttx_multileptons-{row['physical_name']}"
        assert (output / "cards" / row["destination_root_name"]).read_bytes() == (
            source / "cards" / f"{stem}.root").read_bytes()
        source_card = (source / "cards" / f"{stem}.txt").read_bytes()
        expected_card = source_card.replace((stem + ".root").encode(),
                                            row["destination_root_name"].encode(), 1)
        assert (output / "cards" / row["destination_txt_name"]).read_bytes() == expected_card
    expected_scalings = combined._combine_scalings(*(
        json.loads((source / "scalings.json").read_text()) for source in sources), expected_mapping)
    assert (output / "scalings.json").read_bytes() == combined._scalings_bytes(expected_scalings)
    assert combined.certify_combined_package(output, *sources)["result"] == "pass"
