"""Exercise the maintained fit-input consumer with a standardized package."""

import hashlib
import json

import pytest

from analysis.topeft_run2.build_fit_input_package import build_fit_input_package
from analysis.topeft_run2.standardize_systematics import standardize_package
from test_standardize_systematics_release_contract import make_package


def test_same_consumer_accepts_source_and_standardized_roots(tmp_path):
    source, ordered = make_package(
        tmp_path, ("Run2_alpha.txt", "Run3_beta.txt", "Run2_gamma.txt")
    )
    standardized = tmp_path / "standardized"
    provenance = standardize_package(source, standardized)
    assert {path.name for path in standardized.iterdir()} == {
        "cards", "scalings.json", "ordered_card_inputs.txt",
        "combined_mapping_manifest.json", "package-provenance.json", "README.md",
    }
    assert (standardized / "scalings.json").read_bytes() == (source / "scalings.json").read_bytes()
    assert (standardized / "ordered_card_inputs.txt").read_bytes() == ordered.encode()
    source_mapping = json.loads((source / "combined_mapping_manifest.json").read_text())
    standardized_mapping = json.loads((standardized / "combined_mapping_manifest.json").read_text())
    assert standardized_mapping["rows"] == source_mapping["rows"]
    assert standardized_mapping["package_root"] == str(standardized)
    assert provenance["manifest_sha256"] == hashlib.sha256(
        (standardized / "combined_mapping_manifest.json").read_bytes()
    ).hexdigest()
    assert provenance["standardization"]["accepted_mapping_commit"] == (
        "013d9de2b53ad2f0eb5cfae5e47e5d796b07fe1f"
    )
    assert {entry["path"] for entry in provenance["standardization"]["member_inventory"]} == {
        path.relative_to(standardized).as_posix()
        for path in standardized.rglob("*") if path.is_file() and path.name != "package-provenance.json"
    }
    config = tmp_path / "fit.yml"
    config.write_text(
        "analysis: TOP_26_006\nwc_population:\n  scientific: [ctW]\n"
        "wc_ranges:\n  ctW: [-3, 3]\n"
    )
    source_fit = build_fit_input_package(source, config, tmp_path / "source-fit")
    standardized_fit = build_fit_input_package(standardized, config, tmp_path / "standardized-fit")
    assert source_fit["fit_config"] == standardized_fit["fit_config"]
    assert source_fit["datacards"]["channel_mapping"] == standardized_fit["datacards"]["channel_mapping"]
    assert source_fit["datacards"]["input_snapshot_sha256"] != (
        standardized_fit["datacards"]["input_snapshot_sha256"]
    )
    for name in ("scalings.json", "ordered_card_inputs.txt"):
        assert (tmp_path / "source-fit" / "datacards" / name).read_bytes() == (
            tmp_path / "standardized-fit" / "datacards" / name
        ).read_bytes()
    for forbidden in ("selectedWCs.txt", "combinedcard.txt", "workspace.root"):
        assert not (standardized / forbidden).exists()


def test_invalid_source_metadata_does_not_publish(tmp_path):
    source, _ = make_package(tmp_path, ("Run2_alpha.txt",))
    provenance_path = source / "package-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["scalings_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance) + "\n")
    output = tmp_path / "standardized"
    with pytest.raises(ValueError, match="hash mismatch"):
        standardize_package(source, output)
    assert not output.exists()


def test_arbitrary_card_names_use_mapping_era(tmp_path):
    names = ("nebula_axis.txt", "quartz_window.txt", "cobalt_leaf.txt")
    source, ordered = make_package(tmp_path, names, eras=("run2", "run3", "run2"))
    output = tmp_path / "standardized"
    provenance = standardize_package(source, output)
    assert (output / "ordered_card_inputs.txt").read_text() == ordered
    assert provenance["packaged_txt_count"] == len(names)
    assert {path.name for path in (output / "cards").glob("*.txt")} == set(names)
    for name, domain in zip(names, ("13TeV", "13p6TeV", "13TeV")):
        assert f"CMS_pileup_{domain} shape" in (output / "cards" / name).read_text()
