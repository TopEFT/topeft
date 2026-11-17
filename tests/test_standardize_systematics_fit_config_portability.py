"""Check the derivative's package-local fit configuration contract."""

import hashlib
import json

import hist
import pytest
import uproot

from analysis.topeft_run2.standardize_systematics import standardize_package
from test_standardize_systematics_release_contract import make_package


def add_current_config(source):
    config = source / "config" / "TOP_26_006.yml"
    config.parent.mkdir()
    content = b"analysis: TOP_26_006\nwc_population:\n  scientific: [ctW]\n"
    config.write_bytes(content)
    provenance_path = source / "package-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["fit_config"] = {
        "logical_path": "config/TOP_26_006.yml",
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_repository": "topeft",
        "source_path": "topeft/params/fits/TOP_26_006.yml",
        "source_commit": "a" * 40,
    }
    provenance_path.write_text(json.dumps(provenance) + "\n")
    return config


def test_current_derivative_carries_exact_local_config_and_lineage(tmp_path):
    source, ordered = make_package(tmp_path, ("Run2_tau.txt", "Run3_tau.txt"))
    config = add_current_config(source)
    source_provenance_bytes = (source / "package-provenance.json").read_bytes()
    output = tmp_path / "standardized"
    standardize_package(source, output)

    copied = output / "config" / "TOP_26_006.yml"
    assert {path.name for path in output.iterdir()} == {
        "cards", "config", "scalings.json", "ordered_card_inputs.txt",
        "combined_mapping_manifest.json", "package-provenance.json", "README.md",
    }
    assert {path.name for path in copied.parent.iterdir()} == {copied.name}
    assert copied.read_bytes() == config.read_bytes()
    observed_sha = hashlib.sha256(copied.read_bytes()).hexdigest()
    assert observed_sha == hashlib.sha256(config.read_bytes()).hexdigest()
    provenance = json.loads((output / "package-provenance.json").read_text())
    assert provenance["fit_config"]["logical_path"] == "config/TOP_26_006.yml"
    assert provenance["fit_config"]["sha256"] == observed_sha
    assert provenance["fit_config"]["source_commit"] == "a" * 40
    assert provenance["standardization"]["source_package_provenance_sha256"] == hashlib.sha256(
        source_provenance_bytes
    ).hexdigest()
    assert {entry["path"]: entry["sha256"] for entry in provenance["standardization"]["member_inventory"]}[
        "config/TOP_26_006.yml"
    ] == observed_sha
    assert (output / "ordered_card_inputs.txt").read_text() == ordered
    assert (output / "scalings.json").read_bytes() == (source / "scalings.json").read_bytes()


@pytest.mark.parametrize("fault", ("missing", "empty", "symlink", "directory"))
def test_advertised_current_config_must_be_regular_and_nonempty(tmp_path, fault):
    source, _ = make_package(tmp_path, ("Run2_tau.txt",))
    config = add_current_config(source)
    if fault == "missing":
        config.unlink()
    elif fault == "empty":
        config.write_bytes(b"")
    elif fault == "symlink":
        config.unlink()
        config.symlink_to(source / "README.md")
    else:
        config.unlink()
        config.mkdir()
    output = tmp_path / "standardized"
    with pytest.raises(ValueError, match="Package-local fit config"):
        standardize_package(source, output)
    assert not output.exists()


def test_current_config_digest_warning_is_repaired_in_derivative_provenance(tmp_path):
    source, _ = make_package(tmp_path, ("Run2_tau.txt",))
    config = add_current_config(source)
    source_provenance_path = source / "package-provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text())
    source_provenance["fit_config"]["sha256"] = "0" * 64
    source_provenance_path.write_text(json.dumps(source_provenance) + "\n")
    output = tmp_path / "standardized"
    standardize_package(source, output)
    provenance = json.loads((output / "package-provenance.json").read_text())
    assert provenance["fit_config"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert provenance["standardization"]["source_package_provenance_sha256"] == hashlib.sha256(
        source_provenance_path.read_bytes()
    ).hexdigest()


def test_sparse_tau_and_per_run_mva_remain_separate(tmp_path):
    source, _ = make_package(tmp_path, ("Run2_tau.txt", "Run3_tau.txt"))
    add_current_config(source)
    tau_names = {
        "13TeV": "CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016",
        "13p6TeV": "CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022",
    }
    histogram = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    histogram.fill(0.5, weight=2.0)
    for card_name, domain in (("Run2_tau.txt", "13TeV"), ("Run3_tau.txt", "13p6TeV")):
        tau_name = tau_names[domain]
        card = source / "cards" / card_name
        with card.open("a") as stream:
            stream.write(f"{tau_name} shape {'1' if domain == '13TeV' else '-'}\n")
            stream.write(f"lepSF_elec_mva_{domain} shape 1\n")
        root = source / "cards" / card_name.replace(".txt", "_shapes.root")
        with uproot.update(root) as stream:
            for direction in ("Up", "Down"):
                stream[f"ttH_sm_lepSF_elec_mva_{domain}{direction}"] = histogram
                if domain == "13TeV":
                    stream[f"ttH_sm_{tau_name}{direction}"] = histogram
    output = tmp_path / "standardized"
    standardize_package(source, output)
    for card_name, domain in (("Run2_tau.txt", "13TeV"), ("Run3_tau.txt", "13p6TeV")):
        tau_name = tau_names[domain]
        card = (output / "cards" / card_name).read_text()
        root = output / "cards" / card_name.replace(".txt", "_shapes.root")
        assert f"lepSF_elec_mva_{domain} shape 1" in card
        other_domain = "13p6TeV" if domain == "13TeV" else "13TeV"
        assert f"lepSF_elec_mva_{other_domain}" not in card
        assert f"{tau_name} shape {'1' if domain == '13TeV' else '-'}" in card
        with uproot.open(root) as stream:
            keys = set(stream.keys(cycle=False))
            assert f"ttH_sm_lepSF_elec_mva_{domain}Up" in keys
            assert f"ttH_sm_lepSF_elec_mva_{domain}Down" in keys
            assert f"ttH_sm_lepSF_elec_mva_{other_domain}Up" not in keys
            assert f"ttH_sm_lepSF_elec_mva_{other_domain}Down" not in keys
            assert (f"ttH_sm_{tau_name}Up" in keys) == (domain == "13TeV")
            assert (f"ttH_sm_{tau_name}Down" in keys) == (domain == "13TeV")
            if domain == "13TeV":
                for direction in ("Up", "Down"):
                    assert stream[f"ttH_sm_{tau_name}{direction}"].values(flow=True).tolist() == [0.0, 2.0, 0.0]
