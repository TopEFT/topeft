import hashlib
import json

import hist
import pytest
import uproot

from analysis.topeft_run2.standardize_systematics import (
    SYST_NAMING_MAP,
    _preflight_package,
    remap_name,
    resolve_nuisance_name,
    standardize_package,
)
from topeft.modules.datacard_tools import cross_run_mva_identities


@pytest.mark.parametrize(
    "run,domain", (("run2", "13TeV"), ("run3", "13p6TeV"))
)
def test_pu_met_and_accepted_aliases(run, domain):
    assert resolve_nuisance_name("PU", run) == f"CMS_pileup_{domain}"
    assert resolve_nuisance_name("MET_UnclusteredEnergy", run) == (
        f"CMS_scale_met_Unclustred_energy_{domain}"
    )
    assert resolve_nuisance_name("PreFiring", run) == "CMS_l1_ecal_prefiring"
    for source in ("lepSF_elec_non_mva", "lepSF_muon_non_mva"):
        assert resolve_nuisance_name(source, run) == f"{source}_{domain}"
    for shared in cross_run_mva_identities:
        assert resolve_nuisance_name(shared, run) == shared
        assert resolve_nuisance_name(f"{shared}_{domain}", run) == f"{shared}_{domain}"
        for direction in ("Up", "Down"):
            assert remap_name(f"ttH_sm_{shared}{direction}", run) == f"ttH_sm_{shared}{direction}"
            assert remap_name(f"ttH_sm_{shared}_{domain}{direction}", run) == f"ttH_sm_{shared}_{domain}{direction}"
    for invalid in (
        "lepSF_elec_mva_13TeV_13p6TeV",
        "lepSF_muon_mva_13TeV_13p6TeV",
        "lepSF_elec_mva_13TeV_13p6TeV_extra",
        f"lepSF_elec_mva_{'13p6TeV' if run == 'run2' else '13TeV'}",
        "lepSF_muon_mva_unknown",
    ):
        with pytest.raises(RuntimeError, match="not found"):
            resolve_nuisance_name(invalid, run)


@pytest.mark.parametrize("dm", (0, 1, 10, 11))
@pytest.mark.parametrize("period", ("2022EE", "2023BPix"))
def test_retired_efes_aliases_rejected(dm, period):
    source = f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron_{period}"
    assert source not in SYST_NAMING_MAP["run3"]
    with pytest.raises(RuntimeError, match="not found"):
        resolve_nuisance_name(source, "run3")
    year = period[:4]
    valid = f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron_{year}"
    assert resolve_nuisance_name(valid, "run3") == valid


def make_package(tmp_path, card_names=("Run3_second.txt", "Run2_first.txt"), eras=None):
    source = tmp_path / "source"
    cards = source / "cards"
    cards.mkdir(parents=True)
    histogram = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    histogram.fill(0.5)
    for card_name in card_names:
        root_name = card_name.replace(".txt", "_shapes.root")
        (cards / card_name).write_text(
            f"shapes * * {root_name} $PROCESS $PROCESS_$SYSTEMATIC\n"
            "PU shape 1\nMET_UnclusteredEnergy shape 1\n"
        )
        with uproot.recreate(cards / root_name) as root_file:
            for source_name in ("PU", "MET_UnclusteredEnergy"):
                for direction in ("Up", "Down"):
                    root_file[f"ttH_sm_{source_name}{direction}"] = histogram
    ordered = "".join(f"cards/{card_name}\n" for card_name in card_names)
    (source / "ordered_card_inputs.txt").write_text(ordered)
    (source / "scalings.json").write_text('[{"channel": "ch1"}]\n')
    (source / "README.md").write_text("Source datacard package\n")
    mapping = {
        "schema": "TOP26006_v1", "artifact_type": "combined_mapping_manifest",
        "package_root": str(source), "rows": [
            {"era": eras[index - 1] if eras is not None else
                    ("run3" if name.startswith("Run3_") else "run2"),
             "physical_name": name.removesuffix(".txt"),
             "per_era_chN": f"ch{index}", "combined_chN": f"ch{index}",
             "combined_order_index": index, "destination_txt_name": name,
             "destination_root_name": name.replace(".txt", "_shapes.root")}
            for index, name in enumerate(card_names, 1)
        ],
    }
    (source / "combined_mapping_manifest.json").write_text(json.dumps(mapping) + "\n")
    provenance = {
        "schema": "TOP26006_v1", "artifact_type": "package_provenance",
        "analysis": "TOP-26-006", "package_root": str(source),
        "package_version": "v1", "package_date": "261005",
        "assembler_commit": "a" * 40, "assembler_source_sha256": "b" * 64,
        "source_mapping_sha256": {"run2": "c" * 64, "run3": "d" * 64},
        "source_scalings_sha256": {"run2": "e" * 64, "run3": "f" * 64},
        "manifest_sha256": hashlib.sha256((source / "combined_mapping_manifest.json").read_bytes()).hexdigest(),
        "ordered_card_inputs_sha256": hashlib.sha256(ordered.encode()).hexdigest(),
        "scalings_sha256": hashlib.sha256((source / "scalings.json").read_bytes()).hexdigest(),
        "packaged_txt_count": len(card_names), "packaged_root_count": len(card_names),
    }
    (source / "package-provenance.json").write_text(json.dumps(provenance) + "\n")
    return source, ordered


def test_ordered_pairing_provenance_and_existing_output(tmp_path):
    source, ordered = make_package(tmp_path)
    output = tmp_path / "derivative"
    provenance = standardize_package(source, output)
    assert (output / "ordered_card_inputs.txt").read_text() == ordered
    assert {path.name for path in (output / "cards").iterdir()} == {
        "Run3_second.txt", "Run2_first.txt",
        "Run3_second_shapes.root", "Run2_first_shapes.root",
    }
    assert provenance == json.loads((output / "package-provenance.json").read_text())
    assert provenance["standardization"]["source_package_root"] == str(source)
    assert provenance["standardization"]["source_package_provenance_sha256"] == hashlib.sha256(
        (source / "package-provenance.json").read_bytes()
    ).hexdigest()
    assert provenance["packaged_txt_count"] == provenance["packaged_root_count"] == 2
    readme = (output / "README.md").read_text()
    assert "Drop-in datacard-package input" in readme
    assert "not a combined card, workspace, or fit result" in readme
    for card_name, domain in (("Run3_second.txt", "13p6TeV"),
                              ("Run2_first.txt", "13TeV")):
        card_text = (output / "cards" / card_name).read_text()
        root_name = card_name.replace(".txt", "_shapes.root")
        assert root_name in card_text
        assert f"CMS_pileup_{domain} shape" in card_text
        with uproot.open(output / "cards" / root_name) as root_file:
            keys = set(root_file.keys(cycle=False))
            assert f"ttH_sm_CMS_pileup_{domain}Up" in keys
            assert f"ttH_sm_CMS_pileup_{domain}Down" in keys
            assert f"ttH_sm_CMS_scale_met_Unclustred_energy_{domain}Up" in keys
    with pytest.raises(FileExistsError):
        standardize_package(source, output)


def test_distinct_final_nuisances_cannot_collapse(tmp_path, monkeypatch):
    source, _ = make_package(tmp_path, ("Run2_first.txt",))
    monkeypatch.setitem(SYST_NAMING_MAP["run2"], "MET_UnclusteredEnergy", "CMS_pileup_13TeV")
    with pytest.raises(ValueError, match="Nuisance collision"):
        standardize_package(source, tmp_path / "derivative")
    assert not (tmp_path / "derivative").exists()


def test_output_cannot_be_inside_source_package(tmp_path):
    source, _ = make_package(tmp_path, ("Run2_first.txt",))
    with pytest.raises(ValueError, match="outside the input package"):
        standardize_package(source, source / "derivative")
    assert not (source / "derivative").exists()


def test_root_key_collision_fails_before_output(tmp_path, monkeypatch):
    source, _ = make_package(tmp_path, ("Run2_first.txt",))
    root_path = source / "cards" / "Run2_first_shapes.root"
    histogram = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    with uproot.update(root_path) as root_file:
        for direction in ("Up", "Down"):
            root_file[f"ttH_sm_JES_Absolute{direction}"] = histogram
    monkeypatch.setitem(SYST_NAMING_MAP["run2"], "JES_Absolute", "CMS_pileup_13TeV")
    with pytest.raises(ValueError, match="Nuisance collision|ROOT key collision"):
        standardize_package(source, tmp_path / "derivative")
    assert not (tmp_path / "derivative").exists()


def _add_mva_shape(source, card_name, identity):
    card = source / "cards" / card_name
    with card.open("a") as output:
        output.write(f"{identity} shape 1\n")
    root = source / "cards" / card_name.replace(".txt", "_shapes.root")
    histogram = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    histogram.fill(0.5, weight=2.0)
    with uproot.update(root) as output:
        for direction in ("Up", "Down"):
            output[f"ttH_sm_{identity}{direction}"] = histogram


def test_shared_mva_names_survive_paired_package_standardization(tmp_path):
    source, _ = make_package(tmp_path)
    for card_name in ("Run2_first.txt", "Run3_second.txt"):
        for shared in cross_run_mva_identities:
            _add_mva_shape(source, card_name, shared)
    output = tmp_path / "derivative"
    standardize_package(source, output)
    for card_name in ("Run2_first.txt", "Run3_second.txt"):
        card = output / "cards" / card_name
        root = output / "cards" / card_name.replace(".txt", "_shapes.root")
        for shared in cross_run_mva_identities:
            assert f"{shared} shape 1" in card.read_text()
            with uproot.open(root) as source_root, uproot.open(
                source / "cards" / root.name
            ) as original_root:
                for direction in ("Up", "Down"):
                    key = f"ttH_sm_{shared}{direction}"
                    assert key in source_root
                    assert source_root[key].values(flow=True).tolist() == original_root[key].values(flow=True).tolist()
                    assert source_root[key].variances(flow=True).tolist() == original_root[key].variances(flow=True).tolist()


@pytest.mark.parametrize("base", cross_run_mva_identities)
@pytest.mark.parametrize("shared_card,per_card,domain", (
    ("Run2_first.txt", "Run3_second.txt", "13p6TeV"),
    ("Run3_second.txt", "Run2_first.txt", "13TeV"),
))
def test_mixed_mva_topology_rejected_without_absence_false_positive(
    tmp_path, base, shared_card, per_card, domain,
):
    source, _ = make_package(tmp_path)
    _add_mva_shape(source, shared_card, base)
    _add_mva_shape(source, per_card, f"{base}_{domain}")
    with pytest.raises(ValueError, match="Mixed lepton MVA correlation identities"):
        standardize_package(source, tmp_path / "derivative")
    assert not (tmp_path / "derivative").exists()
    source_2, _ = make_package(tmp_path / "separate")
    _add_mva_shape(source_2, shared_card, base)
    standardize_package(source_2, tmp_path / "separate_derivative")


def test_per_run_mva_names_survive_paired_package_standardization(tmp_path):
    source, _ = make_package(tmp_path)
    for card_name, domain in (("Run2_first.txt", "13TeV"),
                              ("Run3_second.txt", "13p6TeV")):
        for base in cross_run_mva_identities:
            _add_mva_shape(source, card_name, f"{base}_{domain}")
    output = tmp_path / "derivative"
    standardize_package(source, output)
    for card_name, domain in (("Run2_first.txt", "13TeV"),
                              ("Run3_second.txt", "13p6TeV")):
        card = output / "cards" / card_name
        root = output / "cards" / card_name.replace(".txt", "_shapes.root")
        with uproot.open(root) as output_root:
            original_root = source / "cards" / root.name
            for base in cross_run_mva_identities:
                identity = f"{base}_{domain}"
                assert f"{identity} shape 1" in card.read_text()
                with uproot.open(original_root) as source_root:
                    for direction in ("Up", "Down"):
                        key = f"ttH_sm_{identity}{direction}"
                        assert key in output_root
                        assert output_root[key].values(flow=True).tolist() == source_root[key].values(flow=True).tolist()
                        assert output_root[key].variances(flow=True).tolist() == source_root[key].variances(flow=True).tolist()


@pytest.mark.parametrize("card_name,invalid", (
    ("Run2_first.txt", "lepSF_elec_mva_13p6TeV"),
    ("Run3_second.txt", "lepSF_muon_mva_13TeV"),
    ("Run2_first.txt", "lepSF_elec_mva_13TeV_13p6TeV"),
    ("Run3_second.txt", "lepSF_muon_mva_13TeV_13p6TeV"),
    ("Run3_second.txt", "lepSF_elec_mva_unknown"),
))
def test_invalid_mva_package_identity_rejected_before_output(tmp_path, card_name, invalid):
    source, _ = make_package(tmp_path)
    _add_mva_shape(source, card_name, invalid)
    output = tmp_path / "derivative"
    with pytest.raises(RuntimeError, match="not found"):
        standardize_package(source, output)
    assert not output.exists()


@pytest.mark.parametrize("base", cross_run_mva_identities)
def test_root_only_mixed_mva_topology_rejected_before_output(tmp_path, base):
    source, _ = make_package(tmp_path)
    _add_mva_shape(source, "Run2_first.txt", base)
    root = source / "cards" / "Run3_second_shapes.root"
    histogram = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    with uproot.update(root) as root_file:
        for direction in ("Up", "Down"):
            root_file[f"ttH_sm_{base}_13p6TeV{direction}"] = histogram
    output = tmp_path / "derivative"
    with pytest.raises(ValueError, match="Mixed lepton MVA correlation identities"):
        standardize_package(source, output)
    assert not output.exists()
