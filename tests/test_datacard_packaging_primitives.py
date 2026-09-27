"""Focused contracts for the maintained, unwired packaging primitives."""

import ast
import copy
import json
from pathlib import Path

import pytest

from topeft.modules import datacard_packaging as packaging


_fixtures = Path(__file__).parent / "data" / "datacard_packages"


def _golden(name):
    return json.loads((_fixtures / f"top26006_{name}_golden_v1.json").read_text(encoding="utf-8"))


def _mappings():
    run2 = _golden("run2_per_era")["physical_to_chN"]
    run3 = _golden("run3_per_era")["physical_to_chN"]
    return run2, run3


def _source_scalings():
    return [[
        {"channel": "alpha_ptz", "process": "ttH", "parameters": ["cSM[1]"], "scaling": [[1.0, 0.25]]},
        {"channel": "beta_lj0pt", "process": "ttH", "parameters": ["cSM[1]"], "scaling": [[1.0, -0.5]]},
    ]]


def test_sha256_file(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abc")
    assert packaging.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_per_era_mapping_golden_and_fault_isolation(monkeypatch):
    assert packaging.build_per_era_mapping(["alpha_ptz", "beta_lj0pt"]) == [
        {"physical_name": "alpha_ptz", "per_era_chN": "ch1"},
        {"physical_name": "beta_lj0pt", "per_era_chN": "ch2"},
    ]
    with pytest.raises(ValueError):
        packaging.build_per_era_mapping(["alpha_ptz", "alpha_ptz"])
    for mapping in _mappings():
        names = [row["physical_name"] for row in mapping]
        assert packaging.build_per_era_mapping(names) == mapping
        assert packaging.verify_per_era_mapping(mapping, names)
    malformed = copy.deepcopy(_mappings()[0])
    malformed[1]["per_era_chN"] = "ch130"
    with pytest.raises(ValueError):
        packaging.verify_per_era_mapping(malformed, [row["physical_name"] for row in malformed])
    wrong = [{"physical_name": "alpha_ptz", "per_era_chN": "ch2"},
             {"physical_name": "beta_lj0pt", "per_era_chN": "ch1"}]
    monkeypatch.setattr(packaging, "build_per_era_mapping", lambda _: wrong)
    with pytest.raises(ValueError):
        packaging.verify_per_era_mapping(packaging.build_per_era_mapping([]), ["alpha_ptz", "beta_lj0pt"])


def test_selected_wcs_union_duplicates_and_fault_isolation(monkeypatch):
    units = [{"ttH": ["ctp", "ctG"], "tHq": ["ctW"]},
             {"ttH": ["ctG", "cpt"], "tHq": ["ctW", "ctp"]}]
    expected = {"ttH": ["ctp", "ctG", "cpt"], "tHq": ["ctW", "ctp"]}
    assert packaging.consolidate_selected_wcs(units) == expected
    assert packaging.verify_per_era_selected_wcs(expected, units)
    assert packaging.verify_per_era_selected_wcs(dict(reversed(list(expected.items()))), units)
    with pytest.raises(ValueError):
        packaging.consolidate_selected_wcs([{"ttH": ["ctp", "ctp"]}])
    with pytest.raises(ValueError):
        packaging.consolidate_selected_wcs([{"ttH": "ctp"}])
    monkeypatch.setattr(packaging, "consolidate_selected_wcs", lambda _: {"ttH": ["ctG", "ctp", "cpt"], "tHq": ["ctW", "ctp"]})
    with pytest.raises(ValueError):
        packaging.verify_per_era_selected_wcs(packaging.consolidate_selected_wcs(units), units)


def test_scaling_relabelling_payload_and_fault_isolation(monkeypatch):
    source = _source_scalings()
    mapping = packaging.build_per_era_mapping(["alpha_ptz", "beta_lj0pt"])
    output = packaging.consolidate_scaling_records(source, mapping)
    assert [row["channel"] for row in output] == ["ch1", "ch2"]
    assert [row["scaling"] for row in output] == [row["scaling"] for row in source[0]]
    assert packaging.verify_per_era_scalings(output, source, mapping)
    with pytest.raises(ValueError):
        packaging.consolidate_scaling_records(source, mapping[:1])
    with pytest.raises(ValueError):
        packaging.consolidate_scaling_records(source + [source[0]], mapping)
    corrupted = copy.deepcopy(output)
    corrupted[0]["scaling"][0][1] = 0.75
    monkeypatch.setattr(packaging, "consolidate_scaling_records", lambda *_: corrupted)
    with pytest.raises(ValueError):
        packaging.verify_per_era_scalings(packaging.consolidate_scaling_records(source, mapping), source, mapping)


@pytest.mark.parametrize("field", ["channel", "process"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_scaling_rejects_blank_identities(field, blank):
    record = copy.deepcopy(_source_scalings()[0][0])
    record[field] = blank
    with pytest.raises(ValueError):
        packaging._scaling_record(record)


def test_scaling_preserves_valid_identity_and_owns_nested_records():
    source = _source_scalings()
    source[0][0]["process"] = " ttH "
    source[0][0]["metadata"] = {"tags": ["first"]}
    source[0][1]["metadata"] = {"tags": ["second"]}
    source_before = json.dumps(source)
    mapping = [
        {"physical_name": "alpha_ptz", "per_era_chN": "ch1"},
        {"physical_name": "beta_lj0pt", "per_era_chN": "ch2"},
    ]

    output = packaging.consolidate_scaling_records(source, mapping)
    assert json.dumps(source) == source_before
    assert [record["process"] for record in output] == [" ttH ", "ttH"]
    assert output[0]["metadata"] == source[0][0]["metadata"]
    assert packaging.verify_per_era_scalings(output, source, mapping)
    second_before = copy.deepcopy(output[1])

    output[0]["scaling"][0][1] = 9.0
    output[0]["parameters"].append("ctG")
    output[0]["metadata"]["tags"].append("changed")
    assert json.dumps(source) == source_before
    assert output[1] == second_before


def test_combined_mapping_golden_offset_and_fault_isolation(monkeypatch):
    run2, run3 = _mappings()
    golden = _golden("combined")["combined_mapping"]
    assert packaging.build_combined_mapping(run2, run3) == golden
    assert packaging.verify_combined_mapping(golden, run2, run3)
    small = packaging.build_combined_mapping(run2[:2], run3[:1])
    assert small[-1]["combined_chN"] == "ch3"
    for bad in (run2[:1] + run2[:1], [run2[0], run2[2]]):
        with pytest.raises(ValueError):
            packaging.build_combined_mapping(bad, run3[:1])
    with pytest.raises(ValueError):
        packaging.build_combined_mapping(run2[:2][::-1], run3[:1])
    corrupted = copy.deepcopy(golden)
    corrupted[129]["destination_txt_name"] = "Run3_wrong.txt"
    monkeypatch.setattr(packaging, "build_combined_mapping", lambda *_: corrupted)
    with pytest.raises(ValueError):
        packaging.verify_combined_mapping(packaging.build_combined_mapping(run2, run3), run2, run3)


def test_combined_verifier_rejects_reinterpreted_source_order():
    run2_source = [
        {"physical_name": "beta", "per_era_chN": "ch2"},
        {"physical_name": "alpha", "per_era_chN": "ch1"},
    ]
    run3_source = [{"physical_name": "gamma", "per_era_chN": "ch1"}]
    observed = [
        {"era": "run2", "physical_name": "beta", "per_era_chN": "ch1",
         "combined_chN": "ch1", "combined_order_index": 1,
         "destination_txt_name": "Run2_ttx_multileptons-beta.txt",
         "destination_root_name": "Run2_ttx_multileptons-beta.root"},
        {"era": "run2", "physical_name": "alpha", "per_era_chN": "ch2",
         "combined_chN": "ch2", "combined_order_index": 2,
         "destination_txt_name": "Run2_ttx_multileptons-alpha.txt",
         "destination_root_name": "Run2_ttx_multileptons-alpha.root"},
        {"era": "run3", "physical_name": "gamma", "per_era_chN": "ch1",
         "combined_chN": "ch3", "combined_order_index": 3,
         "destination_txt_name": "Run3_ttx_multileptons-gamma.txt",
         "destination_root_name": "Run3_ttx_multileptons-gamma.root"},
    ]
    with pytest.raises(ValueError, match="source mapping rows are out of canonical order"):
        packaging.verify_combined_mapping(observed, run2_source, run3_source)

    canonical_run2 = [
        {"physical_name": "beta", "per_era_chN": "ch1"},
        {"physical_name": "alpha", "per_era_chN": "ch2"},
    ]
    assert packaging.verify_combined_mapping(observed, canonical_run2, run3_source)


def test_ordered_inputs_golden_and_fault_isolation(monkeypatch):
    run2, run3 = _mappings()
    mapping = packaging.build_combined_mapping(run2, run3)
    basenames = _golden("combined")["ordered_card_inputs"]["basenames"]
    expected = ["cards/" + name for name in basenames]
    assert packaging.build_ordered_card_inputs(mapping) == expected
    assert packaging.verify_ordered_card_inputs("\n".join(expected) + "\n", mapping)
    bad_cases = [expected[1:2] + expected[:1] + expected[2:], expected[:-1],
                 expected[:-1] + expected[:1], [line[6:] for line in expected]]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            packaging.verify_ordered_card_inputs(bad, mapping)
    monkeypatch.setattr(packaging, "build_ordered_card_inputs", lambda _: expected[::-1])
    with pytest.raises(ValueError):
        packaging.verify_ordered_card_inputs(packaging.build_ordered_card_inputs(mapping), mapping)


def test_prohibited_call_edges_absent_transitively():
    tree = ast.parse(Path(packaging.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = {
        name: {node.func.id for node in ast.walk(function) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id in functions}
        for name, function in functions.items()
    }
    pairs = {
        "verify_per_era_mapping": "build_per_era_mapping",
        "verify_per_era_selected_wcs": "consolidate_selected_wcs",
        "verify_per_era_scalings": "consolidate_scaling_records",
        "verify_combined_mapping": "build_combined_mapping",
        "verify_ordered_card_inputs": "build_ordered_card_inputs",
    }
    for verifier, writer in pairs.items():
        reachable = set()
        pending = list(calls[verifier])
        while pending:
            name = pending.pop()
            if name not in reachable:
                reachable.add(name)
                pending.extend(calls[name])
        assert writer not in reachable
