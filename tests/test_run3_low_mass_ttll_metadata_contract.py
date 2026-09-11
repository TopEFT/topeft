from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
TARGET_XSEC = Decimal("0.025442626")
ORDINARY_TARGET_XSEC = Decimal("0.366888675")
EXPECTED_LABELS = {
    "2022": "ttll_private2022",
    "2022EE": "ttll_private2022EE",
    "2023": "ttll_private2023",
    "2023BPix": "ttll_private2023BPix",
}
LOW_MASS_PATHS = (
    "input_samples/sample_jsons/signal_samples/ND_skim2022/ttll_mll4to10_2022.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2022EE/ttll_mll4to10_2022EE.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2023/ttll_mll4to10_2023.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2023BPix/ttll_mll4to10_2023BPix.json",
    "input_samples/sample_jsons/signal_nAOD/2022/ttll_mll4to10_2022.json",
    "input_samples/sample_jsons/signal_nAOD/2022EE/ttll_mll4to10_2022EE.json",
    "input_samples/sample_jsons/signal_nAOD/2023/ttll_mll4to10_2023.json",
    "input_samples/sample_jsons/signal_nAOD/2023BPix/ttll_mll4to10_2023BPix.json",
)
ORDINARY_PATHS = (
    "input_samples/sample_jsons/signal_samples/ND_SRskim2022/ttll_NDSkim_2022.json",
    "input_samples/sample_jsons/signal_samples/ND_SRskim2022EE/ttll_NDSkim_2022EE.json",
    "input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttll_NDSkim_2023.json",
    "input_samples/sample_jsons/signal_samples/ND_SRskim2023BPix/ttll_NDSkim_2023BPix.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2022/ttll_2022.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2022EE/ttll_2022EE.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2023/ttll_2023.json",
    "input_samples/sample_jsons/signal_samples/ND_skim2023BPix/ttll_2023BPix.json",
    "input_samples/sample_jsons/signal_nAOD/2022/ttll_2022.json",
    "input_samples/sample_jsons/signal_nAOD/2022EE/ttll_2022EE.json",
    "input_samples/sample_jsons/signal_nAOD/2023/ttll_2023.json",
    "input_samples/sample_jsons/signal_nAOD/2023BPix/ttll_2023BPix.json",
)
ZERO_EVENT_PROVENANCE_HASHES = {
    LOW_MASS_PATHS[0]: "375848c3d9f13a7a65514b26668bc228982ed3e23c21ec19a7ec7ac68e4c5158",
    LOW_MASS_PATHS[1]: "1676e33c1f67a865e7e9d43859f37e6a99461f070eae2095e557631e6df31e50",
    LOW_MASS_PATHS[2]: "311da43d2521ef7d54b80ebe82140be4148031fd294c244fa7c7211022d15287",
    LOW_MASS_PATHS[3]: "f8813888d8631b7fdc1aa04214e4eb0702b7bf15e886cb45441fb7b182fc70dc",
}
ACTIVE_CR_CONFIGS = (
    "input_samples/cfgs/NDSkim_2022_mc_signal_samples.cfg",
    "input_samples/cfgs/NDSkim_2022EE_mc_signal_samples.cfg",
    "input_samples/cfgs/NDSkim_2023_mc_signal_samples.cfg",
    "input_samples/cfgs/NDSkim_2023BPix_mc_signal_samples.cfg",
)
T0_SR_CONFIG_HASHES = {
    "input_samples/cfgs/NDSkim_2022_mc_signal_samples_sr.cfg": "e5bdd401d7f6c1964b33353bd42d509173e9d3e7e04ca6d87f899b904bca7b85",
    "input_samples/cfgs/NDSkim_2022EE_mc_signal_samples_sr.cfg": "033213aec8b498eb036b2709763f8151904dac9d7d37d7a0eb830e0b44a59269",
    "input_samples/cfgs/NDSkim_2023_mc_signal_samples_sr.cfg": "fad4af6f25342209e6d350d0fc30fb9aaf07b3ff449e21ecb6826c5e62518683",
    "input_samples/cfgs/NDSkim_2023BPix_mc_signal_samples_sr.cfg": "22c792e24829947e4c290f4b18d88a216cd4852886f0e6c538509d3bc46b9bb8",
}
EXPECTED_FILES_HASHES = {
    LOW_MASS_PATHS[0]: "e98afd0a7e35effa3dda3b0c8c9ecef740c1ecbf2b6f5c287a276ec648c654c8",
    LOW_MASS_PATHS[1]: "75dcd7fa9f20483228e71f025f49a58cb2a3c7d0f0ae230800e74604ce551c63",
    LOW_MASS_PATHS[2]: "b3855bbe517479a0b0695560ef57eda3d0e1e7d8739f3603558cfd0570424ae3",
    LOW_MASS_PATHS[3]: "9fe1d6bcfe0f99d10ce937a5e7a9eb9974db9338aaeb4f276e15867c51729b32",
    LOW_MASS_PATHS[4]: "52e9b8cad3bbd0225ccf688413e803c3e661217bc4bcaeeb82c2c795cd0f4fa1",
    LOW_MASS_PATHS[5]: "ea385dd23f78ef55a5c498e29b015e5abf6167d941ecd8b5c244916df0c64f63",
    LOW_MASS_PATHS[6]: "d6254d49d55101b62c846ad172434d329027bf6c58fefe5b35ec1993b288e988",
    LOW_MASS_PATHS[7]: "209f8fb688f8c38a476f37621a2fe19063d94ee21b98f590853e9dd9456382ff",
}
EXPECTED_WC_HASH = "86e7e95978f769fcc0674bdacdf4a347f4be8586dc8b320e93a1dd0976a0a2e1"


def _canonical_hash(value):
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load(path):
    return json.loads((REPO / path).read_text(encoding="utf-8"), parse_float=Decimal)


def _assert_contract(payload):
    assert payload["xsec"] == TARGET_XSEC
    assert payload["histAxisName"] == EXPECTED_LABELS[payload["year"]]
    assert payload["eft_treatment"] == "sm_only"
    assert payload["isData"] is False
    assert len(payload["WCnames"]) == 32
    assert len(set(payload["WCnames"])) == 32


@pytest.mark.parametrize("path", LOW_MASS_PATHS)
def test_every_low_mass_metadata_copy_has_the_exact_contract(path):
    payload = _load(path)
    _assert_contract(payload)
    assert _canonical_hash(payload["WCnames"]) == EXPECTED_WC_HASH
    assert _canonical_hash(payload["files"]) == EXPECTED_FILES_HASHES[path]


def test_each_year_has_two_maintained_copies_with_identical_role_and_wc_basis():
    by_year = {}
    for path in LOW_MASS_PATHS:
        payload = _load(path)
        by_year.setdefault(payload["year"], []).append(payload)
    assert set(by_year) == set(EXPECTED_LABELS)
    for year, payloads in by_year.items():
        assert len(payloads) == 2
        assert {payload["histAxisName"] for payload in payloads} == {EXPECTED_LABELS[year]}
        assert {tuple(payload["WCnames"]) for payload in payloads} == {tuple(payloads[0]["WCnames"])}


def test_global_wc_union_is_unchanged_from_ordinary_ttll():
    ordinary = _load(ORDINARY_PATHS[0])
    expected = tuple(ordinary["WCnames"])
    assert len(expected) == 32
    for path in LOW_MASS_PATHS:
        assert tuple(_load(path)["WCnames"]) == expected


@pytest.mark.parametrize("path", ORDINARY_PATHS)
def test_ordinary_ttll_metadata_uses_the_run3_cross_section(path):
    assert _load(path)["xsec"] == ORDINARY_TARGET_XSEC


@pytest.mark.parametrize("path,expected_hash", ZERO_EVENT_PROVENANCE_HASHES.items())
def test_zero_event_low_mass_provenance_metadata_is_preserved(path, expected_hash):
    assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected_hash


@pytest.mark.parametrize("path", ACTIVE_CR_CONFIGS)
def test_active_cr_signal_configs_exclude_low_mass(path):
    assert "ttll_mll4to10" not in (REPO / path).read_text(encoding="utf-8")


@pytest.mark.parametrize("path,expected_hash", T0_SR_CONFIG_HASHES.items())
def test_t0_sr_signal_configs_remain_unchanged_and_exclude_low_mass(
    path,
    expected_hash,
):
    contents = (REPO / path).read_bytes()
    assert hashlib.sha256(contents).hexdigest() == expected_hash
    assert b"ttll_mll4to10" not in contents


def test_contract_rejects_any_other_low_mass_cross_section():
    payload = _load(LOW_MASS_PATHS[0])
    payload["xsec"] = Decimal("0.025442625")
    with pytest.raises(AssertionError):
        _assert_contract(payload)
