from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import correctionlib
import json
import numpy as np
import pytest
from topcoffea.modules.paths import topcoffea_path

import topeft.modules.corrections as corrections


REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "topeft" / "params" / "params.json"
PROCESSOR_PATH = REPO_ROOT / "analysis" / "topeft_run2" / "analysis_processor.py"
CORRECTIONS_PATH = REPO_ROOT / "topeft" / "modules" / "corrections.py"


class _RecordingCorrection:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def evaluate(self, *args):
        self.calls.append(args)
        return np.ones(len(args[0]), dtype=np.float32)


class _RecordingEvaluator:
    def __init__(self):
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)

        def evaluate(first, *args):
            return ak.ones_like(first, dtype=np.float32)

        return evaluate


def _tau_record(year, gen_part_flav):
    if year.startswith("201"):
        discriminator_fields = {
            "idDeepTau2017v2p1VSjet": 16,
            "idDeepTau2017v2p1VSe": 2,
            "idDeepTau2017v2p1VSmu": 8,
        }
    else:
        discriminator_fields = {
            "idDeepTau2018v2p5VSjet": 5,
            "idDeepTau2018v2p5VSe": 2,
            "idDeepTau2018v2p5VSmu": 4,
        }

    return ak.Array(
        [[
            {
                "pt": 35.0,
                "mass": 1.2,
                "eta": 0.5,
                "decayMode": 0,
                "genPartFlav": gen_part_flav,
                "isLoose": 1,
                "isMedium": 1,
                "iseTight": 1,
                "ismTight": 1,
                **discriminator_fields,
            }
        ]]
    )


@pytest.mark.parametrize(
    ("year", "vsjet_correction_name", "fake_sf_key"),
    [
        ("2018", "DeepTau2017v2p1VSjet", "TauFakeSF"),
        ("2022", "DeepTau2018v2p5VSjet", "TauFakeSF_Run3"),
    ],
)
def test_tau_vsjet_payload_uses_aligned_medium_wp_and_fake_sf_stays_separate(
    monkeypatch,
    year,
    vsjet_correction_name,
    fake_sf_key,
):
    recording_corrections = {
        vsjet_correction_name: _RecordingCorrection(vsjet_correction_name),
        "DeepTau2018v2p5VSe": _RecordingCorrection("DeepTau2018v2p5VSe"),
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    recording_evaluator = _RecordingEvaluator()
    monkeypatch.setattr(corrections, "SFevaluator", recording_evaluator)

    corrections.AttachTauSF(
        {},
        _tau_record(year, gen_part_flav=5),
        year,
        vsJetWP=corrections.get_te_param(
            "run2_tau_t_tag" if year.startswith("201") else "run3_tau_t_tag"
        ),
    )

    vsjet_calls = recording_corrections[vsjet_correction_name].calls
    assert vsjet_calls
    assert all(call[3] == "Medium" for call in vsjet_calls)
    if year.startswith("201"):
        assert vsjet_calls[0][4:] == ("VVLoose", "nom", "dm")
    assert fake_sf_key in recording_evaluator.keys


@pytest.mark.parametrize(
    ("year", "vsjet_correction_name", "fake_sf_key"),
    [
        ("2018", "DeepTau2017v2p1VSjet", "TauFakeSF"),
        ("2022", "DeepTau2018v2p5VSjet", "TauFakeSF_Run3"),
    ],
)
def test_jet_faking_tau_sf_keeps_its_dedicated_non_pog_payload(
    monkeypatch,
    year,
    vsjet_correction_name,
    fake_sf_key,
):
    recording_corrections = {
        vsjet_correction_name: _RecordingCorrection(vsjet_correction_name),
        "DeepTau2018v2p5VSe": _RecordingCorrection("DeepTau2018v2p5VSe"),
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    recording_evaluator = _RecordingEvaluator()
    monkeypatch.setattr(corrections, "SFevaluator", recording_evaluator)

    corrections.AttachTauSF(
        {},
        _tau_record(year, gen_part_flav=0),
        year,
        vsJetWP=corrections.get_te_param(
            "run2_tau_t_tag" if year.startswith("201") else "run3_tau_t_tag"
        ),
    )

    assert fake_sf_key in recording_evaluator.keys


def test_params_json_defines_run2_and_run3_tau_tags():
    params = json.loads(PARAMS_PATH.read_text())
    stale_key = "tau_" + "pog_vsjet_wp"

    assert params["run2_tau_t_tag"] == "Medium"
    assert params["run3_tau_t_tag"] == "Medium"
    assert params["run2_tau_fo_tag"] == "Loose"
    assert params["run3_tau_fo_tag"] == "Loose"
    assert stale_key not in params


def test_run2_deeptau2017_vsjet_schema_matches_evaluate_call_order():
    expected_input_names = [
        "pt",
        "dm",
        "genmatch",
        "wp",
        "wp_VSe",
        "syst",
        "flag",
    ]
    for year in ("2016APV", "2016", "2017", "2018"):
        clib_year = corrections.clib_year_map[year]
        payload_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
        corr = correctionlib.CorrectionSet.from_file(payload_path)[
            "DeepTau2017v2p1VSjet"
        ]
        assert [input_.name for input_ in corr.inputs] == expected_input_names


def test_processor_uses_params_tau_wp_and_loose_tau_fo_tag_for_run2_and_run3():
    processor_source = PROCESSOR_PATH.read_text()

    assert "tau_T_tag = get_te_param(" in processor_source
    assert "tau_fo_tag = get_te_param(" in processor_source
    for key in (
        "run2_tau_t_tag",
        "run3_tau_t_tag",
        "run2_tau_fo_tag",
        "run3_tau_fo_tag",
    ):
        assert key in processor_source
    assert 'tau_fo_tag = "Loose"' not in processor_source
    assert 'tau_fo_tag = "VLoose" if is_run2 else "Loose"' not in processor_source
    assert 'tau_T_tag = "Loose" if is_run2 else "Medium"' not in processor_source


def test_tau_tag_params_use_direct_access_without_redundant_helpers():
    corrections_source = CORRECTIONS_PATH.read_text()
    processor_source = PROCESSOR_PATH.read_text()
    rejected_getter = "get_tau_" + "pog_vsjet_wp"
    rejected_resolver = "_resolve_tau_" + "pog_vsjet_wp"
    stale_key = "tau_" + "pog_vsjet_wp"

    assert f"def {rejected_getter}" not in corrections_source
    assert f"def {rejected_resolver}" not in corrections_source
    assert rejected_getter not in processor_source
    assert rejected_resolver not in corrections_source
    assert stale_key not in corrections_source
    assert stale_key not in processor_source
    assert 'get_te_param("run2_tau_t_tag" if is_run2 else "run3_tau_t_tag")' in corrections_source
