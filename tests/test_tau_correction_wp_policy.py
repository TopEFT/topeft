from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import correctionlib
import json
import numpy as np
import pytest
from topcoffea.modules.paths import topcoffea_path

import topeft.modules.corrections as corrections
from topeft.modules.object_selection import run3TauSelection


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


class _DirectionalCorrection(_RecordingCorrection):
    def evaluate(self, *args):
        self.calls.append(args)
        systematic = args[-2] if args[-1] == "dm" else args[-1]
        if systematic == "nom":
            value = 2.0
        elif systematic.endswith("up"):
            value = 4.0
        elif systematic.endswith("down"):
            value = 1.0
        else:
            raise AssertionError(f"Unexpected systematic {systematic}")
        return np.full(len(args[0]), value, dtype=np.float32)


class _RecordingEvaluator:
    def __init__(self):
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)

        def evaluate(first, *args):
            return ak.ones_like(first, dtype=np.float32)

        return evaluate


class _MissingKeyEvaluator(_RecordingEvaluator):
    def __init__(self, missing_keys):
        super().__init__()
        self.missing_keys = set(missing_keys)

    def __getitem__(self, key):
        if key in self.missing_keys:
            raise KeyError(key)
        return super().__getitem__(key)


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
                "isTight": 1,
                "isVLoose": 1,
                "iseTight": 1,
                "ismTight": 1,
                **discriminator_fields,
            }
        ]]
    )


def _run3_tau_records(etas, gen_part_flav, decay_modes=None):
    if decay_modes is None:
        decay_modes = (0,) * len(etas)
    return ak.Array(
        [
            [
            {
                "pt": 35.0,
                "mass": 1.2,
                "eta": eta,
                "decayMode": decay_mode,
                "genPartFlav": gen_part_flav,
                "isMedium": 1,
                "iseTight": 1,
                "ismTight": 1,
                "idDeepTau2018v2p5VSjet": 5,
                "idDeepTau2018v2p5VSe": 2,
                "idDeepTau2018v2p5VSmu": 4,
            }
            ]
            for eta, decay_mode in zip(etas, decay_modes)
        ]
    )


@pytest.mark.parametrize(
    ("source", "gen_part_flav", "etas", "expected_tokens"),
    [
        (
            "VSe",
            1,
            (1.4599, 1.46, 1.5599, 1.56, 2.5),
            (
                "abseta0to1p46",
                "abseta1p46to1p56",
                "abseta1p46to1p56",
                "abseta1p56to2p5",
                "abseta1p56to2p5",
            ),
        ),
        (
            "VSmu",
            2,
            (0.3999, 0.4, 1.6999, 1.7, 2.4),
            (
                "abseta0to0p4",
                "abseta0p4to0p8",
                "abseta1p2to1p7",
                "abseta1p7to2p4",
                "abseta1p7to2p4",
            ),
        ),
    ],
)
def test_run3_eta_nuisance_masks_are_source_specific_and_boundary_exact(
    monkeypatch, source, gen_part_flav, etas, expected_tokens
):
    recording_corrections = {
        name: _DirectionalCorrection(name)
        for name in (
            "DeepTau2018v2p5VSjet",
            "DeepTau2018v2p5VSe",
            "DeepTau2018v2p5VSmu",
        )
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    monkeypatch.setattr(corrections, "SFevaluator", _RecordingEvaluator())
    weights = corrections.AttachTauSF(
        {},
        _run3_tau_records(etas, gen_part_flav),
        "2022",
        vsJetWP="Medium",
    )

    for index, expected_token in enumerate(expected_tokens):
        if source == "VSe":
            expected_name = (
                "CMS_fake_t_DeepTau2018v2p5_VSe_DM0_"
                f"{expected_token}_2022"
            )
        else:
            expected_name = (
                f"CMS_fake_t_DeepTau2018v2p5_{source}_{expected_token}_2022"
            )
        for name, directions in weights["variations"].items():
            if f"_{source}_" not in name:
                continue
            expected_ratio = 2.0 if name == expected_name else 1.0
            assert directions["up"][index] == pytest.approx(expected_ratio)
        other_source = "VSmu" if source == "VSe" else "VSe"
        assert all(
            directions["up"][index] == pytest.approx(1.0)
            for name, directions in weights["variations"].items()
            if f"_{other_source}_" in name
        )


def test_run3_vse_selection_and_correction_both_use_vvloose(monkeypatch):
    selection = run3TauSelection()
    assert ak.to_list(selection.iseTightTau(ak.Array([1, 2]))) == [False, True]

    recording_corrections = {
        name: _RecordingCorrection(name)
        for name in (
            "DeepTau2018v2p5VSjet",
            "DeepTau2018v2p5VSe",
            "DeepTau2018v2p5VSmu",
        )
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    monkeypatch.setattr(corrections, "SFevaluator", _RecordingEvaluator())

    corrections.AttachTauSF(
        {},
        _tau_record("2022", gen_part_flav=1),
        "2022",
        vsJetWP="Medium",
    )

    vse_calls = recording_corrections["DeepTau2018v2p5VSe"].calls
    assert corrections.TAU_VSE_WORKING_POINT == "VVLoose"
    assert vse_calls
    assert all(call[3] == "VVLoose" for call in vse_calls)


def test_run3_vse_nuisance_masks_are_decay_mode_specific(monkeypatch):
    recording_corrections = {
        name: _DirectionalCorrection(name)
        for name in (
            "DeepTau2018v2p5VSjet",
            "DeepTau2018v2p5VSe",
            "DeepTau2018v2p5VSmu",
        )
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    monkeypatch.setattr(corrections, "SFevaluator", _RecordingEvaluator())
    selected_dms = (0, 1, 10, 11)
    weights = corrections.AttachTauSF(
        {},
        _run3_tau_records(
            (0.5,) * len(selected_dms),
            gen_part_flav=1,
            decay_modes=selected_dms,
        ),
        "2022",
        vsJetWP="Medium",
    )

    vse_variations = {
        name: directions
        for name, directions in weights["variations"].items()
        if "_VSe_" in name
    }
    for index, decay_mode in enumerate(selected_dms):
        expected_name = (
            "CMS_fake_t_DeepTau2018v2p5_VSe_"
            f"DM{decay_mode}_abseta0to1p46_2022"
        )
        for name, directions in vse_variations.items():
            expected_ratio = 2.0 if name == expected_name else 1.0
            assert directions["up"][index] == pytest.approx(expected_ratio)


@pytest.mark.parametrize(
    ("year", "vsjet_correction_name", "expected_vsjet_wp", "fake_sf_key"),
    [
        ("2018", "DeepTau2017v2p1VSjet", "Loose", "TauFakeSFL"),
        ("2022", "DeepTau2018v2p5VSjet", "Medium", "TauFakeSF_Run3"),
    ],
)
def test_tau_vsjet_payload_uses_configured_wp_and_fake_sf_stays_separate(
    monkeypatch,
    year,
    vsjet_correction_name,
    expected_vsjet_wp,
    fake_sf_key,
):
    recording_corrections = {
        vsjet_correction_name: _RecordingCorrection(vsjet_correction_name),
        "DeepTau2018v2p5VSe": _RecordingCorrection("DeepTau2018v2p5VSe"),
        "DeepTau2018v2p5VSmu": _RecordingCorrection("DeepTau2018v2p5VSmu"),
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
    assert all(call[3] == expected_vsjet_wp for call in vsjet_calls)
    if year.startswith("201"):
        assert vsjet_calls[0][4:] == ("VVLoose", "nom", "dm")
    assert fake_sf_key in recording_evaluator.keys


@pytest.mark.parametrize(
    ("vsJetWP", "expected_key"),
    [
        ("Loose", "TauFakeSFL"),
        ("Medium", "TauFakeSFM"),
    ],
)
def test_run2_tau_fake_sf_resolver_selects_supported_payloads(vsJetWP, expected_key):
    assert corrections.get_run2_tau_fake_sf_name(vsJetWP) == expected_key


@pytest.mark.parametrize(
    "vsJetWP",
    [
        "loose",
        "medium",
        "Tight",
        "VLoose",
        "VVLoose",
        "VTight",
        "VVTight",
        "unexpected",
        None,
        "",
    ],
)
def test_run2_tau_fake_sf_resolver_rejects_unsupported_or_stale_payloads(vsJetWP):
    with pytest.raises(ValueError) as excinfo:
        corrections.get_run2_tau_fake_sf_name(vsJetWP)

    message = str(excinfo.value)
    assert "Supported" in message
    assert "Loose" in message
    assert "Medium" in message
    assert "canonical" in message
    assert "case-sensitive" in message
    assert "legacy/stale" in message
    assert "TauFakeSF/Tight" in message


@pytest.mark.parametrize(
    ("year", "vsjet_correction_name", "fake_sf_key"),
    [
        ("2018", "DeepTau2017v2p1VSjet", "TauFakeSFL"),
        ("2022", "DeepTau2018v2p5VSjet", "TauFakeSF_Run3"),
    ],
)
def test_jet_faking_tau_sf_uses_configured_payload(
    monkeypatch,
    year,
    vsjet_correction_name,
    fake_sf_key,
):
    recording_corrections = {
        vsjet_correction_name: _RecordingCorrection(vsjet_correction_name),
        "DeepTau2018v2p5VSe": _RecordingCorrection("DeepTau2018v2p5VSe"),
        "DeepTau2018v2p5VSmu": _RecordingCorrection("DeepTau2018v2p5VSmu"),
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


@pytest.mark.parametrize(
    ("vsJetWP", "expected_keys"),
    [
        ("Loose", {"TauFakeSFL", "TauFakeSFL_up", "TauFakeSFL_down"}),
        ("Medium", {"TauFakeSFM", "TauFakeSFM_up", "TauFakeSFM_down"}),
    ],
)
def test_run2_jet_faking_tau_sf_selects_wp_specific_payloads(
    monkeypatch,
    vsJetWP,
    expected_keys,
):
    recording_corrections = {
        "DeepTau2017v2p1VSjet": _RecordingCorrection("DeepTau2017v2p1VSjet"),
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
        _tau_record("2018", gen_part_flav=0),
        "2018",
        vsJetWP=vsJetWP,
    )

    assert expected_keys <= set(recording_evaluator.keys)


def test_run2_default_jet_faking_tau_sf_uses_loose_payload(monkeypatch):
    recording_corrections = {
        "DeepTau2017v2p1VSjet": _RecordingCorrection("DeepTau2017v2p1VSjet"),
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
        _tau_record("2018", gen_part_flav=0),
        "2018",
        vsJetWP=None,
    )

    assert {"TauFakeSFL", "TauFakeSFL_up", "TauFakeSFL_down"} <= set(
        recording_evaluator.keys
    )
    assert "TauFakeSF" not in recording_evaluator.keys


@pytest.mark.parametrize("vsJetWP", ["Tight", "VLoose"])
def test_run2_unsupported_fake_tau_wp_does_not_fall_back_to_legacy_payload(
    monkeypatch,
    vsJetWP,
):
    recording_corrections = {
        "DeepTau2017v2p1VSjet": _RecordingCorrection("DeepTau2017v2p1VSjet"),
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    recording_evaluator = _RecordingEvaluator()
    monkeypatch.setattr(corrections, "SFevaluator", recording_evaluator)

    with pytest.raises(ValueError) as excinfo:
        corrections.AttachTauSF(
            {},
            _tau_record("2018", gen_part_flav=0),
            "2018",
            vsJetWP=vsJetWP,
        )

    message = str(excinfo.value)
    assert vsJetWP in message
    assert "Loose" in message
    assert "Medium" in message
    assert "canonical" in message
    assert "case-sensitive" in message
    assert "legacy/stale" in message
    assert "TauFakeSF/Tight" in message
    assert "TauFakeSF" not in recording_evaluator.keys


@pytest.mark.parametrize("vsJetWP", ["loose", "medium"])
def test_run2_lowercase_fake_tau_wp_fails_before_tau_mask_lookup(
    monkeypatch,
    vsJetWP,
):
    recording_corrections = {
        "DeepTau2017v2p1VSjet": _RecordingCorrection("DeepTau2017v2p1VSjet"),
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    recording_evaluator = _RecordingEvaluator()
    monkeypatch.setattr(corrections, "SFevaluator", recording_evaluator)

    with pytest.raises(ValueError) as excinfo:
        corrections.AttachTauSF(
            {},
            _tau_record("2018", gen_part_flav=0),
            "2018",
            vsJetWP=vsJetWP,
        )

    message = str(excinfo.value)
    assert vsJetWP in message
    assert "canonical" in message
    assert "case-sensitive" in message
    assert "Loose" in message
    assert "Medium" in message
    assert recording_corrections["DeepTau2017v2p1VSjet"].calls == []
    assert recording_evaluator.keys == []


def test_run2_missing_fake_tau_evaluator_key_fails_clearly(monkeypatch):
    recording_corrections = {
        "DeepTau2017v2p1VSjet": _RecordingCorrection("DeepTau2017v2p1VSjet"),
    }
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        SimpleNamespace(from_file=lambda path: recording_corrections),
    )
    recording_evaluator = _MissingKeyEvaluator({"TauFakeSFL_down"})
    monkeypatch.setattr(corrections, "SFevaluator", recording_evaluator)

    with pytest.raises(RuntimeError) as excinfo:
        corrections.AttachTauSF(
            {},
            _tau_record("2018", gen_part_flav=0),
            "2018",
            vsJetWP="Loose",
        )

    message = str(excinfo.value)
    assert "TauFakeSFL" in message
    assert "TauFakeSFL_down" in message


def test_run3_year_specific_fake_tau_sf_still_uses_split_payload(monkeypatch):
    recording_corrections = {
        "DeepTau2018v2p5VSjet": _RecordingCorrection("DeepTau2018v2p5VSjet"),
        "DeepTau2018v2p5VSe": _RecordingCorrection("DeepTau2018v2p5VSe"),
        "DeepTau2018v2p5VSmu": _RecordingCorrection("DeepTau2018v2p5VSmu"),
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
        _tau_record("2022", gen_part_flav=0),
        "2022",
        vsJetWP="Medium",
        run3_fake_split=True,
    )

    assert "TauFakeSF_2022" in recording_evaluator.keys
    assert "TauFakeSF_2022_up" in recording_evaluator.keys
    assert "TauFakeSF_2022_down" in recording_evaluator.keys


def test_run2_fake_tau_sf_registrations_use_distinct_medium_and_loose_payloads():
    corrections_source = CORRECTIONS_PATH.read_text()

    for base_name, payload_name in (
        ("TauFakeSFM", "TauFakeSFM.json"),
        ("TauFakeSFL", "TauFakeSFL.json"),
    ):
        assert (
            f"{base_name} TauSF/pt_value %s\"%topcoffea_path('data/TauSF/{payload_name}')"
            in corrections_source
        )
        assert (
            f"{base_name}_up TauSF/pt_up %s\"%topcoffea_path('data/TauSF/{payload_name}')"
            in corrections_source
        )
        assert (
            f"{base_name}_down TauSF/pt_down %s\"%topcoffea_path('data/TauSF/{payload_name}')"
            in corrections_source
        )
        assert (
            f"{base_name} TauSF/pt_value %s\"%topcoffea_path('data/TauSF/TauFakeSF.json')"
            not in corrections_source
        )


def test_params_json_defines_run2_and_run3_tau_tags():
    params = json.loads(PARAMS_PATH.read_text())
    stale_key = "tau_" + "pog_vsjet_wp"

    assert params["run2_tau_t_tag"] == "Loose"
    assert params["run3_tau_t_tag"] == "Medium"
    assert params["run2_tau_fo_tag"] == "VLoose"
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
