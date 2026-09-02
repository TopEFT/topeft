from pathlib import Path

import pytest

from analysis.topeft_run2 import analysis_processor as ap
from topeft.modules.corrections import get_tau_weight_variation_names


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_SOURCE = REPO_ROOT / "analysis/topeft_run2/analysis_processor.py"
CORRECTIONS_SOURCE = REPO_ROOT / "topeft/modules/corrections.py"


@pytest.mark.parametrize(
    ("tau_run_mode", "enable_tau_blocks", "is_data", "expected"),
    [
        ("standard", True, False, True),
        ("taufitter", True, False, False),
        ("standard", False, False, False),
        ("standard", True, True, False),
    ],
)
def test_should_apply_fake_tau_sf_policy(
    tau_run_mode,
    enable_tau_blocks,
    is_data,
    expected,
):
    assert (
        ap.should_apply_fake_tau_sf(
            tau_run_mode,
            enable_tau_blocks=enable_tau_blocks,
            is_data=is_data,
        )
        is expected
    )


def test_should_apply_fake_tau_sf_rejects_unknown_active_mode():
    with pytest.raises(ValueError, match="Unknown tau_run_mode"):
        ap.should_apply_fake_tau_sf(
            "unexpected",
            enable_tau_blocks=True,
            is_data=False,
        )


def test_analysis_jet_fake_weight_is_the_only_tau_component_gated_by_mode():
    source = PROCESSOR_SOURCE.read_text()
    run2_standard = get_tau_weight_variation_names("2018", include_jet_fake=True)
    run2_taufitter = get_tau_weight_variation_names("2018", include_jet_fake=False)

    assert set(run2_standard) - set(run2_taufitter) == {"lepSF_taus_fake_run2"}
    assert all(name.startswith("CMS_") for name in run2_taufitter)
    assert "include_jet_fake=apply_fake_tau_sf" in source
    assert '"nominal" if apply_fake_tau_sf else "nominal_without_jet_fake"' in source
    assert 'weights_dict[ch_name].add("tauSF_nominal", tau_nominal)' in source
    assert 'weights_dict[ch_name].add("lepSF_taus_real"' not in source
    assert 'weights_dict[ch_name].add("lepSF_taus_fake"' not in source
    assert source.count('weights_dict[ch_name].add("lepSF_muon"') == 4
    assert source.count('weights_dict[ch_name].add("lepSF_elec"') == 4


def test_run3_vse_tau_sf_uses_selected_vvloose_wp():
    source = CORRECTIONS_SOURCE.read_text()
    assert 'ceval[f"{tagger}VSe"]' in source
    assert 'TAU_VSE_WORKING_POINT = "VVLoose"' in source
    assert source.count(
        "(flat_eta, flat_dm, flat_gen, TAU_VSE_WORKING_POINT)"
    ) == 3
    assert '(flat_eta, flat_dm, flat_gen, "Tight")' not in source
    assert (
        "real_args = (flat_pt, flat_dm, flat_gen, vsJetWP, "
        "TAU_VSE_WORKING_POINT)"
    ) in source
