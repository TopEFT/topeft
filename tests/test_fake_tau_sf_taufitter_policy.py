import re
from pathlib import Path

import pytest

from analysis.topeft_run2 import analysis_processor as ap


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


def test_fake_tau_weight_insertions_are_guarded_by_policy_helper():
    source = PROCESSOR_SOURCE.read_text()
    helper_assignment = source.index("apply_fake_tau_sf = should_apply_fake_tau_sf(")

    fake_syst = source.index('wgt_correction_syst_lst.append("lepSF_taus_fakeUp")')
    assert source.rfind("if apply_fake_tau_sf:", helper_assignment, fake_syst) != -1

    fake_weight_calls = [
        match.start()
        for match in re.finditer(
            re.escape('weights_dict[ch_name].add("lepSF_taus_fake"'),
            source,
        )
    ]
    assert len(fake_weight_calls) == 3
    for fake_weight_call in fake_weight_calls:
        previous_real_tau = source.rfind(
            'weights_dict[ch_name].add("lepSF_taus_real"',
            0,
            fake_weight_call,
        )
        previous_fake_guard = source.rfind(
            "if apply_fake_tau_sf:",
            0,
            fake_weight_call,
        )
        assert previous_real_tau != -1
        assert previous_fake_guard > previous_real_tau

    assert source.count('weights_dict[ch_name].add("lepSF_taus_real"') == 3
    assert source.count('weights_dict[ch_name].add("lepSF_muon"') == 4
    assert source.count('weights_dict[ch_name].add("lepSF_elec"') == 4


def test_run3_vse_tau_sf_uses_tight_wp():
    source = CORRECTIONS_SOURCE.read_text()
    deep_tau_cuts = source.index("deep_tau_cuts = [")
    loop_start = source.index("for idx, deep_tau_cut", deep_tau_cuts)
    block = source[deep_tau_cuts:loop_start]

    assert '"DeepTau2018v2p5VSe"' in block
    assert '(flat_eta, flat_dm, flat_gen, "Tight")' in block
    assert '(flat_eta, flat_dm, flat_gen, "VVLoose")' not in block
    assert '(flat_pt, flat_dm, flat_gen, vsJetWP, "VVLoose")' in block
