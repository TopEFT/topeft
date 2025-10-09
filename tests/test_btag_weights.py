from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

if not hasattr(np, "AxisError"):
    np.AxisError = np.exceptions.AxisError

import awkward as ak

from topeft.modules.btag_weights import register_btag_sf_weights


class FakeCorrections:
    def __init__(self, sf_variations):
        self.sf_variations = sf_variations

    def get_method1a_wgt_doublewp(self, effA, effB, sfA, sfB, cutA, cutB, cutC):
        effA_data = effA * sfA
        effB_data = effB * sfB

        pMC = ak.prod(effA[cutA], axis=-1) * ak.prod(effB[cutB] - effA[cutB], axis=-1) * ak.prod(
            1 - effB[cutC], axis=-1
        )
        pMC = ak.where(pMC == 0, 1, pMC)
        pData = ak.prod(effA_data[cutA], axis=-1) * ak.prod(effB_data[cutB] - effA_data[cutB], axis=-1) * ak.prod(
            1 - effB_data[cutC], axis=-1
        )
        return pData, pMC

    def btag_sf_eval(self, jets, wp, year, method, syst):
        key = (jets.label, wp, year, method, syst)
        return self.sf_variations[key]


def _build_common_inputs():
    jets_light = SimpleNamespace(label="light")
    jets_bc = SimpleNamespace(label="bc")

    light_mask = ak.Array([[True, False], [True, False]])
    bc_mask = ~light_mask

    is_loose = ak.Array([[True, True], [True, True]])
    is_medium = ak.Array([[True, False], [False, True]])
    is_not_loose = ~is_loose
    is_loose_not_medium = is_loose & ~is_medium

    efficiencies = {
        "light": {
            "M": ak.Array([[0.6], [0.65]]),
            "L": ak.Array([[0.8], [0.82]]),
        },
        "bc": {
            "M": ak.Array([[0.7], [0.72]]),
            "L": ak.Array([[0.85], [0.86]]),
        },
    }

    sf_central = {
        "light": {
            "M": ak.Array([[0.95], [1.05]]),
            "L": ak.Array([[1.0], [1.02]]),
        },
        "bc": {
            "M": ak.Array([[0.98], [1.01]]),
            "L": ak.Array([[1.04], [1.03]]),
        },
    }

    sf_variations = {
        ("light", "L", "2016APV", "deepJet_incl", "up_uncorrelated"): ak.Array([[1.05], [1.07]]),
        ("light", "L", "2016APV", "deepJet_incl", "down_uncorrelated"): ak.Array([[0.95], [0.97]]),
        ("light", "M", "2016APV", "deepJet_incl", "up_uncorrelated"): ak.Array([[1.08], [1.09]]),
        ("light", "M", "2016APV", "deepJet_incl", "down_uncorrelated"): ak.Array([[0.92], [0.93]]),
        ("bc", "L", "2018", "deepJet_comb", "up_correlated"): ak.Array([[1.06], [1.08]]),
        ("bc", "L", "2018", "deepJet_comb", "down_correlated"): ak.Array([[0.96], [0.94]]),
        ("bc", "M", "2018", "deepJet_comb", "up_correlated"): ak.Array([[1.07], [1.09]]),
        ("bc", "M", "2018", "deepJet_comb", "down_correlated"): ak.Array([[0.93], [0.91]]),
    }

    corrections = FakeCorrections(sf_variations)

    pData_light, pMC_light = corrections.get_method1a_wgt_doublewp(
        efficiencies["light"]["M"],
        efficiencies["light"]["L"],
        sf_central["light"]["M"],
        sf_central["light"]["L"],
        is_medium[light_mask],
        is_loose_not_medium[light_mask],
        is_not_loose[light_mask],
    )
    pData_bc, pMC_bc = corrections.get_method1a_wgt_doublewp(
        efficiencies["bc"]["M"],
        efficiencies["bc"]["L"],
        sf_central["bc"]["M"],
        sf_central["bc"]["L"],
        is_medium[bc_mask],
        is_loose_not_medium[bc_mask],
        is_not_loose[bc_mask],
    )

    central_values = {
        "light": {"weight": pData_light / pMC_light, "pMC": pMC_light},
        "bc": {"weight": pData_bc / pMC_bc, "pMC": pMC_bc},
    }

    selection_masks = {
        "medium": is_medium,
        "loose_not_medium": is_loose_not_medium,
        "not_loose": is_not_loose,
        "light": light_mask,
        "bc": bc_mask,
    }

    years = {"light": "2016APV", "bc": "2018"}

    return (
        corrections,
        jets_light,
        jets_bc,
        efficiencies,
        sf_central,
        central_values,
        selection_masks,
        years,
    )


def test_register_btag_sf_weights_nominal_returns_central_only():
    (
        corrections,
        jets_light,
        jets_bc,
        efficiencies,
        sf_central,
        central_values,
        selection_masks,
        years,
    ) = _build_common_inputs()

    result = register_btag_sf_weights(
        jets_light=jets_light,
        jets_bc=jets_bc,
        efficiencies={
            "light": {"M": efficiencies["light"]["M"], "L": efficiencies["light"]["L"]},
            "bc": {"M": efficiencies["bc"]["M"], "L": efficiencies["bc"]["L"]},
        },
        central_values=central_values,
        selection_masks=selection_masks,
        years=years,
        systematic_descriptor={"has_systematics": False, "object_variation": "nominal"},
        corrections_api=corrections,
    )

    expected_central = central_values["light"]["weight"] * central_values["bc"]["weight"]
    assert np.allclose(ak.to_numpy(result.central), ak.to_numpy(expected_central))
    assert result.variation_label is None


def test_register_btag_sf_weights_light_uncorrelated_variation():
    (
        corrections,
        jets_light,
        jets_bc,
        efficiencies,
        sf_central,
        central_values,
        selection_masks,
        years,
    ) = _build_common_inputs()

    result = register_btag_sf_weights(
        jets_light=jets_light,
        jets_bc=jets_bc,
        efficiencies={
            "light": {"M": efficiencies["light"]["M"], "L": efficiencies["light"]["L"]},
            "bc": {"M": efficiencies["bc"]["M"], "L": efficiencies["bc"]["L"]},
        },
        central_values=central_values,
        selection_masks=selection_masks,
        years=years,
        systematic_descriptor={
            "has_systematics": True,
            "object_variation": "nominal",
            "variation_name": "btagSFlight_2016Up",
        },
        corrections_api=corrections,
    )

    assert result.variation_label == "btagSFlight_2016"

    fixed_weight = central_values["bc"]["weight"]
    pMC_light = central_values["light"]["pMC"]
    central_weight = central_values["light"]["weight"] * fixed_weight

    btag_sfL_up = corrections.btag_sf_eval(jets_light, "L", years["light"], "deepJet_incl", "up_uncorrelated")
    btag_sfM_up = corrections.btag_sf_eval(jets_light, "M", years["light"], "deepJet_incl", "up_uncorrelated")
    pData_up, _ = corrections.get_method1a_wgt_doublewp(
        efficiencies["light"]["M"],
        efficiencies["light"]["L"],
        btag_sfM_up,
        btag_sfL_up,
        selection_masks["medium"][selection_masks["light"]],
        selection_masks["loose_not_medium"][selection_masks["light"]],
        selection_masks["not_loose"][selection_masks["light"]],
    )

    btag_sfL_down = corrections.btag_sf_eval(jets_light, "L", years["light"], "deepJet_incl", "down_uncorrelated")
    btag_sfM_down = corrections.btag_sf_eval(jets_light, "M", years["light"], "deepJet_incl", "down_uncorrelated")
    pData_down, _ = corrections.get_method1a_wgt_doublewp(
        efficiencies["light"]["M"],
        efficiencies["light"]["L"],
        btag_sfM_down,
        btag_sfL_down,
        selection_masks["medium"][selection_masks["light"]],
        selection_masks["loose_not_medium"][selection_masks["light"]],
        selection_masks["not_loose"][selection_masks["light"]],
    )

    expected_up = fixed_weight * (pData_up / pMC_light) / central_weight
    expected_down = fixed_weight * (pData_down / pMC_light) / central_weight

    assert np.allclose(ak.to_numpy(result.variation_up), ak.to_numpy(expected_up))
    assert np.allclose(ak.to_numpy(result.variation_down), ak.to_numpy(expected_down))


def test_register_btag_sf_weights_bc_correlated_variation():
    (
        corrections,
        jets_light,
        jets_bc,
        efficiencies,
        sf_central,
        central_values,
        selection_masks,
        years,
    ) = _build_common_inputs()

    result = register_btag_sf_weights(
        jets_light=jets_light,
        jets_bc=jets_bc,
        efficiencies={
            "light": {"M": efficiencies["light"]["M"], "L": efficiencies["light"]["L"]},
            "bc": {"M": efficiencies["bc"]["M"], "L": efficiencies["bc"]["L"]},
        },
        central_values=central_values,
        selection_masks=selection_masks,
        years=years,
        systematic_descriptor={
            "has_systematics": True,
            "object_variation": "nominal",
            "variation_name": "btagSFbc_corrUp",
        },
        corrections_api=corrections,
    )

    assert result.variation_label == "btagSFbc_corr"

    fixed_weight = central_values["light"]["weight"]
    pMC_bc = central_values["bc"]["pMC"]
    central_weight = fixed_weight * central_values["bc"]["weight"]

    btag_sfL_up = corrections.btag_sf_eval(jets_bc, "L", years["bc"], "deepJet_comb", "up_correlated")
    btag_sfM_up = corrections.btag_sf_eval(jets_bc, "M", years["bc"], "deepJet_comb", "up_correlated")
    pData_up, _ = corrections.get_method1a_wgt_doublewp(
        efficiencies["bc"]["M"],
        efficiencies["bc"]["L"],
        btag_sfM_up,
        btag_sfL_up,
        selection_masks["medium"][selection_masks["bc"]],
        selection_masks["loose_not_medium"][selection_masks["bc"]],
        selection_masks["not_loose"][selection_masks["bc"]],
    )

    btag_sfL_down = corrections.btag_sf_eval(jets_bc, "L", years["bc"], "deepJet_comb", "down_correlated")
    btag_sfM_down = corrections.btag_sf_eval(jets_bc, "M", years["bc"], "deepJet_comb", "down_correlated")
    pData_down, _ = corrections.get_method1a_wgt_doublewp(
        efficiencies["bc"]["M"],
        efficiencies["bc"]["L"],
        btag_sfM_down,
        btag_sfL_down,
        selection_masks["medium"][selection_masks["bc"]],
        selection_masks["loose_not_medium"][selection_masks["bc"]],
        selection_masks["not_loose"][selection_masks["bc"]],
    )

    expected_up = fixed_weight * (pData_up / pMC_bc) / central_weight
    expected_down = fixed_weight * (pData_down / pMC_bc) / central_weight

    assert np.allclose(ak.to_numpy(result.variation_up), ak.to_numpy(expected_up))
    assert np.allclose(ak.to_numpy(result.variation_down), ak.to_numpy(expected_down))
