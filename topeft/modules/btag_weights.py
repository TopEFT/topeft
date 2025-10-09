from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional

import awkward as ak


@dataclass
class BTagWeightResult:
    """Container for the central and optional systematic b-tag weights."""

    central: ak.Array
    variation_label: Optional[str] = None
    variation_up: Optional[ak.Array] = None
    variation_down: Optional[ak.Array] = None


def register_btag_sf_weights(
    *,
    jets_light,
    jets_bc,
    efficiencies: Mapping[str, Mapping[str, ak.Array]],
    central_values: Mapping[str, Mapping[str, ak.Array]],
    selection_masks: Mapping[str, ak.Array],
    years: Mapping[str, str],
    systematic_descriptor: Optional[Mapping[str, object]] = None,
    corrections_api=None,
) -> BTagWeightResult:
    """Reconstruct the Method-1a b-tag event weights.

    Parameters
    ----------
    jets_light, jets_bc:
        Jet collections split by hadron flavour.
    efficiencies:
        Mapping keyed by flavour (``"light"``/``"bc"``) containing
        the method-1a efficiencies for the loose (``"L"``) and medium
        (``"M"``) working points.
    central_values:
        Mapping keyed by flavour that provides the pre-computed
        ``weight`` and ``pMC`` arrays obtained from the central
        Method-1a evaluation.
    selection_masks:
        Mapping containing the boolean masks used in the Method-1a
        construction.  The masks ``"medium"``, ``"loose_not_medium"``
        and ``"not_loose"`` correspond to the selections applied to the
        full jet collection, while ``"light"`` and ``"bc"`` select the
        respective flavour subsets.
    years:
        Mapping containing the year to use for each flavour when
        evaluating the correction library.
    systematic_descriptor:
        Optional metadata describing which systematic variation is
        currently being processed.  When the descriptor requests a
        ``btagSF*`` variation, the corresponding up/down weights are
        reconstructed and returned alongside the central value.
    corrections_api:
        Object that provides ``btag_sf_eval`` and
        ``get_method1a_wgt_doublewp``.  Defaults to the implementation in
        :mod:`topcoffea.modules.corrections` but can be replaced for
        testing.
    """

    if corrections_api is None:
        import topcoffea.modules.corrections as tc_cor  # Local import to avoid heavy dependencies during testing

        corrections_api = tc_cor

    flavour_info: MutableMapping[str, MutableMapping[str, object]] = {
        "light": {
            "jets": jets_light,
            "mask": selection_masks["light"],
            "effM": efficiencies["light"]["M"],
            "effL": efficiencies["light"]["L"],
            "pMC": central_values["light"]["pMC"],
            "weight": central_values["light"]["weight"],
            "year": years["light"],
            "tag": "incl",
        },
        "bc": {
            "jets": jets_bc,
            "mask": selection_masks["bc"],
            "effM": efficiencies["bc"]["M"],
            "effL": efficiencies["bc"]["L"],
            "pMC": central_values["bc"]["pMC"],
            "weight": central_values["bc"]["weight"],
            "year": years["bc"],
            "tag": "comb",
        },
    }

    central_weight = flavour_info["light"]["weight"] * flavour_info["bc"]["weight"]
    result = BTagWeightResult(central=central_weight)

    descriptor = systematic_descriptor or {}
    has_systematics = bool(descriptor.get("has_systematics"))
    object_variation = descriptor.get("object_variation", "nominal")
    variation_name = descriptor.get("variation_name")
    requested_suffix = descriptor.get("requested_suffix")

    if requested_suffix is None and variation_name and variation_name.startswith("btagSF"):
        requested_suffix = variation_name[len("btagSF") :]

    if not (has_systematics and object_variation == "nominal" and requested_suffix):
        return result

    directionless_suffix = requested_suffix.rstrip("Up").rstrip("Down")
    corrtype = "correlated" if directionless_suffix.endswith("_corr") else "uncorrelated"

    flavour_key = "light" if requested_suffix.startswith("light_") else "bc"
    flavour_data = flavour_info[flavour_key]
    other_flavour_key = "bc" if flavour_key == "light" else "light"
    fixed_weight = flavour_info[other_flavour_key]["weight"]

    medium_mask = selection_masks["medium"][flavour_data["mask"]]
    loose_not_medium_mask = selection_masks["loose_not_medium"][flavour_data["mask"]]
    not_loose_mask = selection_masks["not_loose"][flavour_data["mask"]]

    btag_sfL_up = corrections_api.btag_sf_eval(
        flavour_data["jets"], "L", flavour_data["year"], f"deepJet_{flavour_data['tag']}", f"up_{corrtype}"
    )
    btag_sfL_down = corrections_api.btag_sf_eval(
        flavour_data["jets"], "L", flavour_data["year"], f"deepJet_{flavour_data['tag']}", f"down_{corrtype}"
    )
    btag_sfM_up = corrections_api.btag_sf_eval(
        flavour_data["jets"], "M", flavour_data["year"], f"deepJet_{flavour_data['tag']}", f"up_{corrtype}"
    )
    btag_sfM_down = corrections_api.btag_sf_eval(
        flavour_data["jets"], "M", flavour_data["year"], f"deepJet_{flavour_data['tag']}", f"down_{corrtype}"
    )

    pData_up, pMC_up = corrections_api.get_method1a_wgt_doublewp(
        flavour_data["effM"],
        flavour_data["effL"],
        btag_sfM_up,
        btag_sfL_up,
        medium_mask,
        loose_not_medium_mask,
        not_loose_mask,
    )

    pData_down, pMC_down = corrections_api.get_method1a_wgt_doublewp(
        flavour_data["effM"],
        flavour_data["effL"],
        btag_sfM_down,
        btag_sfL_down,
        medium_mask,
        loose_not_medium_mask,
        not_loose_mask,
    )

    pMC_central = flavour_data["pMC"]
    btag_w_up = fixed_weight * (pData_up / pMC_central) / central_weight
    btag_w_down = fixed_weight * (pData_down / pMC_central) / central_weight

    variation_label_base = variation_name
    if variation_label_base:
        for direction in ("Up", "Down"):
            if variation_label_base.endswith(direction):
                variation_label_base = variation_label_base[: -len(direction)]
                break
    else:
        variation_label_base = f"btagSF{directionless_suffix}"

    result.variation_label = variation_label_base
    result.variation_up = btag_w_up
    result.variation_down = btag_w_down
    return result
