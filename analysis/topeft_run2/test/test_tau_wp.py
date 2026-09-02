import pytest
import correctionlib

ak = pytest.importorskip("awkward")
from topcoffea.modules.paths import topcoffea_path

from topeft.modules.object_selection import (
    RUN2_VSMU_TIGHT_BIT,
    RUN3_VSMU_TIGHT_THRESHOLD,
    run2TauSelection,
    run3TauSelection,
)
from topeft.modules.corrections import AttachTauSF, SFevaluator


def test_run2_muon_wp_and_sf_alignment():
    selection = run2TauSelection()

    pt = ak.Array([25.0, 25.0])
    eta = ak.Array([0.1, -0.2])
    dxy = ak.Array([0.0, 0.0])
    dz = ak.Array([0.0, 0.0])
    vs_jet = ak.Array([1 << 3, 1 << 3])
    vs_e = ak.Array([1 << 1, 1 << 1])
    vs_mu = ak.Array([1 << RUN2_VSMU_TIGHT_BIT, 0])

    tight_mask = selection.ismTightTau(vs_mu)
    assert ak.to_list(tight_mask) == [1, 0]

    pres_mask = selection.isPresTau(pt, eta, dxy, dz, vs_jet, vs_e, vs_mu, vsJetWP="Loose")
    assert ak.to_list(pres_mask) == [True, False]

    taus = ak.Array(
        [
            [
                {
                    "pt": 30.0,
                    "eta": 0.35,
                    "mass": 1.2,
                    "decayMode": 0,
                    "genPartFlav": 2,
                    "isLoose": 1,
                    "iseTight": 0,
                    "ismTight": 1,
                    "idDeepTau2017v2p1VSjet": 1 << 3,
                    "idDeepTau2017v2p1VSmu": 1 << RUN2_VSMU_TIGHT_BIT,
                    "idDeepTau2017v2p1VSe": 1 << 1,
                },
                {
                    "pt": 30.0,
                    "eta": -0.45,
                    "mass": 1.2,
                    "decayMode": 0,
                    "genPartFlav": 2,
                    "isLoose": 1,
                    "iseTight": 0,
                    "ismTight": 0,
                    "idDeepTau2017v2p1VSjet": 1 << 3,
                    "idDeepTau2017v2p1VSmu": 0,
                    "idDeepTau2017v2p1VSe": 1 << 1,
                },
            ]
        ]
    )

    events = ak.Array([{}])
    AttachTauSF(events, taus, year="2018", vsJetWP="Loose")

    expected_mu_sf = SFevaluator["Tau_muonFakeSF_2018"](abs(float(taus.eta[0, 0])))
    assert taus["sf_tau_fake"][0, 0] == pytest.approx(expected_mu_sf)
    assert taus["sf_tau_fake"][0, 1] == pytest.approx(1.0)


def test_run3_muon_wp_threshold():
    selection = run3TauSelection()

    pt = ak.Array([25.0, 25.0])
    eta = ak.Array([0.3, -0.4])
    dxy = ak.Array([0.0, 0.0])
    dz = ak.Array([0.0, 0.0])
    vs_jet = ak.Array([4, 4])
    vs_e = ak.Array([2, 2])
    vs_mu = ak.Array([RUN3_VSMU_TIGHT_THRESHOLD, RUN3_VSMU_TIGHT_THRESHOLD - 1])

    tight_mask = selection.ismTightTau(vs_mu)
    assert ak.to_list(tight_mask) == [1, 0]

    pres_mask = selection.isPresTau(pt, eta, dxy, dz, vs_jet, vs_e, vs_mu, vsJetWP="Loose")
    assert ak.to_list(pres_mask) == [True, False]


def test_run3_tight_vsmu_sf_is_nonunity_and_varied_independently():
    taus = ak.Array(
        [[
            {
                "pt": 35.0,
                "eta": 0.35,
                "mass": 1.2,
                "decayMode": 0,
                "genPartFlav": 2,
                "isMedium": 1,
                "iseTight": 1,
                "ismTight": 1,
                "idDeepTau2018v2p5VSjet": 5,
                "idDeepTau2018v2p5VSe": 2,
                "idDeepTau2018v2p5VSmu": RUN3_VSMU_TIGHT_THRESHOLD,
            }
        ]]
    )
    events = {}
    weights = AttachTauSF(events, taus, year="2022", vsJetWP="Medium")
    payload = correctionlib.CorrectionSet.from_file(
        topcoffea_path("data/POG/TAU/2022_Summer22/tau.json.gz")
    )["DeepTau2018v2p5VSmu"]
    expected_nominal = payload.evaluate(0.35, 2, "Tight", "nom")
    expected_up = payload.evaluate(0.35, 2, "Tight", "up")
    expected_down = payload.evaluate(0.35, 2, "Tight", "down")
    nuisance = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022"

    assert expected_nominal != pytest.approx(1.0)
    assert taus.sf_tau_fake[0, 0] == pytest.approx(expected_nominal)
    assert weights["nominal_without_jet_fake"][0] == pytest.approx(expected_nominal)
    assert weights["variations"][nuisance]["up"][0] == pytest.approx(
        expected_up / expected_nominal
    )
    assert weights["variations"][nuisance]["down"][0] == pytest.approx(
        expected_down / expected_nominal
    )
    vse_nuisances = [
        name for name in weights["variations"] if "_VSe_" in name
    ]
    assert all(weights["variations"][name]["up"][0] == 1.0 for name in vse_nuisances)
