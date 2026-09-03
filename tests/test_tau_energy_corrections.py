from itertools import product
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

import topeft.modules.corrections as corrections
from topeft.modules.corrections import (
    ApplyTauEnergySystematics,
    AttachTauEnergyCorrections,
    get_tau_energy_fields,
    get_tau_energy_variation_specs,
    get_supported_tau_energy_systematics,
)


PROCESSOR_DIR = (
    Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"
)


def _taus(records):
    return ak.Array([records])


def _assert_close(actual, expected):
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(actual, axis=None)),
        ak.to_numpy(ak.flatten(expected, axis=None)),
        rtol=1e-6,
        atol=1e-7,
    )


def _energy_variations(year):
    return tuple(get_supported_tau_energy_systematics(year, isData=False))


def _base_records():
    return [
        {
            "pt": 30.0,
            "mass": 1.2,
            "eta": 0.7,
            "decayMode": 0,
            "genPartFlav": 5,
        },
        {
            "pt": 30.0,
            "mass": 1.3,
            "eta": -1.8,
            "decayMode": 10,
            "genPartFlav": 1,
        },
        {
            "pt": 30.0,
            "mass": 1.4,
            "eta": 0.7,
            "decayMode": 0,
            "genPartFlav": 0,
        },
    ]


def test_attaches_all_complete_tau_energy_fields_and_preserves_raw_fields():
    tau = _taus(_base_records())
    attached = AttachTauEnergyCorrections(
        "2022", tau, False, vsJetWP="Medium"
    )

    expected_fields = {"pt_raw", "mass_raw"}
    for variation in ("nominal",) + _energy_variations("2022"):
        pt_field, mass_field = get_tau_energy_fields(variation)
        expected_fields.update((pt_field, mass_field))
    assert expected_fields <= set(ak.fields(attached))
    _assert_close(attached.pt_raw, tau.pt)
    _assert_close(attached.mass_raw, tau.mass)


def test_complete_views_apply_only_the_targeted_category_variation():
    tau = _taus(_base_records())
    attached = AttachTauEnergyCorrections(
        "2022", tau, False, vsJetWP="Medium"
    )

    tes = "CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022Up"
    fes = "CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2022Up"
    tes_pt, _ = get_tau_energy_fields(tes)
    fes_pt, _ = get_tau_energy_fields(fes)

    assert attached[tes_pt][0, 0] != pytest.approx(attached.pt_nom[0, 0])
    _assert_close(attached[tes_pt][:, 1], attached.pt_nom[:, 1])
    _assert_close(attached[tes_pt][:, 2], attached.pt_raw[:, 2])

    _assert_close(attached[fes_pt][:, 0], attached.pt_nom[:, 0])
    assert attached[fes_pt][0, 1] != pytest.approx(attached.pt_nom[0, 1])
    _assert_close(attached[fes_pt][:, 2], attached.pt_raw[:, 2])

    for variation in ("nominal",) + _energy_variations("2022"):
        pt_field, mass_field = get_tau_energy_fields(variation)
        _assert_close(
            attached[mass_field] / attached.mass_raw,
            attached[pt_field] / attached.pt_raw,
        )


def test_legacy_tau_energy_helpers_are_not_public():
    assert not hasattr(corrections, "ApplyTES")
    assert not hasattr(corrections, "ApplyTESSystematic")
    assert not hasattr(corrections, "ApplyFESSystematic")


def test_selector_uses_attached_views_and_resets_from_nominal():
    attached = AttachTauEnergyCorrections(
        "2022", _taus(_base_records()), False, vsJetWP="Medium"
    )
    nominal = ApplyTauEnergySystematics(attached, "nominal")
    variation = "CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022Up"
    varied_after_nominal = ApplyTauEnergySystematics(nominal, variation)
    varied_pt, varied_mass = get_tau_energy_fields(variation)

    _assert_close(nominal.pt, attached.pt_nom)
    _assert_close(nominal.mass, attached.mass_nom)
    _assert_close(varied_after_nominal.pt, attached[varied_pt])
    _assert_close(varied_after_nominal.mass, attached[varied_mass])
    _assert_close(varied_after_nominal.pt_raw, attached.pt_raw)
    _assert_close(varied_after_nominal.mass_raw, attached.mass_raw)

    non_tau = ApplyTauEnergySystematics(varied_after_nominal, "JESUp")
    _assert_close(non_tau.pt, attached.pt_nom)
    _assert_close(non_tau.mass, attached.mass_nom)


def test_selector_fails_loudly_when_requested_view_is_missing():
    tau = _taus(
        [
            {
                "pt": 30.0,
                "mass": 1.2,
                "pt_nom": 29.0,
                "mass_nom": 1.16,
            }
        ]
    )
    variation = "CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022Up"
    with pytest.raises(ValueError, match="pt_CMS_scale_t_.*mass_CMS_scale_t_"):
        ApplyTauEnergySystematics(tau, variation)


def test_data_is_nominal_only_and_all_attached_views_are_raw():
    tau = _taus(
        [
            {
                "pt": 30.0,
                "mass": 1.2,
                "eta": 0.7,
                "decayMode": 0,
            }
        ]
    )
    attached = AttachTauEnergyCorrections("2022", tau, True)

    assert get_supported_tau_energy_systematics("2022", isData=True) == []
    assert set(ak.fields(attached)) == set(ak.fields(tau)) | {
        "pt_raw",
        "mass_raw",
        "pt_nom",
        "mass_nom",
    }
    active = ApplyTauEnergySystematics(attached, "nominal")
    _assert_close(active.pt, tau.pt)
    _assert_close(active.mass, tau.mass)


def test_mc_tau_energy_systematics_are_advertised_for_run2_and_run3():
    run2 = get_supported_tau_energy_systematics("2018", isData=False)
    run3 = get_supported_tau_energy_systematics("2022", isData=False)
    assert len(run2) == 2 * len(get_tau_energy_variation_specs("2018")) == 20
    assert len(run3) == 2 * len(get_tau_energy_variation_specs("2022")) == 24
    assert not {"TESUp", "TESDown", "FESUp", "FESDown"} & set(run2 + run3)


@pytest.mark.parametrize(
    ("year", "tagger", "run", "wp"),
    [
        ("2018", "DeepTau2017v2p1", "Run2", "Loose"),
        ("2022", "DeepTau2018v2p5", "Run3", "Medium"),
    ],
)
def test_muon_fake_energy_scale_is_dm_specific_and_shared_within_run(
    year, tagger, run, wp
):
    records = [
        {
            "pt": 30.0,
            "mass": 1.2,
            "eta": 0.7,
            "decayMode": dm,
            "genPartFlav": gen,
        }
        for dm in (0, 1, 10, 11)
        for gen in (2, 4)
    ]
    attached = AttachTauEnergyCorrections(year, _taus(records), False, vsJetWP=wp)
    _assert_close(attached.pt_nom, attached.pt_raw)
    for target_dm in (0, 1, 10, 11):
        base = f"CMS_scale_t_{tagger}_DM{target_dm}_genMuon_{run}"
        up_pt, _ = get_tau_energy_fields(f"{base}Up")
        down_pt, _ = get_tau_energy_fields(f"{base}Down")
        for index, record in enumerate(records):
            expected_up = 1.01 if record["decayMode"] == target_dm else 1.0
            expected_down = 0.99 if record["decayMode"] == target_dm else 1.0
            assert attached[up_pt][0, index] == pytest.approx(30.0 * expected_up)
            assert attached[down_pt][0, index] == pytest.approx(
                30.0 * expected_down
            )


def test_run2_fes_uses_absolute_eta_for_nominal_and_variations():
    tau = _taus(
        [
            {
                "pt": 30.0,
                "mass": 1.2,
                "eta": 1.8,
                "decayMode": 0,
                "genPartFlav": 1,
            },
            {
                "pt": 30.0,
                "mass": 1.2,
                "eta": -1.8,
                "decayMode": 0,
                "genPartFlav": 1,
            },
        ]
    )
    attached = AttachTauEnergyCorrections(
        "2018", tau, False, vsJetWP="Loose"
    )

    base = "CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2018"
    for variation in ("nominal", f"{base}Up", f"{base}Down"):
        pt_field, mass_field = get_tau_energy_fields(variation)
        _assert_close(attached[pt_field][:, 0], attached[pt_field][:, 1])
        _assert_close(
            attached[mass_field][:, 0], attached[mass_field][:, 1]
        )


@pytest.mark.parametrize("year", ["2022", "2022EE", "2023", "2023BPix"])
def test_run3_fes_variations_cover_all_reachable_selected_decay_modes(year):
    tau = _taus(
        [
            {
                "pt": 30.0,
                "mass": 1.2,
                "eta": 0.7,
                "decayMode": dm,
                "genPartFlav": 1,
            }
            for dm in (0, 1, 2, 10, 11)
        ]
    )
    attached = AttachTauEnergyCorrections(
        year, tau, False, vsJetWP="Medium"
    )

    for index, dm in enumerate((0, 1, 2, 10, 11)):
        if dm == 2:
            assert attached.pt_nom[0, index] == pytest.approx(
                attached.pt_raw[0, index]
            )
            continue
        base = f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron_{year}"
        up_pt, _ = get_tau_energy_fields(f"{base}Up")
        down_pt, _ = get_tau_energy_fields(f"{base}Down")
        assert attached[up_pt][0, index] != pytest.approx(attached.pt_nom[0, index])
        assert attached[down_pt][0, index] != pytest.approx(attached.pt_nom[0, index])


def test_raw_based_threshold_case_keeps_tes_up_independent_of_nominal():
    tau = _taus(
        [
            {
                "pt": 20.05,
                "mass": 1.2,
                "eta": 0.7,
                "decayMode": 0,
                "genPartFlav": 5,
            }
        ]
    )
    attached = AttachTauEnergyCorrections(
        "2022", tau, False, vsJetWP="Medium"
    )

    variation = "CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022Up"
    varied_pt, _ = get_tau_energy_fields(variation)
    assert attached.pt_nom[0, 0] < 20
    assert attached[varied_pt][0, 0] > 20
    assert attached.pt_nom[0, 0] == pytest.approx(19.869549933075906)
    assert attached[varied_pt][0, 0] == pytest.approx(20.170300841331482)


def test_correction_applicability_uses_strict_raw_pt_range():
    tau = _taus(
        [
            {
                "pt": pt,
                "mass": 1.2,
                "eta": 0.7,
                "decayMode": 0,
                "genPartFlav": 5,
            }
            for pt in (19.95, 20.0, 20.01, 204.99, 205.0, 205.01)
        ]
    )
    attached = AttachTauEnergyCorrections(
        "2022", tau, False, vsJetWP="Medium"
    )

    for index in (0, 1, 4, 5):
        assert attached.pt_nom[0, index] == pytest.approx(
            attached.pt_raw[0, index]
        )
    assert attached.pt_nom[0, 2] != pytest.approx(attached.pt_raw[0, 2])


@pytest.mark.parametrize("year,wp", [("2018", "Loose"), ("2022", "Medium")])
def test_pt_and_mass_receive_identical_scale_for_every_view(year, wp):
    attached = AttachTauEnergyCorrections(
        year, _taus(_base_records()), False, vsJetWP=wp
    )
    for variation in ("nominal",) + _energy_variations(year):
        pt_field, mass_field = get_tau_energy_fields(variation)
        pt_scale = attached[pt_field] / attached.pt_raw
        mass_scale = attached[mass_field] / attached.mass_raw
        _assert_close(pt_scale, mass_scale)


@pytest.mark.parametrize("year,wp", [("2018", "Loose"), ("2022", "Medium")])
def test_synthetic_distribution_grid_is_finite_and_category_complete(year, wp):
    records = [
        {
            "pt": pt,
            "mass": 1.2,
            "eta": eta,
            "decayMode": dm,
            "genPartFlav": gen,
        }
        for pt, eta, dm, gen in product(
            (19.95, 20.0, 20.01, 20.05, 30.0, 204.99, 205.0, 205.01),
            (0.7, -0.7, 1.8, -1.8),
            (0, 1, 2, 10, 11),
            (0, 1, 2, 3, 4, 5),
        )
    ]
    attached = AttachTauEnergyCorrections(
        year, _taus(records), False, vsJetWP=wp
    )

    assert len(records) == 960
    for variation in ("nominal",) + _energy_variations(year):
        pt_field, mass_field = get_tau_energy_fields(variation)
        pt = ak.to_numpy(ak.flatten(attached[pt_field]))
        mass = ak.to_numpy(ak.flatten(attached[mass_field]))
        assert np.all(np.isfinite(pt))
        assert np.all(np.isfinite(mass))
        assert np.count_nonzero(pt > 20) > 0
        assert np.count_nonzero(pt < 205) > 0

    gen = ak.to_numpy(ak.flatten(attached.genPartFlav))
    dm = ak.to_numpy(ak.flatten(attached.decayMode))
    raw = ak.to_numpy(ak.flatten(attached.pt_raw))
    nominal = ak.to_numpy(ak.flatten(attached.pt_nom))
    untargeted = gen == 0
    np.testing.assert_allclose(nominal[untargeted], raw[untargeted])

    if year == "2018":
        fake_unsupported_dm = (
            (gen >= 1)
            & (gen <= 4)
            & ((dm == 2) | (dm == 10) | (dm == 11))
        )
        np.testing.assert_allclose(
            nominal[fake_unsupported_dm], raw[fake_unsupported_dm]
        )


def test_processors_attach_once_and_select_before_tau_acceptance():
    for processor_name in (
        "analysis_processor.py",
        "analysis_processor_diboson.py",
    ):
        source = (PROCESSOR_DIR / processor_name).read_text()
        syst_loop = source.index("for syst_var in syst_var_list:")
        attach = source.index(
            "taus = AttachTauEnergyCorrections("
        )
        selector = source.index(
            "tau = ApplyTauEnergySystematics(taus, syst_var)",
            syst_loop,
        )
        tau_pres = source.index('tau["isPres', selector)
        tau_clean = source.index('tau["isClean"]', tau_pres)
        tau_good = source.index('tau["isGood"]', tau_clean)
        tau_select = source.index("tau = tau[tau.isGood]", tau_good)
        dm_flag = source.index('tau["DMflag"]', tau_select)
        dm_select = source.index("tau = tau[tau.DMflag]", dm_flag)
        tau_sf = source.index("AttachTauSF(", dm_select)
        jet_cleaning = source.index("tmp = ak.cartesian", tau_select)

        assert attach < syst_loop < selector < tau_pres
        assert selector < tau_pres < tau_clean < tau_good < tau_select
        assert tau_select < dm_flag < dm_select < tau_sf < jet_cleaning
        assert "ApplyTES" not in source
        assert "ApplyTESSystematic" not in source
        assert "ApplyFESSystematic" not in source
        assert "tau_energy_views" not in source
        assert "ApplyMETSystematics" not in source[selector:tau_pres]
