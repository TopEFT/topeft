from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from topeft.modules import corrections

_RUN3_VARIATIONS = [
    "MuonScaleUp",
    "MuonScaleDown",
    "MuonResolutionUp",
    "MuonResolutionDown",
]


class _ThresholdSelection:
    def coneptMuon(self, muons):
        return muons.pt

    def isPresMuon(self, muons):
        return muons.pt > 10.0


def _muons(pt=9.5):
    return ak.Array(
        [
            [
                {
                    "pt": pt,
                    "eta": 0.2,
                    "phi": 0.1,
                    "charge": 1,
                    "nTrackerLayers": 12,
                }
            ]
        ]
    )


def _run3_muons():
    return ak.Array(
        [
            [
                {
                    "pt": 30.0,
                    "eta": 0.2,
                    "phi": 0.1,
                    "charge": 1,
                    "nTrackerLayers": 12,
                },
                {
                    "pt": 45.0,
                    "eta": -1.1,
                    "phi": -0.4,
                    "charge": -1,
                    "nTrackerLayers": 14,
                },
            ],
            [],
            [
                {
                    "pt": 60.0,
                    "eta": 1.3,
                    "phi": 2.0,
                    "charge": 1,
                    "nTrackerLayers": 10,
                }
            ],
        ]
    )


def _assert_finite_run3_shape(corrected):
    assert ak.to_list(ak.num(corrected)) == [2, 0, 1]
    assert np.all(np.isfinite(ak.to_numpy(ak.flatten(corrected))))


def _processor_source(name):
    repo = Path(__file__).resolve().parents[1]
    return (repo / "analysis" / "topeft_run2" / name).read_text()


def test_corrected_pt_is_used_for_conept_and_selection_threshold():
    muons = _muons()
    prepared = ak.with_field(muons, muons.pt, "pt_raw")
    prepared = ak.with_field(prepared, muons.pt + 1.0, "pt")
    prepared = ak.with_field(
        prepared, _ThresholdSelection().coneptMuon(prepared), "conept"
    )

    assert ak.to_list(prepared.pt_raw) == [[9.5]]
    assert ak.to_list(prepared.pt) == [[10.5]]
    assert ak.to_list(prepared.conept) == [[10.5]]
    assert ak.to_list(_ThresholdSelection().isPresMuon(prepared)) == [[True]]


@pytest.mark.parametrize("year", ["2016APV", "2016", "2017", "2018"])
def test_run2_dispatch_preserves_rochester_path(monkeypatch, year):
    calls = []

    def _fake_rochester(year, muons, is_data):
        calls.append((year, is_data))
        return muons.pt + 0.5

    monkeypatch.setattr(
        corrections, "ApplyRochesterCorrections", _fake_rochester
    )

    corrected = corrections.apply_muon_momentum_corrections(
        year, _muons(), False
    )

    assert calls == [(year, False)]
    assert ak.to_list(corrected) == [[10.0]]


@pytest.mark.parametrize("year", ["2022", "2022EE", "2023", "2023BPix"])
def test_run3_nominal_data_uses_default_backend_and_payload(year):
    corrected = corrections.apply_muon_momentum_corrections(
        year,
        _run3_muons(),
        True,
    )

    _assert_finite_run3_shape(corrected)


@pytest.mark.parametrize("year", ["2022", "2022EE", "2023", "2023BPix"])
def test_run3_nominal_mc_uses_default_backend_and_payload(year):
    muons = _run3_muons()
    data_corrected = corrections.apply_muon_momentum_corrections(
        year,
        muons,
        True,
    )
    corrected = corrections.apply_muon_momentum_corrections(
        year,
        muons,
        False,
        event_numbers=ak.Array([1001, 1002, 1003]),
        luminosity_blocks=ak.Array([11, 12, 13]),
    )

    _assert_finite_run3_shape(corrected)
    assert not np.allclose(
        ak.to_numpy(ak.flatten(data_corrected)),
        ak.to_numpy(ak.flatten(corrected)),
    )


@pytest.mark.parametrize("variation", _RUN3_VARIATIONS)
def test_run3_mc_variations_use_default_backend_and_payload(variation):
    corrected = corrections.apply_muon_momentum_corrections(
        "2022",
        _run3_muons(),
        False,
        event_numbers=ak.Array([1001, 1002, 1003]),
        luminosity_blocks=ak.Array([11, 12, 13]),
        variation=variation,
    )

    _assert_finite_run3_shape(corrected)


@pytest.mark.parametrize("variation", _RUN3_VARIATIONS)
def test_run3_data_rejects_muon_momentum_variations(variation):
    with pytest.raises(ValueError, match="not applicable to data"):
        corrections.apply_muon_momentum_corrections(
            "2022",
            _run3_muons(),
            True,
            variation=variation,
        )


@pytest.mark.parametrize("variation", _RUN3_VARIATIONS)
def test_run2_variation_requests_remain_unsupported(variation):
    with pytest.raises(ValueError, match="Run 2 Rochester.*does not support"):
        corrections.apply_muon_momentum_corrections(
            "2018",
            _muons(),
            False,
            variation=variation,
        )


def test_run3_mc_requires_event_and_lumi_inputs():
    with pytest.raises(ValueError, match="event_numbers and luminosity_blocks"):
        corrections.apply_muon_momentum_corrections(
            "2022",
            _run3_muons(),
            False,
        )


def test_unsupported_year_fails_loudly():
    with pytest.raises(ValueError, match="Unsupported Run 3.*2024"):
        corrections.apply_muon_momentum_corrections(
            "2024",
            _run3_muons(),
            True,
        )


def test_muon_momentum_systematic_list_is_run3_mc_only():
    assert corrections.RUN3_MUON_MOMENTUM_SYSTEMATICS == tuple(_RUN3_VARIATIONS)
    assert (
        corrections.get_supported_muon_momentum_systematics("2022", isData=False)
        == _RUN3_VARIATIONS
    )
    assert corrections.get_supported_muon_momentum_systematics("2022", isData=True) == []
    assert corrections.get_supported_muon_momentum_systematics("2018", isData=False) == []


def test_run3_mc_attachment_adds_nominal_and_all_variation_fields(monkeypatch):
    calls = []
    offsets = {
        "nominal": 1.0,
        "MuonScaleUp": 2.0,
        "MuonScaleDown": 3.0,
        "MuonResolutionUp": 4.0,
        "MuonResolutionDown": 5.0,
    }

    def _fake_apply(
        year,
        muons,
        is_data,
        *,
        event_numbers=None,
        luminosity_blocks=None,
        variation="nominal",
    ):
        calls.append(
            (
                year,
                is_data,
                variation,
                ak.to_list(event_numbers),
                ak.to_list(luminosity_blocks),
            )
        )
        return muons.pt + offsets[variation]

    monkeypatch.setattr(
        corrections, "apply_muon_momentum_corrections", _fake_apply
    )
    attached = corrections.AttachMuonMomentumCorrections(
        "2022",
        _muons(),
        False,
        event_numbers=ak.Array([101]),
        luminosity_blocks=ak.Array([7]),
    )

    assert [call[2] for call in calls] == [
        "nominal",
        *_RUN3_VARIATIONS,
    ]
    assert ak.to_list(attached.pt_nom) == [[10.5]]
    for variation in _RUN3_VARIATIONS:
        field = corrections.MUON_MOMENTUM_PT_FIELDS[variation]
        assert field in ak.fields(attached)
        assert ak.to_list(attached[field]) == [[9.5 + offsets[variation]]]


def test_data_attachment_only_adds_nominal_field(monkeypatch):
    calls = []

    def _fake_apply(year, muons, is_data, **kwargs):
        calls.append((year, is_data, kwargs["variation"]))
        return muons.pt + 1.0

    monkeypatch.setattr(
        corrections, "apply_muon_momentum_corrections", _fake_apply
    )
    attached = corrections.AttachMuonMomentumCorrections(
        "2022", _muons(), True
    )

    assert calls == [("2022", True, "nominal")]
    assert "pt_nom" in ak.fields(attached)
    for variation in _RUN3_VARIATIONS:
        assert corrections.MUON_MOMENTUM_PT_FIELDS[variation] not in ak.fields(
            attached
        )


def test_run2_attachment_only_adds_rochester_nominal_field(monkeypatch):
    calls = []

    def _fake_apply(year, muons, is_data, **kwargs):
        calls.append((year, is_data, kwargs["variation"]))
        return muons.pt + 0.5

    monkeypatch.setattr(
        corrections, "apply_muon_momentum_corrections", _fake_apply
    )
    attached = corrections.AttachMuonMomentumCorrections(
        "2018", _muons(), False
    )

    assert calls == [("2018", False, "nominal")]
    assert ak.to_list(attached.pt_nom) == [[10.0]]
    assert set(ak.fields(attached)).isdisjoint(
        {
            corrections.MUON_MOMENTUM_PT_FIELDS[variation]
            for variation in _RUN3_VARIATIONS
        }
    )


def _muons_with_attached_pt_fields():
    muons = ak.with_field(_muons(), ak.Array([[999.0]]), "pt")
    muons = ak.with_field(muons, ak.Array([[10.0]]), "pt_nom")
    for idx, variation in enumerate(_RUN3_VARIATIONS, start=1):
        muons = ak.with_field(
            muons,
            ak.Array([[10.0 + idx]]),
            corrections.MUON_MOMENTUM_PT_FIELDS[variation],
        )
    return muons


@pytest.mark.parametrize(
    ("syst_var", "expected"),
    [
        ("nominal", 10.0),
        ("JES_TotalUp", 10.0),
        ("MuonScaleUp", 11.0),
        ("MuonScaleDown", 12.0),
        ("MuonResolutionUp", 13.0),
        ("MuonResolutionDown", 14.0),
    ],
)
def test_muon_systematic_selector_uses_attached_fields(syst_var, expected):
    selected = corrections.ApplyMuonMomentumSystematics(
        "2022", _muons_with_attached_pt_fields(), syst_var
    )

    assert ak.to_list(selected.pt) == [[expected]]


def test_muon_systematic_selector_fails_when_requested_field_is_missing():
    with pytest.raises(ValueError, match="pt_MuonScaleUp.*not attached"):
        corrections.ApplyMuonMomentumSystematics(
            "2022",
            ak.with_field(_muons(), ak.Array([[10.0]]), "pt_nom"),
            "MuonScaleUp",
        )


def test_run2_muon_systematic_selector_rejects_variations():
    with pytest.raises(ValueError, match="Run 2 Rochester.*does not support"):
        corrections.ApplyMuonMomentumSystematics(
            "2018",
            ak.with_field(_muons(), ak.Array([[10.0]]), "pt_nom"),
            "MuonScaleDown",
        )


def test_nominal_attachment_selector_matches_previous_direct_path():
    muons = _run3_muons()
    events = ak.Array([1001, 1002, 1003])
    lumis = ak.Array([11, 12, 13])
    direct = corrections.apply_muon_momentum_corrections(
        "2022",
        muons,
        False,
        event_numbers=events,
        luminosity_blocks=lumis,
    )
    muons = ak.with_field(muons, muons.pt, "pt_raw")
    attached = corrections.AttachMuonMomentumCorrections(
        "2022",
        muons,
        False,
        event_numbers=events,
        luminosity_blocks=lumis,
    )
    selected = corrections.ApplyMuonMomentumSystematics(
        "2022", attached, "nominal"
    )

    assert ak.to_list(selected.pt_raw) == ak.to_list(muons.pt_raw)
    assert np.allclose(
        ak.to_numpy(ak.flatten(selected.pt)),
        ak.to_numpy(ak.flatten(direct)),
    )


@pytest.mark.parametrize("variation", _RUN3_VARIATIONS)
def test_muon_systematics_are_jet_noops(variation):
    jets = object()
    assert corrections.ApplyJetSystematics("2022", jets, variation) is jets


@pytest.mark.parametrize(
    "processor_name",
    ["analysis_processor.py", "analysis_processor_diboson.py"],
)
def test_processors_attach_muon_variations_before_systematic_loop(processor_name):
    source = _processor_source(processor_name)

    preserve_raw = source.index('mu["pt_raw"] = mu.pt')
    attachment = source.index("mu = AttachMuonMomentumCorrections(")
    syst_loop = source.index("for syst_var in syst_var_list:")

    assert preserve_raw < attachment < syst_loop
    assert "mu_base = mu" not in source
    assert "apply_muon_momentum_corrections" not in source


@pytest.mark.parametrize(
    "processor_name",
    ["analysis_processor.py", "analysis_processor_diboson.py"],
)
def test_processors_build_loop_local_lepton_state(processor_name):
    source = _processor_source(processor_name)
    syst_loop = source.index("for syst_var in syst_var_list:")
    selector = source.index(
        "mu = ApplyMuonMomentumSystematics(year, mu, syst_var)", syst_loop
    )
    compute_conept = source.index(
        'mu["conept"] = leptonSelection.coneptMuon(mu)', selector
    )
    mu_pres = source.index(
        'mu["isPres"] = leptonSelection.isPresMuon(mu)', compute_conept
    )
    mu_fo = source.index(
        'mu["isFO"] = leptonSelection.isFOMuon(mu, year)', mu_pres
    )
    electron_selection = source.index(
        'ele["isPres"] = leptonSelection.isPresElec(ele)', mu_fo
    )
    m_loose = source.index("m_loose = mu[", electron_selection)
    l_loose = source.index("l_loose = ak.with_name(", m_loose)
    min_mll = source.index("min_mll_afas = ak.min(", l_loose)
    m_fo = source.index("m_fo = mu[", min_mll)
    l_fo = source.index("l_fo = ak.with_name(", m_fo)
    l_fo_sorted = source.index("l_fo_conept_sorted = l_fo[", l_fo)
    jet_cleaning = source.index("tmp = ak.cartesian", l_fo_sorted)
    event_leptons = source.index(
        'events["l_fo_conept_sorted"] = l_fo_conept_sorted',
        jet_cleaning,
    )
    event_selection = source.index("te_es.add", event_leptons)

    assert syst_loop < selector < compute_conept < mu_pres < mu_fo
    assert mu_fo < electron_selection < m_loose < l_loose < min_mll
    assert min_mll < m_fo < l_fo < l_fo_sorted < jet_cleaning
    assert jet_cleaning < event_leptons < event_selection
    assert source.index("AttachMuonSF(m_fo", m_fo) < l_fo
    assert source.index("AttachPerLeptonFR(m_fo", m_fo) < l_fo

    forbidden_names = [
        "varied_mu",
        "varied_tau",
        "m_loose_for_syst",
        "m_fo_for_syst",
        "l_loose_for_syst",
        "l_fo_for_syst",
        "l_fo_conept_sorted_for_syst",
        "min_mll_afas_for_syst",
    ]
    for name in forbidden_names:
        assert name not in source
    assert "if is_muon_momentum_systematic(syst_var):" not in source

    muon_selection_block = source[selector:l_fo_sorted]
    assert "ApplyMETSystematics" not in muon_selection_block
    assert "get_selected_met" not in muon_selection_block


@pytest.mark.parametrize(
    "processor_name",
    ["analysis_processor.py", "analysis_processor_diboson.py"],
)
def test_processors_apply_tau_shifts_before_loop_local_selection(processor_name):
    source = _processor_source(processor_name)
    syst_loop = source.index("for syst_var in syst_var_list:")
    attach = source.index("taus = AttachTauEnergyCorrections(")
    selector = source.index(
        "tau = ApplyTauEnergySystematics(taus, syst_var)",
        syst_loop,
    )
    tau_pres = source.index('tau["isPres', selector)
    tau_clean = source.index(
        'tau["isClean"] = te_os.isClean(tau, l_fo', tau_pres
    )
    tau_good = source.index('tau["isGood"]', tau_clean)
    tau_select = source.index("tau = tau[tau.isGood]", tau_good)
    jet_cleaning = source.index("tmp = ak.cartesian", tau_select)

    assert attach < syst_loop < selector < tau_pres
    assert tau_pres < tau_clean < tau_good < tau_select
    assert tau_select < jet_cleaning
    assert "ApplyTES" not in source
    assert "ApplyTESSystematic" not in source
    assert "ApplyFESSystematic" not in source
    assert "tau_energy_views" not in source


def test_legacy_tau_correction_helpers_are_removed():
    assert not hasattr(corrections, "ApplyTES")
    assert not hasattr(corrections, "ApplyTESSystematic")
    assert not hasattr(corrections, "ApplyFESSystematic")
