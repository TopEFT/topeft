import math


def _uniform_processing(bins, start, stop):
    return {"kind": "uniform", "bins": bins, "start": start, "stop": stop}


def _edge_processing(edges):
    return {"kind": "edges", "edges": list(edges)}


_lj0pt_override_channels = (
    *(
        f"2lss_{charge}_1tau_offZ_{njets}j"
        for charge in ("m", "p")
        for njets in (3, 4, 5, 6)
    ),
    *(
        f"3l_1tau_{nbtags}b_{njets}j"
        for nbtags in (1, 2)
        for njets in (2, 3, 4, 5)
    ),
    *(
        f"3l_{charge}_offZ_none_{nbtags}b_{njets}j"
        for charge in ("m", "p")
        for nbtags in (1, 2)
        for njets in (2, 3, 4, 5)
    ),
)

_ptll_channels = tuple(
    f"3l_{charge}_offZ_{pt_region}_{nbtags}b_{njets}j"
    for charge in ("m", "p")
    for pt_region in ("low", "high")
    for nbtags in (1, 2)
    for njets in (2, 3, 4, 5)
)

_lt_override_channels = tuple(
    f"3l_{charge}_offZ_2b_fwd_{njets}j"
    for charge in ("m", "p")
    for njets in (2, 3, 4)
)


info = {
    "npvs": {
        "processing": _uniform_processing(50, 0, 100),
        "label": r"Number of reco primary vertices ",
    },
    "npvsGood": {
        "processing": _uniform_processing(50, 0, 100),
        "label": r"Number of Good reco primary vertices ",
    },
    "invmass": {
        "processing": _uniform_processing(40, 60, 140),
        "label": r"$m_{\ell\ell}$ (GeV) ",
    },
    "ptbl": {
        "processing": _edge_processing([0, 100, 200, 400]),
        "label": r"$p_{T}^{b\mathrm{-}jet+\ell_{min(dR)}}$ (GeV) ",
    },
    "ptz": {
        "processing": _uniform_processing(12, 0, 600),
        "fitting": {
            "default": [0, 200, 300, 400, 500],
        },
        "label": r"$p_{T}$ Z (GeV) ",
    },
    "ptll": {
        "processing": _uniform_processing(12, 0, 600),
        "fitting": {
            "default": [0, 50, 100, 200, 300],
            "channels": {
                channel: [0, 50, 100, 200, 300]
                for channel in _ptll_channels
            },
        },
        "label": r"$p_{T}^{\ell\ell}$ (GeV) ",
    },
    "njets": {
        "processing": _uniform_processing(7, 0, 7),
        "label": r"Jet multiplicity ",
    },
    "nbtagsl": {
        "processing": _uniform_processing(5, 0, 5),
        "label": r"Loose btagged jet multiplicity ",
    },
    "nbtagsm": {
        "processing": _uniform_processing(5, 0, 5),
        "label": r"Medium btagged jet multiplicity ",
    },
    "l0pt": {
        "processing": _uniform_processing(25, 0, 250),
        "label": r"Leading lepton raw $p_{T}$ (GeV) ",
    },
    "l0ptcorr": {
        "processing": _uniform_processing(25, 0, 250),
        "label": r"Leading corrected lepton $p_{T}$ (GeV) ",
    },
    "l0conept": {
        "processing": _uniform_processing(25, 0, 250),
        "label": r"Leading lepton cone-$p_{T}$ (GeV) ",
    },
    "l0eta": {
        "processing": _uniform_processing(20, -2.5, 2.5),
        "label": r"Leading lepton $\eta$ ",
    },
    "l1pt": {
        "processing": _uniform_processing(15, 0, 150),
        "label": r"Subleading lepton raw $p_{T}$ (GeV) ",
    },
    "l1ptcorr": {
        "processing": _uniform_processing(15, 0, 150),
        "label": r"Subleading lepton corrected $p_{T}$ (GeV) ",
    },
    "l1conept": {
        "processing": _edge_processing([10, 20, 30, 40, 50, 60, 80, 100]),
        "label": r"Subleading lepton cone-$p_{T}$ (GeV) ",
    },
    "l1eta": {
        "processing": _uniform_processing(10, -2.5, 2.5),
        "label": r"Subleading lepton $\eta$ ",
    },
    "j0pt": {
        "processing": _uniform_processing(15, 0, 300),
        "label": r"Leading jet $p_{T}$ (GeV) ",
    },
    "fwd0pt": {
        "processing": _uniform_processing(10, 0, 200),
        "label": r"Leading forward jet $p_{T}$ (GeV) ",
    },
    "b0pt": {
        "processing": _uniform_processing(50, 0, 500),
        "label": r"Leading b-jet $p_{T}$ (GeV) ",
    },
    "j0eta": {
        "processing": _uniform_processing(15, -3, 3),
        "label": r"Leading jet $\eta$ ",
    },
    "fwd0eta": {
        "processing": _edge_processing(
            [-5, -4.5, -4, -3.6, -3.2, -2.8, -2.4, 2.4, 2.8, 3.2, 3.6, 4, 4.5, 5]
        ),
        "label": r"Leading forward jet $\eta$ ",
    },
    "ht": {
        "processing": _edge_processing([0, 300, 500, 800]),
        "label": r"H$_{T}$ (GeV) ",
    },
    "met": {
        "processing": _uniform_processing(20, 0, 200),
        "label": r"MET (GeV)",
    },
    "ljptsum": {
        "processing": _edge_processing([0, 400, 600, 1000]),
        "label": r"S$_{T}$ (GeV) ",
    },
    "o0pt": {
        "processing": _edge_processing([0, 100, 200, 400]),
        "label": r"Leading l or b jet $p_{T}$ (GeV)",
    },
    "bl0pt": {
        "processing": _edge_processing([0, 100, 200, 400]),
        "label": r"Leading (b+l) $p_{T}$ (GeV) ",
    },
    "lj0pt": {
        "processing": _uniform_processing(12, 0, 600),
        "fitting": {
            "default": [0, 150, 250, 500],
            "channels": {
                channel: [0, 150, 250, 350]
                for channel in _lj0pt_override_channels
            },
        },
        "label": r"Leading $p_{T}$ of pair from the ($\ell$+j+$\tau$) collection $p_{T}^{\ell j 0}$ (GeV) ",
    },
    "ptz_wtau": {
        "processing": _uniform_processing(12, 0, 600),
        "fitting": {"default": [0, 50, 100, 150]},
        "label": r"$p_{T}$ of $\ell+\tau_h$ pair (GeV) ",
    },
    "tau0Tpt": {
        "processing": _uniform_processing(20, 0, 200),
        "label": r"$p_{T}$ of leading tight hadronic tau (GeV) ",
    },
    "tau0Fpt": {
        "processing": _uniform_processing(20, 0, 200),
        "label": r"$p_{T}$ of leading FO hadronic tau (GeV) ",
    },
    "lt": {
        "processing": _uniform_processing(12, 0, 600),
        "fitting": {
            "default": [0, 150, 250, 500],
            "channels": {
                channel: [0, 250, 400, 500]
                for channel in _lt_override_channels
            },
        },
        "label": r"Scalar sum of MET and leading leptons (GeV)",
    },
}


_pdg_labels = {
    "l0_gen_pdgId": "pdgid of l0 genparticle",
    "l1_gen_pdgId": "pdgid of l1 genparticle",
    "l2_gen_pdgId": "pdgid of l2 genparticle",
    "l0_genParent_pdgId": "pdgid of l0 genparent",
    "l1_genParent_pdgId": "pdgid of l1 genparent",
    "l2_genParent_pdgId": "pdgid of l2 genparent",
}
for _name, _label in _pdg_labels.items():
    info[_name] = {
        "processing": _uniform_processing(28, -1.5, 26.5),
        "label": _label,
    }


_flavor_labels = {
    "b0l_hFlav": "Hadron Flavor of leading loose b jet",
    "b0l_pFlav": "Parton Flavor of leading loose b jet",
    "b0m_hFlav": "Hadron Flavor of leading medium b jet",
    "b0m_pFlav": "Parton Flavor of leading medium b jet",
    "b1l_hFlav": "Hadron Flavor of subleading loose b jet",
    "b1l_pFlav": "Parton Flavor of subleading loose b jet",
    "b1m_hFlav": "Hadron Flavor of subleading medium b jet",
    "b1m_pFlav": "Parton Flavor of subleading medium b jet",
    "b0l_genhFlav": "GenHadron Flavor of leading loose b jet",
    "b0l_genpFlav": "GenParton Flavor of leading loose b jet",
    "b0m_genhFlav": "GenHadron Flavor of leading medium b jet",
    "b0m_genpFlav": "GenParton Flavor of leading medium b jet",
    "b1l_genhFlav": "GenHadron Flavor of subleading loose b jet",
    "b1l_genpFlav": "GenParton Flavor of subleading loose b jet",
    "b1m_genhFlav": "GenHadron Flavor of subleading medium b jet",
    "b1m_genpFlav": "GenParton Flavor of subleading medium b jet",
}
for _name, _label in _flavor_labels.items():
    info[_name] = {
        "processing": _uniform_processing(28, -1.5, 26.5),
        "label": _label,
    }


def _two_dimensional_axis(name, bins, start, stop, label):
    return {
        "name": name,
        "processing": _uniform_processing(bins, start, stop),
        "label": label,
    }


info_2d = {
    "lepton_pt_vs_eta": {
        "axes": [
            _two_dimensional_axis(
                "lepton_pt_vs_eta_pt", 25, 0, 250, r"Leading lep $p_{T}$ (GeV) "
            ),
            _two_dimensional_axis(
                "lepton_pt_vs_eta_abseta", 25, 0, 2.5, r"Leading lep $|\eta|$ "
            ),
        ],
    },
    "l0_SeedEtaOrX_vs_SeedPhiOrY": {
        "axes": [
            _two_dimensional_axis(
                "l0_SeedEtaOrX_vs_SeedPhiOrY_SeedEtaOrX",
                400,
                -200,
                200,
                r"Leading lep seed $\eta / x$ ",
            ),
            _two_dimensional_axis(
                "l0_SeedEtaOrX_vs_SeedPhiOrY_SeedPhiOrY",
                500,
                0,
                500,
                r"Leading lep seed $\phi / y$ ",
            ),
        ],
    },
    "l0_eta_vs_phi": {
        "axes": [
            _two_dimensional_axis("l0_eta_vs_phi_eta", 120, -3, 3, r"Leading lep $\eta$ "),
            _two_dimensional_axis("l0_eta_vs_phi_phi", 160, -4, 4, r"Leading lep $\phi$ "),
        ],
    },
    "l1_SeedEtaOrX_vs_SeedPhiOrY": {
        "axes": [
            _two_dimensional_axis(
                "l1_SeedEtaOrX_vs_SeedPhiOrY_SeedEtaOrX",
                400,
                -200,
                200,
                r"Subleading lep seed $\eta / x$ ",
            ),
            _two_dimensional_axis(
                "l1_SeedEtaOrX_vs_SeedPhiOrY_SeedPhiOrY",
                500,
                0,
                500,
                r"Subleading lep seed $\phi / y$ ",
            ),
        ],
    },
    "l1_eta_vs_phi": {
        "axes": [
            _two_dimensional_axis("l1_eta_vs_phi_eta", 120, -3, 3, r"Subleading lep $\eta$ "),
            _two_dimensional_axis("l1_eta_vs_phi_phi", 160, -4, 4, r"Subleading lep $\phi$ "),
        ],
    },
    "jet_eta_phi_before_veto": {
        "axes": [
            _two_dimensional_axis(
                "jet_eta_phi_before_veto_eta", 104, -5.2, 5.2, r"Jet $\eta$ before veto "
            ),
            _two_dimensional_axis(
                "jet_eta_phi_before_veto_phi", 72, -math.pi, math.pi, r"Jet $\phi$ before veto "
            ),
        ],
    },
    "jet_eta_phi_after_veto": {
        "axes": [
            _two_dimensional_axis(
                "jet_eta_phi_after_veto_eta", 104, -5.2, 5.2, r"Jet $\eta$ after veto "
            ),
            _two_dimensional_axis(
                "jet_eta_phi_after_veto_phi", 72, -math.pi, math.pi, r"Jet $\phi$ after veto "
            ),
        ],
    },
}
