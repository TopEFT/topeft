info = {
    "npvs": {
        "regular": (100, 0, 100),
        "label": r"Number of reco primary vertices ",
    },
    "npvsGood": {
        "regular": (100, 0, 100),
        "label": r"Number of Good reco primary vertices ",
    },
    "invmass": {
        "regular": (40, 60, 140),
        "label": r"$m_{\ell\ell}$ (GeV) ",
    },
    "ptbl": {
        "regular": (40, 0, 1000),
        "variable": [0, 100, 200, 400],
        "label": r"$p_{T}^{b\mathrm{-}jet+\ell_{min(dR)}}$ (GeV) ",
    },
    "ptz": {
        "regular": (12, 0, 600),
        "variable": [0, 200, 300, 400, 500],
        "label": r"$p_{T}$ Z (GeV) ",
    },
    "njets": {
        "regular": (6, 0, 6),
        "variable_multi": {
            "2l": [4, 5, 6, 7],
            "3l": [2, 3, 4, 5],
            "4l": [2, 3, 4],
        },
        "label": r"Jet multiplicity ",
    },
    "nbtagsl": {
        "regular": (5, 0, 5),
        "label": r"Loose btag multiplicity "},
    "l0pt": {
        "regular": (25, 0, 250),
        "label": r"Leading lep raw $p_{T}$ (GeV) ",
    },
    "l0ptcorr": {
        "regular": (25, 0, 250),
        "label": r"Leading corrected lep $p_{T}$ (GeV) ",
    },
    "l0conept": {
        "regular": (25, 0, 250),
        "label": r"Leading lep cone-$p_{T}$ (GeV) ",
    },
    "l0eta": {
        "regular": (20, -2.5, 2.5),
        "label": r"Leading lep $\eta$ "
    },
    "l1pt": {
        "regular": (15, 0, 150),
        "label": r"Subleading lep raw $p_{T}$ (GeV) "
    },
    "l1ptcorr": {
        "regular": (15, 0, 150),
        "label": r"Subleading lep corrected $p_{T}$ (GeV) "
    },
    "l1conept": {
        "regular": (15, 0, 150),
        "label": r"Subleading lep cone-$p_{T}$ (GeV) "
    },
    "l1eta": {
        "regular": (20, -2.5, 2.5),
        "label": r"Subleading $\eta$ "
    },
    "j0pt": {
        "regular": (50, 0, 500),
        "label": r"Leading jet  $p_{T}$ (GeV) "
    },
    "b0pt": {
        "regular": (50, 0, 500),
        "label": r"Leading b jet  $p_{T}$ (GeV) "
    },
    "j0eta": {
        "regular": (30, -3, 3),
        "label": r"Leading jet  $\eta$ "
    },
    "ht": {
        "regular": (100, 0, 1000),
        "variable": [0, 300, 500, 800],
        "label": r"H$_{T}$ (GeV) ",
    },
    "met": {"regular": (40, 0, 400), "label": r"MET (GeV)"},
    "ljptsum": {
        "regular": (11, 0, 1100),
        "variable": [0, 400, 600, 1000],
        "label": r"S$_{T}$ (GeV) ",
    },
    "o0pt": {
        "regular": (10, 0, 500),
        "variable": [0, 100, 200, 400],
        "label": r"Leading l or b jet $p_{T}$ (GeV)",
    },
    "bl0pt": {
        "regular": (10, 0, 500),
        "variable": [0, 100, 200, 400],
        "label": r"Leading (b+l) $p_{T}$ (GeV) ",
    },
    "lj0pt": {
        "regular": (12, 0, 600),
        "variable": [0, 150, 250, 500],
        "label": r"Leading pt of pair from l+j collection (GeV) ",
    },
    "ptz_wtau": {
        "regular": (12, 0, 600),
        "variable": [0, 150, 250, 500],
        "label": r"pt of lepton hadronic tau pair (GeV) ",
    },
    "tau0pt": {
        "regular": (60, 0, 600),
        "variable": [0, 150, 250, 500],
        "label": r"pt of leading hadronic tau (GeV) ",
    },
    "lt": {
        "regular": (12, 0, 600),
        "variable": [0,150,250,500],
        "label": r"Scalar sum of met at leading leptons (GeV)",
    },

    "l0_gen_pdgId": {
       "regular": (28, -1.5, 26.5),
       "label": r"pdgid of l0 genparticle",
    },
    "l1_gen_pdgId": {
       "regular": (28, -1.5, 26.5),
       "label": r"pdgid of l1 genparticle",
    },
    "l2_gen_pdgId": {
       "regular": (28, -1.5, 26.5),
       "label": r"pdgid of l2 genparticle",
    },
    "l0_genParent_pdgId": {
       "regular": (28, -1.5, 26.5),
       "label": r"pdgid of l0 genparent",
    },
    "l1_genParent_pdgId": {
       "regular": (28, -1.5, 26.5),
       "label": r"pdgid of l1 genparent",
    },
    "l2_genParent_pdgId": {
       "regular": (28, -1.5, 26.5),
       "label": r"pdgid of l2 genparent",
    },

    "b0l_hFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Hadron Flavor of leading loose b jet",
   },
    "b0l_pFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Parton Flavor of leading loose b jet",
   },
    "b0m_hFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Hadron Flavor of leading medium b jet",
   },
    "b0m_pFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Parton Flavor of leading medium b jet",
   },
    "b1l_hFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Hadron Flavor of subleading loose b jet",
   },
    "b1l_pFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Parton Flavor of subleading loose b jet",
   },
    "b1m_hFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Hadron Flavor of subleading medium b jet",
   },
    "b1m_pFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"Parton Flavor of subleading medium b jet",
   },
    "b0l_genhFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenHadron Flavor of leading loose b jet",
   },
    "b0l_genpFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenParton Flavor of leading loose b jet",
   },
    "b0m_genhFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenHadron Flavor of leading medium b jet",
   },
    "b0m_genpFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenParton Flavor of leading medium b jet",
   },
    "b1l_genhFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenHadron Flavor of subleading loose b jet",
   },
    "b1l_genpFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenParton Flavor of subleading loose b jet",
   },
    "b1m_genhFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenHadron Flavor of subleading medium b jet",
   },
    "b1m_genpFlav": {
       "regular": (28, -1.5, 26.5),
       "label": r"GenParton Flavor of subleading medium b jet",
    }
}

info_2d = {
    "lepton_pt_vs_eta": {
        "axes": [
            {
                "name": "lepton_pt_vs_eta_pt",
                "regular": (25, 0, 250),
                "label": r"Leading lep $p_{T}$ (GeV) ",
            },
            {
                "name": "lepton_pt_vs_eta_abseta",
                "regular": (25, 0, 2.5),
                "label": r"Leading lep $|\eta|$ ",
            },
        ],
    },
    "l0_SeedEtaOrX_vs_SeedPhiOrY": {
        "axes": [
            {
                "name": "l0_SeedEtaOrX_vs_SeedPhiOrY_SeedEtaOrX",
                "regular": (400, -200, 200),
                "label": r"Leading lep seed $\eta / x$ ",
            },
            {
                "name": "l0_SeedEtaOrX_vs_SeedPhiOrY_SeedPhiOrY",
                "regular": (500, 0, 500),
                "label": r"Leading lep seed $\phi / y$ ",
            },
        ],
    },
    "l0_eta_vs_phi": {
        "axes": [
            {
                "name": "l0_eta_vs_phi_eta",
                "regular": (120, -3, 3),
                "label": r"Leading lep $\eta$ ",
            },
            {
                "name": "l0_eta_vs_phi_phi",
                "regular": (160, -4, 4),
                "label": r"Leading lep $\phi$ ",
            },
        ],
    },
    "l1_SeedEtaOrX_vs_SeedPhiOrY": {
        "axes": [
            {
                "name": "l1_SeedEtaOrX_vs_SeedPhiOrY_SeedEtaOrX",
                "regular": (400, -200, 200),
                "label": r"Subleading lep seed $\eta / x$ ",
            },
            {
                "name": "l1_SeedEtaOrX_vs_SeedPhiOrY_SeedPhiOrY",
                "regular": (500, 0, 500),
                "label": r"Subleading lep seed $\phi / y$ ",
            },
        ],
    },
    "l1_eta_vs_phi": {
        "axes": [
            {
                "name": "l1_eta_vs_phi_eta",
                "regular": (120, -3, 3),
                "label": r"Subleading lep $\eta$ ",
            },
            {
                "name": "l1_eta_vs_phi_phi",
                "regular": (160, -4, 4),
                "label": r"Subleading lep $\phi$ ",
            },
        ],
    },
}
