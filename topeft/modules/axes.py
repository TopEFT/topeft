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
        "regular": (50, 0, 1000),
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
        "regular": (10, 0, 10),
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
       "regular": (27, -0.5, 26.5),
       "label": r"pdgid of l0 genparticle",
    },
    "l1_gen_pdgId": {
       "regular": (27, -0.5, 26.5),
       "label": r"pdgid of l1 genparticle",
    },
    "l2_gen_pdgId": {
       "regular": (27, -0.5, 26.5),
       "label": r"pdgid of l2 genparticle",
    },
    "l0_genParent_pdgId": {
       "regular": (27, -0.5, 26.5),
       "label": r"pdgid of l0 genparent",
    },
    "l1_genParent_pdgId": {
       "regular": (27, -0.5, 26.5),
       "label": r"pdgid of l1 genparent",
    },
    "l2_genParent_pdgId": {
       "regular": (27, -0.5, 26.5),
       "label": r"pdgid of l2 genparent",
    }
}
