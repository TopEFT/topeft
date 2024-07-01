info = {
    "invmass": {
        "regular": (20, 0, 1000),
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
    "nbtagsl": {"regular": (5, 0, 5), "label": r"Loose btag multiplicity "},
    "l0pt": {
        "regular": (10, 0, 500),
        "variable": [0, 50, 100, 200],
        "label": r"Leading lep $p_{T}$ (GeV) ",
    },
    "l1pt": {"regular": (10, 0, 100), "label": r"Subleading lep $p_{T}$ (GeV) "},
    "l1eta": {"regular": (20, -2.5, 2.5), "label": r"Subleading $\eta$ "},
    "j0pt": {"regular": (10, 0, 500), "label": r"Leading jet  $p_{T}$ (GeV) "},
    "b0pt": {"regular": (10, 0, 500), "label": r"Leading b jet  $p_{T}$ (GeV) "},
    "l0eta": {"regular": (20, -2.5, 2.5), "label": r"Leading lep $\eta$ "},
    "j0eta": {"regular": (30, -3, 3), "label": r"Leading jet  $\eta$ "},
    "photon_pt": {
        "regular": (20, 0, 400),
        "variable": [20,35,50,70,100,170,200,250,300],
        "label": "$p_{T}$ $\gamma$ (GeV)"
     },
    "nPhoton": {"regular": (7, 0, 7), "label": "Photon multiplicity"},
    "photon_relPFchIso": {"regular": (50,0,7), "label": "PF relative ch. had. isolation (GeV)"},
    "photon_PFchIso": {"regular": (100,0,15), "label": "PF ch. had. isolation (GeV)"},
    "cutBased": {"regular": (6, 0, 6), "label": "$p_{T}$ $\gamma$ (GeV)"},
    "pp_mass": {"regular": (60, 0, 600), "label": "$m_{\gamma\gamma}$ (GeV)"},
    "invmass_llgamma": {"regular": (28,60,200), "label": "$m_{\ell\ell\gamma}$ (GeV)"},
    "ht": {
        "regular": (20, 0, 1000),
        "variable": [0, 300, 500, 800],
        "label": r"H$_{T}$ (GeV) ",
    },
    "met": {"regular": (20, 0, 400), "label": r"MET (GeV)"},
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
}
