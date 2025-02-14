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
    "l1pt": {
        "regular": (10, 0, 100), 
        "variable": [0, 50, 100, 200],
        "label": r"Subleading lep $p_{T}$ (GeV) "
    },
    "l1eta": {
        "regular": (20, -2.5, 2.5), 
        "variable": [-2.4, -1.44, -0.8, 0, 0.8, 1.44, 2.4],
        "label": r"Subleading $\eta$ "
    },
    "j0pt": {
        "regular": (10, 0, 500), 
        "variable": [0, 50, 100, 200],
        "label": r"Leading jet  $p_{T}$ (GeV) "
    },
    "b0pt": {
        "regular": (10, 0, 500), 
        "variable": [0, 50, 100, 200],
        "label": r"Leading b jet  $p_{T}$ (GeV) "
    },
    "l0eta": {
        "regular": (20, -2.5, 2.5), 
        "variable": [-2.4, -1.44, -0.8, 0, 0.8, 1.44, 2.4],
        "label": r"Leading lep $\eta$ "
    },
    "j0eta": {
        "regular": (30, -3, 3), 
        "variable": [-2.4, -1.44, -0.8, 0, 0.8, 1.44, 2.4],
        "label": r"Leading jet  $\eta$ "
    },
    "njet_bjet": {
        "regular": (15, 0, 15), 
        "label": "(njets,nbjets)"
    },
    "photon_pt_eta": {
        "pt": {
            "variable": [20,30,45,70,120],
            "label": "$p_{T}$ $\gamma$ (GeV)"
        },
        "abseta": {
            "variable" : [0,0.435,0.783,1.13,1.50],
            "label": "Photon abs. $\eta$"
        }
    },
    "photon_pt": {
        "regular": (20, 0, 400),
        "variable": [20,35,50,70,100,170,200,250,300],
        "label": "$p_{T}$ $\gamma$ (GeV)"
    },
    "photon_pt2": {
        "variable": [20,30,45,70,120],
        "label": "$p_{T}$ $\gamma$ (GeV)"
    },
    "photon_eta": { "regular": (15, -1.5, 1.5), "label": "Photon $\eta$"},
    "photon_eta2": { "variable" : [0,0.435,0.783,1.13,1.50], "label": "Photon abs. $\eta$"},
    "nPhoton": {"regular": (7, 0, 7), "label": "Photon multiplicity"},
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
    "ptz_wtau": {
        "regular": (12, 0, 600),
        "variable": [0, 150, 250, 500],
        "label": r"pt of lepton hadronic tau pair (GeV) ",
    },
    "tau0pt": {
        "regular": (12, 0, 600),
        "variable": [0, 150, 250, 500],
        "label": r"pt of leading hadronic tau (GeV) ",
    },
    "lt": {
        "regular": (12, 0, 600),
        "variable": [0,150,250,500],
        "label": r"Scalar sum of met at leading leptons (GeV)",
    },
}
