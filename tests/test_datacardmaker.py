import numpy as np
from coffea import hist
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth
import cloudpickle
import os
import gzip
import subprocess
import json

# Let's generate some fake data to use for testing
nevts = 1
rng = np.random.default_rng()
#eft_fit_coeffs = rng.normal(0.3, 0.5, (nevts,276))
eft_fit_coeffs = np.ones((nevts,276))*1.3
sums = np.sum(eft_fit_coeffs, axis=0)
tolerance = 1e-10

wc_names_lst = [
    "cpt",
    "ctp",
    "cptb",
    "cQlMi",
    "cQq81",
    "cQq11",
    "cQl3i",
    "ctq8",
    "ctlTi",
    "ctq1",
    "ctli",
    "cQq13",
    "cbW",
    "cpQM",
    "cpQ3",
    "ctei",
    "cQei",
    "ctW",
    "ctlSi",
    "cQq83",
    "ctZ",
    "ctG"
]

hists = {}
hists["ptbl"] =  HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ptbl",    "$p_{T}^{b-jet+l_{min(dR)}}$ (GeV) ", 200, 0, 2000))
hists["ptz"]  = HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ptz",      "$p_{T}$ Z (GeV)", 25, 0, 1000))
hists["njets"] =  HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("njets",   "Jet multiplicity ", 10, 0, 10))
hists["nbtagsl"] =  HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("nbtagsl",   "Jet multiplicity ", 10, 0, 10))
hists["ht"] =  HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ht",      "H$_{T}$ (GeV)", 200, 0, 2000))
hists["met"] =  HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("met",     "MET (GeV)", 40, 0, 400))

# Fill the EFT histogram
weight_val = 1.0
hists["njets"].fill(njets=4, sample='ttHJet_privateUL17', channel='2lss_2b_p', appl='isSR_2lSS', systematic='nominal', weight=nevts, eft_coeff=eft_fit_coeffs)
hists["njets"].fill(njets=4, sample='ttHJet_privateUL17', channel='2lss_2b_p', appl='isSR_2lSS', systematic='testUp', weight=nevts, eft_coeff=eft_fit_coeffs)
hists["ptbl"].fill(ptbl=40, sample='ttHJet_privateUL17', channel='2lss_2b_p', appl='isSR_2lSS', systematic='nominal', weight=nevts, eft_coeff=eft_fit_coeffs)
hists["ptbl"].fill(ptbl=4, sample='ttHJet_privateUL17', channel='2lss_2b_p', appl='isSR_2lSS', systematic='testUp', weight=nevts, eft_coeff=eft_fit_coeffs)
sm_weight = np.zeros(nevts)
sm_weight[0] = 1
sm_njets4 = sums*sm_weight

with open('topcoffea/json/lumi.json') as jf:
    lumi = json.load(jf)
    lumi = lumi
lumi = {year : 1000*lumi for year,lumi in lumi.items()}

def test_datacard_pkl():
    assert(np.all(hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4] - sm_njets4) < tolerance) # Testing SM value

    out_pkl_file = os.path.join("tests/test_datacard.pkl.gz")
    with gzip.open(out_pkl_file, "wb") as fout:
      cloudpickle.dump(hists, fout)

def test_datacard_maker():
    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "tests/test_datacard.pkl.gz",
        "--var-lst",
        "njets", # Only checking an njets card, so don't need to make cards for the other variables
        "-j",
        "0",
        "--do-nuisance"
    ]

    # Run datacard maker
    subprocess.run(args)

def test_datacard_results():
    '''
    These lines parse the 2lss_2b_p datacard (njets) and
    extract a dictionary of {process: value} for all histograms
    '''
    f = open('histos/ttx_multileptons-2lss_2b_p.txt', 'r')
    fin = f.readlines()
    process = []
    rate = []
    found_process = False # Hack to skip process number
    for line in fin:
        if 'process' in line and not found_process:
            found_process = True
            line = line.split()[1:]
            process = line
        if 'rate' in line:
            line = line.split()[1:]
            line = [float(l) for l in line]
            rate = line
    d = dict(zip(process, rate))

    wcs = ['ctW']

    hists['njets'].scale(lumi['2017'] )

    # Test ttH SM
    sm_val = hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4]
    assert((hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4] - d['ttH_sm']) < tolerance)

    '''
    ``S`` is theSM
    ``S+L_i+Q_i`` sets ``WC_i=1`` and the rest to ``0``
    ``S+L_i+L_j+Q_i+Q_j+2 M_IJ`` set ``WC_i=1``, ``WC_j=1`` and the rest to ``0``
    '''

    # Lets just check ctW for now
    lin_names = [k for k in d if 'lin' in k and any(wc in k for wc in wcs)]
    quad_names = [k for k in d if 'quad' in k and 'mix' not in k and any(wc in k for wc in wcs)]
    mix_names = [k for k in d if 'mix' in k and any(wc in k for wc in wcs)]
    lin_vals = {wc:d[wc] for wc in lin_names}
    wcs_index = {wc:hists['njets']._wcnames.index(wc) for wc in wcs}
    coeffs = np.zeros(len(wc_names_lst))
    coeffs[wcs_index['ctW']] = 1
    lin_val = hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4]
    coeffs[wcs_index['ctW']] = 2
    quad_val = hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4]
    pred_val = sm_val + lin_val + quad_val
    hists['njets'].set_wilson_coeff_from_array(coeffs)
    dc_val = lin_vals['ttH_lin_ctW']
    assert(np.abs(pred_val - dc_val) < tolerance)

    # Check bilinear terms (mixed quadratic)
    mix_vals = {wc:d[wc] for wc in mix_names}
    quad_vals = {wc:d[wc] for wc in quad_names}
    wcs_mix = []
    # Build list of binlinear terms
    for mix in mix_names:
        wcs_mix.append([wc for wc in mix.split('_')[-2:] if wc != wcs[0]][0])
    wcs_index = {wc:hists['njets']._wcnames.index(wc) for wc in wcs_mix+wcs}
    # Loop over each term
    for mix in wcs_mix:
        coeffs = np.zeros(len(wc_names_lst))
        coeffs[wcs_index['ctW']] = 1
        hists['njets'].set_wilson_coeff_from_array(coeffs)
        lin_val = hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4]
        coeffs[wcs_index[mix]] = 1
        hists['njets'].set_wilson_coeff_from_array(coeffs)
        mix_val = hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4]
        dc_val = mix_vals[[name for name in mix_names if mix in name][0]]
        assert(np.abs(mix_val - dc_val) < tolerance)

    # Check pure quadratic term (e.g. `ctW*ctW`)
    coeffs = np.zeros(len(wc_names_lst))
    coeffs[wcs_index['ctW']] = 2
    hists['njets'].set_wilson_coeff_from_array(coeffs)
    quad_val = hists['njets'].integrate('sample', 'ttHJet_privateUL17').integrate('channel', '2lss_2b_p').integrate('appl', 'isSR_2lSS').integrate('systematic', 'nominal').values()[()][4]
    quad_val += -2*lin_val + sm_val
    quad_val /= 2
    dc_val = quad_vals[[name for name in quad_names if 'ctW' in name][0]]
    assert(np.abs(quad_val - dc_val) < tolerance)
