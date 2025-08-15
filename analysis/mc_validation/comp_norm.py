'''
This script produces plots comparing two input pkl files made from `gen_processor.py`.
The minimum set of inputs is the paths to each pkl file and the json for one of them.
If the json file contains the starting points ("StPt") it will draw a green curve reweighted to that value,
otherwise it will set all WCs to 1 for the green curve.

Variables:
You can specify a single variable using `--var <your variable>`,
otherwise the script will loop over _all_ variables listed at the bottom.

Comparing two private EFT files:
By default the script expects the first pkl file to be an EFT file and the second to be a SM central file.
If you pass the argument `--private` it will understand that the _second_ pkl file is also a private sample.
There is a special flag `--skip` which tells the script _not_ to draw a reweighted histogram for the second pkl file.
This is useful when comparing Run 3 to Run 2 (at the SM) since some of the WC names have changed.


Labels:
You can modify the labels by using `--str1 <your string>` and/or `--str2 <your other string>`.

Density and flow bins:
The flag `--density` will normalize the main plots to unity and draw them on a log scale.
This is useful if your normalization is off but you want a larger plot of the shapes.
The flag `--flow` will draw the under and overflow bins.

Lumi:
The script will try to infer the luminosity of each file based on the names.
If the names don't contain `UL1x` or `202x` it will _not_ rescale anything.
If it detects _different_ lumis, it will scale to the larger one.
Use `--no-lumi` to skip this check/scaling.


Exampe:
python comp_norm.py histos/2022_tllq_NewStPt4_zero.pkl.gz histos/2022_tllq_fixed0p1.pkl.gz ../../input_samples/sample_jsons/signal_samples/private_UL/2022_tllq_fixed0p1_nanoGEN.json --var t_pt --private --skip --str2 "nanoGEN gen-level fixed +/-1"
'''
import os
import re
import pickle
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import mplhep as hep
from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))

#Load hists from pickle file created by TopCoffea
hists1={}
hists2={}

parser = argparse.ArgumentParser(description='You can select which file to run over')
parser.add_argument('fin1'   , default='analysis/topEFT/histos/mar03_central17_pdf_np.pkl.gz' , help = 'Variable to run over')
parser.add_argument('fin2'   , default='analysis/topEFT/histos/mar03_central17_pdf_np.pkl.gz' , help = 'Variable to run over')
parser.add_argument('--str1'   , default='nanoGEN gen-level EFT' , help = 'String 1')
parser.add_argument('--str2'   , default='nanoAOD gen-level central' , help = 'String 2')
parser.add_argument('--density', action='store_true' , help = 'Normalize')
parser.add_argument('--flow'   , action='store_true' , help = 'Overflow')
parser.add_argument('--var'    , default=None , help = 'Variable to plot')
parser.add_argument('--private', action='store_true' , help = 'Use private key for second hist')
parser.add_argument('--skip'   , action='store_true' , help = 'Use private key for second hist')
parser.add_argument('json'     , default='', help = 'Json file(s) containing files and metadata')
parser.add_argument('--small'   , action='store_true', help = 'Remove all |WCs| >100')
parser.add_argument('--no-lumi' , action='store_true', help = 'Don\t rescale the lumi')
args  = parser.parse_args()
fin1   = args.fin1
fin2   = args.fin2

with gzip.open(fin1) as fin1:
    hin = pickle.load(fin1)
    for k in hin.keys():
        if k in hists1: hists1[k]+=hin[k]
        else:               hists1[k]=hin[k]
with gzip.open(fin2) as fin2:
    hin = pickle.load(fin2)
    for k in hin.keys():
        if k in hists2: hists2[k]+=hin[k]
        else:               hists2[k]=hin[k]

'''
h = hists['met'] #grab the MET plot, others are available
h = hists['njets'] #grab the MET plot, others are available
h = hists['njetsnbtags'] #grab the njets,nbtags plot, others are available
ch3l = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ'] #defin1e 3l channel
fig, ax = plt.subplots(1, 1, figsize=(7,7)) #1 panel plot, 700x700
hist.plot1d(h.integrate('channel', ch3l).sum('sample'), overlay='cut', stack=False) #create a 1D plot
plt.show() #took me longer than I’d like to admit to get Python to draw the canvas!

#last = np.array(list(h.values().values()))[np.arange(65).reshape(1,13,5)[0][12][4]]

ch2lss = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
h = hists['njets'].integrate('sample', 'ttHnobb').integrate('channel', ch2lss).integrate('cut', '4j2b')
ax = hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7])
print(h.values()[()])
h.rebin('njets', ax)
print(h.values()[()])
'''
#fname = '/afs/crc.nd.edu/user/{user[0]}/{user}/new_topcoffea/topeft/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root'
#events = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema).events()
#print(events)
var = 'l0pt'
var = 'lhe_photon_pt'
var = 'photon_pt'
var = args.var

density=True
density=False
density=args.density
flow='none'
flow='show'
flow = 'show' if args.flow else 'none'
printout = True

def plot(var=None, fin1=None, fin2=None, flow=None, private=False, hists1=None, hists2=None, hists3=None):
    chan = '2lss_3j'
    appl = 'isSR_2lSS'
    chan = 'incl_sfos30_dral0p4_0j'
    chan = 'incl_draj0p5_dral0p4_0j'
    chan = 'incl_2los_ph_0j'
    chan = 'incl_2los_ph_dral0p4_0j'
    chan = 'incl_2los_ph_sfos30_dral0p4_0j'
    chan = 'incl_draj_dral0p4_0j'
    chan = '3l_1j'
    appl = 'isAR_3l'
    chan = '2los_ph_1j'
    appl = 'isSR_2lOS'
    chan = '2lss_2j'
    appl = 'isSR_2lSS'
    chan = 'incl_0j'
    appl = 'isAR_incl'

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    #hists1[var] *=  1 / 2.2471 / (1000*41.48)
    #hists2[var] *=  1 / 2.2471 / (1000*41.48)
    #hists2[var] *=  41.48/59.83
    #hists1[var] *=  1.513 / 2.2471
    #hists1[var] /=  3.75 / 2.2471
    #hists1[var] /=  3.75 / 2.629
    #hists1[var] *=  3.489 / 2.2471
    #hists1[var] /=  0.8548255479147789 #dral
    #hists1[var] *=  1.33158521443235 #dral central/private
    #hists1[var] *=  1.3120242975018286 #dral dressed leptons central/private
    #hists1[var] *=  1.58040479981269 # 2los ph 3j
    #hists1[var] /= 0.7440378616686412
    #hists1[var] /=  0.44528
    #hists1[var] /=  1.426547908273073
    #hists1[var] *=  2.629 / 2.2471
    #hists1[var] *=  2.629 / 3.75
    #hists1[var] /=  1.513 / 2.2471
    #hists1[var] /=  1.3904 / 2.2471
    #hists1[var] *=  3.754 / 2.2471
    #hists1[var] *=  2.499 / 2.2471

    if var == 'njets':
        chan = 'incl'

    proc2 = 'private'
    proc = 'TLLQ_2022_nanoGEN'
    proc = 'THQ_2022_nanoGEN'
    proc = 'private'
    proc2 = 'private'
    proc = ''
    proc2 = ''
    str2 = args.str2
    if args.private:
        str2 = str2.replace('central', 'TOP-22-006')

    lumi = 1000*41.48

    # Infer lumi from file name
    def extract_year(s):
        match = re.search(r'(?:UL(\d{2})|(\d{4}))', s)
        if match:
            if match.group(1):  # Matched ULxx
                return int("20" + match.group(1))  # Convert UL17 -> 2017
            else:  # Matched yyyy
                return int(match.group(2))
        return None
    lumi1, lumi2 = 0, 0
    year1 = extract_year(args.fin1)
    year2 = extract_year(args.fin2)
    if year1 is not None and not args.no_lumi:
        lumi1 = 1000.0*get_tc_param(f"lumi_{year1}")
    if year2 is not None and not args.no_lumi:
        lumi2 = 1000.0*get_tc_param(f"lumi_{year2}")
    if lumi1 > 0 and lumi2 > 0 and not args.density:
        if lumi1 > lumi2:
            print(f'Scaling {args.fin2} from {round(lumi2/1000)} pb^-1 to {round(lumi1/1000)} pb^-1')
            hists2[var] *= lumi1/lumi2
        elif lumi2 > lumi1:
            print(f'Scaling {args.fin1} from {round(lumi1/1000)} pb^-1 to {round(lumi2/1000)} pb^-1')
            hists1[var] *= lumi2/lumi1
    elif not args.density:
        print(f'\n\nWARNING: Could not infer the luminosity.\n         Make sure {args.fin1} and {args.fin2} are normalized the same!\n\n')

    #hists1[var] /= lumi
    #hists1[var] *= 41.48/7.9804
    #hists2[var] /= lumi

    if args.private: sm = hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    else: sm = hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    err = np.sqrt(sm.variances()[()])/np.sum(sm.values()[()])
    err = np.sqrt(sm.variances()[()])#/np.sum(sm.values()[()])
    err = np.sqrt(sm.variances(flow=(flow=='show'))[()])#/np.sum(sm.values()[()])
    if flow=='show':
        err = err[1:]
    if not args.private: hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=err, label=str2 + ' SM', ax=ax, density=density, flow=flow)
    else: hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=err, label=str2, ax=ax, density=density, flow=flow)

    #hists1[var] *=  143723.62086377045 / 47358.43419695908
    #hists1[var] *= 1 + 47358.43419695908 / 143723.62086377045

    #s = np.sum(sm.values()) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values())
    #hists1[var] *= s

    global printout
    if printout and var=='dral':
        printout = False
        print('hist1 / lumi', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / lumi)
        print('hist1 / lumi / xsec / k', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / 2.2471 / lumi)
        print('hist1 / lumi / 3.75', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / 3.75 / lumi)
        #print('hist1 * sow / lumi / xsec / k', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=False)) / 2.2471 / lumi *  143723.62086377045)
        #hists1[var] *=  1.513/ 2.2471 # undo k-factor
        print('hist1 / lumi / xsec', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / 1.513 / lumi)
        #lumi = (1000*59.83)
        print('hist2 / lumi / xsec / k', np.sum(sm.values(flow=True)) / 2.2471 / lumi)
        print('hist2 / lumi / xsec', np.sum(sm.values(flow=True)) / 1.513 / lumi)
        #print('hist2 / lumi / xsec / k', np.sum(sm.values(flow=True)) / 2.2471 / (1000*41.48))
        #print('hist2 / lumi / xsec', np.sum(sm.values(flow=True)) / 1.513 / (1000*41.48))
        print('Removing dral<0.1: hist1 / (hist>0.1) / xsec', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / 2.2471)
        print('Removing dral<0.1: hist1 / (hist>0.1)', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]))
        print('Removing dral<0.1: (hist>0.1)', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / lumi)
        print('Removing dral<0.1: (hist2>0.1)/(hist1>0.1)', np.sum(hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]))
        print('Removing dral<0.1: (hist2>0.1)', np.sum(hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / lumi)
        print('Removing dral<0.4: (hist>0.4)', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[40:]) / lumi)
    if printout and var=='photon_pt':# and args.private:
        if args.private: c = hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True) / hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)
        else: c = hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True) / hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)
        print('photon pT corrections', c)
    if printout and var=='dral':
        print('Removing lj0pt<10: hist1 / (hist>10) / xsec', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / 2.2471)
    if printout:
        print(f'{var} int(hist1)=', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values()), 'int(hist2)=', np.sum(sm.values()))
    s = np.sum(sm.values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True))
    #s = np.sum(sm.values()) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values())
    print(f'Ratio {s}')
    #hists1[var] *= 1.26
    #hists1[var] *= s
    #hists1[var] /= 0.3141 / 0.04406 # tWA ratio of MG / SM xsec
    #print(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True) / hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True))
    #hists1[var] *= 3
    #hists1[var] *= 10.7 / 1.513
    eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}]
    err = np.sqrt(eft.as_hist({}).variances()[()])#/np.sum(eft.values()[()])

    #eft.as_hist({}).plot1d(yerr=False, label='nanoAOD gen-level to SM', ax=ax, density=density)#, flow='show')
    st_pt = {'ctq1': 0.64, 'ctq8': 0.86, 'cQq81': 0.79, 'cQq83': -1.19, 'ctW': 1.49, 'cpQM': -1.02, 'ctZ': 0.94, 'ctG': 0.24, 'cQq13': 0.51, 'cQq11': -0.53, 'cpt': 1.5}
    #st_pt = {'ctq1': 0.64, 'ctq8': 0.86, 'cQq81': 0.79, 'cQq83': -1.19, 'ctW': 1.49, 'ctZ': 0.94, 'ctG': 0.24, 'cQq13': 0.51, 'cQq11': -0.53, 'cpt': 1.5}
    st_pt = {"cpQM": 62.000000, "ctW": 1.580000, "ctq1": 1.190000, "cQq81": 2.430000, "ctZ": 2.560000, "cQq83": 2.780000, "ctG": 0.310000, "ctq8": 2.020000, "cQq13": 1.340000, "cQq11": 1.350000, "cpt": 32.930000} # dim6top
    st_pt = {"cHQ1": 62.000000, "ctWRe": 1.580000, "ctj1": 1.190000, "cQj18": 2.430000, "ctBRe": 2.560000, "cQj38": 2.780000, "ctGRe": 0.310000, "ctj8": 2.020000, "cQj31": 1.340000, "cQj11": 1.350000, "cHt": 32.930000} # SMEFTsim
    #st_pt = {"ctW": 1.580000, "ctq1": 1.190000, "cQq81": 2.430000, "ctZ": 2.560000, "cQq83": 2.780000, "ctG": 0.310000, "ctq8": 2.020000, "cQq13": 1.340000, "cQq11": 1.350000}
    #st_pt = {"cHQ1": 62.000000, "ctWRe": 1.580000, "ctj1": 1.190000, "cQj18": 2.430000, "ctBRe": 2.560000, "cQj38": 2.780000, "ctGRe": 0.310000, "ctj8": 2.020000, "cQj31": 1.340000, "cQj11": 1.350000} # Run3 TTGamma missing "cHt" bunt run1 has it
    #wc = ["ctHRe", "cHQ1", "ctWRe", "ctBRe", "ctGRe", "cbWRe", "cHQ3", "cHtbRe", "cHt", "cQl3", "cQl1", "cQe", "ctl", "cte", "cleQt1Re", "cleQt3Re", "cQj31", "cQj38", "cQj11", "ctj1", "cQj18", "ctj8", "ctt", "cQQ1", "cQt1", "cQt8"]
    wc = ['cpt', 'ctp', 'ctt1', 'cptb', 'ctG', 'cQq11', 'cQl3i', 'ctlSi', 'ctq8', 'ctZ', 'cQq83', 'ctlTi', 'ctq1', 'cpQM', 'cQq13', 'cQt1', 'cbW', 'ctli', 'cQt8', 'ctei', 'cQq81', 'cQlMi', 'cQQ1', 'cpQ3', 'cQei', 'ctW']
    val = [-1.66, 100.0, 0.13, 100.0, -2.07, 0.3, 0.71, -1.35, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.05, -0.21, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

    wc = ['cQq11', 'cptb', 'ctlTi', 'ctZ', 'ctq1', 'cQl3i', 'cQlMi', 'cpQ3', 'ctW', 'ctp', 'cQq13', 'cbB', 'cbG', 'cpt', 'ctlSi', 'cbW', 'cpQM', 'ctq8', 'ctG', 'ctei', 'cQq81', 'cQei', 'ctli', 'cQq83']
    val = [-1.66, 1.0, 1.0, -2.07, 0.3, 0.71, -1.35, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, -0.21, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    wc = ['cpt', 'ctp', 'cptb', 'cQlMi', 'cQq81', 'cQq11', 'cQl3i', 'ctq8', 'ctlTi', 'ctq1', 'ctli', 'cQq13', 'cbW', 'cpQM', 'cpQ3', 'ctei', 'cQei', 'ctW', 'ctlSi', 'cQq83', 'ctZ', 'ctG']
    wc = ['ctW', 'ctG', 'ctZ']
    #wc = ['cbBRe', 'cQl1', 'cld', 'cQu8', 'cHQ1', 'cQu1', 'ctt', 'cleQt1Re33', 'cQd8', 'clu', 'ctWRe', 'cHt', 'clj1', 'cleQt3Re22', 'ctd8', 'cQt8', 'cleQt3Re33', 'ctj1', 'cleQt1Re11', 'cQj31', 'ctHRe', 'cQe', 'cQj18', 'ctj8', 'ctu1', 'ctGRe', 'cleQt1Re22', 'cHbox', 'cbGRe', 'cQQ1', 'cQl3', 'cQt1', 'ctu8', 'cQj38', 'ctl', 'ctBRe', 'cleQt3Re11', 'cHQ3', 'ctb8', 'cte', 'ctd1', 'cHtbRe', 'cbWRe', 'cQj11', 'cQb8', 'cQd1']
    #wc = ['cQq11', 'ctq8', 'ctq1', 'ctW', 'cQq81', 'cQq13', 'ctZ', 'cQq83', 'ctG']
    #wc = ['cbBRe', 'cQl1', 'cld', 'cQu8', 'cHQ1', 'cQu1', 'ctt', 'cleQt1Re33', 'cQd8', 'clu', 'ctWRe', 'cHt', 'clj1', 'cleQt3Re22', 'ctd8', 'cQt8', 'cleQt3Re33', 'ctj1', 'cleQt1Re11', 'cQj31', 'ctHRe', 'cQe', 'cQj18', 'ctj8', 'ctu1', 'ctGRe', 'cleQt1Re22', 'cHbox', 'cbGRe', 'cQQ1', 'cQl3', 'cQt1', 'ctu8', 'cQj38', 'ctl', 'ctBRe', 'cleQt3Re11', 'cHQ3', 'ctb8', 'cte', 'ctd1', 'cHtbRe', 'cbWRe', 'cQj11', 'cQb8', 'cQd1']

    if not wc:
        hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=False, label=args.str1 + ' @ SM', ax=ax, density=density, flow=flow)
    else:
        hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=False, label=args.str1 + ' @ SM', ax=ax, density=density, flow=flow)
    #hists1[var][{'process': [s for s in hists1[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=False, label='nanoAOD central pt', ax=ax, density=density)#, flow='show')

    st_pt = None

    if args.json != '':
        with open(args.json) as fin:
            print(f'Loading {args.json}')
            j = json.load(fin)
            wc = j['WCnames']
            if 'StPt' in j:
                st_pt = j['StPt']

    val = [1.0] * len(wc)

    if st_pt is None:
        st_pt = dict(zip(wc, val))
    if args.small:
        st_pt = {wc:(val if abs(val) < 100 else 0) for wc,val in st_pt.items()}

    print(f'Using {st_pt=}')
    eft_err = np.sqrt(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt).variances(flow=(flow=='show')))
    eft_err = False
    #if flow=='show': eft_err = eft_err[1:]
    hep.histplot(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt), label=f'{args.str1} pt.', ax=ax, density=density, flow=flow, ls='--', yerr=eft_err)
    if args.private and not args.skip: #FIXME
        hep.histplot(hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt), label=f'{str2} EFT pt.', ax=ax, density=density, flow=flow, yerr=False, ls='--')
    #eft_st = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt).values()
    #eft_err = np.sqrt(eft_st)/np.sum(eft_st)
    #eft_err = np.sqrt(eft_st)
    yerr = hists2[var][{'process': sum, 'channel': chan, 'systematic': 'nominal', 'appl': appl}].as_hist({}).values()[()]

    err = np.sqrt(sm.variances(flow=(flow=='show'))[()])/sm.values(flow=(flow=='show'))[()]
    if flow=='show': err = err[1:]
    eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    #eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    plt.sca(rax)
    (sm/sm).plot1d(yerr=err, ax=rax, flow=flow)
    norm = np.sum(sm.values()) / np.sum(eft.values())
    #norm = 1
    (eft/sm * norm).plot1d(yerr=False, ax=rax, flow=flow)

    ax2 = fig.add_axes([0.7, 0.55, 0.15, 0.15])
    eb1 = ax2.errorbar([1], 1, xerr=0.05, yerr=np.sqrt(np.sum(sm.values(flow=True)))/np.sum(sm.values(flow=True)))
    #eb1 = ax2.errorbar([1], np.sum(sm.values(flow=True)), xerr=0.05, yerr=np.sqrt(np.sum(sm.values(flow=True))))
    eft_sm_norm = np.sum(eft.values(flow=False)[()]) #/ sm_scale
    eb2 = ax2.errorbar([1], eft_sm_norm / np.sum(sm.values(flow=True)), xerr=0.05)
    plt.gca().set_xticks([])

    eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt)
    eft_err = np.sqrt(eft.variances(flow=(flow=='show')))/np.sum(eft.values(flow=(flow=='show')))
    if flow=='show': eft_err = eft_err[1:]
    norm = np.sum(sm.values()) / np.sum(eft.values())
    (eft/sm * norm).plot1d(yerr=eft_err, ax=rax, flow=flow, ls='--')

    eft_start_norm = np.sum(eft.values(flow=True)[()]) #/ sm_scale
    if 'fixed' in args.fin2:
        eb3 = ax2.errorbar([1], eft_start_norm / np.sum(sm.values(flow=True)), xerr=0.05, linestyle='--')

    if args.private and wc and not args.skip: #FIXME
        eft = hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt)
        norm = np.sum(sm.values()) / np.sum(eft.values())
        (eft/sm * norm).plot1d(yerr=False, ax=rax, flow=flow, ls='--')

    #ax = plt.gca()
    if density:
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 5e-2)
    rax.set_ylim(0.5, 1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Yield')
    var_label = var
    if '_pt' in var_label and 'lhe' in var_label:
        var_label = var_label.split('_')[1]
    elif '_pt' in var_label:
        var_label = var_label.split('_')[0]
    else:
        var_label = var_label.split('pt')[0]
    if norm == 1: rax.set_ylabel(r'$\frac{dN_{\rm{EFT}}}{d p_{\rm{T}}} / \frac{dN_{\rm{ref}}}{d p_{\rm{T}}}$')
    else: rax.set_ylabel(r'$(\frac{1}{N_{\rm{EFT}}} \frac{dN_{\rm{EFT}}}{d p_{\rm{T}}}) / (\frac{1}{N_{\rm{ref}}} \frac{dN_{\rm{ref}}}{d p_{\rm{T}}})$')
    #else: rax.set_ylabel(r'$(\frac{1}{N_{\rm{EFT}}} \frac{dN_{\rm{EFT}}}{d p_{\rm{T}} \gamma}) / (\frac{1}{N_{\rm{SM}}} \frac{dN_{\rm{SM}}}{d p_{\rm{T}} \gamma})$')
    plt.sca(ax)
    plt.legend()
    plt.show()
    user = os.getlogin()
    #plt.savefig('/afs/crc.nd.edu/user/{user[0]}/{user}/www/comp.png')
    plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/comp_{var}.png')
    plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/comp_{var}.pdf')
    plt.close()

if __name__ == '__main__':
    if args.var is None:
        plot('photon_pt', fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        ##plot('lhe_photon_pt', fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        #plot('lhe_t_pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('t_pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        ##plot('lhe_l0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('l0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('j0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('bj0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('lj0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('dral'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        #plot('dral_sec'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('draj'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('mll'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('njets'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        #plot('photon_eta'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
    else:
        plot(var        , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
