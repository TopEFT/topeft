'''
This script produces plots comparing two input pkl files made from `gen_processor.py`.
The minimum set of inputs is the paths to each pkl file and the json for one of them.
If the json file contains the starting points ("StPt") it will draw a green curve reweighted to that value,
otherwise it will set all WCs to 1 for the green curve.
You can get the official starting points from the GridpackGeneration repo
e.g. Run 3 dim6top: https://github.com/TopEFT/GridpackGeneration/blob/dim6top_run3/mcgeneration/scanfiles/startpts_scale_by_1p1_Run3_dim6top.json

Variables:
You can specify a single variable using `--var <your variable>`,
otherwise the script will loop over _all_ variables listed at the bottom.

Comparing two private EFT files:
By default the script expects the first pkl file to be an EFT file and the second to be a SM central file.
If you pass the argument `--private` it will understand that the _second_ pkl file is also a private sample.
There is a special flag `--skip` which tells the script _not_ to draw a reweighted histogram for the second pkl file.
This is useful when comparing Run 3 to Run 2 (at the SM) since some of the WC names have changed.
If you pass the argument `--central` it will assume _both_ pkl files are central samples.
By default, any missing WCs in the starting point are set to `100`, since this is what we do when making the gridpacks.
Use `--zero` to keep these missing WCs fixded to `0`.

The flag `--abs` will not normalize the ratios.


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
import math
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
parser.add_argument('--central', action='store_true' , help = 'Use central key for first hist')
parser.add_argument('--abs'    , action='store_true' , help = 'Use absolute scale for ratio')
parser.add_argument('--skip'   , action='store_true' , help = 'Skip plotting EFT points')
parser.add_argument('--zero'   , action='store_true' , help = 'Set missing WCs to 0')
parser.add_argument('json'     , default='', help = 'Json file(s) containing files and metadata')
parser.add_argument('--small'   , action='store_true', help = 'Remove all |WCs| >100')
parser.add_argument('--no-lumi' , action='store_true', help = 'Don\t rescale the lumi')
parser.add_argument('--info'    , action='store_true', help = 'Print summary info')
parser.add_argument('--start'   , action='store_true', help = 'Use starting point')
args  = parser.parse_args()
fin1   = args.fin1
fin2   = args.fin2

assert not args.private or not args.central, 'Please use either `--private` OR `--central`, not both'

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

    hep.style.use("CMS")

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12,12),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

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
    if 'fixed' in args.fin2:
        str2 = 'nanoGEN gen-level fixed +/-1'

    lumi = 1000*41.48

    # Infer lumi from file name
    def extract_year(s):
        match = re.search(r'(?:UL(\d{2})|(\d{4}))(?:[A-Za-z]+)?', s)
        if match:
            if match.group(1):  # Matched ULxx
                return "20" + match.group(1)  # Convert UL17 -> 2017
            if match.group(0):  # Matched ULxx
                return match.group(0)
            else:  # Matched yyyy
                return match.group(2)
        return None
    lumi1, lumi2 = 0, 0
    year1 = extract_year(args.fin1)
    year2 = extract_year(args.fin2)
    if year1 is not None:
        lumi1 = 1000.0*get_tc_param(f"lumi_{year1}")
    if year2 is not None:
        lumi2 = 1000.0*get_tc_param(f"lumi_{year2}")
    if lumi1 > 0 and lumi2 > 0 and not args.no_lumi:
        if lumi1 > lumi2:
            print(f'Scaling {args.fin2} from {round(lumi2/1000)} pb^-1 to {round(lumi1/1000)} pb^-1')
            hists2[var] *= lumi1/lumi2
        elif lumi2 > lumi1:
            print(f'Scaling {args.fin1} from {round(lumi1/1000)} pb^-1 to {round(lumi2/1000)} pb^-1')
            hists1[var] *= lumi2/lumi1
    elif not args.density:
        print(f'\n\nWARNING: Could not infer the luminosity.\n         Make sure {args.fin1} and {args.fin2} are normalized the same!\n\n')

    if args.private: sm = hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    else: sm = hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    err = np.sqrt(sm.variances()[()])/np.sum(sm.values()[()])
    err = np.sqrt(sm.variances()[()])#/np.sum(sm.values()[()])
    err = np.sqrt(sm.variances(flow=(flow=='show'))[()])#/np.sum(sm.values()[()])
    if flow=='show':
        err = err[1:]
    if not args.private: hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=err, label=str2 + ' SM', ax=ax, density=density, flow=flow)
    else: hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=err, label=str2 + ' @ SM', ax=ax, density=density, flow=flow)

    global printout
    if printout and var=='dral':
        printout = False
        print('hist1 / lumi', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / lumi)
        print('hist1 / lumi / xsec / k', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / 2.2471 / lumi)
        print('hist1 / lumi / 3.75', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / 3.75 / lumi)
        print('hist1 / lumi / xsec', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / 1.513 / lumi)
        print('hist2 / lumi / xsec / k', np.sum(sm.values(flow=True)) / 2.2471 / lumi)
        print('hist2 / lumi / xsec', np.sum(sm.values(flow=True)) / 1.513 / lumi)
        print('Removing dral<0.1: hist1 / (hist>0.1) / xsec', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / 2.2471)
        print('Removing dral<0.1: hist1 / (hist>0.1)', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]))
        print('Removing dral<0.1: (hist>0.1)', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / lumi)
        print('Removing dral<0.1: (hist2>0.1)/(hist1>0.1)', np.sum(hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]))
        print('Removing dral<0.1: (hist2>0.1)', np.sum(hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / lumi)
        print('Removing dral<0.4: (hist>0.4)', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[40:]) / lumi)
    if printout and var=='photon_pt':# and args.private:
        if args.private: c = hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True) / hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)
        else: c = hists2[var][{'process': [s for s in hists2[var].axes['process'] if 'central' in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True) / hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)
        print('photon pT corrections (shape only)', c / (np.sum(hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True))))
    if printout and var=='dral':
        print('Removing lj0pt<10: hist1 / (hist>10) / xsec', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True)[10:]) / 2.2471)
    if printout:
        print(f'{var} int(hist1)=', np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values()), 'int(hist2)=', np.sum(sm.values()))
    s = np.sum(sm.values(flow=True)) / np.sum(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).values(flow=True))
    print(f'Ratio {s}')
    eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}]
    err = np.sqrt(eft.as_hist({}).variances()[()])#/np.sum(eft.values()[()])

    max_lumi = max(lumi1, lumi2)
    if args.info:
        tab = '    '
        print('\n\nSummary information:')
        if args.no_lumi:
            print(f'{tab}{args.fin1} with lumi removed {np.round(np.sum(eft.eval({})[()]) / lumi1, 3)}')
        else:
            print(f'{tab}{args.fin1} with lumi removed {np.round(np.sum(eft.eval({})[()]) / max_lumi, 3)}')
        if args.no_lumi:
            print(f'{tab}{args.fin2} with lumi removed {np.round(np.sum(sm.values(flow=True)[()]) / lumi2, 3)}')
        else:
            print(f'{tab}{args.fin2} with lumi removed {np.round(np.sum(sm.values(flow=True)[()]) / max_lumi, 3)}')
        print('\n\n')

    hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({}).plot1d(yerr=False, label=args.str1 + ' @ SM', ax=ax, density=density, flow=flow)

    st_pt = None

    if args.json != '':
        with open(args.json) as fin:
            print(f'Loading {args.json}')
            j = json.load(fin)
            wc = j['WCnames']
            if 'StPt' in j:
                st_pt = j['StPt']
    for wc in hists1[var].wc_names:
        if st_pt is not None and wc not in st_pt and not args.zero:
            st_pt[wc] = 100.

    if not args.start:
        val = [1.0] * len(wc)

    lab = 'st pt.'
    if st_pt is None:
        lab = 'pt.'
        wcs = hists1[var].wc_names
        val = [1.0] * len(wcs)
        st_pt = dict(zip(wcs, val))
    if args.small:
        lab = 'non-SM pt.'
        st_pt = {wc:(val*.1 if abs(val) < 100 else 0) for wc,val in st_pt.items()}

    if not args.central: print(f'Using {st_pt=}')
    if not args.central and not args.skip: eft_err = np.sqrt(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt).variances(flow=(flow=='show')))
    eft_err = False
    #if flow=='show': eft_err = eft_err[1:] # FIXME fix overflow, some vars need the extra bin
    if not args.central and not args.skip: hep.histplot(hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt), label=f'{args.str1} {lab}', ax=ax, density=density, flow=flow, ls='--', yerr=eft_err)
    if args.private and not args.skip and not args.central: #FIXME
        hep.histplot(hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt), label=f'{str2} {lab}', ax=ax, density=density, flow=flow, yerr=False, ls='-.')
    yerr = hists2[var][{'process': sum, 'channel': chan, 'systematic': 'nominal', 'appl': appl}].as_hist({}).values()[()]

    err = np.sqrt(sm.variances(flow=(flow=='show'))[()])/sm.values(flow=(flow=='show'))[()]
    if flow=='show': err = err[1:]
    eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist({})
    plt.sca(rax)
    (sm/sm).plot1d(yerr=err, ax=rax, flow=flow)
    norm = np.sum(sm.values()) / np.sum(eft.values())
    if args.abs: norm = 1
    (eft/sm * norm).plot1d(yerr=False, ax=rax, flow=flow)

    ax2 = fig.add_axes([0.7, 0.55, 0.15, 0.15])
    eb1 = ax2.errorbar([1], 1, xerr=0.05, yerr=np.sqrt(np.sum(sm.values(flow=True)))/np.sum(sm.values(flow=True)))
    eft_sm_norm = np.sum(eft.values(flow=True)[()]) #/ sm_scale
    eb2 = ax2.errorbar([1], eft_sm_norm / np.sum(sm.values(flow=True)), xerr=0.05)
    plt.gca().set_xticks([])

    if not args.central and not args.skip: eft = hists1[var][{'process': [s for s in hists1[var].axes['process'] if proc in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt)
    eft_err = np.sqrt(eft.variances(flow=(flow=='show')))/np.sum(eft.values(flow=(flow=='show')))
    if flow=='show': eft_err = eft_err[1:]
    norm = np.sum(sm.values()) / np.sum(eft.values())
    if args.abs: norm = 1
    if not args.central and not args.skip: (eft/sm * norm).plot1d(yerr=eft_err, ax=rax, flow=flow, ls='-.')

    eft_start_norm = np.sum(eft.values(flow=True)[()]) #/ sm_scale
    if args.abs: norm = 1
    if 'fixed' in args.fin2:
        eb3 = ax2.errorbar([1], eft_start_norm / np.sum(sm.values(flow=True)), xerr=0.05, linestyle='--')

    if args.private and wc and not args.skip and not args.central: #FIXME
        eft = hists2[var][{'process': [s for s in hists2[var].axes['process'] if proc2 in s], 'channel': chan, 'systematic': 'nominal', 'appl': appl}][{'process': sum}].as_hist(st_pt)
        norm = np.sum(sm.values()) / np.sum(eft.values())
        if args.abs: norm = 1
        (eft/sm * norm).plot1d(yerr=False, ax=rax, flow=flow, ls='--')

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
    if args.abs: rax.set_ylabel(r'$\frac{dN_{\rm{EFT}}}{d p_{\rm{T}}} / \frac{dN_{\rm{ref}}}{d p_{\rm{T}}}$')
    else: rax.set_ylabel(r'$(\frac{1}{N_{\rm{EFT}}} \frac{dN_{\rm{EFT}}}{d p_{\rm{T}}}) / (\frac{1}{N_{\rm{ref}}} \frac{dN_{\rm{ref}}}{d p_{\rm{T}}})$')
    plt.sca(ax)
    plt.legend()
    plt.show()
    user = os.getlogin()
    com = '13' if 'UL' in args.fin1 else '13.6'
    com = '13.6 vs 13' if 'UL' in args.fin2 and 'UL' not in args.fin1 else com
    n_dec = 3 - math.ceil(np.log10(max_lumi/1000.))
    hep.cms.label(lumi=np.round(max_lumi/1000., n_dec), com=com)
    plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/comp_{var}.png')
    plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/comp_{var}.pdf')
    plt.close()

if __name__ == '__main__':
    if args.var is None:
        plot('photon_pt', fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('lhe_t_pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('t_pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('l0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('j0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('bj0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('lj0pt'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('dral'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('draj'     , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('mll'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('invm'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('invm_ttX'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('invm_tX'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('invm_4t'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
        plot('njets'      , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
    else:
        plot(var        , fin1=fin1, fin2=fin2, flow=flow, private=args.private, hists1=hists1, hists2=hists2)
