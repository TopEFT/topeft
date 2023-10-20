'''
Script for comparing two pkl files
example:
`python comp.py histos/newHist.pkl.gz histos/anatest25.pkl.gz -n | tee log_hist_ana`
-n tells it the first pkl file is using teh new histEFT based on scikithep hist
'''
import pickle
from coffea.hist import Bin
import gzip
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mplhep as hep
from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))
import numba
from numba.typed import Dict

BINNING = {
    "ptbl":    [0,100,200,400],
    "ht":      [0,300,500,800],
    "ljptsum": [0,400,600,1000],
    "ptz":     [0,200,300,400,500],
    "o0pt":    [0,100,200,400],
    "bl0pt":   [0,100,200,400],
    "l0pt":    [0,50,100,200],
    "lj0pt":   [0,150,250,500]
}


def comp(fin1, fin2, hists1, hists2, newHist):
    Hist = True if 'kmohrman' in fin2 else False # Official anatest25 was made _before_ the luminosity step was added
    if '_np' in fin1 and '_np' in fin2:
        for hname in hists2:
            h = hists2[hname]
            if h.empty(): continue
            if hname not in hists2:
                raise Exception(f'{hname} found in {fin1} but missing in {fin2}!')
            for proc in h.axis('sample').identifiers():
                proc = proc.name
                year = '20' + proc.split('UL')[1].split()[0]
                lumi = 1000.0*get_tc_param(f"lumi_{year}")
                yields1[proc] = {}
                yields2[proc] = {}
                for chan in h.axis('channel').identifiers():
                    chan = chan.name
                    yields1[proc][chan] = {}
                    yields2[proc][chan] = {}
                    for syst in h.axis('systematic').identifiers():
                        if 'nominal' not in syst.name: continue
                        syst = syst.name
                        h1 = hists1[hname]
                        h2 = hists2[hname]
                        if Hist and 'data' not in proc: h2.scale(lumi)
                        if newHist: v1 = np.sum(h1.integrate('process', proc).integrate('channel', chan).integrate('systematic', syst).eval({})[()])
                        else: v1 = np.sum(h1.integrate('sample', proc).integrate('channel', chan).integrate('systematic', syst).values()[()])
                        v2 = h2.integrate('sample', proc).integrate('channel', chan).integrate('systematic', syst).values()[()]
                        # Rebin old histogram to match new variable binning
                        if 'njets' in hname:
                            bins = BINNING[hname]
                            v2 = h2.rebin(hname, Bin(hname, h2.axis(hname).label, bins)).integrate('sample', proc).integrate('channel', chan).integrate('systematic', syst).values()[()]
                        yields1[proc][chan][syst] = v1
                        yields2[proc][chan][syst] = v2
                        if np.any((np.nan_to_num(np.abs(v1 - v2)/v1, 0) > 1e-3) & ((v1-v2) != 0)):
                            d = [str(round(x*100, 2))+'%' for x in np.nan_to_num((v1-v2)/v1, 0)]
                            print(f'Diff in {proc} {chan} {syst} greater than 1e-5\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n!')

    else:
        match = True
        for hname in hists2:
            if 'njets' not in hname: continue
            h = hists2[hname]
            if h.empty(): continue
            if hname not in hists2:
                raise Exception(f'{hname} found in {fin1} but missing in {fin2}!')
            for proc in h.axis('sample').identifiers():
                proc = proc.name
                if any(proc in p for p in ["2l_CRflip", "2l_CR", "3l_CR", "2los_CRtt", "2los_CRZ"]): continue
                year = '20' + proc.split('UL')[1].split()[0]
                lumi = 1000.0*get_tc_param(f"lumi_{year}")
                yields1[proc] = {}
                yields2[proc] = {}
                for chan in h.axis('channel').identifiers():
                    chan = chan.name
                    yields1[proc][chan] = {}
                    yields2[proc][chan] = {}
                    for appl in h.axis('appl').identifiers():
                        appl = appl.name
                        yields1[proc][chan][appl] = {}
                        yields2[proc][chan][appl] = {}
                        for syst in h.axis('systematic').identifiers():
                            syst = syst.name
                            if 'nominal' not in syst: continue
                            h1 = hists1[hname]
                            h2 = hists2[hname]
                            if Hist and 'data' not in proc: h2.scale(lumi)
                            if not any(appl in a.name for a in h2.integrate('sample', proc).integrate('channel', chan).axis('appl').identifiers()):
                                c = appl.split('_')[1]
                                print(f'Checking {appl} for {chan} {c}')
                                if c not in chan:
                                    print(f'Skipping {appl} for {chan} {c}')
                                    continue
                                print(f'{appl} not found!')
                                continue
                            if not any(appl in a for a in h1.integrate('process', proc).integrate('channel', chan).axes['appl']):
                                c = appl.split('_')[1]
                                if c not in proc: continue
                                print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                continue
                            if newHist:
                                if not h1.integrate('process', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).eval({}):
                                    c = appl.split('_')[1]
                                    print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                    continue
                            else:
                                if not h1.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values():
                                    c = appl.split('_')[1]
                                    print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                    continue
                            if newHist: v1 = h1.integrate('process', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).eval({})[()]
                            else: v1 = h1.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values(overflow='all')[()]
                            v2 = h2.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values(overflow='all')[()]
                            # Rebin old histogram to match new variable binning
                            if 'njets' not in hname:
                                bins = BINNING[hname]
                                v2 = h2.rebin(hname, Bin(hname, h2.axis(hname).label, bins)).integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values(overflow='all')[()]
                            yields1[proc][chan][appl][syst] = v1
                            yields2[proc][chan][appl][syst] = v2
                            if np.any((np.nan_to_num(np.abs(v1 - v2)/v1, 0) > 1e-3) & ((v1-v2) != 0)):
                                d = [str(round(x*100, 2))+'%' for x in np.nan_to_num((v1-v2)/v1, 0)]
                                print(f'Diff in {proc} {chan} {appl} {syst} greater than 1e-5\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n!')
                                match = False
    if match:
        print('All processes match!')

if __name__ == '__main__':
    #Load hists1 from pickle file created by TopCoffea
    hists1={}
    hists2={}
    yields1={}
    yields2={}

    parser = argparse.ArgumentParser(description='You can select which file to run over')
    parser.add_argument('fin1'   , default='analysis/topEFT/histos/mar03_central17_pdf_np.pkl.gz' , help = 'Variable to run over')
    parser.add_argument('fin2'   , default='analysis/topEFT/histos/mar03_central17_pdf_np.pkl.gz' , help = 'Variable to run over')
    parser.add_argument('--newHist', '-n', action='store_true', help='First file was made with the new histEFT')
    args    = parser.parse_args()
    fin1    = args.fin1
    fin2    = args.fin2
    newHist = args.newHist

    if ('_np' in fin1 and '_np' not in fin2) or ('_np' in fin2 and '_np' not in fin1):
        raise Exception("Looks like you're trying to compare a non-prompt subtracted file to one without!")

    with gzip.open(fin1) as fin:
        hin = pickle.load(fin)
        for k in hin.keys():
            if k in hists1: hists1[k]+=hin[k]
            else:               hists1[k]=hin[k]

    with gzip.open(fin2) as fin:
        hin = pickle.load(fin)
        for k in hin.keys():
            if k in hists2: hists2[k]+=hin[k]
            else:               hists2[k]=hin[k]

    comp(fin1, fin2, hists1, hists2, newHist)
