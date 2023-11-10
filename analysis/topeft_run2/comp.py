'''
Script for comparing two pkl files
example:
`python comp.py histos/example_name_np.pkl.gz ~/../../k/kmohrman/Public/fullR2_files/pkl_files/feb15_fullRun2_withSys_anatest25_np.pkl.gz --newHist1`
--newHist1 tells it the first pkl file is using the new histEFT based on scikithep hist
--newHist2 would tell it the second pkl file is using the new histEFT based on scikithep hist
'''
import pickle
from coffea.hist import Bin
import gzip
import numpy as np
import argparse
import json
from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))
from topeft.modules.axes import info as axes_info

BINNING = {k: v['variable'] for k,v in axes_info.items() if 'variable' in v}

def comp(fin1, fin2, hists1, hists2, newHist1, newHist2):
    old_hist = True if 'kmohrman' in fin2 else False # Official anatest25 was made _before_ the luminosity step was added
    match = True
    if '_np' in fin1 and '_np' in fin2:
        for hname in hists2:
            if 'njets' in hname: continue
            h = hists2[hname]
            if h.empty(): continue
            if hname not in hists2:
                raise Exception(f'{hname} found in {fin1} but missing in {fin2}!')
            ax_proc = h.axes['process'] if newHist2 else h.axis('sample').identifiers()
            #for proc in h.axes['process']:
            for proc in ax_proc:
                if not newHist2:
                    proc = proc.name
                if any(proc in p for p in ["2l_CRflip", "2l_CR", "3l_CR", "2los_CRtt", "2los_CRZ"]): continue
                year = '20' + proc.split('UL')[1].split()[0]
                lumi = 1000.0*get_tc_param(f"lumi_{year}")
                yields1[proc] = {}
                yields2[proc] = {}
                #for chan in h.axes['channel']:
                ax_chan = h.axes['channel'] if newHist2 else h.axis('channel').identifiers()
                for chan in ax_chan:
                    #chan = chan
                    if not newHist2:
                        chan = chan.name
                    #if '2lss_4t_m_7j' not in chan: continue
                    if 'flips' in proc and '2l' not in chan: continue
                    if 'nonprompt' in proc and '4l' in chan: continue
                    yields1[proc][chan] = {}
                    yields2[proc][chan] = {}
                    #for syst in h.axes['systematic']:
                    ax_syst = h.axes['systematic'] if newHist2 else h.axis('systematic').identifiers()
                    for syst in ax_syst:
                        #if 'nominal' not in syst.name: continue
                        if 'data' in proc and syst != 'nominal': continue # Data-driven
                        if 'nonprompt' in proc and syst != 'nominal': continue # Data-driven
                        if not newHist2:
                            syst = syst.name
                        if year not in syst and '_20' in syst: continue
                        if 'APV' in year and not 'APV' in syst: continue
                        if not 'APV' in year and 'APV' in syst: continue
                        if 'JER_2016' not in syst: continue
                        h1 = hists1[hname]
                        h2 = hists2[hname]
                        #if old_hist and 'data' not in proc: h2.scale(lumi)
                        if chan not in h1.integrate('process', proc).axes['channel']:
                            print(f'Skipping {proc} {chan} {syst} - {chan} missing')
                            continue
                        if syst not in list(h1.axes['systematic']) or syst not in list(h1.integrate('process', proc).integrate('channel', chan).axes['systematic']):
                            print(f'Skipping {proc} {chan} {syst} - {syst} missing')
                            continue
                        if newHist1: v1 = h1.integrate('process', proc).integrate('channel', chan).integrate('systematic', syst).eval({})[()]
                        else: v1 = h1.integrate('sample', proc).integrate('channel', chan).integrate('systematic', syst).values(overflow='all')[()]
                        if newHist2: v2 = h2.integrate('process', proc).integrate('channel', chan).integrate('systematic', syst).eval({})[()]
                        else: v2 = h2.integrate('sample', proc).integrate('channel', chan).integrate('systematic', syst).values(overflow='all')[()]
                        # Rebin old histogram to match new variable binning
                        if 'njets' not in hname and not newHist2:
                            bins = BINNING[hname]
                            v2 = h2.rebin(hname, Bin(hname, h2.axis(hname).label, bins)).integrate('sample', proc).integrate('channel', chan).integrate('systematic', syst).values(overflow='all')[()]
                        if old_hist and 'data' not in proc: v2 = v2*lumi
                        yields1[proc][chan][syst] = v1
                        yields2[proc][chan][syst] = v2
                        if np.any((np.nan_to_num(np.abs(v1 - v2)/v1, 0) > 1e0) & ((v1-v2) != 0)):
                            d = [str(round(x*100, 2))+'%' for x in np.nan_to_num((v1-v2)/v1, 0)]
                            print(f'Diff in {proc} {chan} {syst} greater than 1e-5!\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n')
                            match = False

                        yields1[proc][chan][syst] = list(v1)
                        yields2[proc][chan][syst] = list(v2)
    else:
        for hname in hists2:
            if 'njets' in hname: continue
            #if 'njets' not in hname: continue
            h = hists2[hname]
            if h.empty(): continue
            if hname not in hists2:
                raise Exception(f'{hname} found in {fin1} but missing in {fin2}!')
            for proc in h.axis('sample').identifiers():
                if not newHist2:
                    proc = proc.name
                if any(proc in p for p in ["2l_CRflip", "2l_CR", "3l_CR", "2los_CRtt", "2los_CRZ"]): continue
                year = '20' + proc.split('UL')[1].split()[0]
                lumi = 1000.0*get_tc_param(f"lumi_{year}")
                yields1[proc] = {}
                yields2[proc] = {}
                for chan in h.axis('channel').identifiers():
                    if not newHist2:
                        chan = chan.name
                    yields1[proc][chan] = {}
                    yields2[proc][chan] = {}
                    for appl in h.axis('appl').identifiers():
                        if not newHist2:
                            appl = appl.name
                        yields1[proc][chan][appl] = {}
                        yields2[proc][chan][appl] = {}
                        for syst in h.axis('systematic').identifiers():
                            syst = syst.name
                            if 'nominal' not in syst: continue
                            h1 = hists1[hname]
                            h2 = hists2[hname]
                            #if old_hist and 'data' not in proc: h2.scale(lumi)
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
                            if newHist1:
                                if not h1.integrate('process', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).eval({}):
                                    c = appl.split('_')[1]
                                    print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                    continue
                            else:
                                if not h1.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values():
                                    c = appl.split('_')[1]
                                    print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                    continue
                            if newHist1: v1 = h1.integrate('process', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).eval({})[()]
                            else: v1 = h1.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values(overflow='all')[()]
                            v2 = h2.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values(overflow='all')[()]
                            # Rebin old histogram to match new variable binning
                            if 'njets' not in hname:
                                bins = BINNING[hname]
                                v2 = h2.rebin(hname, Bin(hname, h2.axis(hname).label, bins)).integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values(overflow='all')[()]
                            if old_hist and 'data' not in proc: v2*lumi
                            yields1[proc][chan][appl][syst] = v1
                            yields2[proc][chan][appl][syst] = v2
                            if np.any((np.nan_to_num(np.abs(v1 - v2)/v1, 0) > 1e-3) & ((v1-v2) != 0)):
                                d = [str(round(x*100, 2))+'%' for x in np.nan_to_num((v1-v2)/v1, 0)]
                                print(f'Diff in {proc} {chan} {appl} {syst} greater than 1e-5\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n!')
                                match = False
                            yields1[proc][chan][appl][syst] = list(v1)
                            yields2[proc][chan][appl][syst] = list(v2)
    with open(fin1.split('/')[-1].replace('pkl.gz', 'json'), "w") as out_file:
        json.dump(yields1, out_file, indent=4)
    with open(fin2.split('/')[-1].replace('pkl.gz', 'json'), "w") as out_file:
        json.dump(yields2, out_file, indent=4)
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
    parser.add_argument('--newHist1', action='store_true', help='First file was made with the new histEFT')
    parser.add_argument('--newHist2', action='store_true', help='Second file was made with the new histEFT')
    args    = parser.parse_args()
    fin1    = args.fin1
    fin2    = args.fin2
    newHist1 = args.newHist1
    newHist2 = args.newHist2

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

    comp(fin1, fin2, hists1, hists2, newHist1, newHist2)
