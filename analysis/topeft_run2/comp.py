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
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import mplhep as hep
import argparse
import time
import json
from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))
from topeft.modules.axes import info as axes_info

BINNING = {k: v['variable'] for k,v in axes_info.items() if 'variable' in v}

def comp(fin1, fin2, hists1, hists2, newHist1, newHist2, tolerance):
    fout = open('comp_log.txt', 'w')
    old_hist = True if 'kmohrman' in fin2 else False # Official anatest25 was made _before_ the luminosity step was added
    match = True
    if '_np' in fin1 and '_np' in fin2:
        for hname in hists2:
            if 'njets' in hname: continue
            h1 = hists1[hname]
            h2 = hists2[hname]
            if h2.empty(): continue
            if hname not in hists2:
                raise Exception(f'{hname} found in {fin1} but missing in {fin2}!')
            ax_proc = h2.axes['process'] if newHist2 else h2.axis('sample').identifiers()
            #for proc in h2.axes['process']:
            for proc in ax_proc:
                if not newHist2:
                    proc = proc.name
                #if proc not in h1.axes['process']:
                #    fout.write(f'{proc} missing from {fin1}!\n')
                #    continue
                if newHist1: h1_proc = h1.integrate('process', proc)
                else: h1_proc = h1.integrate('sample', proc)
                if newHist2: h2_proc = h2.integrate('process', proc)
                else: h2_proc = h2.integrate('sample', proc)
                if any(proc in p for p in ["2l_CRflip", "2l_CR", "3l_CR", "2los_CRtt", "2los_CRZ"]): continue
                year = '20' + proc.split('UL')[1].split()[0]
                lumi = 1000.0*get_tc_param(f"lumi_{year}")
                #if any(proc in p for p in ["ttllJet", "TTZToLLM1to10", "TTToSemiLeptonic", "TTTo2L2Nu"]):
                #    for p in ttll_lst:
                #        proc = proc.replace(p, 'ttll')
                if proc not in yields1:
                    yields1[proc] = {}
                if proc not in yields2:
                    yields2[proc] = {}
                #for chan in h2.axes['channel']:
                ax_chan = h2_proc.axes['channel'] if newHist2 else h2_proc.axis('channel').identifiers()
                for chan in ax_chan:
                    #chan = chan
                    if not newHist2:
                        chan = chan.name
                    #if '2lss_p_4' not in chan: continue
                    #if '2lss_4t_m_7j' not in chan: continue
                    #if 'flips' in proc and '2l' not in chan: continue
                    #if 'nonprompt' in proc and '4l' in chan: continue
                    if newHist1 and chan not in h1_proc.axes['channel']:
                        #print(f'Skipping {proc} {chan} {syst} - {chan} missing')
                        fout.write(f'Skipping {proc} {chan} - {chan} missing\n')
                    if newHist1: h1_chan = h1_proc.integrate('channel', chan)
                    else: h1_chan = h1_proc.integrate('channel', chan)
                    h2_chan = h2_proc.integrate('channel', chan)
                    if chan not in yields1[proc]:
                        yields1[proc][chan] = {}
                    if chan not in yields2[proc]:
                        yields2[proc][chan] = {}
                    #for syst in h2.axes['systematic']:
                    ax_syst = h2_chan.axes['systematic'] if newHist2 else h2_chan.axis('systematic').identifiers()
                    for syst in ax_syst:
                        if not newHist2:
                            syst = syst.name
                        if 'nominal' not in syst: continue
                        if 'data' in proc and syst != 'nominal': continue # Data-driven
                        if 'nonprompt' not in proc and 'FF' in syst: continue # Data-driven
                        if 'nonprompt' in proc and syst != 'nominal' and 'FF' not in syst: continue # Data-driven
                        if 'flips' in proc and syst != 'nominal': continue
                        if year not in syst and '_20' in syst: continue
                        if 'APV' in year and 'APV' not in syst and syst != 'nominal': continue
                        if not 'APV' in year and 'APV' in syst: continue
                        if newHist1 and syst not in h1_chan.axes['systematic']:
                            #print(f'Skipping {proc} {chan} {syst} - {chan} missing')
                            fout.write(f'Skipping {proc} {chan} {syst} - {chan} missing\n')
                            fout.write(f'{h1_chan.axes["systematic"]=}\n')
                            continue
                        #if 'JER_2016' not in syst: continue
                        if newHist1: h1_syst = h1_chan.integrate('systematic', syst)
                        else: h1_syst = h1_chan.integrate('systematic', syst)
                        h2_syst = h2_chan.integrate('systematic', syst)
                        #if old_hist and 'data' not in proc: h2.scale(lumi)
                        '''
                        if chan not in h1.integrate('process', proc).axes['channel']:
                            #print(f'Skipping {proc} {chan} {syst} - {chan} missing')
                            fout.write(f'Skipping {proc} {chan} {syst} - {chan} missing')
                            continue
                        if syst not in list(h1.axes['systematic']) or syst not in list(h1.integrate('process', proc).integrate('channel', chan).axes['systematic']):
                            #print(f'Skipping {proc} {chan} {syst} - {syst} missing')
                            fout.write(f'Skipping {proc} {chan} {syst} - {syst} missing')
                            continue
                        '''
                        for pt in [{}, {'ctW': 1}]:
                            if 'data' in proc and pt != {}:
                                continue
                            if newHist1: v1 = h1_syst.as_hist(pt).values(flow=True)[()]
                            else:
                                h1_syst.set_wilson_coefficients(**pt)
                                v1 = h1_syst.values(overflow='all')[()]
                            if newHist2: v2 = h2_syst.as_hist(pt).values(flow=True)[()]
                            else:
                                h2_syst.set_wilson_coefficients(**pt)
                                v2 = h2_syst.values(overflow='all')[()]
                            # Rebin old histogram to match new variable binning
                            if 'njets' not in hname and not newHist1:
                                v1norebin = h1_syst.values(overflow='all')[()]
                                bins = BINNING[hname]
                                v1rebin = h1_syst.rebin(hname, Bin(hname, h1_syst.axis(hname).label, bins))
                                v1 = v1rebin.values(overflow='all')[()]
                                #v1norebin = h1_syst.rebin(hname, Bin(hname, h1_syst.axis(hname).label, bins)).values(overflow='all')[()]
                            if 'njets' not in hname and not newHist2:
                                bins = BINNING[hname]
                                v2 = h2_syst.rebin(hname, Bin(hname, h2_syst.axis(hname).label, bins)).values(overflow='all')[()]
                            if old_hist and 'data' not in proc:
                                v2 = v2*lumi
                                #print(f'Scaled by {lumi}')
                                #fout.write(f'Scaled by {lumi}')
                            if pt == {}:
                                if syst not in yields1[proc][chan]:
                                    yields1[proc][chan][syst] = v1
                                #elif 'ttll' in proc:
                                #    print(proc, chan, syst, yields1[proc][chan][syst], v1)
                                #    yields1[proc][chan][syst] += v1
                                if syst not in yields2[proc][chan]:
                                    yields2[proc][chan][syst] = v2
                                #elif 'ttll' in proc:
                                #    yields2[proc][chan][syst] += v2
                            if syst == 'nominal' and '2lss_p_4' in chan:# and 'TTTo2L2Nu_centralUL18' in proc:
                            #if pt == {} and syst == 'nominal' and '2lss_p_4' in chan:# and 'TTTo2L2Nu_centralUL18' in proc:
                                if not newHist1:
                                    edg = h1_syst.axes()[0].edges()[:-1]
                                else:
                                    edg = h1_syst.axes[0].edges[:-1]
                                bins = BINNING[hname]#[:-1]
                                #if not newHist1 and 'TTTo2L2Nu_centralUL18' in proc: print(proc, 'full1', edg, v1norebin)
                                #if 'TTTo2L2Nu_centralUL18' in proc: print(proc, 'rebin', bins, v1)
                                #if not newHist1 and 'TTTo2L2Nu_centralUL18' in proc: print(proc, 'manre', bins, [v1norebin[0], sum(v1norebin[1:3]), sum(v1norebin[4:5]), sum(v1norebin[6:11])])
                                #if 'TTTo2L2Nu_centralUL18' in proc: print(proc, 'full2', edg, v2)
                                #if 'TTTo2L2Nu_centralUL18' in proc: print(proc, 'manre', bins, [sum(v2[0:3]), sum(v2[3:5]), sum(v2[5:11])])
                                if not newHist1: hep.histplot(h1_syst.to_hist(), label=fin1.split('/')[-1].split('_')[0], histtype='step', yerr=False, flow='show')
                                else: hep.histplot(h1_syst.as_hist({}), label=fin1.split('/')[-1].split('_')[0], histtype='step', yerr=False, flow='show')
                                if not newHist1: hep.histplot(v1rebin.to_hist(), label=fin1.split('/')[-1].split('_')[0] + ' rebin', histtype='step', yerr=False, flow='show')
                                #if not newHist1 hep.histplot(bins, [sum(v1norebin[0:3]), sum(v1norebin[3:5]), sum(v1norebin[5:11])], label=fin1.split('/')[-1].split('_')[0] + ' manual rebin', ls='-.', histtype='step', yerr=False, flow='show')
                                #hep.histplot([sum(v2[0:3]), sum(v2[3:5]), sum(v2[5:11])], bins=bins, label=fin1.split('/')[-1].split('_')[0] + ' manual rebin', ls='-.', histtype='step', yerr=False, flow='show')
                                edg = h2_syst.axes[0].edges
                                hep.histplot(h2_syst.as_hist(pt), label=fin2.split('/')[-1].split('_')[0], ls='--', histtype='step', yerr=False, flow='show')
                                #if 'regular' in fin2: hep.histplot(bins, [v2[0], sum(v2[1:3]), sum(v2[4:5]), sum(v2[6:11])], label=fin2.split('/')[-1].split('_')[0] + ' manual rebin', ls='--', histtype='step', yerr=False, flow='show')
                                plt.legend()
                                if pt == {}:
                                    tpt = 'sm'
                                else:
                                    tpt = [(k,str(v)) for k,v in pt.items()]
                                    tpt = [v for k in tpt for v in k]
                                    tpt = '_'.join(tpt)
                                plt.savefig(f'{proc}_{chan}_{syst}_{tpt}.png')
                                plt.close()

                            if np.any((np.nan_to_num(np.abs(v1 - v2)/v1, 0) > tolerance) & ((v1-v2) != 0)) and ('nonprompt' not in proc and pt != {}):
                                d = [str(round(x*100, 2))+'%' for x in np.nan_to_num((v1-v2)/v1, 0)]
                                #print(f'Diff in {proc} {chan} {syst} greater than {tolerance}!\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n')
                                fout.write(f'Diff in {proc} {chan} {syst} {pt} greater than {tolerance}!\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n')
                                match = False
                        yields1[proc][chan][syst] = list(yields1[proc][chan][syst])
                        yields2[proc][chan][syst] = list(yields2[proc][chan][syst])

        # Fold in other ttll processes
        for proc in yields1:
            for chan in yields1[proc]:
                for syst in yields1[proc][chan]:
                    if any(p in proc for p in ["TTZToLLM1to10", "TTToSemiLeptonic", "TTTo2L2Nu"]):
                        p_year = proc.split('UL')[1]
                        yields1['ttllJet_privateUL' + p_year][chan][syst] = list(np.array(yields1['ttllJet_privateUL' + p_year][chan][syst]) + np.array(yields1[proc][chan][syst]))
    else:
        for hname in hists2:
            if 'njets' in hname: continue
            #if 'njets' not in hname: continue
            h1 = hists1[hname]
            h2 = hists2[hname]
            if h2.empty(): continue
            if hname not in hists2:
                raise Exception(f'{hname} found in {fin1} but missing in {fin2}!')
            ax_proc = h2.axes['process'] if newHist2 else h2.axis('sample').identifiers()
            for proc in ax_proc:
                if not newHist2:
                    proc = proc.name
                if any(proc in p for p in ["2l_CRflip", "2l_CR", "3l_CR", "2los_CRtt", "2los_CRZ"]): continue
                year = '20' + proc.split('UL')[1].split()[0]
                lumi = 1000.0*get_tc_param(f"lumi_{year}")
                if newHist1: h1_proc = h1.integrate('process', proc)
                else: h1_proc = h1.integrate('sample', proc)
                if newHist2: h2_proc = h2.integrate('process', proc)
                else: h2_proc = h2.integrate('sample', proc)
                yields1[proc] = {}
                yields2[proc] = {}
                ax_chan = h2_proc.axes['channel'] if newHist2 else h2_proc.axis('channel').identifiers()
                for chan in ax_chan:
                    if not newHist2:
                        chan = chan.name
                    yields1[proc][chan] = {}
                    yields2[proc][chan] = {}
                    if newHist1: h1_chan = h1_proc.integrate('channel', chan)
                    else: h1_chan = h1_proc.integrate('channel', chan)
                    h2_chan = h2_proc.integrate('channel', chan)
                    ax_appl = h2_chan.axes['appl'] if newHist2 else h2_chan.axis('appl').identifiers()
                    for appl in ax_appl:
                        if not newHist2:
                            appl = appl.name
                        if appl not in h1_chan.axes['appl']:
                            #print(f'Skipping {proc} {chan} {syst} - {chan} missing')
                            fout.write(f'Skipping {proc} {chan} {appl} - {appl} missing\n')
                            continue
                        yields1[proc][chan][appl] = {}
                        yields2[proc][chan][appl] = {}
                        if newHist1: h1_appl = h1_chan.integrate('appl', appl)
                        else: h1_appl = h1_chan.integrate('appl', appl)
                        h2_appl = h2_chan.integrate('appl', appl)
                        ax_syst = h2_chan.axes['systematic'] if newHist2 else h2_chan.axis('systematic').identifiers()
                        for syst in ax_syst:
                            if not newHist2:
                                syst = syst.name
                            #if 'nominal' not in syst: continue
                            if syst not in h1_chan.axes['systematic']:
                                #print(f'Skipping {proc} {chan} {syst} - {chan} missing')
                                fout.write(f'Skipping {proc} {chan} {syst} - {chan} missing\n')
                                continue
                            if newHist1: h1_syst = h1_appl.integrate('systematic', syst)
                            else: h1_syst = h1_appl.integrate('systematic', syst)
                            h2_syst = h2_appl.integrate('systematic', syst)
                            #if old_hist and 'data' not in proc: h2.scale(lumi)
                            #if not any(appl in a.name for a in h2_syst.integrate('sample', proc).integrate('channel', chan).axis('appl').identifiers()):
                            #    c = appl.split('_')[1]
                            #    #print(f'Checking {appl} for {chan} {c}')
                            #    fout.write(f'Checking {appl} for {chan} {c}\n')
                            #    if c not in chan:
                            #        #print(f'Skipping {appl} for {chan} {c}')
                            #        fout.write(f'Skipping {appl} for {chan} {c}\n')
                            #        continue
                            #    #print(f'{appl} not found!')
                            #    fout.write(f'{appl} not found!\n')
                            #    continue
                            #if not any(appl in a for a in h1.integrate('process', proc).integrate('channel', chan).axes['appl']):
                            #    c = appl.split('_')[1]
                            #    if c not in proc: continue
                            #    #print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                            #    fout.write(f'Skipping {proc} {chan} {c} {appl} {syst}\n')
                            #    continue
                            if newHist1:
                                if not h1.integrate('process', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).eval({}):
                                    c = appl.split('_')[1]
                                    #print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                    fout.write(f'Skipping {proc} {chan} {c} {appl} {syst}\n')
                                    continue
                            else:
                                if not h1.integrate('sample', proc).integrate('channel', chan).integrate('appl', appl).integrate('systematic', syst).values():
                                    c = appl.split('_')[1]
                                    #print(f'Skipping {proc} {chan} {c} {appl} {syst}')
                                    fout.write(f'Skipping {proc} {chan} {c} {appl} {syst}\n')
                                    continue
                            if newHist1: v1 = h1_syst.as_hist(pt).values(flow=True)[()]
                            else: v1 = h1_syst.values(overflow='all')[()]
                            if newHist2: v2 = h2_syst.as_hist(pt).values(flow=True)[()]
                            else: v2 = h2_syst.values(overflow='all')[()]
                            # Rebin old histogram to match new variable binning
                            if 'njets' not in hname and not newHist1:
                                bins = BINNING[hname]
                                v1 = h1_syst.rebin(hname, Bin(hname, h1_syst.axis(hname).label, bins)).values(overflow='all')[()]
                            if 'njets' not in hname and not newHist2:
                                bins = BINNING[hname]
                                v2 = h2_syst.rebin(hname, Bin(hname, h2_syst.axis(hname).label, bins)).values(overflow='all')[()]
                            if old_hist and 'data' not in proc:
                                v2 = v2*lumi
                                #print(f'Scaled {proc} {chan} {appl} {syst} by {lumi}')
                                #fout.write(f'Scaled {proc} {chan} {appl} {syst} by {lumi}')
                            yields1[proc][chan][appl][syst] = v1
                            yields2[proc][chan][appl][syst] = v2
                            if np.any((np.nan_to_num(np.abs(v1 - v2)/v1, 0) > tolerance) & ((v1-v2) != 0)):
                                d = [str(round(x*100, 2))+'%' for x in np.nan_to_num((v1-v2)/v1, 0)]
                                #print(f'Diff in {proc} {chan} {appl} {syst} greater than {tolerance}\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n!')
                                fout.write(f'Diff in {proc} {chan} {appl} {syst} greater than {tolerance}\n{v1}\n{v2}\n{v1-v2}\n{d}\n\n!')
                                match = False
                            yields1[proc][chan][appl][syst] = list(v1)
                            yields2[proc][chan][appl][syst] = list(v2)
    with open(fin1.split('/')[-1].replace('pkl.gz', 'json'), "w") as out_file:
        json.dump(yields1, out_file, indent=4)
    with open(fin2.split('/')[-1].replace('pkl.gz', 'json'), "w") as out_file:
        json.dump(yields2, out_file, indent=4)
    if match:
        print('All processes match!')
        fout.write('All processes match!')
    fout.close()

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
    parser.add_argument('--tolerance', '-t', default='1e-3', help='Tolerance for warnings')
    args    = parser.parse_args()
    fin1    = args.fin1
    fin2    = args.fin2
    newHist1 = args.newHist1
    newHist2 = args.newHist2
    tolerance = float(args.tolerance)

    if ('_np' in fin1 and '_np' not in fin2) or ('_np' in fin2 and '_np' not in fin1):
        raise Exception("Looks like you're trying to compare a non-prompt subtracted file to one without!")

    tic = time.time()
    with gzip.open(fin1) as fin:
        hin = pickle.load(fin)
        for k in hin.keys():
            if k in hists1: hists1[k]+=hin[k]
            else:               hists1[k]=hin[k]
    dt = time.time() - tic
    print(f'Opened {fin1}: {dt:.2f} s')

    tic = time.time()
    with gzip.open(fin2) as fin:
        hin = pickle.load(fin)
        for k in hin.keys():
            if k in hists2: hists2[k]+=hin[k]
            else:               hists2[k]=hin[k]
    dt = time.time() - tic
    print(f'Opened {fin2}: {dt:.2f} s')

    comp(fin1, fin2, hists1, hists2, newHist1, newHist2, tolerance)
