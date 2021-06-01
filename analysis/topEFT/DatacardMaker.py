import pickle
import gzip
from coffea import hist
import topcoffea.modules.HistEFT
from topcoffea.modules.WCPoint import WCPoint
import uproot3
import numpy as np
import os

from ROOT import TFile, TH1D

class HistoReader():
    def __init__(self, infile='', central=False):
        self.hists = {}
        self.rename = {'tZq': 'tllq', 'ttZ': 'ttll', 'ttW': 'ttlnu', 'ttGJets': 'convs', 'WZ': 'Diboson', 'WWW': 'Triboson', 'ttHnobb': 'ttH'} #Used to rename things like ttZ to ttll and ttHnobb to ttH
        self.processDic = {
          'Nonprompt' : 'TTTo2L2Nu,tW_noFullHad, tbarW_noFullHad, WJetsToLNu_MLM, WWTo2L2Nu',
          'DY' : 'DYJetsToLL_M_10to50_MLM, DYJetsToLL_M_50_a',
          'Other': 'WWW,WZG,WWZ,WZZ,ZZZ,tttt,ttWW,ttWZ,ttZH,ttZZ,ttHH,tZq,TTG',
          'WZ' : 'WZTo2L2Q,WZTo3LNu',
          'ZZ' : 'ZZTo2L2Nu,ZZTo2L2Q,ZZTo4L',
          'ttW': 'TTWJetsToLNu',
          'ttZ': 'TTZToLL_M_1to10,TTZToLLNuNu_M_10_a',
          'ttH' : 'ttHnobb,tHq',
          'data' : 'EGamma, SingleMuon, DoubleMuon',
        }
        self.bkglist = ['Nonprompt', 'Other', 'DY',  'ttH', 'WZ', 'ZZ', 'ttZ', 'ttW']
        self.coeffs = ['ctW', 'ctp', 'cpQM', 'ctli', 'cQei', 'ctZ', 'cQlMi', 'cQl3i', 'ctG', 'ctlTi', 'cbW', 'cpQ3', 'ctei', 'cpt', 'ctlSi', 'cptb','cQq13','cQq83','cQq11','ctq1','cQq81','ctq8']
        self.ch2lss = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
        self.ch3l = ['eemSSoffZ', 'mmeSSoffZ', 'eeeSSoffZ', 'mmmSSoffZ']
        self.ch3lsfz = ['eemSSonZ', 'mmeSSonZ', 'eeeSSonZ', 'mmmSSonZ', 'mmmSSoffZ']
        self.levels = ['base', '2jets', '4jets', '4j1b', '4j2b']
        self.channels = {'2lss': self.ch2lss, '3l': self.ch3l, '3l_sfz': self.ch3lsfz, '4l': '4l'}
        self.outf = "EFT_MultiDim_Datacard_combine.txt"
        self.fin = infile
        self.var = ['njets', 'ht']
        self.tolerance = 0.0001


    def read(self):
        '''
        Load pickle file into hist dictionary
        '''
        print(f'Loading {self.fin}')
        with gzip.open(self.fin) as fin:
            self.hists = pickle.load(fin)
        self.coeffs = self.hists['njets']._wcnames
        self.coeffs = ['cpt', 'ctp', 'cptb', 'cQlMi', 'cQq81', 'cQq11', 'cQl3i', 'ctq8', 'ctlTi', 'ctq1', 'ctli', 'cQq13', 'cbW', 'cpQM', 'cpQ3', 'ctei', 'cQei', 'ctW', 'ctlSi', 'cQq83', 'ctZ', 'ctG']

        #Get list of samples and cut levels from histograms
        self.samples = list({k[0]:0 for k in self.hists['njets'].values().keys()})
        self.levels = list({k[2]:0 for k in self.hists['njets'].values().keys()})
        self.charge = list({k[3]:0 for k in self.hists['njets'].values().keys()})
        self.nbjets = list({k[4]:0 for k in self.hists['njets'].values().keys()})
        self.syst = list({k[5]:0 for k in self.hists['njets'].values().keys()})
        self.hsow = self.hists['SumOfEFTweights']
        self.hsow = self.hsow.sum('sample')
        self.hsow.set_wilson_coefficients(np.zeros(self.hsow._nwc))
        self.smsow = self.hsow.values()[()][0]
        self.lumi = 1000*59.7

    def relish(self):
        '''
        Create temporary ROOT files from pickle file
        Files are stored in the ``histos`` dir and start with ``tmp_``
        These files are deleted in the next stage
        '''
        print(f'Making relish from the pickle file')
        print('.', end='', flush=True)
        wcs = self.buildWCString(self.coeffs)
        print('.', end='', flush=True)
        #Integrate out channels
        print('.', end='', flush=True)
        plots = [[self.hists[var].integrate('channel', chan).integrate('cut', 'base') for chan in self.channels.values()] for var in self.var]
        [[h.scale(self.lumi/self.smsow) for h in plot] for plot in plots] #Hack for small test samples
        for nbjet in self.nbjets:
            print('.', end='', flush=True)
            for syst in self.syst:
                if syst != 'nominal': continue
                for ch in self.charge:
                    for ivar,var in enumerate(self.var):
                        print('.', end='', flush=True)
                        #Integrate out jet cuts
                        #print([h.values() for h in plots[ivar]])
                        nbjetplots = [h.integrate('nbjet', nbjet).integrate('systematic', syst).integrate('sumcharge', ch) for h in plots[ivar]]
                        #print([h.values() for h in cutplots])
                        if syst == 'nominal': sys = ''
                        else: sys = '_'+syst
                        charge = 'p' if ch == 'ch+' else 'm'
                        for chan in self.channels.keys():
                            #Create the temp ROOT file
                            print('.', end='', flush=True)
                            fname = f'histos/tmp_ttx_multileptons-{chan}_{charge}_{nbjet}{sys}.root' if var == 'njets' else f'histos/tmp_ttx_multileptons-{chan}_{charge}_{nbjet}{sys}_{var}.root'
                            fout = uproot3.recreate(fname)
                            #Scale each plot to the SM
                            [h.set_wilson_coefficients(np.zeros(h._nwc)) for h in plots[ivar]] #optimized HistEFT
                            for proc,h in zip(self.samples, nbjetplots):
                                print('.', end='', flush=True)
                                h = h.integrate('sample', proc)
                                #Integrate out processes
                                pname = self.rename[proc]+'_' if proc in self.rename else proc+'_'
                                if var == 'njets': h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7,8]))
                                #Save the SM plot
                                h_sm = h#.copy()
                                h_sm.set_wilson_coefficients(np.zeros(h._nwc))
                                fout[pname+'sm'] = hist.export1d(h_sm)
                                #Asimov data: data_obs = MC at SM (all WCs = 0)
                                fout['data_obs'] = hist.export1d(h_sm)
                                
                                h_lin = h; h_quad = []; h_mix = []
                                yields = []
                                for wc,name,wcpt in wcs:
                                    #Scale plot to the WCPoint
                                    w = wcpt.buildMatrix(self.coeffs)
                                    #Handle linear and quadratic terms
                                    if 'lin' in name:
                                        h_lin = h#.copy()
                                        h_lin.set_wilson_coefficients(w)
                                        if np.sum(h_lin.values()[()]) > self.tolerance:
                                            fout[pname+name] = hist.export1d(h_lin)
                                    elif 'quad' in name and 'mix' not in name:
                                        h_quad = h#.copy()
                                        h_quad.set_wilson_coefficients(w)
                                        if np.sum(h_quad.values()[()]) > self.tolerance:
                                            fout[pname+name] = hist.export1d(h_quad)
                                    else:
                                        h_mix = h#.copy()
                                        h_mix.set_wilson_coefficients(w)
                                        if np.sum(h_mix.values()[()]) > self.tolerance:
                                            fout[pname+name] = hist.export1d(h_mix)
        
                        fout.close()
        print('.')

    def buildWCString(self, wc):
        '''
        Builds a set of WC strings
        Linear terms (single WC set to 1)
        Quadratic terms (pair of WCs set to 2)
        '''
        wcpt = []
        if len(wc)==0:
            wcpt = None
        #Case for a single wc
        elif isinstance(wc, str):
            wcpt.append([wc, f'lin_{wc}', WCPoint(f'EFTrwgt0_{wc}_{1}')])
        elif len(wc)==1:
            wcpt.append([wc, f'lin_{wc[0]}', WCPoint(f'EFTrwgt0_{wc[0]}_{1}')])
        #Case for 2+ wcs
        else:
            pairs = [[wc[w1],wc[w2]] for w1 in range(len(wc)) for w2 in range(0, w1+1)]
            wcpt = []
            lin = []
            quad = []
            mixed = []
            #linear terms
            for n,w in enumerate(wc):
                wcpt.append([w, f'lin_{w}', WCPoint(f'EFTrwgt0_{w}_{1}')])
            #quadratic terms
                for m,w in enumerate([[w,wc[w2]] for w2 in range(0, n+1)]):
                    wc1 = w[0]
                    wc2 = w[1]
                    if(wc1==wc2):  wcpt.append([[wc1,wc2], f'quad_{wc1}', WCPoint(f'EFTrwgt0_{wc1}_{2}')])
                    else: wcpt.append([[wc1,wc2], f'quad_mixed_{wc1}_{wc2}', WCPoint(f'EFTrwgt0_{wc1}_{1}_{wc2}_{2}')])
        return wcpt

    def makeCard(self):
        '''
        Create datacard files from temp uproot outputs
        Creates histograms for ``combine``:
        ``S`` is theSM
        ``S+L_i+Q_i`` sets ``WC_i=1`` and the rest to ``0``
        ``S+L_i+L_j+Q_i+Q_j+2 M_IJ`` set ``WC_i=1``, ``WC_j=1`` and the rest to ``0``
        '''
        print(f'Making the datacard')
        for nbjet in self.nbjets:
            for syst in self.syst:
                if syst != 'nominal': continue
                print(f'Systematic: {syst}')
                for ch in self.charge:
                    print(f'Charge: {ch}')
                    for ivar,var in enumerate(self.var):
                        print(f'Category: {nbjet}')
                        print(f'Variable: {var}')
                        if syst == 'nominal': sys = ''
                        else: sys = '_'+syst
                        charge = 'p' if ch == 'ch+' else 'm'
                        for chan in self.channels.keys():
                            #Open temp ROOT file
                            fname = f'histos/tmp_ttx_multileptons-{chan}_{charge}_{nbjet}{sys}.root' if var == 'njets' else f'histos/tmp_ttx_multileptons-{chan}_{charge}_{nbjet}{sys}_{var}.root'
                            fin = TFile(fname)
                            d_hists = {k.GetName(): fin.Get(k.GetName()) for k in fin.GetListOfKeys()}
                            [h.SetDirectory(0) for h in d_hists.values()]
                            fin.Close()
                            #os.system(f'rm {fname}')
                            #Delete temp ROOT file
                            #Create the ROOT file
                            fname = f'histos/ttx_multileptons-{chan}_{charge}_{nbjet}{sys}.root' if var == 'njets' else f'histos/ttx_multileptons-{chan}_{charge}_{nbjet}{sys}_{var}.root'
                            fout = TFile(fname, 'recreate')
                            for proc in self.samples:
                                p = self.rename[proc] if proc in self.rename else proc
                                print(f'Process: {p}')
                                signalcount=0; bkgcount=0; iproc = {}; allyields = {'data_obs' : 1.}
                                name = 'data_obs'
                                data_obs = d_hists[name]
                                if name not in d_hists:
                                    continue
                                data_obs.SetDirectory(fout)
                                data_obs.Write()
                                allyields[name] = data_obs.Integral()
                                pname = self.rename[proc]+'_' if proc in self.rename else proc+'_'
                                name = '_'.join([pname[:-1],'sm'])
                                if name not in d_hists:
                                    continue
                                h_sm = d_hists[name]
                                h_sm.SetDirectory(fout)
                                h_sm.Write()
                                for n,wc in enumerate(self.coeffs):
                                    name = '_'.join([pname[:-1],'lin',wc])
                                    if name not in d_hists:
                                        print(f'Histogram {name} not found!')
                                        continue
                                    h_lin = d_hists[name]
                                    signalcount -= 1
                                    if h_lin.Integral() > self.tolerance:
                                        h_lin.SetDirectory(fout)
                                        h_lin.Write()
                                        iproc[name] = signalcount
                                        allyields[name] = h_lin.Integral()
                                        if allyields[name] < 0:
                                            allyields[name] = 0.

                                    h_lin.Scale(-2)
                                    name = '_'.join([pname[:-1],'quad',wc])
                                    if name not in d_hists:
                                        print(f'Histogram {name} not found!')
                                        continue
                                    h_quad = d_hists[name]
                                    h_quad.Add(h_sm)
                                    h_quad.Add(h_lin)
                                    h_quad.Scale(0.5)
                                    if h_quad.Integral() > self.tolerance:
                                        h_quad.SetDirectory(fout)
                                        h_quad.Write()
                                        iproc[name] = signalcount
                                        allyields[name] = h_quad.Integral()
                                        if allyields[name] < 0:
                                            allyields[name] = 0.

                                    for wc2 in [self.coeffs[w2] for w2 in range(n)]:
                                        name = '_'.join([pname[:-1],'quad_mixed',wc,wc2])
                                        if name not in d_hists:
                                            print(f'Histogram {name} not found!')
                                            continue
                                        h_mix = d_hists[name]
                                        if h_mix.Integral() > self.tolerance:
                                            h_mix.SetDirectory(fout)
                                            h_mix.Write()
                                            iproc[name] = signalcount
                                            allyields[name] = h_mix.Integral()
                                            if allyields[name] < 0:
                                                allyields[name] = 0.

                            #Write datacard
                            if syst == 'nominal':
                                cat = '_'.join([chan, charge, nbjet]) if var == 'njets' else '_'.join([chan, charge, nbjet, var])
                            else:
                                cat = '_'.join([chan, charge, nbjet, syst]) if var == 'njets' else '_'.join([chan, charge, nbjet, syst, var])
                            datacard = open("histos/ttx_multileptons-%s.txt"%cat, "w"); 
                            datacard.write("shapes *        * ttx_multileptons-%s.root $PROCESS $PROCESS_$SYSTEMATIC\n" % cat)
                            cat = 'bin_'+cat
                            datacard.write('##----------------------------------\n')
                            datacard.write('bin         %s\n' % cat)
                            datacard.write('observation %s\n' % allyields['data_obs'])
                            datacard.write('##----------------------------------\n')
                            klen = max([7, len(cat)]+[len(p) for p in iproc.keys()])
                            kpatt = " %%%ds "  % klen
                            fpatt = " %%%d.%df " % (klen,np.abs(int(np.format_float_scientific(self.tolerance).split('e')[1])))#3)
                            #npatt = "%%-%ds " % max([len('process')]+map(len,nuisances))
                            npatt = "%%-%ds " % max([len('process')])
                            datacard.write('##----------------------------------\n')
                            procs = iproc.keys()
                            datacard.write((npatt % 'bin    ')+(" "*6)+(" ".join([kpatt % cat      for p in procs]))+"\n")
                            datacard.write((npatt % 'process')+(" "*6)+(" ".join([kpatt % p        for p in procs]))+"\n")
                            datacard.write((npatt % 'process')+(" "*6)+(" ".join([kpatt % iproc[p] for p in procs]))+"\n")
                            datacard.write((npatt % 'rate   ')+(" "*6)+(" ".join([fpatt % allyields[p] for p in procs]))+"\n")
                            datacard.write('##----------------------------------\n')
        
                        fout.Close()

if __name__ == '__main__':
    #hr = HistoReader('/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/hist_pkl_files/plotsTopEFT_nanoOnly_lobster_20210426_1006.pkl.gz')
    #hr = HistoReader('histos/plotsTopEFT.pkl.gz')
    hr = HistoReader('histos/ttH_top19001.pkl.gz')
    #hr = HistoReader('histos/ttH_top19001_1file.pkl.gz')
    #hr = HistoReader('/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/hist_pkl_files/optimized-histeft/plotsTopEFT_private_UL17_10files.pkl.gz')
    #hr = HistoReader('/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/hist_pkl_files/optimized-histeft/plotsTopEFT_private_top19001-1files.pkl.gz')
    #hr = HistoReader('/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/hist_pkl_files/plotsTopEFT_central_UL17_5_files_each.pkl.gz')
    hr.read()
    hr.relish()
    hr.makeCard()
