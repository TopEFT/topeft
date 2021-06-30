import pickle
import gzip
from coffea import hist
import topcoffea.modules.HistEFT
from topcoffea.modules.WCPoint import WCPoint
import uproot3
import numpy as np
import os
import re
import json

from ROOT import TFile, TH1D

class HistoReader():
    def __init__(self, infile='', analysisList=[], central=False, year=2018, lumiJson='topcoffea/json/lumi.json', do_nuisance=False):
        self.hists = {}
        self.analysisList = analysisList
        self.rename = {'tZq': 'tllq', 'tllq_privateUL17': 'tllq', 'ttZ': 'ttll', 'ttll_TOP-19-001': 'ttll', 'ttW': 'ttlnu', 'ttGJets': 'convs', 'WZ': 'Diboson', 'WWW': 'Triboson', 'ttHnobb': 'ttH', 'ttH_TOP-19-001': 'ttH', "tHq_privateUL17": "tHq", "tllq_privateUL17": "tllq", "ttHJet_privateUL17": "ttH", "ttllJet_privateUL17": "ttll", "ttlnuJet_privateUL17": "ttlnu"} #Used to rename things like ttZ to ttll and ttHnobb to ttH
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
        self.syst_terms =['LF', 'JES', 'MURMUF', 'CERR1', 'MUR', 'CERR2', 'PSISR', 'HFSTATS1', 'Q2RF', 'FR_FF', 'HFSTATS2', 'LFSTATS1', 'TRG', 'LFSTATS2', 'MUF', 'PDF', 'HF', 'PU', 'LEPID']
        self.coeffs = ['ctW', 'ctp', 'cpQM', 'ctli', 'cQei', 'ctZ', 'cQlMi', 'cQl3i', 'ctG', 'ctlTi', 'cbW', 'cpQ3', 'ctei', 'cpt', 'ctlSi', 'cptb','cQq13','cQq83','cQq11','ctq1','cQq81','ctq8']
        self.coeffs = ['cpt', 'ctp', 'cptb', 'cQlMi', 'cQq81', 'cQq11', 'cQl3i', 'ctq8', 'ctlTi', 'ctq1', 'ctli', 'cQq13', 'cbW', 'cpQM', 'cpQ3', 'ctei', 'cQei', 'ctW', 'ctlSi', 'cQq83', 'ctZ', 'ctG']
        self.ch2lss = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
        self.ch3l = ['eemSSoffZ', 'mmeSSoffZ', 'eeeSSoffZ', 'mmmSSoffZ']
        self.ch3lsfz = ['eemSSonZ', 'mmeSSonZ', 'eeeSSonZ', 'mmmSSonZ', 'mmmSSoffZ']
        self.ch4l =['eeee','eeem','eemm','mmme','mmmm']
        self.levels = ['base', '2jets', '4jets', '4j1b', '4j2b']
        self.channels = {'2lss': self.ch2lss, '3l': self.ch3l, '3l_sfz': self.ch3lsfz, '4l': self.ch4l}
        self.outf = "EFT_MultiDim_Datacard_combine.txt"
        self.fin = infile
        self.var = ['njets', 'ht']
        self.tolerance = 0.001
        self.do_nuisance = do_nuisance


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
        #self.nbjets = list({k[4]:0 for k in self.hists['njets'].values().keys()})
        self.syst = list({k[4]:0 for k in self.hists['njets'].values().keys()})
        self.hsow = self.hists['SumOfEFTweights']
        self.hsow.set_wilson_coefficients(np.zeros(self.hsow._nwc))
        self.smsow = {proc: self.hsow.integrate('sample', proc).values()[()][0] for proc in self.samples}
        with open(lumiJson) as jf:
            lumi = json.load(jf)
            lumi = lumi[year]
        self.lumi = 1000*lumi

    def analyzeChannel(self, channel='2lss', cuts='2+bm', charges=['ch+','ch-'], systematics='nominal', variable='njets'):
        if isinstance(channel, str) and channel not in self.channels:
           raise Exception(f'{channel} not found in self.channels!')
        if isinstance(channel, list) and not all(ch in self.channels for ch in self.channels.keys()):
           print(self.channels.keys())
           print([[ch, ch in self.channels.keys()] for ch in channel])
           raise Exception(f'At least one channel in {channels} is not found in self.channels!')
        h = self.hists[variable].integrate('channel', self.channels[channel]).integrate('cut', cuts).integrate('sumcharge', charges).integrate('systematic', systematics)
        all_str = ' '.join([f'{v}' for v in locals().values() if v != self.hists])
        all_str = f'{channel} {cuts} {charges} {systematics} {variable}'
        print(f'Making relish from the pickle file for {all_str}')
        if isinstance(charges, str): charge = charges
        else: charge = ''
        charge = 'p' if charge == 'ch+' else 'm'
        result = [e for e in re.split("[^0-9]", cuts) if e != '']
        maxb = str(max(map(int, result))) + 'b'
        if systematics == 'nominal': sys = ''
        else: sys = '_'+systematics
        if variable == 'njets':
            if isinstance(charge, str):
                cat = '_'.join([channel, charge, maxb])  
            else:
                cat = '_'.join([channel, maxb])  
        else:
            if isinstance(charge, str):
                '_'.join([channel, charge, maxb, variable])
            else:
                '_'.join([channel, maxb, variable])
        fname = f'histos/tmp_ttx_multileptons-{cat}.root'
        fout = uproot3.recreate(fname)
        #Scale each plot to the SM
        for proc in self.samples:
            #Integrate out processes
            h_base = h.integrate('sample', proc)
            if h_base == {}:
                print(f'Issue with {proc}')
                continue
            nwc = self.hsow._nwc
            if nwc > 0:
                h_base.scale(self.lumi/self.smsow[proc])
            pname = self.rename[proc]+'_' if proc in self.rename else proc+'_'
            if variable == 'njets':
                if '2l' in channel: h_base = h_base.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7]))
                elif '3l' in channel: h_base = h_base.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4,5]))
                elif '4l' in channel: h_base = h_base.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4]))
            #Save the SM plot
            h_sm = h_base#.copy()
            h_sm.set_wilson_coefficients(np.zeros(h_base._nwc))
            fout[pname+'sm'] = hist.export1d(h_sm)
            #Asimov data: data_obs = MC at SM (all WCs = 0)
            fout['data_obs'] = hist.export1d(h_sm)
            
            h_lin = h_base; h_quad = []; h_mix = []
            yields = []
            for wc,name,wcpt,wl in self.wcs:
                #Scale plot to the WCPoint
                #w = wcpt.buildMatrix(self.coeffs)
                #Handle linear and quadratic terms
                if 'lin' in name:
                    h_lin = h_base#.copy()
                    h_lin.set_wilson_coefficients(wl)
                    if np.sum(h_lin.values()[()]) > self.tolerance:
                        fout[pname+name] = hist.export1d(h_lin)
                        if 'ttH' in pname or True: #FIXME
                            if variable == 'njets':
                                if isinstance(charge, str):
                                    cat = '_'.join([channel, charge, ])  
                                else:
                                    cat = '_'.join([channel, maxb])  
                            else:
                                if isinstance(charge, str):
                                    '_'.join([channel, charge, maxb, variable])
                                else:
                                    '_'.join([channel, maxb, variable])
                elif 'quad' in name and 'mix' not in name:
                    h_quad = h_base#.copy()
                    h_quad.set_wilson_coefficients(wl)
                    if np.sum(h_quad.values()[()]) > self.tolerance:
                        fout[pname+name] = hist.export1d(h_quad)
                else:
                    h_mix = h_base#.copy()
                    h_mix.set_wilson_coefficients(wl)
                    if np.sum(h_mix.values()[()]) > self.tolerance:
                        fout[pname+name] = hist.export1d(h_mix)
        
        fout.close()
        self.makeCardLevel(channel=channel, cuts=cuts, charges=charges, nbjet=maxb, systematics=systematics, variable=variable)

    def makeCardLevel(self, channel='2lss', cuts='base', charges=['ch+','ch-'], nbjet='2+bm', systematics='nominal', variable='njets'):
        '''
        Create datacard files from temp uproot outputs
        Creates histograms for ``combine``:
        ``S`` is theSM
        ``S+L_i+Q_i`` sets ``WC_i=1`` and the rest to ``0``
        ``S+L_i+L_j+Q_i+Q_j+2 M_IJ`` set ``WC_i=1``, ``WC_j=1`` and the rest to ``0``
        '''
        def processSyst(hist, systMap,fout):
            for s in self.syst_terms:
                if s in systMap:
                    systMap[s].append(name)
                else:
                    systMap[s] = [name]
                '''
                This part is a hack to inject fact systematics for testing only!
                '''
                h_sys = data_obs.Clone(name+'_'+s+'Up')
                h_sys.Scale(1.1/h_sys.Integral())
                h_sys.SetDirectory(fout)
                h_sys.Write()
                h_sys.SetName(h_sys.GetName().replace('Up', 'Down'))
                h_sys.Scale(0.9/h_sys.Integral())
                h_sys.Write()
        print(f'Making the datacard')
        if isinstance(charges, str): charge = charges
        else: charge = ''
        charge = 'p' if charge == 'ch+' else 'm'
        if systematics == 'nominal': sys = ''
        else: sys = '_'+systematics
        if variable == 'njets':
            if isinstance(charge, str):
                cat = '_'.join([channel, charge, nbjet])  
            else:
                cat = '_'.join([channel, nbjet])  
        else:
            if isinstance(charge, str):
                cat = '_'.join([channel, charge, nbjet, variable])
            else:
                cat = '_'.join([channel, nbjet, variable])
        #Open temp ROOT file
        fname = f'histos/tmp_ttx_multileptons-{cat}.root'
        fin = TFile(fname)
        d_hists = {k.GetName(): fin.Get(k.GetName()) for k in fin.GetListOfKeys()}
        [h.SetDirectory(0) for h in d_hists.values()]
        fin.Close()
        #Delete temp ROOT file
        os.system(f'rm {fname}')
        #Create the ROOT file
        fname = f'histos/ttx_multileptons-{cat}.root'
        fout = TFile(fname, 'recreate')
        signalcount=0; bkgcount=0; iproc = {}; systMap = {}; allyields = {'data_obs' : 0.}
        data_obs = []
        for proc in self.samples:
            p = self.rename[proc] if proc in self.rename else proc
            print(f'Process: {p}')
            name = 'data_obs'
            if name not in d_hists:
                print(f'{name} not found!')
                continue
            '''
            These lines are for testing only, and create Asimov data based on all processes provided
            '''
            if proc == self.samples[0]:
                data_obs = d_hists[p+'_sm'].Clone('data_obs')
            else:
                data_obs += d_hists[p+'_sm'].Clone('data_obs')
            allyields[name] += data_obs.Integral()
            asimov = np.random.poisson(int(data_obs.Integral()))
            data_obs.SetDirectory(fout)
            if proc == self.samples[-1]:
                data_obs.Scale(allyields['data_obs'] / data_obs.Integral())
                data_obs.Write()
            pname = self.rename[proc]+'_' if proc in self.rename else proc+'_'
            name = pname + 'sm'
            if name not in d_hists:
                print(f'{name} not found!')
                continue
            h_sm = d_hists[name]
            h_sm.SetDirectory(fout)
            h_sm.Write()
            if h_sm.Integral() > self.tolerance:
                h_sm.SetDirectory(fout)
                h_sm.Write()
                signalcount -= 1
                iproc[name] = signalcount
                allyields[name] = h_sm.Integral()
                if allyields[name] < 0:
                    allyields[name] = 0.
                processSyst(h_sm, systMap,fout)
            for n,wc in enumerate(self.coeffs):
                name = '_'.join([pname[:-1],'lin',wc])
                if name not in d_hists:
                    print(f'Histogram {name} not found! Probably below the tolerance. If so, ignore this message!')
                    continue
                h_lin = d_hists[name]
                if h_lin.Integral() > self.tolerance:
                    h_lin.SetDirectory(fout)
                    h_lin.Write()
                    signalcount -= 1
                    iproc[name] = signalcount
                    allyields[name] = h_lin.Integral()
                    if allyields[name] < 0:
                        allyields[name] = 0.

                    processSyst(h_lin, systMap,fout)
                name = '_'.join([pname[:-1],'quad',wc])
                if name not in d_hists:
                    print(f'Histogram {name} not found! Probably below the tolerance. If so, ignore this message!')
                    continue
                h_quad = d_hists[name]
                h_quad.Add(h_lin, -2)
                h_quad.Add(h_sm)
                h_quad.Scale(0.5)
                if h_quad.Integral() > self.tolerance:
                    h_quad.SetDirectory(fout)
                    h_quad.Write()
                    signalcount -= 1
                    iproc[name] = signalcount
                    allyields[name] = h_quad.Integral()
                    if allyields[name] < 0:
                        allyields[name] = 0.
                    processSyst(h_quad, systMap,fout)

                for wc2 in [self.coeffs[w2] for w2 in range(n)]:
                    name = '_'.join([pname[:-1],'quad_mixed',wc,wc2])
                    if name not in d_hists:
                        print(f'Histogram {name} not found! Probably below the tolerance. If so, ignore this message!')
                        continue
                    h_mix = d_hists[name]
                    if h_mix.Integral() > self.tolerance:
                        h_mix.SetDirectory(fout)
                        h_mix.Write()
                        signalcount -= 1
                        iproc[name] = signalcount
                        allyields[name] = h_mix.Integral()
                        if allyields[name] < 0:
                            allyields[name] = 0.
                        processSyst(h_mix, systMap,fout)

        #Write datacard
        if systematics != 'nominal':
            cat = cat + '_' + systematics
        nuisances = [syst for syst in systMap]
        datacard = open("histos/ttx_multileptons-%s.txt"%cat, "w"); 
        datacard.write("shapes *        * ttx_multileptons-%s.root $PROCESS $PROCESS_$SYSTEMATIC\n" % cat)
        cat = 'bin_'+cat
        datacard.write('##----------------------------------\n')
        datacard.write('bin         %s\n' % cat)
        datacard.write('observation %%.%df\n' % np.abs(int(np.format_float_scientific(self.tolerance).split('e')[1])) % allyields['data_obs'])
        datacard.write('##----------------------------------\n')
        klen = max([7, len(cat)]+[len(p[0]) for p in iproc.keys()])
        kpatt = " %%%ds "  % klen
        fpatt = " %%%d.%df " % (klen,np.abs(int(np.format_float_scientific(self.tolerance).split('e')[1])))#3)
        npatt = "%%-%ds " % max([len('process')]+list(map(len,nuisances)))
        #npatt = "%%-%ds " % max([len('process')])
        datacard.write('##----------------------------------\n')
        procs = iproc.keys()
        datacard.write((npatt % 'bin    ')+(" "*6)+(" ".join([kpatt % cat      for p in procs]))+"\n")
        datacard.write((npatt % 'process')+(" "*6)+(" ".join([kpatt % p        for p in procs]))+"\n")
        datacard.write((npatt % 'process')+(" "*6)+(" ".join([kpatt % iproc[p] for p in procs]))+"\n")
        datacard.write((npatt % 'rate   ')+(" "*6)+(" ".join([fpatt % allyields[p] for p in procs]))+"\n")
        datacard.write('##----------------------------------\n')
        # Uncomment for nuisance parameter testing, or final nuisance paramter values
        if self.do_nuisance:
            for name in nuisances:
                systEff = dict((p,"1" if p in systMap[name] else "-") for p in procs)
                datacard.write(('%s %5s' % (npatt % name,'shape')) + " ".join([kpatt % systEff[p]  for p in procs]) +"\n")
        
        fout.Close()

    def buildWCString(self, wc=''):
        '''
        Builds a set of WC strings
        Linear terms (single WC set to 1)
        Quadratic terms (pair of WCs set to 2)
        '''
        if wc == '': wc = self.coeffs
        wcpt = []
        if len(wc)==0:
            wcpt = None
        #Case for a single wc
        elif isinstance(wc, str):
            wcpt.append([wc, f'lin_{wc}', WCPoint(f'EFTrwgt0_{wc}_{1}')])
            wl = {k:0 for k in self.coeffs}
            wl[wc] = 1.
            wl = np.array(list(wl.values()))
            wcpt[-1].append(wl)
        elif len(wc)==1:
            wcpt.append([wc, f'lin_{wc[0]}', WCPoint(f'EFTrwgt0_{wc[0]}_{1}')])
            wl = {k:0 for k in self.coeffs}
            wl[wc] = 1.
            wl = np.array(list(wl.values()))
            wcpt[-1].append(wl)
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
                wl = {k:0 for k in self.coeffs}
                wl[w] = 1.
                wl = np.array(list(wl.values()))
                wcpt[-1].append(wl)
            #quadratic terms
                for m,w in enumerate([[w,wc[w2]] for w2 in range(0, n+1)]):
                    wc1 = w[0]
                    wc2 = w[1]
                    if(wc1==wc2):  wcpt.append([[wc1,wc2], f'quad_{wc1}', WCPoint(f'EFTrwgt0_{wc1}_{2}')])
                    else: wcpt.append([[wc1,wc2], f'quad_mixed_{wc1}_{wc2}', WCPoint(f'EFTrwgt0_{wc1}_{1}_{wc2}_{1}')])
                    wl = {k:0 for k in self.coeffs}
                    if(wc1==wc2):
                        wl[wc1] = 2.
                    else:
                        wl[wc1] = 1.; wl[wc2] = 1.;
                    wl = np.array(list(wl.values()))
                    wcpt[-1].append(wl)
        self.wcs     = wcpt
        return wcpt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='You can select which file to run over')
    parser.add_argument('pklfile'           , nargs='?', default=''           , help = 'Pickle file with histograms')
    parser.add_argument('--year',     '-y', default='2018'                         , help = 'Run year to access lumi')
    parser.add_argument('--lumiJson', '-l', default='topcoffea/json/lumi.json'     , help = 'Lumi json file')
    parser.add_argument('--do-nuisance',    action='store_true', help = 'Include nuisance parameters')
    args = parser.parse_args()
    pklfile  = args.pklfile
    year = args.year
    lumiJson = args.lumiJson
    do_nuisance = args.do_nuisance
    if pklfile == '':
        raise Exception('Please specify a pkl file!')
    hr = HistoReader(pklfile, year, lumiJson, do_nuisance)
    hr.read()
    hr.buildWCString()
    hr.analyzeChannel(channel='2lss', cuts='1+bm2+bl', charges='ch+', systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='2lss', cuts='1+bm2+bl', charges='ch-', systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='3l', cuts='1bm', charges='ch+', systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='3l', cuts='1bm', charges='ch-', systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='3l', cuts='2+bm', charges='ch+', systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='3l', cuts='2+bm', charges='ch-', systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='3l_sfz', cuts='2+bm', charges=['ch+','ch-'], systematics='nominal', variable='njets')
    hr.analyzeChannel(channel='4l', cuts='1+bm2+bl', charges=['ch+','ch0','ch-'], systematics='nominal', variable='njets')
