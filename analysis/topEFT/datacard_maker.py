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

from ROOT import TFile, TH1D, TH2D

class DatacardMaker():
    def __init__(self, infile='', year=2018, lumiJson='topcoffea/json/lumi.json', do_nuisance=False):
        self.hists = {}
        self.rename = {'tZq': 'tllq', 'tllq_privateUL17': 'tllq', 'ttZ': 'ttll'} #Used to rename things like ttZ to ttll and ttHnobb to ttH
        self.syst_terms =['LF', 'JES', 'MURMUF', 'CERR1', 'MUR', 'CERR2', 'PSISR', 'HFSTATS1', 'Q2RF', 'FR_FF', 'HFSTATS2', 'LFSTATS1', 'TRG', 'LFSTATS2', 'MUF', 'PDF', 'HF', 'PU', 'LEPID']
        self.ch2lss = ['2lss_p_4j','2lss_p_5j','2lss_p_6j','2lss_p_7j','2lss_m_4j','2lss_m_5j','2lss_m_6j','2lss_m_7j']
        self.ch2lss_p = ['2lss_p_4j','2lss_p_5j','2lss_p_6j','2lss_p_7j']
        self.ch2lss_m = ['2lss_m_4j','2lss_m_5j','2lss_m_6j','2lss_m_7j']
        self.ch3l = ['3l_p_offZ_2j_1b','3l_p_offZ_3j_1b','3l_p_offZ_4j_1b','3l_p_offZ_5j_1b', '3l_m_offZ_2j_1b','3l_m_offZ_3j_1b','3l_m_offZ_4j_1b','3l_m_offZ_5j_1b', '3l_p_offZ_2j_2b','3l_p_offZ_3j_2b','3l_p_offZ_4j_2b','3l_p_offZ_5j_2b', '3l_m_offZ_2j_2b','3l_m_offZ_3j_2b','3l_m_offZ_4j_2b','3l_m_offZ_5j_2b']
        self.ch3l1b = ['3l_p_offZ_2j_1b','3l_p_offZ_3j_1b','3l_p_offZ_4j_1b','3l_p_offZ_5j_1b', '3l_m_offZ_2j_1b','3l_m_offZ_3j_1b','3l_m_offZ_4j_1b','3l_m_offZ_5j_1b']
        self.ch3l1b_p = ['3l_p_offZ_2j_1b','3l_p_offZ_3j_1b','3l_p_offZ_4j_1b','3l_p_offZ_5j_1b']
        self.ch3l1b_m = ['3l_m_offZ_2j_1b','3l_m_offZ_3j_1b','3l_m_offZ_4j_1b','3l_m_offZ_5j_1b']
        self.ch3l2b = ['3l_p_offZ_2j_2b','3l_p_offZ_3j_2b','3l_p_offZ_4j_2b','3l_p_offZ_5j_2b', '3l_m_offZ_2j_2b','3l_m_offZ_3j_2b','3l_m_offZ_4j_2b','3l_m_offZ_5j_2b']
        self.ch3l2b_p = ['3l_p_offZ_2j_2b','3l_p_offZ_3j_2b','3l_p_offZ_4j_2b','3l_p_offZ_5j_2b']
        self.ch3l2b_m = ['3l_m_offZ_2j_2b','3l_m_offZ_3j_2b','3l_m_offZ_4j_2b','3l_m_offZ_5j_2b']
        self.ch3lsfz = ['3l_onZ_2j_1b','3l_onZ_3j_1b','3l_onZ_4j_1b','3l_onZ_5j_1b', '3l_onZ_2j_2b','3l_onZ_3j_2b','3l_onZ_4j_2b','3l_onZ_5j_2b']
        self.ch4l =['4l_2j','4l_3j','4l_4j']
        self.levels = ['base', '2jets', '4jets', '4j1b', '4j2b']
        self.channels = {'2lss': self.ch2lss, '2lss_p': self.ch2lss_p, '2lss_m': self.ch2lss_m, '3l1b': self.ch3l1b, '3l1b_p': self.ch3l1b_p, '3l1b_m': self.ch3l1b_m, '3l2b': self.ch3l2b,  '3l2b_p': self.ch3l2b_p, '3l2b_m': self.ch3l2b_m,'3l_sfz': self.ch3lsfz, '4l': self.ch4l}
        self.fin = infile
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

        #Get list of samples and cut levels from histograms
        self.signal = ['ttH','tllq','ttll','ttlnu','tHq','tttt']
        self.samples = list({k[0]:0 for k in self.hists['njets'].values().keys()})
        rename = {l: re.split('(Jet)?_[a-zA-Z]*1[6-8]', l)[0] for l in self.samples}
        rename = {k: 'Triboson' if bool(re.search('[WZ]{3}', v)) else v for k,v in rename.items()}
        rename = {k: 'Diboson' if bool(re.search('[WZ]{2}', v)) else v for k,v in rename.items()}
        rename = {k: 'convs' if bool(re.search('TTG', v)) else v for k,v in rename.items()}
        rename = {k: 'fakes' if bool(re.search('(^tW)|(^tbarW)|(^WJet)|(^t_)|(^tbar_)|(^TT)|(^DY)', v)) else v for k,v in rename.items()}
        self.rename = {**self.rename, **rename}
        print(self.rename)
        self.levels = list({k[2]:0 for k in self.hists['njets'].values().keys()})
        self.syst = list({k[3]:0 for k in self.hists['njets'].values().keys()})
        self.hsow = self.hists['SumOfEFTweights']
        self.hsow.set_sm()
        self.smsow = {proc: self.hsow.integrate('sample', proc).values()[()][0] for proc in self.samples}
        with open(lumiJson) as jf:
            lumi = json.load(jf)
            lumi = lumi[year]
        self.lumi = 1000*lumi

    def analyzeChannel(self, channel=[], appl='isSR_2lss', charges=['ch+','ch-'], systematics='nominal', variable='njets'):
        def export2d(h):
            return h.to_hist().to_numpy()
        if isinstance(channel, str) and channel not in self.channels:
           raise Exception(f'{channel} not found in self.channels!')
        if isinstance(channel, list) and not all(ch in self.channels for ch in self.channels.keys()):
           print(self.channels.keys())
           print([[ch, ch in self.channels.keys()] for ch in channel])
           raise Exception(f'At least one channel in {channels} is not found in self.channels!')
        h = self.hists[variable].integrate('appl', appl).integrate('systematic', systematics)
        if variable == 'njets':
            if isinstance(charges, str):
                charge = 'p' if charges == 'ch+' else 'm'
                h = h.integrate('channel', self.channels[channel+'_'+charge])
            else:
                h = h.integrate('channel', self.channels[channel])
        else:
            if isinstance(charges, str):
                charge = 'p' if charges == 'ch+' else 'm'
                ch = {v:v for v in self.channels[channel+'_'+charge]}
                h = h.group('channel', hist.Cat('channel', {channel+'_'+charge: ch}),ch )
            else:
                ch = {v:v for v in self.channels[channel]}
                h = h.group('channel', hist.Cat('channel', {channel: ch}),ch )
        all_str = ' '.join([f'{v}' for v in locals().values() if v != self.hists])
        channel = re.split('\db', channel)[0]
        all_str = f'{channel} {charges} {systematics} {variable}'
        print(f'Making relish from the pickle file for {all_str}')
        if isinstance(charges, str): charge = charges
        else: charge = ''
        charge = 'p' if charge == 'ch+' else 'm'
        if 'b' in channel:
            maxb = channel[-2:] # E.g. '3l1b' -> '1b'
        else:
            maxb = '2b' # 2lss and 4l cases
        if systematics == 'nominal': sys = ''
        else: sys = '_'+systematics
        if variable == 'njets':
            if isinstance(charges, str):
                cat = '_'.join([channel, charge, maxb])  
            else:
                cat = '_'.join([channel, maxb])  
        else:
            if isinstance(charges, str):
                cat = '_'.join([channel, charge, maxb, variable])
            else:
                cat = '_'.join([channel, maxb, variable])
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
            if 'njet' in variable:
                if   '2l' in channel: h_base = h_base.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7]))
                elif '3l' in channel: h_base = h_base.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4,5]))
                elif '4l' in channel: h_base = h_base.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4]))
            if 'ht' in variable:
                h_base = h_base.rebin('ht', hist.Bin("ht", "H$_{T}$ (GeV)", 50, 0, 1000))
            if 'pbl' in variable:
                h_base = h_base.rebin('pbl', hist.Bin("pbl", "$p_{T}^{b\mathrm{-}jet+\ell_{min(dR)}}$", 10, 0, 500))
            #Save the SM plot
            h_sm = h_base
            h_sm.set_sm()
            if len(h_base.axes())>1:
                fout[pname+'sm'] = export2d(h_sm)
            else:
                fout[pname+'sm'] = hist.export1d(h_sm)
            #Asimov data: data_obs = MC at SM (all WCs = 0)
            if len(h_base.axes())>1:
                fout['data_obs'] = export2d(h_sm)
            else:
                fout['data_obs'] = hist.export1d(h_sm)
            
            if proc in self.signal or self.rename[proc] in self.signal:
                h_lin = h_base; h_quad = []; h_mix = []
                yields = []
                for name,wcpt in self.wcs:
                    #Scale plot to the WCPoint
                    #Handle linear and quadratic terms
                    if 'lin' in name:
                        h_lin = h_base
                        h_lin.set_wilson_coeff_from_array(wcpt)
                        if True or np.sum(h_lin.values()[()]) > self.tolerance or True:
                            if len(h_base.axes())>1:
                                fout[pname+name] = export2d(h_lin)
                            else:
                                fout[pname+name] = hist.export1d(h_lin)
                            if variable == 'njets':
                                if isinstance(charges, str):
                                    cat = '_'.join([channel, charge, ])  
                                else:
                                    cat = '_'.join([channel, maxb])  
                            else:
                                if isinstance(charges, str):
                                    '_'.join([channel, charge, maxb, variable])
                                else:
                                    '_'.join([channel, maxb, variable])
                    elif 'quad' in name and 'mix' not in name:
                        h_quad = h_base
                        h_quad.set_wilson_coeff_from_array(wcpt)
                        if True or np.sum(h_quad.values()[()]) > self.tolerance or True:
                            if len(h_base.axes())>1:
                                fout[pname+name] = export2d(h_quad)
                            else:
                                fout[pname+name] = hist.export1d(h_quad)
                    else:
                        h_mix = h_base
                        h_mix.set_wilson_coeff_from_array(wcpt)
                        if True or np.sum(h_mix.values()[()]) > self.tolerance or True:
                            if len(h_base.axes())>1:
                                fout[pname+name] = export2d(h_mix)
                            else:
                                fout[pname+name] = hist.export1d(h_mix)
        
        fout.close()
        self.makeCardLevel(channel=channel, appl=appl, charges=charges, nbjet=maxb, systematics=systematics, variable=variable)

    def makeCardLevel(self, channel=[], appl='isSR_2lss', charges=['ch+','ch-'], nbjet='2+bm', systematics='nominal', variable='njets'):
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
            if isinstance(charges, str):
                cat = '_'.join([channel, charge, nbjet])  
            else:
                cat = '_'.join([channel, nbjet])  
        else:
            if isinstance(charges, str):
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
            print(f'Process: {proc} -> {p}')
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
                data_obs.Add(d_hists[p+'_sm'].Clone('data_obs'))
            asimov = np.random.poisson(int(data_obs.Integral()))
            data_obs.SetDirectory(fout)
            if proc == self.samples[-1]:
                xmin = data_obs.GetXaxis().GetXmin()
                xmax = data_obs.GetXaxis().GetXmax()
                xwidth = data_obs.GetXaxis().GetBinWidth(1)
                data_obs.GetXaxis().SetRangeUser(xmin, xmax + xwidth) #Include overflow bin in ROOT
                allyields[name] = data_obs.Integral()
                data_obs.Scale(allyields['data_obs'] / data_obs.Integral())
                data_obs.Write()
            pname = self.rename[proc]+'_' if proc in self.rename else proc+'_'
            name = pname + 'sm'
            if name not in d_hists:
                print(f'{name} not found!')
                continue
            h_sm = d_hists[name]
            if h_sm.Integral() > self.tolerance or p not in self.signal:
                if p in self.signal:
                    signalcount -= 1
                    iproc[name] = signalcount
                    allyields[name] = h_sm.Integral()
                else:
                    if name in iproc:
                        allyields[name] += h_sm.Integral()
                        h_sm.Add(fout.Get(name))
                        fout.Delete(name+';1')
                    else:
                        iproc[name] = bkgcount
                        allyields[name] = h_sm.Integral()
                        bkgcount += 1
                if allyields[name] < 0 or allyields[name] < self.tolerance:
                    allyields[name] = 0.
                #processSyst(h_sm, systMap,fout)
                h_sm.SetDirectory(fout)
                h_sm.Write()
                if p not in self.signal:
                    continue
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

                    #processSyst(h_lin, systMap,fout)
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
                    #processSyst(h_quad, systMap,fout)

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
                        #processSyst(h_mix, systMap,fout)

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
            wl = {k:0 for k in self.coeffs}
            wl[wc] = 1.
            wl = np.array(list(wl.values()))
            wcpt.append([f'lin_{wc}', wl])
        elif len(wc)==1:
            wl = {k:0 for k in self.coeffs}
            wl[wc] = 1.
            wl = np.array(list(wl.values()))
            wcpt.append([f'lin_{wc}', wl])
        #Case for 2+ wcs
        else:
            pairs = [[wc[w1],wc[w2]] for w1 in range(len(wc)) for w2 in range(0, w1+1)]
            wcpt = []
            lin = []
            quad = []
            mixed = []
            #linear terms
            for n,w in enumerate(wc):
                wl = {k:0 for k in self.coeffs}
                wl[w] = 1.
                wl = np.array(list(wl.values()))
                wcpt.append([f'lin_{w}', wl])
            #quadratic terms
                for m,w in enumerate([[w,wc[w2]] for w2 in range(0, n+1)]):
                    wc1 = w[0]
                    wc2 = w[1]
                    wl = {k:0 for k in self.coeffs}
                    if(wc1==wc2):
                        wl[wc1] = 2.
                    else:
                        wl[wc1] = 1.; wl[wc2] = 1.;
                    wl = np.array(list(wl.values()))
                    if(wc1==wc2):  wcpt.append([f'quad_{wc1}', wl])
                    else: wcpt.append([f'quad_mixed_{wc1}_{wc2}', wl])
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
    card = DatacardMaker(pklfile, year, lumiJson, do_nuisance)
    card.read()
    card.buildWCString()
    for var in ['njets','ht','pbl']:#,'njetbpl','njetht']:
        card.analyzeChannel(channel='2lss', appl='isSR_2lss', charges='ch+', systematics='nominal', variable=var)
        card.analyzeChannel(channel='2lss', appl='isSR_2lss', charges='ch-', systematics='nominal', variable=var)
        card.analyzeChannel(channel='3l1b', appl='isSR_3l', charges='ch+', systematics='nominal', variable=var)
        card.analyzeChannel(channel='3l1b', appl='isSR_3l', charges='ch-', systematics='nominal', variable=var)
        card.analyzeChannel(channel='3l2b', appl='isSR_3l', charges='ch+', systematics='nominal', variable=var)
        card.analyzeChannel(channel='3l2b', appl='isSR_3l', charges='ch-', systematics='nominal', variable=var)
        card.analyzeChannel(channel='3l_sfz', appl='isSR_3l', charges=['ch+','ch-'], systematics='nominal', variable=var)
        card.analyzeChannel(channel='4l', appl='isSR_4l', charges=['ch+','ch0','ch-'], systematics='nominal', variable=var)
