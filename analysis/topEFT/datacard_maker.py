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
from copy import deepcopy 

from ROOT import TFile, TH1D, TH2D

class DatacardMaker():

    def __init__(self, infile='', lumiJson='topcoffea/json/lumi.json', do_nuisance=False, wcs=[], single_year='', do_sm=False):
        self.hists = {}
        self.rename = {'tZq': 'tllq', 'tllq_privateUL17': 'tllq', 'ttZ': 'ttll'} #Used to rename things like ttZ to ttll and ttHnobb to ttH
        self.rename = {**self.rename, **{'ttH_centralUL17': 'ttH', 'ttH_centralUL16': 'ttH', 'ttH_centralUL18': 'ttH', 'ttHJetToNonbb_M125_centralUL16': 'ttH', 'ttHJetToNonbb_M125_APV_centralUL16': 'ttH', 'ttW_centralUL17': 'ttW', 'ttZ_centralUL17': 'ttZ', 'tZq_centralUL17': 'tllq', 'ttH_centralUL17': 'ttH', 'ttW_centralUL18': 'ttW', 'ttZ_centralUL18': 'ttZ', 'tZq_centralUL18': 'tllq', 'ttH_centralUL18': 'ttH'}}
        self.syst_terms =['LF', 'JES', 'MURMUF', 'CERR1', 'MUR', 'CERR2', 'PSISR', 'HFSTATS1', 'Q2RF', 'FR_FF', 'HFSTATS2', 'LFSTATS1', 'TRG', 'LFSTATS2', 'MUF', 'PDF', 'HF', 'PU', 'LEPID']
        self.syst_special = {'charge_flips': 0.3} # 30% flat uncertainty for charge flips
        self.ignore = ['DYJetsToLL', 'DY10to50', 'DY50', 'ST_antitop_t-channel', 'ST_top_s-channel', 'ST_top_t-channel', 'tbarW', 'TTJets', 'tW', 'WJetsToLNu']
        self.skip_process_channels = {'nonprompt': '4l'} # E.g. 4l does not include non-prompt background
        # Dictionary of njet bins
        self.fin = infile
        self.tolerance = 1e-5
        self.do_nuisance = do_nuisance
        self.coeffs = wcs if len(wcs)>0 else []
        if len(self.coeffs) > 0: print(f'Using the subset {self.coeffs}')
        self.year = single_year
        self.do_sm = do_sm
        # Variables we have defined a binning for
        self.known_var_lst = ['njets','ptbl','ht','ptz','o0pt','bl0pt','l0pt','lj0pt']

    def read(self):
        '''
        Load pickle file into hist dictionary
        '''
        print(f'Loading {self.fin}')
        with gzip.open(self.fin) as fin:
            self.hists = pickle.load(fin)

        # Variables present in the pkl file
        self.all_var_lst = yt.get_hist_list(self.hists)

        # The bins to use in the analysis
        self.analysis_bins = {'njets': {'2l': [4,5,6,7,self.hists['njets'].axis('njets').edges()[-1]], # Last bin in topeft.py is 10, this should grab the overflow
                                        '3l': [2,3,4,5,self.hists['njets'].axis('njets').edges()[-1]],
                                        '4l': [2,3,4,self.hists['njets'].axis('njets').edges()[-1]] }}
        if 'ptbl' in self.hists:
            self.analysis_bins['ptbl'] = [0, 100, 200, 400, self.hists['ptbl'].axis('ptbl').edges()[-1]]
        if 'ht' in self.hists:
            self.analysis_bins['ht'] = [0, 100, 200, 300, 400, self.hists['ht'].axis('ht').edges()[-1]]
        if 'ptz' in self.hists:
            self.analysis_bins['ptz'] = [0, 80, 200, 320, 440, self.hists['ptz'].axis('ptz').edges()[-1]]
        if 'o0pt' in self.hists:
            self.analysis_bins['o0pt'] = [0, 100, 200, 400, self.hists['o0pt'].axis('o0pt').edges()[-1]]
        if 'bl0pt' in self.hists:
            self.analysis_bins['bl0pt'] = [0, 100, 200, 400, self.hists['bl0pt'].axis('bl0pt').edges()[-1]]
        if 'l0pt' in self.hists:
            self.analysis_bins['l0pt'] = [0, 50, 100, 200, self.hists['l0pt'].axis('l0pt').edges()[-1]]
        if 'lj0pt' in self.hists:
            self.analysis_bins['lj0pt'] = [0, 150, 250, 500, self.hists['lj0pt'].axis('lj0pt').edges()[-1]]

        if len(self.coeffs)==0: self.coeffs = self.hists['njets']._wcnames
        if len(self.coeffs)==0: self.coeffs = self.hists['njets']._wcnames

        # Get list of channels
        self.ch2lss = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '2lss' in k[1]})
        self.ch2lss += list({k[1]:0 for k in self.hists['njets'].values().keys() if '2lss' in k[1]})
        self.ch2lss_p = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '2lss_p' in k[1]})
        self.ch2lss_p += list({k[1]:0 for k in self.hists['njets'].values().keys() if '2lss_p' in k[1]})
        self.ch2lss_m = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '2lss_m' in k[1]})
        self.ch2lss_m += list({k[1]:0 for k in self.hists['njets'].values().keys() if '2lss_m' in k[1]})
        self.ch2lssj  = list(set([j[-2:].replace('j','') for j in self.ch2lss_p if 'j' in j]))
        self.ch2lssj.sort()
        self.ch3l1b = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l' in k[1] and '1b' in k[1] and 'onZ' not in k[1]})
        self.ch3l1b += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l' in k[1] and '1b' in k[1] and 'onZ' not in k[1]})
        self.ch3l1b_p = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l' in k[1] and 'p' in k[1] and '1b' in k[1]})
        self.ch3l1b_p += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l' in k[1] and 'p' in k[1] and '1b' in k[1]})
        self.ch3l1b_m = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l' in k[1] and 'm' in k[1]  and '1b' in k[1]})
        self.ch3l1b_m += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l' in k[1] and 'm' in k[1]  and '1b' in k[1]})
        self.ch3l2b = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l' in k[1] and '2b' in k[1] and 'onZ' not in k[1]})
        self.ch3l2b += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l' in k[1] and '2b' in k[1] and 'onZ' not in k[1]})
        self.ch3l2b_p = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l' in k[1] and 'p' in k[1]  and '2b' in k[1]})
        self.ch3l2b_p += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l' in k[1] and 'p' in k[1]  and '2b' in k[1]})
        self.ch3l2b_m = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l' in k[1] and 'm' in k[1]  and '2b' in k[1]})
        self.ch3l2b_m += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l' in k[1] and 'm' in k[1]  and '2b' in k[1]})
        self.ch3lsfz = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l_onZ' in k[1]})
        self.ch3lsfz += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l_onZ' in k[1]})
        self.ch3lsfz1b = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l_onZ' in k[1] and '1b' in k[1]})
        self.ch3lsfz1b += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l_onZ' in k[1] and '1b' in k[1]})
        self.ch3lsfz2b = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '3l_onZ' in k[1] and '2b' in k[1]})
        self.ch3lsfz2b += list({k[1]:0 for k in self.hists['njets'].values().keys() if '3l_onZ' in k[1] and '2b' in k[1]})
        self.ch3lj  = list(set([j[-2].replace('j','') for j in self.ch3l1b_p if 'j' in j]))
        self.ch3lj.sort()
        self.ch3lsfzj  = list(set([j[-2].replace('j','') for j in self.ch3l1b_p if 'j' in j]))
        self.ch3lsfzj.sort()
        self.ch4l = list({k[1]:0 for k in self.hists['ptbl'].values().keys() if '4l' in k[1]})
        self.ch4l += list({k[1]:0 for k in self.hists['njets'].values().keys() if '4l' in k[1]})
        self.ch4lj = list(set([j[-2:].replace('j','') for j in self.ch4l if 'j' in j]))
        self.ch4lj.sort()
        self.channels = {'2lss': self.ch2lss, '2lss_p': self.ch2lss_p, '2lss_m': self.ch2lss_m, '3l1b': self.ch3l1b, '3l1b_p': self.ch3l1b_p, '3l1b_m': self.ch3l1b_m, '3l_p_offZ_1b': self.ch3l1b_p, '3l_m_offZ_1b': self.ch3l1b_m, '3l_p_offZ_2b': self.ch3l2b_p, '3l_m_offZ_2b': self.ch3l2b_m, '3l2b': self.ch3l2b,  '3l2b_p': self.ch3l2b_p, '3l2b_m': self.ch3l2b_m, '3l_sfz': self.ch3lsfz, '3l_sfz_1b': self.ch3lsfz1b, '3l_sfz_2b': self.ch3lsfz2b, '3l_onZ_1b': self.ch3lsfz1b, '3l_onZ_2b': self.ch3lsfz2b, '4l': self.ch4l}
        self.skip_process_channels = {**self.skip_process_channels, **{'data': [k for k in self.channels]}} # Skip all data!
        self.skip_process_channels = {**self.skip_process_channels, **{'flips': [k for k in self.channels if '2l' not in k]}} # Charge flips only in 2lss channels

        # Get list of samples and cut levels from histograms
        self.signal = ['ttH','tllq','ttll','ttlnu','tHq','tttt']
        self.samples = list({k[0]:0 for k in self.hists['ptbl'].values().keys()})
        if self.year != '':
            print(f'Only running over {year=}! If this was not intended, please remove the --year (or -y) flag.')
            self.sampels = [k for k in self.samples if self.year[2:] in k]
        if self.do_sm:
            print('Only running over SM!')
        rename = {l: re.split('(Jet)?_[a-zA-Z]*1[6-8]', l)[0] for l in self.samples}
        rename = {k: 'Triboson' if bool(re.search('[WZ]{3}', v)) else v for k,v in rename.items()}
        rename = {k: 'Diboson' if bool(re.search('[WZ]{2}', v)) else v for k,v in rename.items()}
        rename = {k: 'convs' if bool(re.search('TTG', v)) else v for k,v in rename.items()}
        rename = {k: 'fakes' if bool(re.search('nonprompt', v)) else v for k,v in rename.items()}
        rename = {k: 'charge_flips' if bool(re.search('flips', v)) else v for k,v in rename.items()}
        self.rename = {**self.rename, **rename}
        rename = {k.split('_')[0]: v for k,v in rename.items()}
        self.rename = {**self.rename, **rename}
        self.has_nonprompt = not any(['appl' in str(a) for a in self.hists['njets'].axes()]) # Check for nonprompt samples by looking for 'appl' axis
        self.syst = list({k[2]:0 for k in self.hists['ptbl'].values().keys()})
        with open(lumiJson) as jf:
            lumi = json.load(jf)
            self.lumi = lumi
        self.lumi = {year : 1000*lumi for year,lumi in self.lumi.items()}

    def should_skip_process(self, proc, channel):
        for proc_skip,channel_skip in self.skip_process_channels.items():
            if proc_skip in proc or proc_skip in self.rename[proc]:
                if isinstance(channel_skip, list):
                    if any(channel_skip in channel for channel_skip in self.skip_process_channels[proc_skip]):
                        return True # Should skip this process for this channel
                elif channel_skip in channel:
                        return True # Should skip this process for this channel
        return False # Nothing to skip

    def analyzeChannel(self, channel=[], appl='isSR_2lSS', charges=['ch+','ch-'], systematics='nominal', variable='njets', bins=[]):
        if variable != 'njets' and isinstance(bins, list) and len(bins)>0:
            for jbin in bins:
                self.analyzeChannel(channel=channel, appl=appl, charges=charges, systematics=systematics, variable=variable, bins=jbin)
            return
        def export1d(h, name, cat, fout):
            if 'data_obs' in name:
                fout['data_obs'] = hist.export1d(h['nominal'])
            else:
                for syst,histo in h.items():
                    rename = self.rename[name] if name in  self.rename else ''
                    if syst == 'nominal':
                        fout[name+cat] = hist.export1d(histo)
                    elif rename not in self.syst_special:
                        fout[name+cat+'_'+syst] = hist.export1d(histo)
        def export2d(h):
            return h.to_hist().to_numpy()
        if isinstance(channel, str) and channel not in self.channels:
           raise Exception(f'{channel} not found in self.channels!')
        if isinstance(channel, list) and not all(ch in self.channels for ch in self.channels.keys()):
           print(self.channels.keys())
           print([[ch, ch in self.channels.keys()] for ch in channel])
           raise Exception(f'At least one channel in {channels} is not found in self.channels!')
        if self.has_nonprompt:
            h = self.hists[variable] # 'appl' axis is removed in nonprmopt samples, everything is 'isSR'
        else:
            h = self.hists[variable].integrate('appl', appl)
        if isinstance(charges, str):
            charge = 'p' if charges == 'ch+' else 'm'
            if isinstance(bins, str):
                if variable == 'njets':
                    chan = [c for c in self.channels[channel+'_'+charge] if 'j' not in c]
                else:
                    chan = [c for c in self.channels[channel+'_'+charge] if bins+'j' in c]
                    channel = chan[0]
                h = h.integrate('channel', chan)
            else:
                if variable == 'njets':
                    chan = [c for c in self.channels[channel+'_'+charge] if 'j' not in c]
                    channel = channel + '_' + charge
                    h = h.integrate('channel', chan)
                else:
                    h = h.integrate('channel', self.channels[channel+'_'+charge])
        else:
            if isinstance(bins, str):
                if variable == 'njets':
                    chan = [c for c in self.channels[channel] if bins in c and 'j' not in c]
                else:
                    chan = [c for c in self.channels[channel] if bins+'j' in c]
                    channel = chan[0]
                h = h.integrate('channel', chan)
            else:
                if variable == 'njets':
                    chan = [c for c in self.channels[channel] if 'j' not in c]
                    h = h.integrate('channel', chan)
                else:
                    h = h.integrate('channel', self.channels[channel])
        all_str = ' '.join([f'{v}' for v in locals().values() if v != self.hists])
        all_str = f'{channel} {systematics} {variable}'
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
            if 'b' in channel:
                cat = channel
            else:
                cat = '_'.join([channel, maxb])  
        else:
            if 'b' in channel:
                cat = '_'.join([channel, variable])  
            else:
                cat = '_'.join([channel, maxb, variable])
        fname = f'histos/tmp_ttx_multileptons-{cat}.root'
        fout = uproot3.recreate(fname)
        # Scale each plot to the SM
        processed = []
        for proc in self.samples:
            if proc in self.ignore or self.rename[proc] in self.ignore: continue # Skip any CR processes that might be in the pkl file
            if self.should_skip_process(proc, channel): continue
            simplified = proc.split('_central')[0].split('_private')[0].replace('_4F','').replace('_ext','')
            if simplified in processed: continue # Only one process name per 3 years
            processed.append(simplified)
            p = proc.split('_')[0]
            ul = {'20'+k.split('UL')[1]:k for k in self.samples if p.replace('_4F','').replace('_ext','') in k}
            # Integrate out processes
            h_base = h.group('sample', hist.Cat('year', 'year'), ul)
            if h_base.values() == {}:
                print(f'Issue with {proc}')
                continue
            years = {year : self.lumi[year] for year in ul}
            h_base.scale(years, axis='year')
            h_base = h_base.integrate('year')
            pname = self.rename[p]+'_' if p in self.rename else p
            pname.replace('_4F','').replace('_ext','')
            if isinstance(self.analysis_bins[variable],dict):
                lep_bin = channel.split('_')[0].split('l')[0] + 'l'
                h_base = h_base.rebin(variable, hist.Bin(variable,  h.axis(variable).label, self.analysis_bins[variable][lep_bin]))
            else:
                h_base = h_base.rebin(variable, hist.Bin(variable,  h.axis(variable).label, self.analysis_bins[variable]))
            # Save the SM plot
            h_bases = {syst: h_base.integrate('systematic', syst) for syst in self.syst}
            h_base = h_base.integrate('systematic', 'nominal')
            h_sm = h_bases
            for hists in h_sm.values():
                hists.set_sm()
            if len(h_base.axes())>1:
                fout[pname+'sm'] = export2d(h_bases)
            else:
                if any([sig in p for sig in self.signal]):
                    export1d(h_sm, pname, 'sm', fout) # Special case for SM b/c background names overlap (p not pname)
                else:
                    export1d(h_sm, p, '_sm', fout) # Special case for SM b/c background names overlap (p not pname)
            # Asimov data: data_obs = MC at SM (all WCs = 0)
            if len(h_base.axes())>1:
                fout['data_obs'] = export2d(h_sm)
            else:
                export1d(h_sm, 'data_obs', 'sm', fout)

            isSignal = p in self.signal or self.rename[p] in self.signal
            if not self.do_sm and isSignal:
                h_lin = h_bases; h_quad = None; h_mix = None
                for name,wcpt in self.wcs:
                    # Scale plot to the WCPoint
                    # Handle linear and quadratic terms
                    if 'lin' in name:
                        h_lin = h_bases
                        for hists in h_lin.values():
                            hists.set_wilson_coefficients(**wcpt)
                        if len(h_base.axes())>1:
                            fout[pname+name] = export2d(h_lin)
                        else:
                            export1d(h_lin, pname, name, fout)
                        if variable == 'njets':
                            if isinstance(charges, str):
                                cat = '_'.join([channel, charge, ])  
                            else:
                                cat = '_'.join([channel, maxb])  
                        else:
                            if 'b' in channel:
                                cat = channel
                            elif isinstance(charges, str):
                                cat = '_'.join([channel, charge, maxb, variable])
                            else:
                                cat = '_'.join([channel, maxb, variable])
                    elif 'quad' in name and 'mix' not in name:
                        h_quad = h_bases
                        for hists in h_quad.values():
                            hists.set_wilson_coefficients(**wcpt)
                        if len(h_base.axes())>1:
                            fout[pname+name] = export2d(h_quad)
                        else:
                            export1d(h_quad, pname, name, fout)
                    else:
                        h_mix = h_bases
                        for hists in h_mix.values():
                            hists.set_wilson_coefficients(**wcpt)
                        if len(h_base.axes())>1:
                            fout[pname+name] = export2d(h_mix)
                        else:
                            export1d(h_mix, pname, name, fout)
        
        fout.close()
        self.makeCardLevel(channel=channel, charges=charges, nbjet=maxb, systematics=systematics, variable=variable)

    def makeCardLevel(self, channel=[], charges=['ch+','ch-'], nbjet='2+bm', systematics='nominal', variable='njets'):
        '''
        Create datacard files from temp uproot outputs
        Creates histograms for ``combine``:
        ``S`` is the SM
        ``S+L_i+Q_i`` sets ``WC_i=1`` and the rest to ``0``
        ``Q`` is built from the ``WC=0``, ``WC=1``, and ``WC=2`` pieces
        ``S+L_i+L_j+Q_i+Q_j+2 M_IJ`` set ``WC_i=1``, ``WC_j=1`` and the rest to ``0``
        '''
        def getHist(d_hists,name):
            h = d_hists[name]
            xmin = h.GetXaxis().GetXmin()
            xmax = h.GetXaxis().GetXmax()
            xwidth = h.GetXaxis().GetBinWidth(1)
            h.GetXaxis().SetRangeUser(xmin, xmax + 1.5*xwidth) #Include overflow bin in ROOT
            return deepcopy(h) # to protect d_hists from modifications 

        def processSyst(process, systMap, d_hists, fout):
            for syst in self.syst:
                if channel in self.skip_process_channels and self.skip_process_channels[channel] in syst: continue
                if any([process+'_'+syst in d for d in d_hists]):
                    h_sys = getHist(d_hists, '_'.join([process,syst]))
                    h_sys.SetDirectory(fout)

                    # Need to handle quad term to get "Q"
                    if (("quad" in process) and ("mixed" not in process)):

                        # Get the parts of the name (from e.g. "ttH_quad_ctG")
                        proc_str = process.split("_")[0]
                        wc_str = process.split("_")[2]

                        # Construct the names for the corresponding sm and lin terms, and get the hists
                        name_s = '_'.join([proc_str,'sm'])
                        name_l = '_'.join([proc_str,'lin',wc])
                        h_s_sys = getHist(d_hists, '_'.join([name_s,syst]))
                        h_l_sys = getHist(d_hists, '_'.join([name_l,syst]))

                        # Find the Q term (Q = (Histo(WC=2) - 2*Histo(WC=1) + Histo(WC=0))/2)
                        h_sys.Add(h_l_sys, -2)
                        h_sys.Add(h_s_sys)
                        h_sys.Scale(0.5)

                    # Write output
                    h_sys.Write()

                    if 'Down' in syst: continue # The datacard only stores the systematic name, and combine tacks on Up/Down later
                    syst = syst.replace('Up', '') # Remove 'Up' to get just the systematic name
                    if syst in systMap:
                        systMap[syst].update({process: round(h_sys.Integral(), 3)})
                    else:
                        systMap[syst] = {process: round(h_sys.Integral(), 3)}
            for syst_special,val in self.syst_special.items():
                if syst_special not in process: continue
                if syst in systMap:
                    systMap[syst_special+'_flat_rate'].update({process: 1+val})
                else:
                    systMap[syst_special+'_flat_rate'] = {process: 1+val}
                    
        print(f'Making the datacard for {channel}')
        if isinstance(charges, str): charge = charges
        else: charge = ''
        charge = 'p' if charge == 'ch+' else 'm'
        if systematics == 'nominal': sys = ''
        else: sys = '_'+systematics
        if variable == 'njets':
            if 'b' in channel:
                cat = channel
            else:
                cat = '_'.join([channel, nbjet])
        else:
            if 'b' in channel:
                cat = '_'.join([channel, variable])  
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
        d_sigs = {} # Store signals for summing
        d_bkgs = {} # Store backgrounds for summing
        samples = list(set([proc.split('_')[0] for proc in self.samples]))
        samples.sort()
        selectedWCsForProc={}
        for proc in samples:
            if proc in self.ignore or self.rename[proc] in self.ignore: continue # Skip any CR processes that might be in the pkl file
            if self.should_skip_process(proc, channel): continue
            p = self.rename[proc] if proc in self.rename else proc
            name = 'data_obs'
            if name not in d_hists:
                print(f'{name} not found in {channel}!')
                continue
            '''
            These lines are for testing only, and create Asimov data based on all processes provided
            '''
            if isinstance(data_obs, list):
                if any([sig in proc for sig in self.signal]):
                    data_obs = getHist(d_hists,p+'_sm').Clone('data_obs')
                else:
                    data_obs = getHist(d_hists,proc+'_sm').Clone('data_obs') # Special case for SM b/c background names overlap
            else:
                if any([sig in proc for sig in self.signal]):
                    data_obs.Add(getHist(d_hists,p+'_sm').Clone('data_obs'))
                else:
                    data_obs.Add(getHist(d_hists,proc+'_sm').Clone('data_obs')) # Special case for SM b/c background names overlap
            data_obs.SetDirectory(fout)
            allyields[name] = data_obs.Integral()
            fout.Delete(name+';1')
            data_obs.Write()
            pname = self.rename[proc]+'_' if proc in self.rename else proc+'_'
            name = pname + 'sm'
            if name not in d_hists and proc+'_sm' not in d_hists:
                print(f'{name} not found in {channel}!')
                continue
            if any([sig in proc for sig in self.signal]):
                h_sm = getHist(d_hists, name)
            else:
                h_sm = getHist(d_hists, proc+'_sm') # Special case for SM b/c background names overlap
            if True or h_sm.Integral() > self.tolerance or p not in self.signal:
                if p in self.signal:
                    if name in iproc:
                        allyields[name] += h_sm.Integral()
                        d_sigs[name].Add(h_sm)
                        fout.Delete(name+';1')
                        h_sm = d_sigs[name]
                    else:
                        signalcount -= 1
                        iproc[name] = signalcount
                        allyields[name] = h_sm.Integral()
                        d_sigs[name] = h_sm
                else:
                    if name in iproc:
                        allyields[name] += h_sm.Integral()
                        d_bkgs[name].Add(h_sm)
                        fout.Delete(name+';1')
                        h_sm = d_bkgs[name]
                    else:
                        iproc[name] = bkgcount
                        allyields[name] = h_sm.Integral()
                        bkgcount += 1
                        d_bkgs[name] = h_sm
                if self.do_nuisance: processSyst(name, systMap, d_hists, fout)
                h_sm.SetDirectory(fout)
                h_sm.SetName(name)
                h_sm.SetTitle(name)
                h_sm.Write()
                if p not in self.signal:
                    continue

            # Let's select now the coefficients that actually have an impact
            selectedWCs=[]

            for n,wc in enumerate(self.coeffs):
                if self.do_sm: break
                
                # Check if linear terms are non null
                name = '_'.join([pname[:-1],'lin',wc])
                tmp = getHist(d_hists, name); tmp.Add(h_sm,-1)

                if abs(tmp.Integral() / h_sm.Integral()) > self.tolerance:
                    selectedWCs.append(wc)
                    continue                    

                # Check if quadratic terms are non null
                name = '_'.join([pname[:-1],'quad',wc])
                tmp = getHist(d_hists, name); tmp.Add(h_sm,-1)
                if abs(tmp.Integral() / h_sm.Integral()) > self.tolerance:
                    selectedWCs.append(wc)
                    continue

                # Check if crossed terms are non null
                anyIsNonZero=False
                for wc2 in [self.coeffs[w2] for w2 in range(n)]:
                    name = '_'.join([pname[:-1],'quad_mixed',wc,wc2])
                    name2 = '_'.join([pname[:-1],'lin',wc2])
                    h_mix = getHist(d_hists, name)
                    h_lin2 = getHist(d_hists, name2)
                    tmp=deepcopy(h_mix); tmp.Add(h_lin2,-1) # should be S+L1+Q1+M12, if its small we can skip it 
                    if abs(tmp.Integral() / h_sm.Integral()) > self.tolerance:
                        selectedWCs.append(wc)
                        break

            # Find the "S+L+Q", "Q", and "S+Li+Lj+Qi+Qj+2Mij" pieces
            selectedWCsForProc[pname[:-1]]=selectedWCs
            for n,wc in enumerate(selectedWCs):
                if self.do_sm: break

                # Get the "S+L+Q" piece
                name = '_'.join([pname[:-1],'lin',wc])
                if name not in d_hists:
                    print(f'Histogram {name} not found in {channel}! Probably below the tolerance. If so, ignore this message!')
                    continue
                h_lin = getHist(d_hists, name)
                if name in iproc:
                    allyields[name] += h_lin.Integral()
                    d_sigs[name].Add(h_lin)
                    fout.Delete(name+';1')
                    h_lin = d_sigs[name]
                else:
                    signalcount -= 1
                    iproc[name] = signalcount
                    allyields[name] = h_lin.Integral()
                    d_sigs[name] = h_lin
                h_lin.SetDirectory(fout)
                h_lin.Write()
                if allyields[name] < 0:
                    raise Exception(f"This value {allyields[name]} should not be negative, check for bugs upstream.")

                if self.do_nuisance: processSyst(name, systMap, d_hists, fout)

                # Get the "Q" piece
                name = '_'.join([pname[:-1],'quad',wc])
                if name not in d_hists:
                    print(f'Histogram {name} not found in {channel}! Probably below the tolerance. If so, ignore this message!')
                    continue
                h_quad = getHist(d_hists, name)
                h_quad.Add(h_lin, -2)
                h_quad.Add(h_sm)
                h_quad.Scale(0.5)
                if name in iproc:
                    allyields[name] += h_quad.Integral()
                    d_sigs[name].Add(h_quad)
                    fout.Delete(name+';1')
                    h_quad = d_sigs[name]
                else:
                    signalcount -= 1
                    iproc[name] = signalcount
                    allyields[name] = h_quad.Integral()
                    d_sigs[name] = h_quad
                h_quad.SetDirectory(fout)
                h_quad.Write()
                if allyields[name] < 0:
                    raise Exception(f"This value {allyields[name]} should not be negative (except potentially due to rounding errors, if the value is tiny), check for bugs upstream.")
                if self.do_nuisance: processSyst(name, systMap, d_hists, fout)
                

                # Get the "S+Li+Lj+Qi+Qj+2Mij" piece
                for wc2 in [selectedWCs[w2] for w2 in range(n)]:
                    
                    name = '_'.join([pname[:-1],'quad_mixed',wc,wc2])
                    name2 = '_'.join([pname[:-1],'lin',wc2])
                    if name not in d_hists:
                        print(f'Histogram {name} not found in {channel}! Probably below the tolerance. If so, ignore this message!')
                        continue
                    h_mix = getHist(d_hists, name)
                    skipMe=False
                    if name in iproc:
                        allyields[name] += h_mix.Integral()
                        d_sigs[name].Add(h_mix)
                        fout.Delete(name+';1')
                        h_mix = d_sigs[name]
                    else:
                        signalcount -= 1
                        iproc[name] = signalcount
                        allyields[name] = h_mix.Integral()
                        d_sigs[name] = h_mix
                    h_mix.SetDirectory(fout)
                    h_mix.Write()
                    allyields[name] = h_mix.Integral()
                    if allyields[name] < 0:
                        raise Exception(f"This value {allyields[name]} should not be negative, check for bugs upstream.")
                    if self.do_nuisance: processSyst(name, systMap, d_hists, fout)

        selectedWCsFile=open(f'histos/selectedWCs-{cat}.txt','w')
        json.dump(selectedWCsForProc, selectedWCsFile)
        selectedWCsFile.close()

        # Write datacard
        allyields = {k : (v if v>0 else 0) for k,v in allyields.items()}
        if systematics != 'nominal':
            cat = cat + '_' + systematics
        nuisances = [syst for syst in systMap]
        datacard = open("histos/ttx_multileptons-%s.txt"%cat, "w"); 
        datacard.write("shapes *        * ttx_multileptons-%s.root $PROCESS $PROCESS_$SYSTEMATIC\n" % cat)
        cat = 'bin_'+cat
        datacard.write('##----------------------------------\n')
        datacard.write('bin         %s\n' % cat)
        datacard.write('observation %%.%df\n' % 3 % allyields['data_obs'])
        #datacard.write('observation %%.%df\n' % np.abs(int(np.format_float_scientific(self.tolerance).split('e')[1])) % allyields['data_obs'])
        datacard.write('##----------------------------------\n')
        klen = max([7, len(cat)]+[len(p[0]) for p in iproc.keys()])
        kpatt = " %%%ds "  % klen
        fpatt = " %%%d.%df " % (klen,np.abs(3))
        #fpatt = " %%%d.%df " % (klen,np.abs(int(np.format_float_scientific(self.tolerance).split('e')[1])))#3)
        npatt = "%%-%ds " % max([len('process')]+list(map(len,nuisances)))
        datacard.write('##----------------------------------\n')
        procs = iproc.keys()
        datacard.write((npatt % 'bin    ')+(" "*6)+(" ".join([kpatt % cat      for p in procs]))+"\n")
        datacard.write((npatt % 'process')+(" "*6)+(" ".join([kpatt % p        for p in procs]))+"\n")
        datacard.write((npatt % 'process')+(" "*6)+(" ".join([kpatt % iproc[p] for p in procs]))+"\n")
        datacard.write((npatt % 'rate   ')+(" "*6)+(" ".join([fpatt % allyields[p] for p in procs]))+"\n")
        datacard.write('##----------------------------------\n')
        if self.do_nuisance:
            for syst in nuisances:
                systEff = dict((p,"1" if p in systMap[syst] else "-") for p in procs if 'rate' not in syst)
                systEffRate = dict((p,systMap[syst][p] if p in systMap[syst] else "-") for p in procs if 'rate' in syst)
                if 'rate' in syst:
                    datacard.write(('%s %5s' % (npatt % syst.replace('_rate',''),'lnN')) + " ".join([kpatt % systEffRate[p]  for p in procs if p in systEffRate]) +"\n")
                else:
                    datacard.write(('%s %5s' % (npatt % syst,'shape')) + " ".join([kpatt % systEff[p]  for p in procs if p in systEff]) +"\n")
        
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
        # Case for a single wc
        elif isinstance(wc, str):
            wl = {k:0 for k in self.coeffs}
            wl[wc] = 1.
            wcpt.append([f'lin_{wc}', wl])
        elif len(wc)==1:
            wl = {k:0 for k in self.coeffs}
            wl[wc] = 1.
            wcpt.append([f'lin_{wc}', wl])
        # Case for 2+ wcs
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
                    if(wc1==wc2):  wcpt.append([f'quad_{wc1}', wl])
                    else: wcpt.append([f'quad_mixed_{wc1}_{wc2}', wl])
        self.wcs     = wcpt
        return wcpt
    def condor_job(self, pklfile, njobs, wcs, do_nuisance, do_sm, var_lst):
        os.system('mkdir -p %s/condor' % os.getcwd())
        os.system('mkdir -p %s/condor/log' % os.getcwd())
        target = '%s/condor_submit.sh' % os.getcwd()
        condorFile = open(target,'w')
        condorFile.write('source %s/miniconda3/etc/profile.d/conda.sh\n' % os.path.expanduser('~'))
        condorFile.write('unset PYTHONPATH\n')
        condorFile.write('conda activate %s\n' % os.environ['CONDA_DEFAULT_ENV'])
        condorFile.write('cluster=$1\n')
        condorFile.write('job=$2\n')
        condorFile.write('\n')
        args = []
        args.append('--var-lst ' + ' '.join(var_lst))
        if do_nuisance: args.append('--do-nuisance')
        if len(wcs) > 0: args.append('--POI ' + ','.join(wcs))
        if do_sm: args.append('--do-sm')
        if len(args) > 0:
            condorFile.write('python analysis/topEFT/datacard_maker.py %s --job "${job}" %s\n' % (pklfile, ' '.join(args)))
        else:
            condorFile.write('python analysis/topEFT/datacard_maker.py %s --job "${job}"\n' % pklfile)
        os.system('chmod 777 condor_submit.sh')
        target = '%s/condor/datacardmaker' % os.getcwd()
        condorFile = open(target,'w')
        condorFile.write('universe              = vanilla\n')
        condorFile.write('executable            = condor_submit.sh\n')
        if do_nuisance: condorFile.write('arguments             = $(ClusterID) $(ProcId) 1\n')
        else: condorFile.write('arguments             = $(ClusterID) $(ProcId)\n')
        condorFile.write('output                = condor/log/$(ClusterID)_$(ProcId).out\n')
        condorFile.write('error                 = condor/log/$(ClusterID)_$(ProcId).err\n')
        condorFile.write('log                   = condor/log/$(ClusterID).log\n')
        condorFile.write('Rank                  = Memory >= 64\n')
        condorFile.write('Request_Memory        = 3 Gb\n')
        condorFile.write('+JobFlavour           = "workday"\n')
        condorFile.write('getenv                = True\n')
        condorFile.write('Should_Transfer_Files = NO\n')
        condorFile.write('queue %d' % njobs)
        condorFile.close()
        os.system('condor_submit %s -batch-name TopCoffea-datacard-maker' % target)
        os.system('rm %s' % target)


if __name__ == '__main__':

    import argparse

    from topcoffea.modules.YieldTools import YieldTools
    yt = YieldTools()

    parser = argparse.ArgumentParser(description='You can select which file to run over')
    parser.add_argument('pklfile'           , nargs='?', default=''           , help = 'Pickle file with histograms')
    parser.add_argument('--lumiJson', '-l', default='topcoffea/json/lumi.json'     , help = 'Lumi json file')
    parser.add_argument('--do-nuisance',    action='store_true', help = 'Include nuisance parameters')
    parser.add_argument('--POI',            default=[],  help = 'List of WCs (comma separated)')
    parser.add_argument('--job',      '-j', default='-1'       , help = 'Job to run')
    parser.add_argument('--year',     '-y', default=''         , help = 'Run over single year')
    parser.add_argument('--do-sm',          action='store_true', help = 'Run over SM only')
    parser.add_argument('--var-lst',        default=[], action='extend', nargs='+', help = 'Specify a list of variables to make cards for.')

    args = parser.parse_args()
    pklfile  = args.pklfile
    lumiJson = args.lumiJson
    do_nuisance = args.do_nuisance
    wcs = args.POI
    job = int(args.job)
    year = args.year
    do_sm = args.do_sm
    var_lst = args.var_lst
    if isinstance(wcs, str): wcs = wcs.split(',')
    if pklfile == '':
        raise Exception('Please specify a pkl file!')
    if do_nuisance: print('Running with nuisance parameters, this will take a bit longer')
    if job > -1: print('Only running one job locally')
    else: print('Submitting all jobs to condor')

    card = DatacardMaker(pklfile, lumiJson, do_nuisance, wcs, year, do_sm)
    card.read()
    card.buildWCString()
    jobs = []

    # Get the list of hists to make datacards for
    # If we don't specify a variable, use all variables in the pkl that we have a binning defined for
    include_var_lst = []
    target_var_lst = var_lst
    if len(var_lst) == 0:
        target_var_lst = yt.get_hist_list(pklfile)
    for var_name in target_var_lst:
        if var_name in card.known_var_lst:
            include_var_lst.append(var_name)
    if len(include_var_lst) == 0: raise Exception("No variables specified")
    print(f"\nMaking cards for: {include_var_lst}\n")

    # Set up cards lst
    for var in include_var_lst:
        if var == 'ptz': continue # This var only applies to a subset of the channels
        cards = [{'channel':'2lss', 'appl':'isSR_2lSS', 'charges':'ch+', 'systematics':'nominal', 'variable':var, 'bins':card.ch2lssj},
                 {'channel':'2lss', 'appl':'isSR_2lSS', 'charges':'ch-', 'systematics':'nominal', 'variable':var, 'bins':card.ch2lssj},
                 {'channel':'3l1b', 'appl':'isSR_3l', 'charges':'ch+', 'systematics':'nominal', 'variable':var, 'bins':card.ch3lj},
                 {'channel':'3l1b', 'appl':'isSR_3l', 'charges':'ch-', 'systematics':'nominal', 'variable':var, 'bins':card.ch3lj},
                 {'channel':'3l2b', 'appl':'isSR_3l', 'charges':'ch+', 'systematics':'nominal', 'variable':var, 'bins':card.ch3lj},
                 {'channel':'3l2b', 'appl':'isSR_3l', 'charges':'ch-', 'systematics':'nominal', 'variable':var, 'bins':card.ch3lj},
                 {'channel':'3l_sfz_1b', 'appl':'isSR_3l', 'charges':['ch+','ch-'], 'systematics':'nominal', 'variable':var, 'bins':card.ch3lsfzj},
                 {'channel':'3l_sfz_2b', 'appl':'isSR_3l', 'charges':['ch+','ch-'], 'systematics':'nominal', 'variable':var, 'bins':card.ch3lsfzj},
                 {'channel':'4l', 'appl':'isSR_4l', 'charges':['ch+','ch0','ch-'], 'systematics':'nominal', 'variable':var, 'bins':card.ch4lj}]
        jobs.append(cards)
    if 'ptz' in include_var_lst:
        cards = [
            {'channel':'3l_sfz_1b', 'appl':'isSR_3l', 'charges':['ch+','ch-'], 'systematics':'nominal', 'variable':var, 'bins':card.ch3lsfzj},
            {'channel':'3l_sfz_2b', 'appl':'isSR_3l', 'charges':['ch+','ch-'], 'systematics':'nominal', 'variable':var, 'bins':card.ch3lsfzj},
        ]
        jobs.append(cards)

    njobs = 0
    for j in jobs:
        njobs = njobs + len(j)
    if job == -1:
        card.condor_job(pklfile, njobs, wcs, do_nuisance, do_sm, include_var_lst)
    elif job < njobs:
        d = jobs[job//len(jobs[0])][job%len(jobs[0])]
        card.analyzeChannel(**d)
    else:
        raise Exception(f'Job number {job} outside of range {njobs}!')
