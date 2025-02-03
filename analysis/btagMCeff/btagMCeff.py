#!/usr/bin/env python
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
#from coffea.arrays import Initialize # Not used and gives error
from coffea import processor
import hist
from coffea.util import load
import coffea.analysis_tools

import topeft.modules.object_selection as te_os
import topcoffea.modules.object_selection as tc_os
from topeft.modules.paths import topeft_path

from topcoffea.modules.get_param_from_jsons import GetParam
get_te_param = GetParam(topeft_path("params/params.json"))

#coffea.deprecations_as_errors = True

# In the future these names will be read from the nanoAOD files

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples):
        self._samples = samples

        # Create the histograms
        # In general, histograms depend on 'sample', 'channel' (final state) and 'cut' (level of selection)
        jpt_axis = hist.axis.Variable([20, 30, 60, 120], name="pt", label="Jet p_{T} (GeV)")
        jetpt_axis = hist.axis.Regular(40, 0, 800, name="pt", label="Jet p_{T} (GeV)")
        jeta_axis = hist.axis.Regular(25, -2.5, 2.5, name="eta", label=r"Jet \eta (GeV)")
        jeta_axis = hist.axis.Regular(25, -2.5, 2.5, name="eta", label=r"Jet \eta (GeV)")
        jaeta_axis = hist.axis.Variable([0, 1, 1.8, 2.4], name="abseta", label=r"Jet \eta (GeV)")
        Flav_axis = hist.axis.StrCategory([], name="Flav", growth=True)
        flav_axis = hist.axis.IntCategory([], name="flav", growth=True)
        wp_axis = hist.axis.StrCategory([], name="WP", growth=True)
        genjet_n_axis = hist.axis.StrCategory([], name="genjet_n", growth=True)
        recojet_n_axis = hist.axis.StrCategory([], name="recojet_n", growth=True)
        misstag_n_axis = hist.axis.StrCategory([], name="misstag_n", growth=True)
        misstag_rate_axis = hist.axis.StrCategory([], name="misstag_rate", growth=True)
        btag_efficiency_axis = hist.axis.StrCategory([], name="btag_efficiency", growth=True)

        self._accumulator = {
            'jetpt'  : hist.Hist(wp_axis, Flav_axis, jpt_axis),
            'jeteta'  : hist.Hist(wp_axis, Flav_axis, jeta_axis),
            'jetpteta'  : hist.Hist(wp_axis, Flav_axis, jpt_axis, jaeta_axis),
            'jetptetaflav'  : hist.Hist(wp_axis, jetpt_axis, jaeta_axis, flav_axis),
            'DeepJet' : hist.Hist(genjet_n_axis, recojet_n_axis, misstag_n_axis, misstag_rate_axis, btag_efficiency_axis),
            'PNet': hist.Hist(genjet_n_axis, recojet_n_axis, misstag_n_axis, misstag_rate_axis, btag_efficiency_axis)
        }

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        events = events[0:10]
        # Dataset parameters
        dataset = events.metadata['dataset']
        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights' ]
        isData = self._samples[dataset]['isData']
        datasets = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets:
            if d in dataset: dataset = dataset.split('_')[0]

        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3

        # Initialize objects
        met = events.MET
        e   = events.Electron
        mu  = events.Muon
        j   = events.Jet

        if is_run3:
            leptonSelection = te_os.run3leptonselection()
            jetsRho = events.Rho["fixedGridRhoFastjetAll"]
        elif is_run2:
            leptonSelection = te_os.run2leptonselection()
            jetsRho = events.fixedGridRhoFastjetAll

        # Muon selection
        mu["conept"] = leptonSelection.coneptMuon(mu)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)
        mu["isPres"] = leptonSelection.isPresMuon(mu)
        mu["isFO"] = leptonSelection.isFOMuon(mu, year)
        mu["isTight"]= leptonSelection.tightSelMuon(mu)
        mu['isGood'] = mu['isPres'] & mu['isTight']

        leading_mu = mu[ak.argmax(mu.pt,axis=-1,keepdims=True)]
        leading_mu = leading_mu[leading_mu.isGood]

        mu = mu[mu.isGood]
        mu_pres = mu[mu.isPres]

        # Electron selection
        e["idEmu"] = te_os.ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = leptonSelection.coneptElec(e)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)
        e["isPres"] = leptonSelection.isPresElec(e)
        e["isFO"] = leptonSelection.isFOElec(e, year)
        e["isTight"] = leptonSelection.tightSelElec(e)
        e["isLooseE"] = leptonSelection.isLooseElec(e)
        e['isClean'] = te_os.isClean(e, mu, drmin=0.05)
        e['isGood']  = e['isPres'] & e['isTight'] & e['isClean']

        leading_e = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        leading_e = leading_e[leading_e.isGood]

        e  =  e[e.isGood]
        e_pres = e[e.isPres & e.isClean]

        nElec = ak.num(e)
        nMuon = ak.num(mu)

        twoLeps   = (nElec+nMuon) == 2
        threeLeps = (nElec+nMuon) == 3
        twoElec   = (nElec == 2)
        twoMuon   = (nMuon == 2)
        e0 = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        m0 = mu[ak.argmax(mu.pt,axis=-1,keepdims=True)]

        # Jet selection

        jetptname = 'pt_nom' if hasattr(j, 'pt_nom') else 'pt'
        j["isGood"] = tc_os.is_tight_jet(getattr(j, jetptname), j.eta, j.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=0)
        j['isClean'] = te_os.isClean(j, e, drmin=0.4)& te_os.isClean(j, mu, drmin=0.4)
        goodJets = j[(j.isClean)&(j.isGood)]

        # Define b-tagging thresholds
        DeepJet_btagcut = 0.3086
        PNet_btagcut    = 0.245

        # Define masks for b-jets and tagged jets using goodJets.partonFlavour
        bjet_mask = (goodJets.partonFlavour == 5) | (goodJets.partonFlavour == -5)
        maskDeepJet = goodJets.btagDeepFlavB > DeepJet_btagcut
        maskPNetB   = goodJets.btagPNetB > PNet_btagcut

        # Count total b-jets per event                                                                                              
        gen_jet_total = ak.sum(bjet_mask)

        Deeptag_jet_total = ak.sum(maskDeepJet & bjet_mask)
        PNet_jet_total    = ak.sum(maskPNetB & bjet_mask)

        # Count untagged b-jets (b-jets that were missed)                                                                           
        uniden_Deepjet = ak.sum(bjet_mask & ~maskDeepJet)
        uniden_PNet    = ak.sum(bjet_mask & ~maskPNetB)

        # Count mistagged jets (jets that are NOT b-jets but are tagged)                                                            
        mistagDeepJet_mask = maskDeepJet & ~bjet_mask
        mistagPNet_mask    = maskPNetB & ~bjet_mask

        miss_Deepjet = ak.sum(mistagDeepJet_mask)
        miss_PNet    = ak.sum(mistagPNet_mask)

        # Compute rates per event (handling division by zero)                                                                       
        Deep_btagefficiency = Deeptag_jet_total / gen_jet_total
        PNet_btagefficiency = PNet_jet_total / gen_jet_total

        Deepmisstag_rate = miss_Deepjet / Deeptag_jet_total
        PNetmisstag_rate = miss_PNet / PNet_jet_total

        # Print results per event if you want to check runs with future, but comment out these lines when running with work_queue_factory!                                                                                                 
        #print(f"Total b-jets: {gen_jet_total}")
        #print(f"DeepJet: Tagged {Deeptag_jet_total}, Missed {uniden_Deepjet}, Mistagged {miss_Deepjet}")
        #print(f"PNet: Tagged {PNet_jet_total}, Missed {uniden_PNet}, Mistagged {miss_PNet}")
        #print(f"DeepJet Efficiency: {Deep_btagefficiency:.4f}, Mistag Rate: {Deepmisstag_rate:.4f}")
        #print(f"PNet Efficiency: {PNet_btagefficiency:.4f}, Mistag Rate: {PNetmisstag_rate:.4f}")

        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]

        ### We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        weights = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        weights.add('norm', np.ones_like(met.pt))

        eftweights = events['EFTfitCoefficients'] if hasattr(events, "EFTfitCoefficients") else []

        hout = self.accumulator
        normweights = weights.weight().flatten()
        #hout['SumOfEFTweights'].fill(eftweights, sample=dataset, SumOfEFTweights=varnames['counts'], weight=normweights)

        flavSelection = {'b': (np.abs(goodJets.hadronFlavour) == 5), 'c': (np.abs(goodJets.hadronFlavour) == 4), 'l': (np.abs(goodJets.hadronFlavour) <= 3) }

        WP = {'all' : -999., 'loose': 0.0490, 'medium': 0.2783, 'tight': 0.7100}
        btagSelection = {}
        for wp, wpvals in WP.items():
            btagSelection[wp] = (goodJets.btagDeepFlavB>wpvals)

        for jetype in ['b', 'c', 'l']:
            for wp in WP.keys():
                mask = (flavSelection[jetype])&(btagSelection[wp])
                selectjets = goodJets[mask]
                pts     = ak.flatten(selectjets.pt)
                etas    = ak.flatten(selectjets.eta)
                absetas = ak.flatten(np.abs(selectjets.eta))
                flavarray = np.zeros_like(pts) if jetype == 'l' else (np.ones_like(pts)*(4 if jetype=='c' else 5))
                weights =  np.ones_like(pts)
                hout['jetpt'].fill(WP=wp, Flav=jetype,  pt=pts, weight=weights)
                hout['jeteta'].fill(WP=wp, Flav=jetype,  eta=etas, weight=weights)
                hout['jetpteta'].fill(WP=wp, Flav=jetype,  pt=pts, abseta=absetas, weight=weights)
                #hout['jetptetaflav'].fill(WP=wp, pt=pts, abseta=absetas, flav=flavarray, weight=weights)
        hout['DeepJet'].fill(genjet_n=gen_jet_total, recojet_n=Deeptag_jet_total, misstag_n=miss_Deepjet, misstag_rate=Deepmisstag_rate, btag_efficiency=Deep_btagefficiency)
        hout['PNet'].fill(genjet_n=gen_jet_total, recojet_n=PNet_jet_total, misstag_n=miss_PNet, misstag_rate=PNetmisstag_rate, btag_efficiency=PNet_btagefficiency)
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)


