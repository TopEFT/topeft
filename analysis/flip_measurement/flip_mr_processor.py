#!/usr/bin/env python
import numpy as np
import awkward as ak
from coffea import hist, processor

import topcoffea.modules.objects as obj

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, dtype=np.float32):

        self._samples = samples
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
            "ptabseta" : hist.Hist(
                "Counts",
                hist.Cat("sample", "sample"),
                hist.Cat("flipstatus", "flipstatus"),
                hist.Bin("pt", "pt", [0, 30.0, 45.0, 60.0, 100.0, 200.0]),
                hist.Bin("abseta", "abseta", [0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5]),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset      = events.metadata["dataset"]
        histAxisName = self._samples[dataset]["histAxisName"]
        year         = self._samples[dataset]["year"]
        isData       = self._samples[dataset]["isData"]
        if isData: raise Exception("Error: Do not run this processor on data.")


        ################### Object selection ####################

        e = events.Electron

        e["gen_pdgId"] = e.matched_gen.pdgId

        e["idEmu"]         = obj.ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"]        = obj.coneptElec(e.pt, e.mvaTTHUL, e.jetRelIso)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)

        e["isPres"]     = obj.isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, getattr(e,"mvaFall17V2noIso_WPL"))
        e["isLooseE"]   = obj.isLooseElec(e.miniPFRelIso_all,e.sip3d,e.lostHits)
        e["isFO"]       = obj.isFOElec(e.pt, e.conept, e.btagDeepFlavB, e.idEmu, e.convVeto, e.lostHits, e.mvaTTHUL, e.jetRelIso, e.mvaFall17V2noIso_WP90, year)
        e["isTightLep"] = obj.tightSelElec(e.isFO, e.mvaTTHUL)

        e_tight = e[e.isPres & e.isLooseE & e.isFO & e.isTightLep]

        # Apply tight charge requirement (this happens in the event selection, so we want to apply it here to be consistent)
        e_tight = e_tight[e_tight.tightCharge>=2]

        ######### Store boolean masks ##########

        # MC truth for flips
        isflip = (e_tight.gen_pdgId == -e_tight.pdgId)
        noflip = (e_tight.gen_pdgId == e_tight.pdgId)
        isprompt = ((e_tight.genPartFlav==1) | (e_tight.genPartFlav == 15))
        truthFlip_mask   = ak.fill_none((isflip & isprompt),False)
        truthNoFlip_mask = ak.fill_none((noflip & isprompt),False)

        #print("isflip",isflip)
        #print("e_tight.gen_pdgId",e_tight.gen_pdgId)
        #print("e_tight.pdgId",e_tight.pdgId)
        #print("isprompt",isprompt)


        ########## Fill the histograms ##########

        hout = self.accumulator.identity()

        # Loop over flip and noflip, and fill the histo
        flipstatus_mask_dict = { "truthFlip" : truthFlip_mask, "truthNoFlip" : truthNoFlip_mask }
        for flipstatus_mask_name, flipstatus_mask in flipstatus_mask_dict.items():

            dense_objs_flat = ak.flatten(e_tight[flipstatus_mask])

            axes_fill_info_dict = {
                "pt"         : dense_objs_flat.pt,
                "abseta"     : abs(dense_objs_flat.eta),
                "flipstatus" : flipstatus_mask_name,
                "sample"     : histAxisName,
            }

            hout["ptabseta"].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
