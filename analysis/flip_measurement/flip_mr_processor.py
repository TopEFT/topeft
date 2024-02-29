#!/usr/bin/env python
import numpy as np
import awkward as ak
import hist
from coffea import processor

import topeft.modules.object_selection as te_os

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, dtype=np.float32):

        self._samples = samples
        self._dtype = dtype

        # Create the histograms
        self._accumulator = {
            "ptabseta" : hist.Hist(
                hist.axis.StrCategory([], name="process", label="process", growth=True),
                hist.axis.StrCategory([], name="flipstatus", label="flipstatus", growth=True),
                hist.axis.Variable([0, 30.0, 45.0, 60.0, 100.0, 200.0], name="pt", label="pt"),
                hist.axis.Variable([0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5], name="abseta", label="abseta"),
            ),
        }

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

        ele = events.Electron

        ele["gen_pdgId"] = ele.matched_gen.pdgId

        ele["idEmu"]         = te_os.ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        ele["conept"]        = te_os.coneptElec(ele.pt, ele.mvaTTHUL, ele.jetRelIso)
        ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, -99)

        ele["isPres"]     = te_os.isPresElec(ele.pt, ele.eta, ele.dxy, ele.dz, ele.miniPFRelIso_all, ele.sip3d, getattr(ele,"mvaFall17V2noIso_WPL"))
        ele["isLooseE"]   = te_os.isLooseElec(ele.miniPFRelIso_all,ele.sip3d,ele.lostHits)
        ele["isFO"]       = te_os.isFOElec(ele.pt, ele.conept, ele.btagDeepFlavB, ele.idEmu, ele.convVeto, ele.lostHits, ele.mvaTTHUL, ele.jetRelIso, ele.mvaFall17V2noIso_WP90, year)
        ele["isTightLep"] = te_os.tightSelElec(ele.isFO, ele.mvaTTHUL)

        e_tight = ele[ele.isPres & ele.isLooseE & ele.isFO & ele.isTightLep]

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

        hout = self.accumulator

        # Loop over flip and noflip, and fill the histo
        flipstatus_mask_dict = { "truthFlip" : truthFlip_mask, "truthNoFlip" : truthNoFlip_mask }
        for flipstatus_mask_name, flipstatus_mask in flipstatus_mask_dict.items():

            dense_objs_flat = ak.flatten(e_tight[flipstatus_mask])

            axes_fill_info_dict = {
                "pt"         : dense_objs_flat.pt,
                "abseta"     : abs(dense_objs_flat.eta),
                "flipstatus" : flipstatus_mask_name,
                "process"     : histAxisName,
            }

            hout["ptabseta"].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
