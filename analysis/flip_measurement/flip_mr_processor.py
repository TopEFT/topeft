#!/usr/bin/env python
import numpy as np
import awkward as ak
from coffea import hist, processor

#import topcoffea.modules.objects as obj
import topeft.modules.object_selection as te_os

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
        if isData:
            raise Exception("Error: Do not run this processor on data.")


        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3

        run_era = None
        if isData:
            if is_run3:
                run_era = self._samples[dataset]["era"]
            else:
                run_era = self._samples["path"].split("/")[2].split("-")[0][-1]

        ################### Object selection ####################

        e = events.Electron
        jets = events.Jet
        
        if is_run3:
            leptonSelection = te_os.run3leptonselection()
            jetsRho = events.Rho["fixedGridRhoFastjetAll"]
            btagAlgo = "btagDeepFlavB"
        elif is_run2:
            leptonSelection = te_os.run2leptonselection()
            jetsRho = events.fixedGridRhoFastjetAll
            btagAlgo = "btagDeepFlavB"
            
        te_os.lepJetBTagAdder(e, jets, btagger=btagAlgo)
            
        e["gen_pdgId"] = e.matched_gen.pdgId

        e["idEmu"]         = te_os.ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"]        = leptonSelection.coneptElec(e)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)

        e["isPres"] = leptonSelection.isPresElec(e)
        e["isLooseE"] = leptonSelection.isLooseElec(e)
        e["isFO"] = leptonSelection.isFOElec(e, year)
        e["isTightLep"] = leptonSelection.tightSelElec(e)
        
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
