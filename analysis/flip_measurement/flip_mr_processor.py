#!/usr/bin/env python
"""Charge flip measurement processor emitting tuple-keyed histograms."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import awkward as ak
import hist
import coffea.processor as processor

import topcoffea.modules.objects as obj
from topeft.modules.runner_output import SUMMARY_KEY, materialise_tuple_dict


def _resolve_nested_field(array, *field_paths):
    for path in field_paths:
        current = array
        for names in path:
            candidate = None
            for name in names:
                if hasattr(current, name):
                    candidate = getattr(current, name)
                    break
                if name in getattr(current, "fields", []):
                    candidate = current[name]
                    break
            if candidate is None:
                current = None
                break
            current = candidate
        if current is not None:
            return current
    return None

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, dtype=np.float32):

        self._samples = samples
        self._dtype = dtype

        self._accumulator = processor.dict_accumulator({})
        self._application_region = "flip_measurement"
        self._systematic = "nominal"
        self._variable = "ptabseta"

        self._pt_bins = (0.0, 30.0, 45.0, 60.0, 100.0, 200.0)
        self._abseta_bins = (0.0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5)

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        source_dataset = events.metadata["dataset"]
        dataset      = source_dataset
        histAxisName = self._samples[dataset]["histAxisName"]
        year         = self._samples[dataset]["year"]
        isData       = self._samples[dataset]["isData"]
        if isData: raise Exception("Error: Do not run this processor on data.")


        ################### Object selection ####################

        e = events.Electron

        gen_pdg = _resolve_nested_field(e, (("matched_gen",), ("pdgId", "pdg_id")))
        if gen_pdg is None:
            raise ValueError(
                f"Missing matched generator PDG IDs for electrons in dataset '{source_dataset}'."
            )

        e["gen_pdgId"] = gen_pdg

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

        flipstatus_mask_dict = {
            "truthFlip": truthFlip_mask,
            "truthNoFlip": truthNoFlip_mask,
        }

        for flipstatus_mask_name, flipstatus_mask in flipstatus_mask_dict.items():
            dense_objs_flat = ak.flatten(e_tight[flipstatus_mask])
            pt_values = ak.to_numpy(dense_objs_flat.pt)
            abseta_values = ak.to_numpy(abs(dense_objs_flat.eta))

            pt_values = np.asarray(pt_values, dtype=self._dtype)
            abseta_values = np.asarray(abseta_values, dtype=self._dtype)

            histogram = self._make_histogram()
            if pt_values.size and abseta_values.size:
                histogram.fill(pt=pt_values, abseta=abseta_values)

            hist_key = self._build_histogram_key(
                flipstatus=flipstatus_mask_name,
                sample=histAxisName,
            )

            if hist_key in hout:
                hout[hist_key] = hout[hist_key] + histogram
            else:
                hout[hist_key] = histogram

        return hout

    def postprocess(self, accumulator):
        tuple_entries: Dict[Tuple[str, str, str, str, str], hist.Hist] = {
            key: value
            for key, value in accumulator.items()
            if isinstance(key, tuple) and len(key) == 5
        }

        ordered_entries: "OrderedDict[Tuple[str, str, str, str, str], hist.Hist]" = OrderedDict(
            sorted(tuple_entries.items(), key=lambda item: item[0])
        )

        summary_payload = materialise_tuple_dict(ordered_entries)
        ordered_entries[SUMMARY_KEY] = summary_payload

        return ordered_entries

    def _make_histogram(self) -> hist.Hist:
        return hist.Hist(
            hist.axis.Variable(self._pt_bins, name="pt", label="pt"),
            hist.axis.Variable(self._abseta_bins, name="abseta", label="abseta"),
            storage=hist.storage.Weight(),
        )

    def _build_histogram_key(
        self,
        *,
        flipstatus: str,
        sample: str,
    ) -> Tuple[str, str, str, str, str]:
        return (
            self._variable,
            flipstatus,
            self._application_region,
            sample,
            self._systematic,
        )
