#!/usr/bin/env python
import numpy as np
import awkward as ak

import hist
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema

from topcoffea.modules.GetValuesFromJsons import get_lumi
from topcoffea.modules.objects import *
# from topcoffea.modules.corrections import get_ht_sf
from topcoffea.modules.selection import *
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth

np.seterr(divide="ignore", invalid="ignore", over="ignore")
NanoAODSchema.warn_missing_crossrefs = False


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(
        self,
        samples,
        wc_names_lst=None,
        hist_lst=None,
        ecut_threshold=None,
        do_errors=False,
        do_systematics=False,
        split_by_lepton_flavor=False,
        skip_signal_regions=False,
        skip_control_regions=False,
        muonSyst="nominal",
        dtype=np.float32,
    ):

        if wc_names_lst is None:
            wc_names_lst = []

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        sample_axis = hist.axis.StrCategory([], name="sample", growth=True)

        histogram_definitions = {
            "mll_fromzg_e": (
                hist.axis.Regular(
                    40,
                    0,
                    200,
                    name="mll_fromzg_e",
                    label="invmass ee from z/gamma",
                ),
            ),
            "mll_fromzg_m": (
                hist.axis.Regular(
                    40,
                    0,
                    200,
                    name="mll_fromzg_m",
                    label="invmass mm from z/gamma",
                ),
            ),
            "mll_fromzg_t": (
                hist.axis.Regular(
                    40,
                    0,
                    200,
                    name="mll_fromzg_t",
                    label="invmass tau tau from z/gamma",
                ),
            ),
            "mll": (
                hist.axis.Regular(
                    60,
                    0,
                    600,
                    name="mll",
                    label="Invmass l0l1",
                ),
            ),
            "ht": (
                hist.axis.Regular(
                    100,
                    0,
                    1000,
                    name="ht",
                    label="Scalar sum of genjet pt",
                ),
            ),
            "ht_clean": (
                hist.axis.Regular(
                    100,
                    0,
                    1000,
                    name="ht_clean",
                    label="Scalar sum of clean genjet pt",
                ),
            ),
            "tops_pt": (
                hist.axis.Regular(
                    50,
                    0,
                    500,
                    name="tops_pt",
                    label="Pt of the sum of the tops",
                ),
            ),
            "tX_pt": (
                hist.axis.Regular(
                    40,
                    0,
                    400,
                    name="tX_pt",
                    label="Pt of the t(t)X system",
                ),
            ),
            "njets": (
                hist.axis.Regular(10, 0, 10, name="njets", label="njets"),
            ),
        }

        self._accumulator = processor.dict_accumulator(
            {
                key: HistEFT(
                    sample_axis,
                    *axes,
                    wc_names=wc_names_lst,
                    label="Events",
                )
                for key, axes in histogram_definitions.items()
            }
        )

        if hist_lst is None:
            self._hist_lst = list(self._accumulator.keys())
        else:
            for hist_to_include in hist_lst:
                if hist_to_include not in self._accumulator:
                    raise Exception(
                        f"Error: Cannot specify hist \"{hist_to_include}\", it is not defined in the processor."
                    )
            self._hist_lst = list(hist_lst)

        self._do_errors = do_errors

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, events):

        dataset = events.metadata["dataset"]

        is_data = self._samples[dataset]["isData"]
        hist_axis_name = self._samples[dataset]["histAxisName"]
        year = self._samples[dataset]["year"]
        xsec = self._samples[dataset]["xsec"]
        sow = self._samples[dataset]["nSumOfWeights"]

        if is_data:
            raise Exception("Error: This processor is not for data")

        genpart = events.GenPart
        genjet = events.GenJet

        is_final_mask = genpart.hasFlags(["fromHardProcess", "isLastCopy"])

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)], 2)

        gen_l = genpart[
            is_final_mask
            & ((abs(genpart.pdgId) == 11) | (abs(genpart.pdgId) == 13) | (abs(genpart.pdgId) == 15))
        ]
        gen_e = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        gen_m = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        gen_t = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]

        gen_l = gen_l[ak.argsort(gen_l.pt, axis=-1, ascending=False)]
        gen_l = ak.pad_none(gen_l, 2)
        gen_e = gen_e[ak.argsort(gen_e.pt, axis=-1, ascending=False)]
        gen_m = gen_m[ak.argsort(gen_m.pt, axis=-1, ascending=False)]
        gen_t = gen_t[ak.argsort(gen_t.pt, axis=-1, ascending=False)]

        gen_l_from_zg = ak.pad_none(
            gen_l[(gen_l.distinctParent.pdgId == 23) | (gen_l.distinctParent.pdgId == 22)], 2
        )
        gen_e_from_zg = ak.pad_none(
            gen_e[(gen_e.distinctParent.pdgId == 23) | (gen_e.distinctParent.pdgId == 22)], 2
        )
        gen_m_from_zg = ak.pad_none(
            gen_m[(gen_m.distinctParent.pdgId == 23) | (gen_m.distinctParent.pdgId == 22)], 2
        )
        gen_t_from_zg = ak.pad_none(
            gen_t[(gen_t.distinctParent.pdgId == 23) | (gen_t.distinctParent.pdgId == 22)], 2
        )

        genjet = genjet[genjet.pt > 30]
        is_clean_jet = (
            isClean(genjet, gen_e, drmin=0.4)
            & isClean(genjet, gen_m, drmin=0.4)
            & isClean(genjet, gen_t, drmin=0.4)
        )
        genjet_clean = genjet[is_clean_jet]
        njets = ak.num(genjet_clean)

        ht = ak.sum(genjet.pt, axis=-1)
        ht_clean = ak.sum(genjet_clean.pt, axis=-1)

        tops_pt = gen_top.sum().pt

        tX_system = ak.concatenate([gen_top, gen_l_from_zg], axis=1)
        tX_pt = tX_system.sum().pt

        mll_e_from_zg = (gen_e_from_zg[:, 0] + gen_e_from_zg[:, 1]).mass
        mll_m_from_zg = (gen_m_from_zg[:, 0] + gen_m_from_zg[:, 1]).mass
        mll_t_from_zg = (gen_t_from_zg[:, 0] + gen_t_from_zg[:, 1]).mass
        mll_l0l1 = (gen_l[:, 0] + gen_l[:, 1]).mass

        dense_axis_dict = {
            "mll_fromzg_e": mll_e_from_zg,
            "mll_fromzg_m": mll_m_from_zg,
            "mll_fromzg_t": mll_t_from_zg,
            "mll": mll_l0l1,
            "ht": ht,
            "ht_clean": ht_clean,
            "tX_pt": tX_pt,
            "tops_pt": tops_pt,
            "njets": njets,
        }

        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            if self._samples[dataset]["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(
                    self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs
                )
        eft_w2_coeffs = (
            efth.calc_w2_coeffs(eft_coeffs, self._dtype)
            if (self._do_errors and eft_coeffs is not None)
            else None
        )

        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events["event"])
        lumi = get_lumi(year) * 1000.0
        event_weight = lumi * xsec * genw / sow

        hout = self.accumulator.identity()

        for dense_axis_name, dense_axis_vals in dense_axis_dict.items():
            if dense_axis_name not in self._hist_lst:
                continue

            not_none_mask = ak.fill_none(dense_axis_vals != None, False)

            selections = PackedSelection()
            selections.add("valid", not_none_mask)

            if not selections.all("valid").any():
                continue

            dense_axis_vals_cut = dense_axis_vals[not_none_mask]
            event_weight_cut = event_weight[not_none_mask]
            eft_coeffs_cut = eft_coeffs[not_none_mask] if eft_coeffs is not None else None
            eft_w2_coeffs_cut = eft_w2_coeffs[not_none_mask] if eft_w2_coeffs is not None else None

            axes_fill_info_dict = {
                dense_axis_name: dense_axis_vals_cut,
                "sample": hist_axis_name,
                "weight": event_weight_cut,
                "eft_coeff": eft_coeffs_cut,
                "eft_err_coeff": eft_w2_coeffs_cut,
            }

            hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
