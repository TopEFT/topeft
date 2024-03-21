import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')

from coffea import processor

import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.corrections as corrections



class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], do_errors=False, dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients

        # Create the histogram
        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        dense_axis = hist.axis.Regular(bins=1, start=0, stop=2, name="SumOfWeights", label="SumOfWeights")
        self._accumulator = {

            "SumOfWeights": HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "counts": hist.Hist(proc_axis, dense_axis),

            "SumOfWeights_ISRUp":   HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_ISRDown": HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_FSRUp":   HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_FSRDown": HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),

            "SumOfWeights_renormUp":       HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_renormDown":     HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_factUp":         HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_factDown":       HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_renormfactUp":   HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),
            "SumOfWeights_renormfactDown": HistEFT(proc_axis, dense_axis, wc_names=wc_names_lst),

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
        dataset = events.metadata["dataset"]    # This should be the name of the .json file (without the .json part)
        isData  = self._samples[dataset]["isData"]

        # Can't think of any reason why we'd want to run this over data, so let's make sure to think twice about it before we do
        if isData: raise Exception("Why are you running this over data?")

        # Get EFT coeffs
        eft_coeffs = None
        eft_w2_coeffs = None
        if hasattr(events, "EFTfitCoefficients"):
            eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"])
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)
            if self._do_errors:
                eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype)

        # Get nominal wgt
        counts = np.ones_like(events['event'])
        wgts = np.ones_like(events['event'])
        if not isData and eft_coeffs is None:
            # Basically any central MC samples
            wgts = events["genWeight"]

        # Attach up/down weights
        corrections.AttachPSWeights(events)
        corrections.AttachScaleWeights(events)

        ###### Fill histograms ######
        hout = self.accumulator

        # Nominal
        hout["SumOfWeights"].fill(process=dataset, SumOfWeights=counts, weight=wgts, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["counts"].fill(process=dataset, SumOfWeights=counts, weight=ak.ones_like(wgts))

        # Fill ISR/FSR histos
        hout["SumOfWeights_ISRUp"].fill(process=dataset,   SumOfWeights=counts, weight=wgts*events.ISRUp,   eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_ISRDown"].fill(process=dataset, SumOfWeights=counts, weight=wgts*events.ISRDown, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_FSRUp"].fill(process=dataset,   SumOfWeights=counts, weight=wgts*events.FSRUp,   eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_FSRDown"].fill(process=dataset, SumOfWeights=counts, weight=wgts*events.FSRDown, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        # Fill renorm/fact histos
        hout["SumOfWeights_renormUp"].fill(process=dataset,       SumOfWeights=counts, weight=wgts*events.renormUp,       eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_renormDown"].fill(process=dataset,     SumOfWeights=counts, weight=wgts*events.renormDown,     eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_factUp"].fill(process=dataset,         SumOfWeights=counts, weight=wgts*events.factUp,         eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_factDown"].fill(process=dataset,       SumOfWeights=counts, weight=wgts*events.factDown,       eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_renormfactUp"].fill(process=dataset,   SumOfWeights=counts, weight=wgts*events.renormfactUp,   eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_renormfactDown"].fill(process=dataset, SumOfWeights=counts, weight=wgts*events.renormfactDown, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        return hout

    def postprocess(self, accumulator):
        return accumulator
