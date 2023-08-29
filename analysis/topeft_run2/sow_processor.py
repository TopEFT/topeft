import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')

from coffea import hist, processor

from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth
import topeft.modules.corrections as corrections



class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], do_errors=False, dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients

        # Create the histogram
        self._accumulator = processor.dict_accumulator({

            "SumOfWeights": HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),

            "SumOfWeights_ISRUp":   HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_ISRDown": HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_FSRUp":   HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_FSRDown": HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),

            "SumOfWeights_renormUp":       HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_renormDown":     HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_factUp":         HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_factDown":       HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_renormfactUp":   HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),
            "SumOfWeights_renormfactDown": HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfWeights", "sow", 1, 0, 2)),

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
        hout = self.accumulator.identity()

        # Nominal
        hout["SumOfWeights"].fill(sample=dataset, SumOfWeights=counts, weight=wgts, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        # Fill ISR/FSR histos
        hout["SumOfWeights_ISRUp"].fill(sample=dataset,   SumOfWeights=counts, weight=wgts*events.ISRUp,   eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_ISRDown"].fill(sample=dataset, SumOfWeights=counts, weight=wgts*events.ISRDown, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_FSRUp"].fill(sample=dataset,   SumOfWeights=counts, weight=wgts*events.FSRUp,   eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_FSRDown"].fill(sample=dataset, SumOfWeights=counts, weight=wgts*events.FSRDown, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        # Fill renorm/fact histos
        hout["SumOfWeights_renormUp"].fill(sample=dataset,       SumOfWeights=counts, weight=wgts*events.renormUp,       eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_renormDown"].fill(sample=dataset,     SumOfWeights=counts, weight=wgts*events.renormDown,     eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_factUp"].fill(sample=dataset,         SumOfWeights=counts, weight=wgts*events.factUp,         eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_factDown"].fill(sample=dataset,       SumOfWeights=counts, weight=wgts*events.factDown,       eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_renormfactUp"].fill(sample=dataset,   SumOfWeights=counts, weight=wgts*events.renormfactUp,   eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
        hout["SumOfWeights_renormfactDown"].fill(sample=dataset, SumOfWeights=counts, weight=wgts*events.renormfactDown, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        return hout

    def postprocess(self, accumulator):
        return accumulator
