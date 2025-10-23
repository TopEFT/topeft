from __future__ import annotations

import numpy as np
import awkward as ak

np.seterr(divide="ignore", invalid="ignore", over="ignore")

from coffea import processor

import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.corrections as corrections


_WEIGHT_VARIATIONS: dict[str, str | None] = {
    "nom": None,
    "ISRUp": "ISRUp",
    "ISRDown": "ISRDown",
    "FSRUp": "FSRUp",
    "FSRDown": "FSRDown",
    "renormUp": "renormUp",
    "renormDown": "renormDown",
    "factUp": "factUp",
    "factDown": "factDown",
    "renormfactUp": "renormfactUp",
    "renormfactDown": "renormfactDown",
}


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(
        self,
        samples,
        wc_names_lst: list[str] | None = None,
        do_errors: bool = False,
        dtype=np.float32,
        debug: bool = False,
    ):

        if wc_names_lst is None:
            wc_names_lst = []

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._do_errors = do_errors  # Whether to calculate and store the w**2 coefficients
        self._debug = debug

        self._accumulator = {
            "sow": self._build_hist_dict("SumOfWeights", "SumOfWeights"),
            "sow_norm": self._build_hist_dict("SumOfWeights_norm", "SumOfWeights (normalized)"),
            "nEvents": self._build_hist_dict("nEvents", "nEvents", variations=("nom",)),
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

        if self._debug:
            print(f"[sow_processor] Processing dataset '{dataset}'")

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
        n_events = len(events)
        counts = np.ones(n_events, dtype=self._dtype)
        wgts = np.ones(n_events, dtype=self._dtype)
        if not isData and eft_coeffs is None:
            # Basically any central MC samples
            wgts = ak.to_numpy(events["genWeight"]).astype(self._dtype, copy=False)

        # Attach up/down weights
        corrections.AttachPSWeights(events)
        corrections.AttachScaleWeights(events)

        # Compute normalization factor (convert to pb when possible)
        sample_info = self._samples[dataset]
        norm_factor = self._dtype(0.0)
        if not isData:
            sample_sow = sample_info.get("nSumOfWeights")
            sample_xsec = sample_info.get("xsec")
            if sample_sow not in (None, 0):
                norm_factor = self._dtype(sample_xsec or 0.0) / self._dtype(sample_sow)
            elif self._debug:
                print(f"[sow_processor] Sample '{dataset}' missing non-zero nSumOfWeights; normalized histograms will be zero.")

        if self._debug:
            print(
                f"[sow_processor] nEvents={n_events}, norm_factor={norm_factor}, has_eft_coeffs={eft_coeffs is not None}"
            )

        variation_factors: dict[str, np.ndarray] = {}
        unity = np.ones(n_events, dtype=self._dtype)
        for variation, attr in _WEIGHT_VARIATIONS.items():
            if attr is None:
                variation_factors[variation] = unity
                continue
            if not hasattr(events, attr):
                if self._debug:
                    print(
                        f"[sow_processor] Missing '{attr}' variation for dataset '{dataset}', defaulting to unity weights."
                    )
                variation_factors[variation] = unity
                continue
            variation_factors[variation] = ak.to_numpy(getattr(events, attr)).astype(
                self._dtype, copy=False
            )

        ###### Fill histograms ######
        hout = self.accumulator

        for variation, factors in variation_factors.items():
            suffix = "" if variation == "nom" else f"_{variation}"

            sow_weight = wgts * factors
            hout["sow"][f"SumOfWeights{suffix}"].fill(
                process=dataset,
                SumOfWeights=counts,
                weight=sow_weight,
                eft_coeff=eft_coeffs,
                eft_err_coeff=eft_w2_coeffs,
            )

            sow_norm_weight = sow_weight * norm_factor
            hout["sow_norm"][f"SumOfWeights_norm{suffix}"].fill(
                process=dataset,
                SumOfWeights_norm=counts,
                weight=sow_norm_weight,
                eft_coeff=eft_coeffs,
                eft_err_coeff=eft_w2_coeffs,
            )

        hout["nEvents"]["nEvents"].fill(
            process=dataset,
            nEvents=counts,
            weight=unity,
            eft_coeff=eft_coeffs,
            eft_err_coeff=eft_w2_coeffs,
        )

        return hout

    def postprocess(self, accumulator):
        return accumulator

    def _build_hist_dict(self, axis_name: str, axis_label: str, variations: tuple[str, ...] | None = None):
        if variations is None:
            variations = tuple(_WEIGHT_VARIATIONS.keys())

        hist_dict: dict[str, HistEFT] = {}
        for variation in variations:
            suffix = "" if variation == "nom" else f"_{variation}"
            proc_axis = hist.axis.StrCategory([], name="process", growth=True)
            dense_axis = hist.axis.Regular(bins=1, start=0, stop=2, name=axis_name, label=axis_label)
            hist_name = f"{axis_name}{suffix}"
            hist_dict[hist_name] = HistEFT(proc_axis, dense_axis, wc_names=self._wc_names_lst)
        return hist_dict
