from __future__ import annotations

import numpy as np
import awkward as ak

np.seterr(divide="ignore", invalid="ignore", over="ignore")

import coffea.processor as processor

import hist
import topcoffea

HistEFT = topcoffea.modules.HistEFT.HistEFT
efth = topcoffea.modules.eft_helper
corrections = topcoffea.modules.corrections

try:  # pragma: no cover - best effort optional import
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    if hasattr(NanoEventsFactory, "warn_missing_crossrefs"):
        NanoEventsFactory.warn_missing_crossrefs = False
    elif hasattr(NanoAODSchema, "warn_missing_crossrefs"):
        NanoAODSchema.warn_missing_crossrefs = False
except ImportError:  # pragma: no cover - coffea nanoevents optional in tests
    pass


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

_PS_VARIATIONS = {"ISRUp", "ISRDown", "FSRUp", "FSRDown"}
_SCALE_VARIATIONS = {
    "renormUp",
    "renormDown",
    "factUp",
    "factDown",
    "renormfactUp",
    "renormfactDown",
}


def _to_numpy_weight(values, dtype):
    if values is None:
        return None
    if not isinstance(values, ak.Array):
        values = ak.Array(values)
    fields = ak.fields(values)
    if fields:
        if "nominal" in fields:
            values = values["nominal"]
        else:
            values = values[fields[0]]
    while isinstance(values, ak.Array) and values.ndim > 1:
        values = ak.flatten(values, axis=1)
    result = ak.to_numpy(values)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


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

        variations = tuple(_WEIGHT_VARIATIONS.keys())
        self._accumulator = {
            "sow": self._build_hist_dict("SumOfWeights", "SumOfWeights", variations=variations),
            "sow_norm": self._build_hist_dict(
                "SumOfWeights_norm", "SumOfWeights (normalized)", variations=variations
            ),
            "nEvents": self._build_hist_dict("nEvents", "nEvents", variations=("nom",)),
            "metadata": {
                "weight_variations": variations,
                "variation_attributes": dict(_WEIGHT_VARIATIONS),
                "datasets": {},
            },
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
        dataset = events.metadata["dataset"]  # This should be the name of the .json file (without the .json part)
        sample_info = self._samples[dataset]
        isData = sample_info["isData"]

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
            if sample_info["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(sample_info["WCnames"], self._wc_names_lst, eft_coeffs)
            if self._do_errors:
                eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype)

        # Get nominal wgt
        n_events = len(events)
        counts = np.ones(n_events, dtype=self._dtype)
        wgts = np.ones(n_events, dtype=self._dtype)
        if not isData and eft_coeffs is None:
            # Basically any central MC samples
            genw = _to_numpy_weight(getattr(events, "genWeight", None), self._dtype)
            if genw is not None:
                wgts = genw

        # Attach up/down weights when necessary
        self._attach_variation_weights(events, dataset, isData)

        # Compute normalization factor (convert to pb when possible)
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
            variation_values = _to_numpy_weight(getattr(events, attr), self._dtype)
            variation_factors[variation] = variation_values if variation_values is not None else unity

        ###### Fill histograms ######
        hout = self.accumulator
        dataset_meta = self._prepare_dataset_metadata(dataset, sample_info, norm_factor)
        dataset_meta["processed_events"] = dataset_meta.get("processed_events", 0) + int(n_events)

        for variation, factors in variation_factors.items():
            suffix = "" if variation == "nom" else f"_{variation}"

            sow_weight = wgts * factors
            sum_total = float(np.sum(sow_weight))
            hout["sow"][f"SumOfWeights{suffix}"].fill(
                process=dataset,
                SumOfWeights=counts,
                weight=sow_weight,
                eft_coeff=eft_coeffs,
                eft_err_coeff=eft_w2_coeffs,
            )

            sow_norm_weight = sow_weight * norm_factor
            norm_total = float(np.sum(sow_norm_weight))
            hout["sow_norm"][f"SumOfWeights_norm{suffix}"].fill(
                process=dataset,
                SumOfWeights_norm=counts,
                weight=sow_norm_weight,
                eft_coeff=eft_coeffs,
                eft_err_coeff=eft_w2_coeffs,
            )

            self._record_variation_metadata(
                dataset_meta,
                variation,
                hist_name=f"SumOfWeights{suffix}",
                norm_hist_name=f"SumOfWeights_norm{suffix}",
                sum_total=sum_total,
                norm_total=norm_total,
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

    def _build_hist_dict(
        self, axis_name: str, axis_label: str, variations: tuple[str, ...] | None = None
    ):
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

    def _prepare_dataset_metadata(self, dataset: str, sample_info: dict, norm_factor: np.ndarray):
        metadata_root = self._accumulator.setdefault("metadata", {})
        metadata_root.setdefault("weight_variations", tuple(_WEIGHT_VARIATIONS.keys()))
        metadata_root.setdefault("variation_attributes", dict(_WEIGHT_VARIATIONS))
        datasets_meta = metadata_root.setdefault("datasets", {})
        dataset_meta = datasets_meta.setdefault(dataset, {})

        dataset_meta.setdefault("is_data", bool(sample_info.get("isData")))
        dataset_meta.setdefault("cross_section", float(sample_info.get("xsec") or 0.0))
        dataset_meta.setdefault("sample_sum_of_weights", float(sample_info.get("nSumOfWeights") or 0.0))
        dataset_meta.setdefault("normalization_factor", float(norm_factor))
        dataset_meta.setdefault("sum_of_weights", {})
        dataset_meta.setdefault("normalized_sum_of_weights", {})
        dataset_meta.setdefault("totals", {})
        dataset_meta.setdefault("normalized_totals", {})
        dataset_meta.setdefault("metadata_keys", {})
        dataset_meta.setdefault("processed_events", 0)

        return dataset_meta

    def _record_variation_metadata(
        self,
        dataset_meta: dict,
        variation: str,
        *,
        hist_name: str,
        norm_hist_name: str,
        sum_total: float,
        norm_total: float,
    ) -> None:
        metadata_key = "nSumOfWeights" if variation == "nom" else f"nSumOfWeights_{variation}"

        sum_map = dataset_meta.setdefault("sum_of_weights", {})
        norm_map = dataset_meta.setdefault("normalized_sum_of_weights", {})
        totals_map = dataset_meta.setdefault("totals", {})
        norm_totals_map = dataset_meta.setdefault("normalized_totals", {})
        metadata_keys = dataset_meta.setdefault("metadata_keys", {})

        sum_map.setdefault(variation, {"histogram": hist_name})
        norm_map.setdefault(variation, {"histogram": norm_hist_name})
        metadata_keys.setdefault(variation, metadata_key)

        totals_map[variation] = totals_map.get(variation, 0.0) + float(sum_total)
        norm_totals_map[variation] = norm_totals_map.get(variation, 0.0) + float(norm_total)

    def _attach_variation_weights(self, events, dataset: str, is_data: bool) -> None:
        if is_data:
            return

        requested_variations = {
            name for name, attr in _WEIGHT_VARIATIONS.items() if attr is not None
        }

        needs_ps = bool(requested_variations & _PS_VARIATIONS)
        needs_scale = bool(requested_variations & _SCALE_VARIATIONS)

        if needs_ps:
            try:
                corrections.AttachPSWeights(events)
            except Exception as exc:
                if self._debug:
                    print(
                        f"[sow_processor] Failed to attach PS weights for dataset '{dataset}': {exc}"
                    )
                raise

        if needs_scale:
            try:
                corrections.AttachScaleWeights(events)
            except Exception as exc:
                if self._debug:
                    print(
                        f"[sow_processor] Failed to attach scale weights for dataset '{dataset}': {exc}"
                    )
                raise
