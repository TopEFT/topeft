#!/usr/bin/env python
"""Coffea processor used by ``run_analysis.py`` and workflow helpers.

The processor responsibilities, expected inputs, and extension hooks are
documented in ``docs/analysis_processing.md``.  That guide also links the
systematic catalogue exposed through ``topeft/params/metadata.yml`` with the
``RunWorkflow`` planner and the YAML quickstart workflow.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
import awkward as ak
import os
import re
import logging

import coffea
import coffea.processor as processor
import hist
import topcoffea
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask
from typing import Dict, List, Optional, Tuple

from topeft.modules.paths import topeft_path
from topeft.modules.corrections import (
    ApplyJetCorrections,
    build_corrected_jets,
    build_corrected_met,
    GetBtagEff,
    AttachMuonSF,
    AttachElectronSF,
    AttachTauSF,
    ApplyTES,
    ApplyTESSystematic,
    ApplyFESSystematic,
    AttachPerLeptonFR,
    ApplyRochesterCorrections,
    ApplyJetSystematics,
)
from topeft.modules.btag_weights import register_btag_sf_weights
import topeft.modules.event_selection as te_es
import topeft.modules.object_selection as te_os
from topeft.modules.systematics import (
    add_fake_factor_weights,
    apply_theory_weight_variations,
    register_lepton_sf_weight,
    register_trigger_sf_weight,
    register_weight_variation,
    validate_data_weight_variations,
)

HistEFT = topcoffea.modules.HistEFT.HistEFT
topcoffea_path = topcoffea.modules.paths.topcoffea_path
efth = topcoffea.modules.eft_helper
tc_es = topcoffea.modules.event_selection
tc_os = topcoffea.modules.object_selection
tc_cor = topcoffea.modules.corrections
GetParam = topcoffea.modules.get_param_from_jsons.GetParam

logger = logging.getLogger(__name__)
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

np.seterr(divide="ignore", invalid="ignore", over="ignore")


# Takes strings as inputs, constructs a string for the full channel name
# Try to construct a channel name like this: [n leptons]_[lepton flavors]_[p or m charge]_[on or off Z]_[n b jets]_[n jets]
# chan_str should look something like "3l_p_offZ_1b", NOTE: This function assumes nlep comes first
# njet_str should look something like "atleast_5j",   NOTE: This function assumes njets comes last
# flav_str should look something like "emm"
def construct_cat_name(chan_str, njet_str=None, flav_str=None):
    # Get the component strings
    nlep_str = chan_str.split("_")[0]  # Assumes n leps comes first in the str
    chan_str = "_".join(
        chan_str.split("_")[1:]
    )  # The rest of the channel name is everything that comes after nlep
    if chan_str == "":
        chan_str = None  # So that we properly skip this in the for loop below
    if flav_str is not None:
        flav_str = flav_str
    if njet_str is not None:
        njet_str = njet_str[
            -2:
        ]  # Assumes number of n jets comes at the end of the string
        if "j" not in njet_str:
            # The njet string should really have a "j" in it
            raise Exception(
                f'Something when wrong while trying to consturct channel name, is "{njet_str}" an njet string?'
            )

    # Put the component strings into the channel name
    ret_str = nlep_str
    for component in [flav_str, chan_str, njet_str]:
        if component is None:
            continue
        ret_str = "_".join([ret_str, component])
    return ret_str


@dataclass(frozen=True)
class DatasetContext:
    dataset: str
    trigger_dataset: str
    hist_axis_name: str
    is_data: bool
    is_eft: bool
    year: str
    xsec: float
    sow: float
    run_era: Optional[str]
    is_run2: bool
    is_run3: bool
    sample_type: str
    is_lo_sample: bool
    lumi_mask: ak.Array
    lumi: float
    eft_coeffs: Optional[np.ndarray]
    eft_w2_coeffs: Optional[np.ndarray]


@dataclass(frozen=True)
class BaseObjectState:
    met: ak.Array
    electrons: ak.Array
    muons: ak.Array
    taus: ak.Array
    jets: ak.Array
    loose_leptons: ak.Array
    fakeable_leptons: ak.Array
    fakeable_sorted: ak.Array
    jets_rho: ak.Array
    lepton_selection: object
    cleaning_taus: Optional[ak.Array]
    n_loose_taus: Optional[ak.Array]
    tau0: Optional[ak.Array]


@dataclass
class VariationObjects:
    met: ak.Array
    electrons: ak.Array
    muons: ak.Array
    taus: ak.Array
    jets: ak.Array
    loose_leptons: ak.Array
    fakeable_leptons: ak.Array
    fakeable_sorted: ak.Array
    cleaning_taus: Optional[ak.Array]
    n_loose_taus: Optional[ak.Array]
    tau0: Optional[ak.Array]
    shifted_cleaning_taus: Optional[ak.Array] = None
    central_cleaning_taus: Optional[ak.Array] = None

    @classmethod
    def from_base(cls, base: BaseObjectState) -> "VariationObjects":
        return cls(
            met=ak.copy(base.met),
            electrons=ak.copy(base.electrons),
            muons=ak.copy(base.muons),
            taus=ak.copy(base.taus),
            jets=ak.copy(base.jets),
            loose_leptons=ak.copy(base.loose_leptons),
            fakeable_leptons=ak.copy(base.fakeable_leptons),
            fakeable_sorted=ak.copy(base.fakeable_sorted),
            cleaning_taus=(
                ak.copy(base.cleaning_taus) if base.cleaning_taus is not None else None
            ),
            n_loose_taus=(
                ak.copy(base.n_loose_taus) if base.n_loose_taus is not None else None
            ),
            tau0=ak.copy(base.tau0) if base.tau0 is not None else None,
            shifted_cleaning_taus=(
                ak.copy(base.cleaning_taus) if base.cleaning_taus is not None else None
            ),
            central_cleaning_taus=(
                ak.copy(base.cleaning_taus) if base.cleaning_taus is not None else None
            ),
        )


@dataclass(frozen=True)
class VariationRequest:
    variation: Optional[object]
    histogram_label: str


@dataclass
class VariationState:
    request: VariationRequest
    name: str
    base: Optional[str]
    variation_type: Optional[str]
    metadata: Mapping[str, object]
    object_variation: str
    weight_variations: List[str]
    requested_data_weight_label: Optional[str]
    sow_variation_key_map: Dict[str, str]
    sow_variations: Dict[str, float]
    objects: VariationObjects
    lepton_selection: object
    jets_rho: ak.Array
    cleaned_jets: Optional[ak.Array] = None
    good_jets: Optional[ak.Array] = None
    fwd_jets: Optional[ak.Array] = None
    njets: Optional[ak.Array] = None
    nfwdj: Optional[ak.Array] = None
    ht: Optional[ak.Array] = None
    j0: Optional[ak.Array] = None
    l_sorted_padded: Optional[ak.Array] = None
    l0: Optional[ak.Array] = None
    l1: Optional[ak.Array] = None
    l2: Optional[ak.Array] = None
    isBtagJetsLoose: Optional[ak.Array] = None
    isNotBtagJetsLoose: Optional[ak.Array] = None
    nbtagsl: Optional[ak.Array] = None
    isBtagJetsMedium: Optional[ak.Array] = None
    isNotBtagJetsMedium: Optional[ak.Array] = None
    nbtagsm: Optional[ak.Array] = None
    isBtagJetsLooseNotMedium: Optional[ak.Array] = None
    jets_light: Optional[ak.Array] = None
    jets_bc: Optional[ak.Array] = None
    light_mask: Optional[ak.Array] = None
    bc_mask: Optional[ak.Array] = None
    has_hadron_flavour: bool = False
    include_muon_sf: bool = False
    include_elec_sf: bool = False
    include_tau_real_sf: bool = False
    include_tau_fake_sf: bool = False


_LEPTON_SF_WEIGHT_SPECS: Dict[str, Tuple[Tuple[str, str, str, str, str], ...]] = {
    "1l": (
        (
            "lepSF_muon",
            "sf_1l_muon",
            "sf_1l_hi_muon",
            "sf_1l_lo_muon",
            "include_muon_sf",
        ),
        (
            "lepSF_elec",
            "sf_1l_elec",
            "sf_1l_hi_elec",
            "sf_1l_lo_elec",
            "include_elec_sf",
        ),
    ),
    "2l": (
        (
            "lepSF_muon",
            "sf_2l_muon",
            "sf_2l_hi_muon",
            "sf_2l_lo_muon",
            "include_muon_sf",
        ),
        (
            "lepSF_elec",
            "sf_2l_elec",
            "sf_2l_hi_elec",
            "sf_2l_lo_elec",
            "include_elec_sf",
        ),
    ),
    "3l": (
        (
            "lepSF_muon",
            "sf_3l_muon",
            "sf_3l_hi_muon",
            "sf_3l_lo_muon",
            "include_muon_sf",
        ),
        (
            "lepSF_elec",
            "sf_3l_elec",
            "sf_3l_hi_elec",
            "sf_3l_lo_elec",
            "include_elec_sf",
        ),
    ),
    "4l": (
        (
            "lepSF_muon",
            "sf_4l_muon",
            "sf_4l_hi_muon",
            "sf_4l_lo_muon",
            "include_muon_sf",
        ),
        (
            "lepSF_elec",
            "sf_4l_elec",
            "sf_4l_hi_elec",
            "sf_4l_lo_elec",
            "include_elec_sf",
        ),
    ),
}

_TAU_SF_WEIGHT_SPECS: Tuple[Tuple[str, str, str, str, str], ...] = (
    (
        "lepSF_taus_real",
        "sf_2l_taus_real",
        "sf_2l_taus_real_hi",
        "sf_2l_taus_real_lo",
        "include_tau_real_sf",
    ),
    (
        "lepSF_taus_fake",
        "sf_2l_taus_fake",
        "sf_2l_taus_fake_hi",
        "sf_2l_taus_fake_lo",
        "include_tau_fake_sf",
    ),
)


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(
        self,
        sample,
        wc_names_lst=[],
        hist_keys=None,
        var_info=None,
        ecut_threshold=None,
        do_errors=False,
        split_by_lepton_flavor=False,
        muonSyst="nominal",
        dtype=np.float32,
        rebin=False,
        channel_dict=None,
        golden_json_path=None,
        systematic_variations=None,
        available_systematics=None,
        metadata_path: Optional[str] = None,
        debug_logging: bool = False,
        executor_mode: Optional[str] = None,
        suppress_debug_prints: Optional[bool] = None,
    ):

        self._sample = sample
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        if channel_dict is None:
            raise ValueError("channel_dict must be provided and cannot be None")

        # ``channel_dict`` is expected to be a flat dictionary with keys
        # ``jet_selection``, ``chan_def_lst``, ``lep_flav_lst`` and ``appl_region``.
        # Previous versions of this processor converted this dictionary into a
        # nested structure with several loops in ``process``.  The new logic
        # operates directly on the flat dictionary, so simply store it.
        self._channel_dict = channel_dict
        channel_features = channel_dict.get("features", ())
        if channel_features is None:
            channel_features = ()
        self._channel_features = frozenset(channel_features)
        self.offZ_3l_split = "offz_split" in self._channel_features
        self.tau_h_analysis = "requires_tau" in self._channel_features
        self.fwd_analysis = "requires_forward" in self._channel_features
        if available_systematics is None:
            raise ValueError(
                "available_systematics must be provided and cannot be None"
            )
        self._available_systematics = {
            key: tuple(value) for key, value in available_systematics.items()
        }
        self._available_systematics_sets = {
            key: set(values) for key, values in self._available_systematics.items()
        }
        self._metadata_path = metadata_path
        self._debug_logging = bool(debug_logging)
        self._executor_mode = (executor_mode or "").strip().lower() or None
        if suppress_debug_prints is None:
            suppress_debug_prints = self._executor_mode == "taskvine" or bool(
                os.environ.get("TOPEFT_SUPPRESS_DEBUG_STDOUT")
            )
        self._suppress_debug_prints = bool(suppress_debug_prints)
        self._golden_json_path = golden_json_path
        if self._sample.get("isData") and not self._golden_json_path:
            raise ValueError("golden_json_path must be provided for data samples")

        if var_info is None:
            raise ValueError("var_info must be provided and cannot be None")

        if hist_keys is None:
            raise ValueError("hist_keys must be provided and cannot be None")

        if not isinstance(hist_keys, dict):
            raise TypeError(
                "hist_keys must be a mapping of variation name to histogram key"
            )

        raw_histogram_key_map = OrderedDict(hist_keys)

        if not raw_histogram_key_map:
            raise ValueError("hist_keys must contain at least one entry")

        histogram_key_map: "OrderedDict[str, Tuple[Tuple[str, ...], ...]]" = (
            OrderedDict()
        )
        for variation_label, key_entries in raw_histogram_key_map.items():
            if (
                isinstance(key_entries, tuple)
                and len(key_entries) == 5
                and not isinstance(key_entries[0], (tuple, list))
            ):
                normalized_entries = (tuple(key_entries),)
            else:
                try:
                    normalized_entries = tuple(tuple(entry) for entry in key_entries)
                except TypeError as exc:
                    raise TypeError(
                        "hist_keys values must be 5-element tuples or an iterable of such tuples"
                    ) from exc
            if not normalized_entries:
                raise ValueError(
                    "hist_keys entries must contain at least one histogram key"
                )
            for entry in normalized_entries:
                if len(entry) != 5:
                    raise ValueError("histogram keys must be 5-element tuples")
            histogram_key_map[variation_label] = normalized_entries

        info = var_info

        histogram = {}
        self._hist_keys_to_fill: List[Tuple[str, ...]] = []
        self._histogram_label_lookup: Dict[str, object] = {}
        self._flavored_channel_lookup: Dict[str, str] = {}

        first_label, first_hist_keys = next(iter(histogram_key_map.items()))
        first_hist_key = first_hist_keys[0]

        var, ch, appl, sample_name, _ = first_hist_key

        self._var = var
        self._channel = ch
        self._appregion = appl
        self._syst = first_label
        self._var_def = info.get("definition")
        if self._var_def is None:
            raise ValueError(f"No definition provided for variable {var}")

        for variation_label, hist_key_entries in histogram_key_map.items():
            base_syst_label = hist_key_entries[0][4]
            self._histogram_label_lookup[variation_label] = base_syst_label

            for idx, hist_key_entry in enumerate(hist_key_entries):
                key_var, key_ch, key_appl, key_sample, syst_label = hist_key_entry
                if key_var != self._var or key_appl != self._appregion:
                    raise ValueError(
                        "All histogram keys must refer to the same variable and application"
                    )
                if key_sample != sample_name:
                    raise ValueError(
                        "Histogram keys must refer to the configured sample"
                    )

                if key_ch != self._channel:
                    mapped_channel = self._flavored_channel_lookup.get(key_ch)
                    if mapped_channel is None:
                        self._flavored_channel_lookup[key_ch] = self._channel
                    elif mapped_channel != self._channel:
                        raise ValueError(
                            f"Histogram key for channel '{key_ch}' does not match base channel '{self._channel}'"
                        )

                if not rebin and "variable" in info:
                    dense_axis = hist.axis.Variable(
                        info["variable"], name=self._var, label=info["label"]
                    )
                else:
                    dense_axis = hist.axis.Regular(
                        *info["regular"], name=self._var, label=info["label"]
                    )

                hist_key = self._build_histogram_key(
                    key_var,
                    key_ch,
                    key_sample,
                    syst_label,
                    application=key_appl,
                )

                histogram[hist_key] = HistEFT(
                    dense_axis,
                    wc_names=wc_names_lst,
                    label=r"Events",
                )

                self._hist_keys_to_fill.append(hist_key)

                if idx == 0:
                    if not rebin and "variable" in info:
                        sumw2_axis = hist.axis.Variable(
                            info["variable"],
                            name=f"{self._var}_sumw2",
                            label=info["label"] + " sum of w^2",
                        )
                    else:
                        sumw2_axis = hist.axis.Regular(
                            *info["regular"],
                            name=f"{self._var}_sumw2",
                            label=info["label"] + " sum of w^2",
                        )

                    sumw2_key = self._build_histogram_key(
                        f"{self._var}_sumw2",
                        self._channel,
                        key_sample,
                        syst_label,
                        application=self._appregion,
                    )

                    histogram[sumw2_key] = HistEFT(
                        sumw2_axis,
                        wc_names=wc_names_lst,
                        label=r"Events",
                    )
                    self._hist_keys_to_fill.append(sumw2_key)

        self._histogram_key_map = histogram_key_map

        self._accumulator = histogram

        # Set the energy threshold to cut on
        self._ecut_threshold = ecut_threshold

        # Set the booleans
        self._do_errors = (
            do_errors  # Whether to calculate and store the w**2 coefficients
        )
        self._split_by_lepton_flavor = split_by_lepton_flavor  # Whether to keep track of lepton flavors individually

        if systematic_variations is None:
            systematic_variations = ()
        else:
            systematic_variations = tuple(systematic_variations)

        self._systematic_variations = systematic_variations
        self._systematic_info = (
            systematic_variations[0] if systematic_variations else None
        )

        if self._systematic_variations and self._systematic_info is None:
            raise ValueError(
                "systematic_variations must contain at least one entry when provided"
            )

    @staticmethod
    def _ensure_ak_array(values, dtype=None):
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
        if dtype is not None:
            values = ak.values_astype(values, dtype)
        return values

    @staticmethod
    def _metadata_to_mapping(metadata: Optional[object]) -> Mapping[str, object]:
        """Return *metadata* as a mapping without mutating the input."""

        if metadata is None:
            return {}
        if isinstance(metadata, Mapping):
            return metadata
        if hasattr(metadata, "items"):
            try:
                return dict(metadata.items())
            except Exception:  # pragma: no cover - defensive fallback
                return dict(metadata)
        if hasattr(metadata, "__iter__") and not isinstance(metadata, (str, bytes)):
            try:
                return dict(metadata)
            except Exception:  # pragma: no cover - defensive fallback
                pass
        if hasattr(metadata, "__dict__"):
            result = {}
            for key in dir(metadata):
                if key.startswith("_"):
                    continue
                value = getattr(metadata, key)
                if callable(value):
                    continue
                result[key] = value
            return result
        return {}

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def sample(self):
        return self._sample

    @property
    def hist_keys_to_fill(self):
        return self._hist_keys_to_fill

    @property
    def var(self):
        return self._var

    @property
    def var_def(self):
        return self._var_def

    def _build_channel_names(self, lep_chan, njet_ch, flav_ch):
        ch_name = construct_cat_name(lep_chan, njet_str=njet_ch, flav_str=flav_ch)
        base_ch_name = construct_cat_name(lep_chan, njet_str=njet_ch, flav_str=None)
        return ch_name, base_ch_name

    def _compute_ptbl(
        self,
        good_jets: ak.Array,
        is_btag_med: ak.Array,
        is_btag_loose: ak.Array,
        leptons: ak.Array,
    ) -> ak.Array:
        ptbl_bjet = good_jets[(is_btag_med | is_btag_loose)]
        ptbl_bjet = ak.with_name(ptbl_bjet, "PtEtaPhiMCandidate")
        leading_b = ak.firsts(ptbl_bjet[ak.argsort(ptbl_bjet.pt, axis=-1, ascending=False)])
        has_btag = ak.num(ptbl_bjet.pt, axis=-1) > 0

        zero_vector = ak.zip(
            {
                "pt": ak.zeros_like(has_btag, dtype=np.float32),
                "eta": ak.zeros_like(has_btag, dtype=np.float32),
                "phi": ak.zeros_like(has_btag, dtype=np.float32),
                "mass": ak.zeros_like(has_btag, dtype=np.float32),
            },
            with_name="PtEtaPhiMCandidate",
        )

        leptons = ak.with_name(leptons, "PtEtaPhiMCandidate")
        leading_b_filled = ak.where(has_btag, leading_b, zero_vector)
        delta_eta = leading_b_filled.eta - leptons.eta
        delta_phi = (leading_b_filled.phi - leptons.phi + np.pi) % (2 * np.pi) - np.pi
        delta_r = np.hypot(delta_eta, delta_phi)
        nearest_lep = leptons[ak.argmin(delta_r, axis=-1)]

        px_b = leading_b_filled.pt * np.cos(leading_b_filled.phi)
        py_b = leading_b_filled.pt * np.sin(leading_b_filled.phi)
        px_l = nearest_lep.pt * np.cos(nearest_lep.phi)
        py_l = nearest_lep.pt * np.sin(nearest_lep.phi)

        pt_sum = np.hypot(px_b + px_l, py_b + py_l)
        pt_sum = ak.firsts(ak.singletons(pt_sum))
        pt_sum = ak.values_astype(ak.fill_none(pt_sum, np.float32(-1.0)), np.float32)
        return ak.where(has_btag, pt_sum, np.float32(-1.0))

    def _build_histogram_key(
        self,
        variable: str,
        channel: str,
        sample: str,
        systematic: str,
        application: Optional[str] = None,
    ) -> Tuple[str, str, str, str, str]:
        return (
            variable,
            channel,
            application if application is not None else self._appregion,
            sample,
            systematic,
        )

    def _build_dataset_context(self, events) -> DatasetContext:
        events_metadata = self._metadata_to_mapping(getattr(events, "metadata", None))
        raw_dataset_name = events_metadata.get("dataset")
        if raw_dataset_name is None:
            raw_dataset_name = self._sample.get("histAxisName")
        if raw_dataset_name is None:
            raise KeyError("Events metadata is missing a 'dataset' entry")
        raw_dataset_name = str(raw_dataset_name)
        dataset, trigger_dataset = self._resolve_dataset_names(raw_dataset_name)

        hist_axis_name = self._sample["histAxisName"]
        is_data = self._sample["isData"]
        year = self._sample["year"]
        xsec = self._sample["xsec"]
        sow = self._sample["nSumOfWeights"]

        is_eft = self._sample["WCnames"] != []
        is_run3 = year.startswith("202")
        is_run2 = not is_run3

        run_era = None
        if is_data:
            run_era = self._sample["path"].split("/")[2].split("-")[0][-1]

        is_lo_sample = hist_axis_name in get_te_param("lo_xsec_samples")

        sample_type = "prompt"
        if is_data:
            sample_type = "data"
        elif hist_axis_name in get_te_param("conv_samples"):
            sample_type = "conversions"
        elif hist_axis_name in get_te_param("prompt_and_conv_samples"):
            sample_type = "prompt_and_conversions"

        lumi_mask = ak.ones_like(events["run"], dtype=bool)
        if is_data:
            lumi_mask = LumiMask(self._golden_json_path)(
                events["run"], events["luminosityBlock"]
            )

        eft_coeffs = (
            ak.to_numpy(events["EFTfitCoefficients"])
            if "EFTfitCoefficients" in events.fields
            else None
        )
        if eft_coeffs is not None and self._sample["WCnames"] != self._wc_names_lst:
            eft_coeffs = efth.remap_coeffs(
                self._sample["WCnames"], self._wc_names_lst, eft_coeffs
            )
        eft_w2_coeffs = (
            efth.calc_w2_coeffs(eft_coeffs, self._dtype)
            if (self._do_errors and eft_coeffs is not None)
            else None
        )

        lumi = 1000.0 * get_tc_param(f"lumi_{year}")

        context = DatasetContext(
            dataset=dataset,
            trigger_dataset=trigger_dataset,
            hist_axis_name=hist_axis_name,
            is_data=is_data,
            is_eft=is_eft,
            year=year,
            xsec=xsec,
            sow=sow,
            run_era=run_era,
            is_run2=is_run2,
            is_run3=is_run3,
            sample_type=sample_type,
            is_lo_sample=is_lo_sample,
            lumi_mask=lumi_mask,
            lumi=lumi,
            eft_coeffs=eft_coeffs,
            eft_w2_coeffs=eft_w2_coeffs,
        )

        if self._debug_logging:
            features = tuple(sorted(self._channel_features))
            self._debug(
                "Resolved dataset context: dataset=%s trigger_dataset=%s features=%s "
                "is_data=%s is_eft=%s sample_type=%s run_era=%s year=%s",
                context.dataset,
                context.trigger_dataset,
                features,
                context.is_data,
                context.is_eft,
                context.sample_type,
                context.run_era,
                context.year,
            )

        return context

    def _select_base_objects(self, events, dataset: DatasetContext) -> BaseObjectState:
        met = events["MET"]
        ele = events["Electron"]
        mu = events["Muon"]
        tau = events["Tau"]
        jets = events["Jet"]

        if dataset.is_run3:
            lepton_selection = te_os.run3leptonselection()
            jets_rho = events["Rho"]["fixedGridRhoFastjetAll"]
        else:
            lepton_selection = te_os.run2leptonselection()
            jets_rho = events["fixedGridRhoFastjetAll"]

        events["nom"] = ak.ones_like(events["MET"].pt)

        ele["idEmu"] = te_os.ttH_idEmu_cuts_E3(
            ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie
        )
        ele["conept"] = lepton_selection.coneptElec(ele)
        mu["conept"] = lepton_selection.coneptMuon(mu)
        ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)
        if not dataset.is_data:
            ele["gen_pdgId"] = ak.fill_none(ele.matched_gen.pdgId, 0)
            mu["gen_pdgId"] = ak.fill_none(mu.matched_gen.pdgId, 0)

        ele["isPres"] = lepton_selection.isPresElec(ele)
        ele["isLooseE"] = lepton_selection.isLooseElec(ele)
        ele["isFO"] = lepton_selection.isFOElec(ele, dataset.year)
        ele["isTightLep"] = lepton_selection.tightSelElec(ele)

        mu["pt"] = ApplyRochesterCorrections(mu, dataset.year, dataset.is_data)
        mu["isPres"] = lepton_selection.isPresMuon(mu)
        mu["isLooseM"] = lepton_selection.isLooseMuon(mu)
        mu["isFO"] = lepton_selection.isFOMuon(mu, dataset.year)
        mu["isTightLep"] = lepton_selection.tightSelMuon(mu)

        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = ele[ele.isPres & ele.isLooseE]
        l_loose = ak.with_name(
            ak.concatenate([e_loose, m_loose], axis=1), "PtEtaPhiMCandidate"
        )

        llpairs = ak.combinations(l_loose, 2, fields=["l0", "l1"])
        events["minMllAFAS"] = ak.min((llpairs.l0 + llpairs.l1).mass, axis=-1)

        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = ele[ele.isPres & ele.isLooseE & ele.isFO]

        AttachElectronSF(
            e_fo, year=dataset.year, looseWP="none" if dataset.is_run3 else "wpLnoiso"
        )
        AttachMuonSF(m_fo, year=dataset.year)
        AttachPerLeptonFR(e_fo, flavor="Elec", year=dataset.year)
        AttachPerLeptonFR(m_fo, flavor="Muon", year=dataset.year)
        m_fo["convVeto"] = ak.ones_like(m_fo.charge)
        m_fo["lostHits"] = ak.zeros_like(m_fo.charge)

        l_fo = ak.with_name(
            ak.concatenate([e_fo, m_fo], axis=1),
            "PtEtaPhiMCandidate",
        )
        l_fo = l_fo[ak.argsort(l_fo.conept, axis=1, ascending=False)]
        l_fo_conept_sorted = l_fo

        cleaning_taus = None
        nLtau = None
        tau0 = None

        if self.tau_h_analysis:
            tau["pt"], tau["mass"] = ApplyTES(dataset.year, tau, dataset.is_data)
            tau["isPres"] = te_os.isPresTau(
                tau.pt,
                tau.eta,
                tau.dxy,
                tau.dz,
                tau.idDeepTau2017v2p1VSjet,
                tau.idDeepTau2017v2p1VSe,
                tau.idDeepTau2017v2p1VSmu,
                minpt=20,
            )
            tau["isClean"] = te_os.isClean(tau, l_fo, drmin=0.3)
            tau["isGood"] = tau["isClean"] & tau["isPres"]
            tau = tau[tau.isGood]

            tau["DMflag"] = (
                (tau.decayMode == 0)
                | (tau.decayMode == 1)
                | (tau.decayMode == 10)
                | (tau.decayMode == 11)
            )
            tau = tau[tau["DMflag"]]
            tau["isVLoose"] = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet)
            tau["isLoose"] = te_os.isLooseTau(tau.idDeepTau2017v2p1VSjet)
            tau["iseTight"] = te_os.iseTightTau(tau.idDeepTau2017v2p1VSe)
            tau["ismTight"] = te_os.ismTightTau(tau.idDeepTau2017v2p1VSmu)

            cleaning_taus = tau[tau["isLoose"] > 0]
            nLtau = ak.num(tau[tau["isLoose"] > 0])
            tau_padded = ak.pad_none(tau, 1)
            tau0 = tau_padded[:, 0]
        else:
            tau["isPres"] = te_os.isPresTau(
                tau.pt,
                tau.eta,
                tau.dxy,
                tau.dz,
                tau.idDeepTau2017v2p1VSjet,
                tau.idDeepTau2017v2p1VSe,
                tau.idDeepTau2017v2p1VSmu,
                minpt=20,
            )
            tau["isClean"] = te_os.isClean(tau, l_loose, drmin=0.3)
            tau["isGood"] = tau["isClean"] & tau["isPres"]
            tau = tau[tau.isGood]
            tau["isTight"] = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet)

        return BaseObjectState(
            met=met,
            electrons=ele,
            muons=mu,
            taus=tau,
            jets=jets,
            loose_leptons=l_loose,
            fakeable_leptons=l_fo,
            fakeable_sorted=l_fo_conept_sorted,
            jets_rho=jets_rho,
            lepton_selection=lepton_selection,
            cleaning_taus=cleaning_taus,
            n_loose_taus=nLtau,
            tau0=tau0,
        )

    def _build_variation_requests(self) -> List[VariationRequest]:
        if self._systematic_variations:
            requests = []
            for variation in self._systematic_variations:
                label = self._histogram_label_lookup.get(variation.name)
                if label is None:
                    raise KeyError(
                        f"Missing histogram label for requested variation '{variation.name}'"
                    )
                requests.append(
                    VariationRequest(variation=variation, histogram_label=label)
                )
        else:
            requests = [VariationRequest(variation=None, histogram_label="nominal")]

        if self._debug_logging:
            summary = [
                (
                    req.variation.name if req.variation is not None else "nominal",
                    req.histogram_label,
                )
                for req in requests
            ]
            self._debug(
                "Prepared %d variation requests: %s",
                len(requests),
                summary,
            )

        return requests

    def _initialize_variation_state(
        self,
        request: VariationRequest,
        base_objects: BaseObjectState,
        dataset: DatasetContext,
        object_systematics: Tuple[str, ...],
        weight_systematics: Tuple[str, ...],
        theory_systematics: Tuple[str, ...],
        data_weight_systematics: Tuple[str, ...],
    ) -> VariationState:
        variation = request.variation
        variation_name = variation.name if variation is not None else "nominal"
        variation_base = variation.base if variation is not None else None
        variation_type = (
            getattr(variation, "type", None) if variation is not None else None
        )
        variation_metadata = self._metadata_to_mapping(
            getattr(variation, "metadata", None) if variation is not None else None
        )

        if self._debug_logging:
            components = (
                tuple(getattr(variation, "components", ()))
                if variation is not None
                else ()
            )
            self._debug(
                "Resolved variation metadata for '%s': type=%s base=%s components=%s metadata=%s",
                variation_name,
                variation_type,
                variation_base,
                components,
                dict(variation_metadata),
            )

        variation_base_str = variation_base or ""
        metadata_lepton_flavor_value = (
            variation_metadata.get("lepton_flavor")
            or variation_metadata.get("lepton_type")
            or variation_metadata.get("flavor")
        )
        metadata_lepton_flavor = (
            str(metadata_lepton_flavor_value).strip().lower()
            if metadata_lepton_flavor_value is not None
            else ""
        )

        include_lep_sf_variations = bool(
            variation_metadata.get("lepton_sf")
            or variation_metadata.get("weight_family") == "lepton_sf"
            or variation_metadata.get("weight_category") == "lepton_sf"
            or variation_base_str.startswith("lepton_sf_")
        )
        include_muon_sf_variations = include_lep_sf_variations and (
            metadata_lepton_flavor in {"mu", "muon", "muons"}
            or variation_base_str.endswith("muon")
        )
        include_elec_sf_variations = include_lep_sf_variations and (
            metadata_lepton_flavor in {"e", "ele", "elec", "electron", "electrons"}
            or variation_base_str.endswith("elec")
            or variation_base_str.endswith("electron")
        )
        include_tau_real_sf_variations = include_lep_sf_variations and (
            metadata_lepton_flavor in {"tau_real", "tau-real"}
            or variation_base_str.endswith("tau_real")
        )
        include_tau_fake_sf_variations = include_lep_sf_variations and (
            metadata_lepton_flavor in {"tau_fake", "tau-fake"}
            or variation_base_str.endswith("tau_fake")
        )

        object_variation = "nominal"
        weight_variations_to_run: List[str] = []
        requested_data_weight_label: Optional[str] = None

        sow_variation_key_map: Dict[str, str] = {}
        requested_sow_variations: set = set()
        sow_variations: Dict[str, float] = {"nominal": dataset.sow}

        if (
            variation is not None
            and self._systematic_variations
            and not dataset.is_data
        ):
            group_mapping = self._metadata_to_mapping(getattr(variation, "group", None))
            group_key = (variation.base, variation.component, variation.year)
            group_info = group_mapping.get(group_key, {})

            if group_mapping:
                self._debug(
                    "Variation group mapping for '%s': mapping=%s key=%s info=%s",
                    variation_name,
                    group_mapping,
                    group_key,
                    group_info,
                )

            if not group_info and variation.metadata.get("sum_of_weights"):
                group_info = {
                    variation.name: {
                        "sum_of_weights": variation.metadata["sum_of_weights"]
                    }
                }

            if group_info:
                requested_sow_variations = set(group_info.keys())
                sow_variation_key_map = {
                    name: info.get("sum_of_weights")
                    for name, info in group_info.items()
                    if info.get("sum_of_weights")
                }

        if self._systematic_variations and not dataset.is_data:
            for sow_label in requested_sow_variations:
                if dataset.is_lo_sample:
                    sow_variations[sow_label] = dataset.sow
                else:
                    key = sow_variation_key_map.get(sow_label)
                    if key is not None and key in self._sample:
                        sow_variations[sow_label] = self._sample[key]

        if variation_type == "object":
            if variation_name not in object_systematics:
                raise ValueError(
                    f"Requested object systematic '{variation_name}' is not available in the mapping"
                )
            object_variation = variation_name
        elif variation_type in {"weight", "theory", "data_weight"}:
            variation_pool = {
                "weight": weight_systematics,
                "theory": theory_systematics,
                "data_weight": data_weight_systematics,
            }[variation_type]

            if variation_name != "nominal":
                if variation_name not in variation_pool:
                    raise ValueError(
                        f"Requested {variation_type} systematic '{variation_name}' is not available in the mapping"
                    )
                weight_variations_to_run = [variation_name]

        if variation_type == "data_weight" and variation_name != "nominal":
            requested_data_weight_label = variation_name
            for _direction in ("Up", "Down"):
                if requested_data_weight_label.endswith(_direction):
                    requested_data_weight_label = requested_data_weight_label[
                        : -len(_direction)
                    ]
                    break

        objects = VariationObjects.from_base(base_objects)

        variation_state = VariationState(
            request=request,
            name=variation_name,
            base=variation_base,
            variation_type=variation_type,
            metadata=variation_metadata,
            object_variation=object_variation,
            weight_variations=weight_variations_to_run,
            requested_data_weight_label=requested_data_weight_label,
            sow_variation_key_map=sow_variation_key_map,
            sow_variations=sow_variations,
            objects=objects,
            lepton_selection=base_objects.lepton_selection,
            jets_rho=base_objects.jets_rho,
            include_muon_sf=include_muon_sf_variations,
            include_elec_sf=include_elec_sf_variations,
            include_tau_real_sf=include_tau_real_sf_variations,
            include_tau_fake_sf=include_tau_fake_sf_variations,
        )
        return variation_state

    def _apply_tau_variations(
        self, variation_state: VariationState, dataset: DatasetContext
    ) -> VariationState:
        if not self.tau_h_analysis:
            return variation_state

        tau = variation_state.objects.taus
        central_cleaning_taus = variation_state.objects.central_cleaning_taus
        if central_cleaning_taus is None:
            central_cleaning_taus = variation_state.objects.cleaning_taus

        if central_cleaning_taus is None:
            central_cleaning_taus = tau[tau["isLoose"] > 0]

        shifted_cleaning_taus = central_cleaning_taus

        if not dataset.is_data:
            tau_pt, tau_mass = ApplyTESSystematic(
                dataset.year, tau, dataset.is_data, variation_state.object_variation
            )
            tau["pt"], tau["mass"] = tau_pt, tau_mass
            tau_pt, tau_mass = ApplyFESSystematic(
                dataset.year, tau, dataset.is_data, variation_state.object_variation
            )
            tau["pt"], tau["mass"] = tau_pt, tau_mass
            shifted_cleaning_taus = tau[tau["isLoose"] > 0]
            variation_state.objects.n_loose_taus = ak.num(shifted_cleaning_taus)
            tau_padded = ak.pad_none(tau, 1)
            variation_state.objects.tau0 = tau_padded[:, 0]

        variation_state.objects.shifted_cleaning_taus = shifted_cleaning_taus
        variation_state.objects.central_cleaning_taus = central_cleaning_taus
        variation_state.objects.cleaning_taus = shifted_cleaning_taus
        variation_state.objects.taus = tau
        return variation_state

    def _build_cleaned_jets(
        self,
        variation_state: VariationState,
        *,
        dataset: DatasetContext,
    ) -> VariationState:
        objects = variation_state.objects
        jets = objects.jets
        l_fo = objects.fakeable_leptons
        l_fo_conept_sorted = objects.fakeable_sorted
        cleaning_taus = (
            objects.shifted_cleaning_taus
            if objects.shifted_cleaning_taus is not None
            else objects.cleaning_taus
        )
        met_raw = objects.met

        def _log_jet_layout(label: str, arr: ak.Array) -> None:
            if not self._debug_logging:
                return

            counts = ak.num(arr)
            preview = ak.to_list(counts[:5])
            self._debug(
                "%s type=%s num[:5]=%s (len=%d)",
                label,
                ak.type(arr),
                preview,
                len(counts),
            )

        _log_jet_layout("jets before cleaning", jets)

        if self.tau_h_analysis:
            vetos_tocleanjets = ak.with_name(
                ak.concatenate([cleaning_taus, l_fo], axis=1),
                "PtEtaPhiMCandidate",
                behavior=ak.behavior,
            )
        else:
            vetos_tocleanjets = ak.with_name(
                l_fo, "PtEtaPhiMCandidate", behavior=ak.behavior
            )

        tmp = ak.cartesian(
            [ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True
        )
        jet_indices, veto_indices = ak.unzip(tmp)
        cleaned_jets = jets[~ak.any(jet_indices == veto_indices, axis=-1)]

        _log_jet_layout("jets after cleaning", cleaned_jets)

        jetptname = "pt_nom" if hasattr(cleaned_jets, "pt_nom") else "pt"

        cleaned_jets["pt_raw"] = (1 - cleaned_jets.rawFactor) * cleaned_jets.pt
        cleaned_jets["mass_raw"] = (1 - cleaned_jets.rawFactor) * cleaned_jets.mass
        cleaned_jets["rho"] = ak.broadcast_arrays(
            variation_state.jets_rho, cleaned_jets.pt
        )[0]

        if not dataset.is_data:
            pt_gen = None

            try:
                matched_gen = cleaned_jets.matched_gen
            except Exception:
                matched_gen = None

            if matched_gen is not None:
                pt_gen = matched_gen.pt
            else:
                genjet_idx = getattr(cleaned_jets, "genJetIdx", None)
                event_record = getattr(cleaned_jets, "_events", None)
                gen_jets = getattr(event_record, "GenJet", None)
                if gen_jets is not None and genjet_idx is not None:
                    try:
                        valid_genjet_idx = genjet_idx >= 0
                        safe_genjet_idx = ak.where(valid_genjet_idx, genjet_idx, None)
                        pt_gen = ak.where(
                            valid_genjet_idx,
                            gen_jets[safe_genjet_idx].pt,
                            ak.zeros_like(cleaned_jets.pt),
                        )
                    except Exception:
                        pt_gen = None

            if pt_gen is None:
                pt_gen = ak.zeros_like(cleaned_jets.pt)

            cleaned_jets["pt_gen"] = ak.values_astype(
                ak.fill_none(pt_gen, 0), np.float32
            )

        jet_lazy_cache = None
        jet_events = getattr(cleaned_jets, "_events", None)
        jet_caches = getattr(jet_events, "caches", None) if jet_events is not None else None
        if jet_caches:
            jet_lazy_cache = jet_caches[0]

        cleaned_jets = build_corrected_jets(
            ApplyJetCorrections(
                dataset.year,
                corr_type="jet",
                isData=dataset.is_data,
                era=dataset.run_era,
            ),
            cleaned_jets,
            lazy_cache=jet_lazy_cache,
        )
        cleaned_jets = ApplyJetSystematics(
            dataset.year, cleaned_jets, variation_state.object_variation
        )
        met = build_corrected_met(
            ApplyJetCorrections(
                dataset.year,
                corr_type="met",
                isData=dataset.is_data,
                era=dataset.run_era,
            ),
            met_raw,
            cleaned_jets,
        )

        objects.met = met
        objects.jets = cleaned_jets
        objects.fakeable_sorted = l_fo_conept_sorted

        _log_jet_layout("jets after JEC/JER", cleaned_jets)

        is_good_mask = tc_os.is_tight_jet(
            getattr(cleaned_jets, jetptname),
            cleaned_jets.eta,
            cleaned_jets.jetId,
            pt_cut=30.0,
            eta_cut=get_te_param("eta_j_cut"),
            id_cut=get_te_param("jet_id_cut"),
        )
        is_good_mask = ak.fill_none(is_good_mask, False)
        is_good_mask = ak.broadcast_arrays(is_good_mask, cleaned_jets.pt)[0]
        cleaned_jets["isGood"] = is_good_mask
        cleaned_jets["isFwd"] = te_os.isFwdJet(
            getattr(cleaned_jets, jetptname),
            cleaned_jets.eta,
            cleaned_jets.jetId,
            jetPtCut=40.0,
        )
        good_jets = cleaned_jets[is_good_mask]
        fwd_jets = cleaned_jets[cleaned_jets.isFwd]

        _log_jet_layout("good jets", good_jets)

        variation_state.cleaned_jets = cleaned_jets
        variation_state.good_jets = good_jets
        variation_state.fwd_jets = fwd_jets
        variation_state.njets = ak.values_astype(
            ak.fill_none(ak.num(good_jets.pt, axis=-1), 0), np.int64
        )
        variation_state.nfwdj = ak.values_astype(
            ak.fill_none(ak.num(fwd_jets.pt, axis=-1), 0), np.int64
        )
        variation_state.ht = (
            ak.sum(good_jets.pt, axis=-1) if "ht" in self._var_def else None
        )
        variation_state.j0 = (
            good_jets[ak.argmax(good_jets.pt, axis=-1, keepdims=True)]
            if "j0" in self._var_def
            else None
        )

        return variation_state

    def _derive_lepton_features(
        self,
        variation_state: VariationState,
        events,
        dataset: DatasetContext,
    ) -> VariationState:
        l_fo_conept_sorted = variation_state.objects.fakeable_sorted

        events["njets"] = variation_state.njets
        events["l_fo_conept_sorted"] = l_fo_conept_sorted

        te_es.add1lMaskAndSFs(
            events, dataset.year, dataset.is_data, dataset.sample_type
        )
        te_es.add2lMaskAndSFs(
            events, dataset.year, dataset.is_data, dataset.sample_type
        )
        te_es.add3lMaskAndSFs(
            events, dataset.year, dataset.is_data, dataset.sample_type
        )
        te_es.add4lMaskAndSFs(events, dataset.year, dataset.is_data)
        te_es.addLepCatMasks(events)

        l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
        variation_state.l_sorted_padded = l_fo_conept_sorted_padded
        variation_state.l0 = l_fo_conept_sorted_padded[:, 0]
        variation_state.l1 = l_fo_conept_sorted_padded[:, 1]
        variation_state.l2 = l_fo_conept_sorted_padded[:, 2]

        return variation_state

    def _apply_object_variations(
        self,
        events,
        dataset: DatasetContext,
        variation_state: VariationState,
    ) -> VariationState:
        variation_state = self._apply_tau_variations(variation_state, dataset)
        variation_state = self._build_cleaned_jets(variation_state, dataset=dataset)
        if (
            self.tau_h_analysis
            and variation_state.objects.shifted_cleaning_taus is not None
        ):
            variation_state.objects.cleaning_taus = (
                variation_state.objects.shifted_cleaning_taus
            )
        variation_state = self._derive_lepton_features(variation_state, events, dataset)

        if self._debug_logging:
            try:
                total_good_jets = (
                    int(ak.sum(variation_state.njets))
                    if variation_state.njets is not None
                    else 0
                )
            except Exception:
                total_good_jets = 0
            try:
                total_fwd_jets = (
                    int(ak.sum(variation_state.nfwdj))
                    if variation_state.nfwdj is not None
                    else 0
                )
            except Exception:
                total_fwd_jets = 0
            self._debug(
                "Prepared objects for variation '%s': object_variation=%s total_good_jets=%d total_fwd_jets=%d",
                variation_state.name,
                variation_state.object_variation,
                total_good_jets,
                total_fwd_jets,
            )

        return variation_state

    def _register_lepton_sf_weights(
        self,
        *,
        events,
        weights_object,
        lepton_category: str,
        channel_prefix: str,
        variation_state: VariationState,
        variation_name: str,
    ) -> None:
        weight_specs = _LEPTON_SF_WEIGHT_SPECS.get(channel_prefix)
        if weight_specs is None:
            raise Exception(f"Unknown channel name: {lepton_category}")

        for (
            label,
            central_attr,
            up_attr,
            down_attr,
            include_attr,
        ) in weight_specs:
            include_variations = getattr(variation_state, include_attr)
            register_lepton_sf_weight(
                weights_object,
                events,
                label,
                central_attr,
                up_attr,
                down_attr,
                include_variations,
                variation_name=variation_name,
                logger_obj=logger,
            )

        if self.tau_h_analysis and channel_prefix in {"1l", "2l", "3l"}:
            for (
                label,
                central_attr,
                up_attr,
                down_attr,
                include_attr,
            ) in _TAU_SF_WEIGHT_SPECS:
                include_variations = getattr(variation_state, include_attr)
                register_lepton_sf_weight(
                    weights_object,
                    events,
                    label,
                    central_attr,
                    up_attr,
                    down_attr,
                    include_variations,
                    variation_name=variation_name,
                    logger_obj=logger,
                )

    def _register_weights_for_variation(
        self,
        events,
        dataset: DatasetContext,
        variation_state: VariationState,
        data_weight_systematics,
        data_weight_systematics_set,
    ):
        weights_object = coffea.analysis_tools.Weights(
            len(events), storeIndividual=True
        )

        goodJets = variation_state.good_jets
        isData = dataset.is_data
        year = dataset.year
        trigger_weight_label = f"triggerSF_{year}"
        sow = dataset.sow
        xsec = dataset.xsec
        lumi = dataset.lumi
        histAxisName = dataset.hist_axis_name
        sampleType = dataset.sample_type
        variation = variation_state.request.variation
        variation_base = variation_state.base
        variation_name = variation_state.name

        nlep_cat = re.match(r"\d+l", self.channel).group(0)
        channel_prefix = nlep_cat[:2]

        if year == "2016":
            year_light = "2016APV"
        else:
            year_light = year

        loose_tag = "btag_wp_loose_" + year.replace("201", "UL1")
        btagwpl = get_tc_param(loose_tag)
        isBtagJetsLoose = goodJets.btagDeepFlavB > btagwpl
        isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
        nbtagsl = ak.sum(isBtagJetsLoose, axis=-1)

        medium_tag = "btag_wp_medium_" + year.replace("201", "UL1")
        btagwpm = get_tc_param(medium_tag)
        isBtagJetsMedium = goodJets.btagDeepFlavB > btagwpm
        isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
        nbtagsm = ak.sum(isBtagJetsMedium, axis=-1)
        isBtagJetsLooseNotMedium = isBtagJetsLoose & isNotBtagJetsMedium

        variation_state.isBtagJetsLoose = isBtagJetsLoose
        variation_state.isNotBtagJetsLoose = isNotBtagJetsLoose
        variation_state.nbtagsl = nbtagsl
        variation_state.isBtagJetsMedium = isBtagJetsMedium
        variation_state.isNotBtagJetsMedium = isNotBtagJetsMedium
        variation_state.nbtagsm = nbtagsm
        variation_state.isBtagJetsLooseNotMedium = isBtagJetsLooseNotMedium

        default_flavour_mask = ak.values_astype(
            ak.zeros_like(goodJets.pt, highlevel=True), np.bool_
        )

        if not isData:
            has_hadron_flavour = hasattr(goodJets, "hadronFlavour")
            variation_state.has_hadron_flavour = has_hadron_flavour
            if has_hadron_flavour:
                light_mask = goodJets.hadronFlavour == 0
                bc_mask = goodJets.hadronFlavour > 0
            else:
                logger.warning(
                    "Missing 'hadronFlavour' for MC sample '%s'; defaulting to empty jet flavour masks.",
                    dataset.dataset,
                )
                light_mask = default_flavour_mask
                bc_mask = default_flavour_mask

            variation_state.light_mask = light_mask
            variation_state.bc_mask = bc_mask
            jets_light = goodJets[light_mask]
            jets_bc = goodJets[bc_mask]
            variation_state.jets_light = jets_light
            variation_state.jets_bc = jets_bc

            if dataset.eft_coeffs is None:
                genw = self._ensure_ak_array(
                    events["genWeight"] if "genWeight" in events.fields else None,
                    dtype=self._dtype,
                )
                if genw is None:
                    genw = ak.ones_like(events["event"], dtype=self._dtype)
            else:
                genw = ak.ones_like(events["event"], dtype=self._dtype)

            weights_object.add("norm", (xsec / sow) * genw * lumi)

            have_systematics = bool(self._systematic_variations)
            tc_cor.AttachScaleWeights(events)

            theory_weight_arguments = apply_theory_weight_variations(
                events=events,
                variation=variation,
                variation_base=variation_base,
                have_systematics=have_systematics,
                sow=sow,
                sow_variations=variation_state.sow_variations,
                sow_variation_key_map=variation_state.sow_variation_key_map,
                is_lo_sample=dataset.is_lo_sample,
                hist_axis_name=histAxisName,
                sample=self._sample,
            )
            for label, args in theory_weight_arguments.items():
                weights_object.add(label, *args)

            if dataset.is_run2:
                l1prefiring_args = [
                    events["L1PreFiringWeight"]["Nom"],
                    events["L1PreFiringWeight"]["Up"],
                    events["L1PreFiringWeight"]["Dn"],
                ]
            else:
                l1prefiring_args = [
                    ak.ones_like(events["nom"]),
                    ak.ones_like(events["nom"]),
                    ak.ones_like(events["nom"]),
                ]

            register_weight_variation(
                weights_object,
                "PreFiring",
                l1prefiring_args[0],
                up=lambda: l1prefiring_args[1],
                down=lambda: l1prefiring_args[2],
                active=have_systematics and variation_state.base == "prefiring",
            )

            pu_central = tc_cor.GetPUSF(events["Pileup"].nTrueInt, year)
            register_weight_variation(
                weights_object,
                "PU",
                pu_central,
                up=lambda: tc_cor.GetPUSF(events["Pileup"].nTrueInt, year, "up"),
                down=lambda: tc_cor.GetPUSF(events["Pileup"].nTrueInt, year, "down"),
                active=have_systematics and variation_state.base == "pileup",
            )

            if variation_state.has_hadron_flavour:
                btag_effM_light = GetBtagEff(variation_state.jets_light, year, "medium")
                btag_effM_bc = GetBtagEff(variation_state.jets_bc, year, "medium")
                btag_effL_light = GetBtagEff(variation_state.jets_light, year, "loose")
                btag_effL_bc = GetBtagEff(variation_state.jets_bc, year, "loose")
                btag_sfM_light = tc_cor.btag_sf_eval(
                    variation_state.jets_light,
                    "M",
                    year_light,
                    "deepJet_incl",
                    "central",
                )
                btag_sfM_bc = tc_cor.btag_sf_eval(
                    variation_state.jets_bc,
                    "M",
                    year,
                    "deepJet_comb",
                    "central",
                )
                btag_sfL_light = tc_cor.btag_sf_eval(
                    variation_state.jets_light,
                    "L",
                    year_light,
                    "deepJet_incl",
                    "central",
                )
                btag_sfL_bc = tc_cor.btag_sf_eval(
                    variation_state.jets_bc,
                    "L",
                    year,
                    "deepJet_comb",
                    "central",
                )

                pData_light, pMC_light = tc_cor.get_method1a_wgt_doublewp(
                    btag_effM_light,
                    btag_effL_light,
                    btag_sfM_light,
                    btag_sfL_light,
                    isBtagJetsMedium[variation_state.light_mask],
                    isBtagJetsLooseNotMedium[variation_state.light_mask],
                    isNotBtagJetsLoose[variation_state.light_mask],
                )
                btag_w_light = pData_light / pMC_light
                pData_bc, pMC_bc = tc_cor.get_method1a_wgt_doublewp(
                    btag_effM_bc,
                    btag_effL_bc,
                    btag_sfM_bc,
                    btag_sfL_bc,
                    isBtagJetsMedium[variation_state.bc_mask],
                    isBtagJetsLooseNotMedium[variation_state.bc_mask],
                    isNotBtagJetsLoose[variation_state.bc_mask],
                )
                btag_w_bc = pData_bc / pMC_bc

                btag_result = register_btag_sf_weights(
                    jets_light=variation_state.jets_light,
                    jets_bc=variation_state.jets_bc,
                    efficiencies={
                        "light": {"M": btag_effM_light, "L": btag_effL_light},
                        "bc": {"M": btag_effM_bc, "L": btag_effL_bc},
                    },
                    central_values={
                        "light": {"weight": btag_w_light, "pMC": pMC_light},
                        "bc": {"weight": btag_w_bc, "pMC": pMC_bc},
                    },
                    selection_masks={
                        "medium": isBtagJetsMedium,
                        "loose_not_medium": isBtagJetsLooseNotMedium,
                        "not_loose": isNotBtagJetsLoose,
                        "light": variation_state.light_mask,
                        "bc": variation_state.bc_mask,
                    },
                    years={"light": year_light, "bc": year},
                    systematic_descriptor={
                        "has_systematics": bool(self._systematic_variations),
                        "object_variation": variation_state.object_variation,
                        "variation_name": variation_name,
                    },
                )

                weights_object.add("btagSF", btag_result.central)

                if btag_result.variation_label is not None:
                    # ``register_btag_sf_weights`` returns variation weights that are already
                    # expressed relative to the central value.  Register them as a unity
                    # nominal correction so ``Weights`` exposes the ``Up``/``Down`` modifiers
                    # required during histogram filling.
                    weights_object.add(
                        btag_result.variation_label,
                        events["nom"],
                        btag_result.variation_up,
                        btag_result.variation_down,
                    )
            else:
                weights_object.add("btagSF", ak.ones_like(events["nom"]))

            self._register_lepton_sf_weights(
                events=events,
                weights_object=weights_object,
                lepton_category=nlep_cat,
                channel_prefix=channel_prefix,
                variation_state=variation_state,
                variation_name=variation_name,
            )

            register_trigger_sf_weight(
                weights_object,
                year=dataset.year,
                events=events,
                lepton0=variation_state.l0,
                lepton1=variation_state.l1,
                label=trigger_weight_label,
                variation_descriptor={
                    "has_systematics": bool(self._systematic_variations),
                    "variation_base": variation_state.base,
                    "variation_name": variation_name,
                },
                logger_obj=logger,
            )

            if self.tau_h_analysis:
                AttachTauSF(
                    events,
                    variation_state.objects.taus,
                    year,
                )
                register_weight_variation(
                    weights_object,
                    "tauID",
                    events["tau_SF_central"],
                    up=lambda: events["tau_SF_up"],
                    down=lambda: events["tau_SF_down"],
                    active=bool(self._systematic_variations)
                    and variation_state.base == "tauID",
                )
                register_weight_variation(
                    weights_object,
                    "tauMisID",
                    events["tau_misID_central"],
                    up=lambda: events["tau_misID_up"],
                    down=lambda: events["tau_misID_down"],
                    active=bool(self._systematic_variations)
                    and variation_state.base == "tauMisID",
                )
        else:
            variation_state.has_hadron_flavour = False

        if channel_prefix in {"1l", "2l", "3l"}:
            add_fake_factor_weights(
                weights_object,
                events,
                channel_prefix,
                year,
                variation_state.requested_data_weight_label,
            )

        if channel_prefix == "2l":
            flipfactor_central = getattr(events, "flipfactor_2l")
            flipfactor_up = getattr(events, "flipfactor_2l_up", None)
            flipfactor_down = getattr(events, "flipfactor_2l_down", None)

            charge_flip_central = flipfactor_central
            charge_flip_up = (
                (lambda: flipfactor_up) if flipfactor_up is not None else None
            )
            charge_flip_down = (
                (lambda: flipfactor_down) if flipfactor_down is not None else None
            )

            if isData and ("os" not in self.channel):

                def _charge_flip_ratio(values):
                    denominator = flipfactor_central
                    ones = ak.ones_like(denominator)
                    nonzero = denominator != 0
                    safe_denominator = ak.where(nonzero, denominator, ones)
                    return ak.where(nonzero, values / safe_denominator, ones)

                charge_flip_central = lambda: ak.ones_like(flipfactor_central)
                if flipfactor_up is not None:
                    charge_flip_up = lambda: _charge_flip_ratio(flipfactor_up)
                if flipfactor_down is not None:
                    charge_flip_down = lambda: _charge_flip_ratio(flipfactor_down)

            register_weight_variation(
                weights_object,
                "charge_flips",
                charge_flip_central,
                up=charge_flip_up,
                down=charge_flip_down,
                active=bool(self._systematic_variations)
                and variation_state.base == "charge_flips",
            )

            if isData and ("os" not in self.channel):
                weights_object.add("fliprate", flipfactor_central)

                central_modifiers = getattr(weights_object, "_weights", None).keys()

                if central_modifiers is None or "fliprate" not in set(
                    central_modifiers
                ):
                    raise AssertionError(
                        "The 2l same-sign data branch must register the central 'fliprate' weight."
                    )

        if isData and self._systematic_variations:
            validate_data_weight_variations(
                weights_object,
                data_weight_systematics,
                variation_state.requested_data_weight_label,
                variation_name,
            )

        if self._debug_logging:
            weight_summary = (
                variation_state.weight_variations
                if variation_state.weight_variations
                else ["nominal"]
            )
            self._debug(
                "Registered weight configuration for '%s': weights=%s data_weight=%s sow_labels=%s",
                variation_state.name,
                weight_summary,
                variation_state.requested_data_weight_label,
                sorted(variation_state.sow_variations.keys()),
            )

        return weights_object

    def _fill_histograms_for_variation(
        self,
        events,
        dataset: DatasetContext,
        variation_state: VariationState,
        weights_object,
        hist_label: str,
        data_weight_systematics_set,
        hout,
    ) -> None:
        goodJets = variation_state.good_jets
        fwdJets = variation_state.fwd_jets
        njets = variation_state.njets
        isBtagJetsLoose = variation_state.isBtagJetsLoose
        isNotBtagJetsLoose = variation_state.isNotBtagJetsLoose
        isBtagJetsMedium = variation_state.isBtagJetsMedium
        isNotBtagJetsMedium = variation_state.isNotBtagJetsMedium
        isBtagJetsLooseNotMedium = variation_state.isBtagJetsLooseNotMedium
        l_fo_conept_sorted = variation_state.objects.fakeable_sorted
        l_fo_conept_sorted_padded = variation_state.l_sorted_padded
        l0 = variation_state.l0
        l1 = variation_state.l1
        l2 = variation_state.l2
        tau = variation_state.objects.taus
        nLtau = variation_state.objects.n_loose_taus
        tau0 = variation_state.objects.tau0

        histAxisName = dataset.hist_axis_name
        trigger_dataset = dataset.trigger_dataset
        year = dataset.year
        isData = dataset.is_data

        sfosz_3l_OnZ_mask = tc_es.get_Z_peak_mask(
            l_fo_conept_sorted_padded[:, 0:3], pt_window=10.0
        )
        sfosz_3l_OffZ_mask = ~sfosz_3l_OnZ_mask
        if self.offZ_3l_split:
            sfosz_3l_OffZ_low_mask = tc_es.get_off_Z_mask_low(
                l_fo_conept_sorted_padded[:, 0:3], pt_window=0.0
            )
            sfosz_3l_OffZ_any_mask = tc_es.get_any_sfos_pair(
                l_fo_conept_sorted_padded[:, 0:3]
            )
        sfosz_2l_mask = tc_es.get_Z_peak_mask(
            l_fo_conept_sorted_padded[:, 0:2], pt_window=10.0
        )
        sfasz_2l_mask = tc_es.get_Z_peak_mask(
            l_fo_conept_sorted_padded[:, 0:2], pt_window=30.0, flavor="as"
        )
        if self.tau_h_analysis and tau0 is not None:
            tl_zpeak_mask = te_es.lt_Z_mask(l0, l1, tau0, 30.0)
        else:
            tl_zpeak_mask = None

        pass_trg = tc_es.trg_pass_no_overlap(
            events,
            isData,
            trigger_dataset,
            str(year),
            te_es.dataset_dict_top22006,
            te_es.exclude_dict_top22006,
        )

        if goodJets is None:
            raise ValueError("goodJets is required to evaluate b-tag categories")
        if isBtagJetsLoose is None or isBtagJetsMedium is None:
            raise ValueError("B-tag jet masks must be populated before histogram filling")

        def _ensure_flat_counts(counts, *, label, fallback=None):
            if counts is None and fallback is not None:
                counts = fallback

            if counts is None:
                raise ValueError(f"{label} must be provided to evaluate selections")

            if ak.fields(counts):
                raise TypeError(
                    f"{label} must be a numeric per-event Awkward array, not a Record"
                )

            counts = ak.values_astype(ak.fill_none(counts, 0), np.int64)
            counts_layout = ak.to_layout(counts, allow_record=False)
            if counts_layout.purelist_depth != 1:
                raise TypeError(
                    f"{label} must be a flat per-event array, not {counts_layout.purelist_depth}D"
                )
            return ak.Array(counts_layout)

        nbtagsl = _ensure_flat_counts(
            ak.sum(isBtagJetsLoose, axis=-1), label="nbtagsl"
        )
        nbtagsm = _ensure_flat_counts(
            ak.sum(isBtagJetsMedium, axis=-1), label="nbtagsm"
        )

        variation_state.nbtagsl = nbtagsl
        variation_state.nbtagsm = nbtagsm

        bmask_atleast1med_atleast2loose = (nbtagsm >= 1) & (nbtagsl >= 2)
        bmask_exactly0med = nbtagsm == 0
        bmask_exactly1med = nbtagsm == 1
        bmask_exactly2med = nbtagsm == 2
        bmask_atleast2med = nbtagsm >= 2
        bmask_atmost2med = nbtagsm < 3
        bmask_atleast3med = nbtagsm >= 3

        if fwdJets is not None and "pt" in ak.fields(fwdJets):
            fwd_mask = ak.ones_like(fwdJets.pt, dtype=bool)
        elif goodJets is not None and "isFwd" in ak.fields(goodJets):
            fwd_mask = ak.fill_none(goodJets.isFwd, False)
        else:
            fwd_mask = None

        if fwd_mask is not None:
            fallback_nfwdj = ak.sum(fwd_mask, axis=-1)
            fwdjet_mask = fallback_nfwdj > 0
        else:
            fallback_nfwdj = ak.zeros_like(events["event"], dtype=np.int64)
            fwdjet_mask = ak.zeros_like(events["event"], dtype=bool)

        variation_state.nfwdj = _ensure_flat_counts(
            variation_state.nfwdj,
            label="nfwdj",
            fallback=fallback_nfwdj,
        )

        variation_state.njets = _ensure_flat_counts(
            variation_state.njets,
            label="njets",
            fallback=ak.num(goodJets.pt, axis=-1),
        )

        nfwdj = variation_state.nfwdj
        njets = variation_state.njets

        chargel0_p = ak.fill_none((l0.charge) > 0, False)
        chargel0_m = ak.fill_none((l0.charge) < 0, False)
        charge2l_0 = ak.fill_none(((l0.charge + l1.charge) == 0), False)
        charge2l_1 = ak.fill_none(((l0.charge + l1.charge) != 0), False)
        charge3l_p = ak.fill_none(((l0.charge + l1.charge + l2.charge) > 0), False)
        charge3l_m = ak.fill_none(((l0.charge + l1.charge + l2.charge) < 0), False)
        if self.tau_h_analysis:
            tau_F_mask = ak.num(tau[tau["isVLoose"] > 0]) >= 1
            tau_L_mask = ak.num(tau[tau["isLoose"] > 0]) >= 1
            no_tau_mask = ak.num(tau[tau["isLoose"] > 0]) == 0
        else:
            tau_F_mask = tau_L_mask = no_tau_mask = None

        selections = PackedSelection(dtype="uint64")
        preselections = PackedSelection(dtype="uint64")
        lumi_mask = dataset.lumi_mask
        selections.add("is_good_lumi", lumi_mask)
        preselections.add("is_good_lumi", lumi_mask)

        preselections.add("chargedl0", (chargel0_p | chargel0_m))
        preselections.add("2l_nozeeveto", (events["is2l_nozeeveto"] & pass_trg))
        preselections.add("2los", charge2l_0)
        preselections.add("2lem", events["is_em"])
        preselections.add("2lee", events["is_ee"])
        preselections.add("2lmm", events["is_mm"])
        preselections.add("2l_onZ_as", sfasz_2l_mask)
        preselections.add("2l_onZ", sfosz_2l_mask)
        preselections.add("bmask_atleast3m", bmask_atleast3med)
        preselections.add("bmask_atleast1m2l", bmask_atleast1med_atleast2loose)
        preselections.add("bmask_atmost2m", bmask_atmost2med)
        preselections.add("fwdjet_mask", fwdjet_mask)
        preselections.add("~fwdjet_mask", ~fwdjet_mask)

        if self.tau_h_analysis:
            preselections.add("1l", (events["is1l"] & pass_trg))
            preselections.add("1tau", tau_L_mask)
            preselections.add("1Ftau", tau_F_mask)
            preselections.add("0tau", no_tau_mask)
            preselections.add("onZ_tau", tl_zpeak_mask)
            preselections.add(
                "offZ_tau",
                ~tl_zpeak_mask if tl_zpeak_mask is not None else tl_zpeak_mask,
            )
        if self.fwd_analysis:
            preselections.add("2lss_fwd", (events["is2l"] & pass_trg & fwdjet_mask))
            preselections.add("2l_fwd_p", (chargel0_p & fwdjet_mask))
            preselections.add("2l_fwd_m", (chargel0_m & fwdjet_mask))

        preselections.add("2lss", (events["is2l"] & pass_trg))
        preselections.add("2l_p", chargel0_p)
        preselections.add("2l_m", chargel0_m)

        preselections.add("3l", (events["is3l"] & pass_trg))
        preselections.add("bmask_exactly0m", bmask_exactly0med)
        preselections.add("bmask_exactly1m", bmask_exactly1med)
        preselections.add("bmask_exactly2m", bmask_exactly2med)
        preselections.add("bmask_atleast2m", bmask_atleast2med)
        preselections.add("3l_p", (events["is3l"] & pass_trg & charge3l_p))
        preselections.add("3l_m", (events["is3l"] & pass_trg & charge3l_m))
        preselections.add("3l_onZ", sfosz_3l_OnZ_mask)

        if self.offZ_3l_split:
            preselections.add(
                "3l_offZ_low",
                (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & sfosz_3l_OffZ_low_mask),
            )
            preselections.add(
                "3l_offZ_high",
                (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & ~sfosz_3l_OffZ_low_mask),
            )
            preselections.add(
                "3l_offZ_none",
                (sfosz_3l_OffZ_mask & ~sfosz_3l_OffZ_any_mask),
            )
            preselections.add(
                "3l_offZ_split",
                (
                    preselections.any("3l_offZ_low")
                    | preselections.any("3l_offZ_high")
                    | preselections.any("3l_offZ_none")
                ),
            )
            preselections.add(
                "3l_offZ",
                (
                    preselections.any("3l_offZ_low")
                    | preselections.any("3l_offZ_high")
                    | preselections.any("3l_offZ_none")
                ),
            )
        else:
            preselections.add("3l_offZ", sfosz_3l_OffZ_mask)

        preselections.add("4l", (events["is4l"] & pass_trg))

        lep_ch = self._channel_dict["chan_def_lst"]
        tempmask = None
        chtag = lep_ch[0]
        for chcut in lep_ch[1:]:
            tempmask = (
                tempmask & preselections.any(chcut)
                if tempmask is not None
                else preselections.any(chcut)
            )
        selections.add(chtag, tempmask)

        del preselections

        selections.add("e", events["is_e"])
        selections.add("m", events["is_m"])
        selections.add("ee", events["is_ee"])
        selections.add("em", events["is_em"])
        selections.add("mm", events["is_mm"])
        selections.add("eee", events["is_eee"])
        selections.add("eem", events["is_eem"])
        selections.add("emm", events["is_emm"])
        selections.add("mmm", events["is_mmm"])
        selections.add(
            "llll",
            (
                events["is_eeee"]
                | events["is_eeem"]
                | events["is_eemm"]
                | events["is_emmm"]
                | events["is_mmmm"]
                | events["is_gr4l"]
            ),
        )

        selections.add("exactly_0j", njets == 0)
        selections.add("exactly_1j", njets == 1)
        selections.add("exactly_2j", njets == 2)
        selections.add("exactly_3j", njets == 3)
        selections.add("exactly_4j", njets == 4)
        selections.add("exactly_5j", njets == 5)
        selections.add("exactly_6j", njets == 6)
        selections.add("atleast_1j", njets >= 1)
        selections.add("atleast_4j", njets >= 4)
        selections.add("atleast_5j", njets >= 5)
        selections.add("atleast_6j", njets >= 6)
        selections.add("atleast_7j", njets >= 7)
        selections.add("atleast_0j", njets >= 0)
        selections.add("atmost_3j", njets <= 3)

        selections.add("isSR_2lSS", (events["is2l_SR"]) & charge2l_1)
        selections.add("isAR_2lSS", (~events["is2l_SR"]) & charge2l_1)
        selections.add("isAR_2lSS_OS", (events["is2l_SR"]) & charge2l_0)
        selections.add("isSR_2lOS", (events["is2l_SR"]) & charge2l_0)
        selections.add("isAR_2lOS", (~events["is2l_SR"]) & charge2l_0)
        if self.tau_h_analysis:
            selections.add("isSR_1l", events["is1l_SR"])
        selections.add("isSR_3l", events["is3l_SR"])
        selections.add("isAR_3l", ~events["is3l_SR"])
        selections.add("isSR_4l", events["is4l_SR"])

        var_def = self.var_def

        if ("ptbl" in var_def) or ("b0pt" in var_def) or ("bl0pt" in var_def):
            ptbl = self._compute_ptbl(
                goodJets, isBtagJetsMedium, isBtagJetsLoose, l_fo_conept_sorted
            )
        else:
            ptbl = None

        if "ptz" in var_def:
            ptz = te_es.get_Z_pt(l_fo_conept_sorted_padded[:, 0:3], 10.0)
            if self.offZ_3l_split:
                ptz = te_es.get_ll_pt(l_fo_conept_sorted_padded[:, 0:3], 10.0)
        else:
            ptz = None
        if "ptz_wtau" in var_def and tau0 is not None:
            ptz_wtau = te_es.get_Zlt_pt(l0, l1, tau0)
        else:
            ptz_wtau = None

        if "bl0pt" in var_def:
            bjetsl = goodJets[isBtagJetsLoose][
                ak.argsort(goodJets[isBtagJetsLoose].pt, axis=-1, ascending=False)
            ]
            bl_pairs = ak.cartesian({"b": bjetsl, "l": l_fo_conept_sorted})
            blpt = (bl_pairs["b"] + bl_pairs["l"]).pt
            bl0pt = ak.flatten(blpt[ak.argmax(blpt, axis=-1, keepdims=True)])
        else:
            bl0pt = None

        need_lj_collection = any(
            token in var_def for token in ["o0pt", "lj0pt", "ljptsum"]
        ) or (self._ecut_threshold is not None)
        if need_lj_collection:
            if self.tau_h_analysis:
                l_j_collection = ak.with_name(
                    ak.concatenate(
                        [
                            l_fo_conept_sorted,
                            goodJets,
                            variation_state.objects.cleaning_taus,
                        ],
                        axis=1,
                    ),
                    "PtEtaPhiMCollection",
                )
            else:
                l_j_collection = ak.with_name(
                    ak.concatenate([l_fo_conept_sorted, goodJets], axis=1),
                    "PtEtaPhiMCollection",
                )
            if "o0pt" in var_def:
                o0pt = ak.max(l_j_collection.pt, axis=-1)
            else:
                o0pt = None
            if ("ljptsum" in var_def) or (self._ecut_threshold is not None):
                ljptsum = ak.sum(l_j_collection.pt, axis=-1)
            else:
                ljptsum = None
            if "lj0pt" in var_def:
                l_j_pairs = ak.combinations(l_j_collection, 2, fields=["o0", "o1"])
                l_j_pairs_pt = (l_j_pairs.o0 + l_j_pairs.o1).pt
                lj0pt = ak.max(l_j_pairs_pt, axis=-1)
            else:
                lj0pt = None
        else:
            o0pt = ljptsum = lj0pt = None

        if "lt" in var_def:
            lt = (
                ak.sum(l_fo_conept_sorted_padded.pt, axis=-1)
                + variation_state.objects.met.pt
            )
        else:
            lt = None

        if "mll_0_1" in var_def:
            mll_0_1 = (l0 + l1).mass
        else:
            mll_0_1 = None

        if self._ecut_threshold is not None:
            if ljptsum is None:
                if self.tau_h_analysis:
                    l_j_collection = ak.with_name(
                        ak.concatenate(
                            [
                                l_fo_conept_sorted,
                                goodJets,
                                variation_state.objects.cleaning_taus,
                            ],
                            axis=1,
                        ),
                        "PtEtaPhiMCollection",
                    )
                else:
                    l_j_collection = ak.with_name(
                        ak.concatenate([l_fo_conept_sorted, goodJets], axis=1),
                        "PtEtaPhiMCollection",
                    )
                ljptsum = ak.sum(l_j_collection.pt, axis=-1)
            ecut_mask = ljptsum < self._ecut_threshold
        else:
            ecut_mask = None

        dense_axis_name = self._var
        dense_axis_vals = eval(self._var_def, {"ak": ak, "np": np}, locals())

        weight_variations_to_run = list(variation_state.weight_variations)
        if weight_variations_to_run:
            wgt_var_lst = []
        else:
            wgt_var_lst = ["nominal"]
        for name in weight_variations_to_run:
            if name not in wgt_var_lst:
                wgt_var_lst.append(name)

        lep_chan = self._channel_dict["chan_def_lst"][0]
        jet_req = self._channel_dict["jet_selection"]
        lep_flav_iter = (
            self._channel_dict["lep_flav_lst"]
            if self._split_by_lepton_flavor
            else [None]
        )

        for wgt_fluct in wgt_var_lst:
            if wgt_fluct == "nominal":
                weight = weights_object.weight(None)
            elif wgt_fluct in weights_object.variations:
                weight = weights_object.weight(wgt_fluct)
            else:
                continue

            if (
                self.appregion.startswith("isSR")
                and wgt_fluct in data_weight_systematics_set
            ):
                continue

            if wgt_fluct == "nominal":
                hist_variation_label = hist_label
            else:
                hist_variation_label = self._histogram_label_lookup.get(
                    wgt_fluct, wgt_fluct
                )

            for lep_flav in lep_flav_iter:
                cuts_lst = [self.appregion, lep_chan]
                flav_ch = None
                njet_ch = None
                if isData:
                    cuts_lst.append("is_good_lumi")
                if self._split_by_lepton_flavor:
                    flav_ch = lep_flav
                    cuts_lst.append(lep_flav)
                if dense_axis_name != "njets":
                    njet_ch = jet_req
                    cuts_lst.append(jet_req)

                ch_name, base_ch_name = self._build_channel_names(
                    lep_chan, njet_ch, flav_ch
                )
                if base_ch_name != self.channel:
                    continue

                if self._debug_logging:
                    cut_pass_info = {cut: selections.all(cut) for cut in cuts_lst}
                    self._debug(
                        "Filling histograms for channel '%s' (base '%s') with cuts %s",
                        ch_name,
                        base_ch_name,
                        cut_pass_info,
                    )

                all_cuts_mask = selections.all(*cuts_lst)
                if ecut_mask is not None:
                    all_cuts_mask = all_cuts_mask & ecut_mask
                if isinstance(all_cuts_mask, ak.Array):
                    mask_numpy = (
                        ak.to_numpy(all_cuts_mask)
                        if hasattr(ak, "to_numpy")
                        else np.asarray(all_cuts_mask)
                    )
                else:
                    mask_numpy = np.asarray(all_cuts_mask)
                weights_flat = np.asarray(weight)[mask_numpy]
                eft_coeffs = dataset.eft_coeffs
                eft_coeffs_cut = (
                    eft_coeffs[all_cuts_mask] if eft_coeffs is not None else None
                )

                axes_fill_info_dict = {
                    dense_axis_name: dense_axis_vals[all_cuts_mask],
                    "weight": weights_flat,
                    "eft_coeff": eft_coeffs_cut,
                }

                histkey = self._build_histogram_key(
                    dense_axis_name,
                    ch_name,
                    dataset.dataset,
                    hist_variation_label,
                    application=self._appregion,
                )

                if histkey not in hout:
                    fallback_histkey = self._build_histogram_key(
                        dense_axis_name,
                        base_ch_name,
                        dataset.dataset,
                        hist_variation_label,
                        application=self._appregion,
                    )
                    if fallback_histkey not in hout:
                        continue
                    histkey = fallback_histkey

                hout[histkey].fill(**axes_fill_info_dict)

                if self._debug_logging:
                    filled_count = int(np.count_nonzero(np.asarray(all_cuts_mask)))
                    self._debug(
                        "Filled histkey %s with %d selected events",
                        histkey,
                        filled_count,
                    )

                axes_fill_info_dict = {
                    dense_axis_name + "_sumw2": dense_axis_vals[all_cuts_mask],
                    "weight": np.square(weights_flat),
                    "eft_coeff": eft_coeffs_cut,
                }
                histkey = self._build_histogram_key(
                    dense_axis_name + "_sumw2",
                    base_ch_name,
                    dataset.dataset,
                    hist_variation_label,
                    application=self._appregion,
                )
                if histkey not in hout.keys():
                    continue
                hout[histkey].fill(**axes_fill_info_dict)

    @property
    def channel(self):
        return self._channel

    @property
    def appregion(self):
        return self._appregion

    @property
    def syst(self):
        return self._syst

    @property
    def systematic_info(self):
        return self._systematic_info

    @property
    def available_systematics(self):
        return self._available_systematics

    @property
    def columns(self):
        return self._columns

    def _resolve_dataset_names(self, dataset_name: str) -> Tuple[str, str]:
        """Return the dataset label for histogram keys and the trigger dataset name."""

        dataset_for_histograms = dataset_name
        dataset_for_triggers = dataset_name

        dataset_prefixes = (
            "Muon",
            "SingleMuon",
            "SingleElectron",
            "EGamma",
            "MuonEG",
            "DoubleMuon",
            "DoubleElectron",
            "DoubleEG",
        )
        for prefix in dataset_prefixes:
            if dataset_for_triggers.startswith(prefix):
                dataset_for_triggers = dataset_for_triggers.split("_")[0]
                break

        return dataset_for_histograms, dataset_for_triggers

    def _debug(self, message: str, *args) -> None:
        if not self._debug_logging:
            return

        logger.debug(message, *args)

        if self._suppress_debug_prints:
            return

        try:
            formatted = message % args if args else message
        except Exception:
            formatted = " ".join([message, *(str(arg) for arg in args)])
        print(formatted, flush=True)

    # Main function: run on a given dataset

    def process(self, events):
        dataset = self._build_dataset_context(events)
        base_objects = self._select_base_objects(events, dataset)
        variation_requests = self._build_variation_requests()

        object_systematics = self._available_systematics.get("object", ())
        weight_systematics = self._available_systematics.get("weight", ())
        theory_systematics = self._available_systematics.get("theory", ())
        data_weight_systematics = self._available_systematics.get("data_weight", ())
        data_weight_systematics_set = self._available_systematics_sets.get(
            "data_weight", set()
        )

        hout = self.accumulator

        for request in variation_requests:
            variation_state = self._initialize_variation_state(
                request,
                base_objects,
                dataset,
                object_systematics,
                weight_systematics,
                theory_systematics,
                data_weight_systematics,
            )

            self._debug(
                "Processing variation '%s' (type: %s, base: %s)",
                variation_state.name,
                variation_state.variation_type,
                variation_state.base,
            )

            variation_state = self._apply_object_variations(
                events,
                dataset,
                variation_state,
            )

            weights_object = self._register_weights_for_variation(
                events,
                dataset,
                variation_state,
                data_weight_systematics,
                data_weight_systematics_set,
            )

            self._fill_histograms_for_variation(
                events,
                dataset,
                variation_state,
                weights_object,
                request.histogram_label,
                data_weight_systematics_set,
                hout,
            )

        return hout

    def postprocess(self, accumulator):
        return accumulator
