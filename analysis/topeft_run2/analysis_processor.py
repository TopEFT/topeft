#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
import json

import hist
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.event_selection as tc_es
import topcoffea.modules.object_selection as tc_os
import topcoffea.modules.corrections as tc_cor

from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.axis_binning import make_processing_axis
from topeft.modules.missing_parton_contract import parse_analysis_njet_token
from topeft.modules.nominal_schema import (
    EFT_NOMINAL_SUFFIX,
    NOMINAL_CONTAINER_LAYOUT,
    NOMINAL_CONTAINER_SCHEMA_VERSION,
    SCALAR_NOMINAL_SUFFIX,
    eft_nominal_key,
    scalar_nominal_key,
)
from topeft.modules.paths import topeft_path
from topeft.modules.sumw2_policy import resolve_nominal_component_availability
from topeft.modules.corrections import ApplyJetCorrections, ApplyMETSystematics, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachElectronCorrections, AttachTauSF, AttachTauEnergyCorrections, ApplyTauEnergySystematics, AttachPerLeptonFR, AttachMuonMomentumCorrections, ApplyMuonMomentumSystematics, get_supported_muon_momentum_systematics, get_supported_tau_energy_systematics, ApplyJetSystematics, GetTriggerSF, ApplyJetVetoMaps, get_selected_met, get_selected_raw_met, get_corr_t1_met_jets, get_supported_jet_systematics, get_supported_met_systematics, is_met_unclustered_systematic, resolve_forward_eta_stochastic_jer_suppression, use_type1_met
import topeft.modules.event_selection as te_es
import topeft.modules.object_selection as te_os
from topeft.modules.ttgamma_photon_history import (
    attach_conversion_overlap_removal_diagnostics,
    attach_photon_history_diagnostics,
    get_ttgamma_sample_role_policy,
)
from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Takes strings as inputs, constructs a string for the full channel name
# Try to construct a channel name like this: [n leptons]_[lepton flavors]_[p or m charge]_[on or off Z]_[n b jets]_[n jets]
    # chan_str should look something like "3l_p_offZ_1b", NOTE: This function assumes nlep comes first
    # njet_str should look something like "atleast_5j",   NOTE: This function assumes njets comes last
    # flav_str should look something like "emm"
def construct_cat_name(chan_str,njet_str=None,flav_str=None):
    # Get the component strings
    nlep_str = chan_str.split("_")[0] # Assumes n leps comes first in the str
    chan_str = "_".join(chan_str.split("_")[1:]) # The rest of the channel name is everything that comes after nlep
    if chan_str == "": chan_str = None # So that we properly skip this in the for loop below
    if flav_str is not None:
        flav_str = flav_str
    if njet_str is not None:
        njet_str = njet_str[-2:] # Assumes number of n jets comes at the end of the string
        if "j" not in njet_str:
            # The njet string should really have a "j" in it
            raise Exception(f"Something when wrong while trying to consturct channel name, is \"{njet_str}\" an njet string?")

    # Put the component strings into the channel name
    ret_str = nlep_str
    for component in [flav_str,chan_str,njet_str]:
        if component is None: continue
        ret_str = "_".join([ret_str,component])
    return ret_str


ANALYSIS_MODE_EXCLUSIVE_ERROR = (
    "Flags are mutually exclusive. Set at most one of: "
    "--offZ-3l-split, --tau-h-analysis, --fwd-analysis, --all-analysis."
)

JVM_ETA_PHI_DIAGNOSTIC_HISTOGRAMS = frozenset(
    {
        "jet_eta_phi_before_veto",
        "jet_eta_phi_after_veto",
    }
)


def flatten_jagged_jet_eta_phi_weights(jets, event_mask, event_weights):
    """Return aligned flattened eta, phi, and per-jet event weights."""

    selected_jets = jets[event_mask]
    selected_event_weights = event_weights[event_mask]
    _, per_jet_weights = ak.broadcast_arrays(selected_jets.eta, selected_event_weights)
    return (
        ak.flatten(selected_jets.eta),
        ak.flatten(selected_jets.phi),
        ak.flatten(per_jet_weights),
    )


def get_jvm_eta_phi_event_mask(base_event_mask, jet_veto_mask, histogram_name):
    """Apply the diagnostic's family-local event-veto policy."""

    if histogram_name == "jet_eta_phi_before_veto":
        return base_event_mask
    if histogram_name == "jet_eta_phi_after_veto":
        return base_event_mask & jet_veto_mask
    raise ValueError(f"Unknown JVM eta-phi diagnostic '{histogram_name}'")


def should_include_jet_veto_in_histogram_selection(histogram_name):
    """Preserve the standard post-veto selection outside the diagnostic pair."""

    return histogram_name not in JVM_ETA_PHI_DIAGNOSTIC_HISTOGRAMS


def should_fill_jvm_eta_phi_diagnostic(is_run3, syst_var, wgt_fluct):
    """Keep reviewer diagnostics to the Run 3 nominal object/weight state."""

    return is_run3 and syst_var == "nominal" and wgt_fluct == "nominal"


def validate_analysis_mode_flags(offz_3l_split, tau_h_analysis, fwd_analysis, all_analysis):
    mode_flags = {
        "offz_3l_split": bool(offz_3l_split),
        "tau_h_analysis": bool(tau_h_analysis),
        "fwd_analysis": bool(fwd_analysis),
        "all_analysis": bool(all_analysis),
    }
    if sum(mode_flags.values()) > 1:
        raise ValueError(ANALYSIS_MODE_EXCLUSIVE_ERROR)
    return mode_flags


def evaluate_eft_coefficients_at_sm(eft_coefficients):
    """Evaluate quadratic EFT coefficients at the zero-WC SM point."""

    coefficients = np.asarray(eft_coefficients)
    if coefficients.ndim < 1:
        raise ValueError("EFT coefficients must have a coefficient dimension")

    coefficient_count = coefficients.shape[-1]
    wc_count = int(efth.n_wc_from_quad(coefficient_count))
    if int(efth.n_quad_terms(wc_count)) != coefficient_count:
        raise ValueError(
            "EFT coefficient count does not match a quadratic polynomial: "
            f"{coefficient_count}"
        )

    sm_coordinates = np.zeros(wc_count, dtype=coefficients.dtype)
    return efth.calc_eft_weights(coefficients, sm_coordinates)


def calculate_sm_sumw2_weights(scalar_weights, eft_coefficients=None):
    """Return squared complete SM event contributions for a companion fill."""

    scalar_weights = np.asarray(scalar_weights)
    if eft_coefficients is None:
        # Preserve the established non-EFT operation exactly.
        return np.square(scalar_weights)

    eft_factor_sm = evaluate_eft_coefficients_at_sm(eft_coefficients)
    if scalar_weights.shape != eft_factor_sm.shape:
        raise ValueError(
            "Scalar weights and evaluated EFT factors must have matching shapes: "
            f"{scalar_weights.shape} != {eft_factor_sm.shape}"
        )
    return np.square(scalar_weights * eft_factor_sm)


SUPPORTED_EFT_TREATMENTS = frozenset({"sm_only"})


def resolve_eft_treatment(sample_metadata, *, sample_name="<unknown>"):
    """Validate and return an explicitly requested EFT source treatment."""

    treatment = sample_metadata.get("eft_treatment")
    if treatment is None:
        # Preserve the established metadata-derived behavior exactly.
        return None

    if not isinstance(treatment, str) or treatment not in SUPPORTED_EFT_TREATMENTS:
        raise ValueError(
            "EFT-TREATMENT-E001: unsupported eft_treatment "
            f"{treatment!r} for sample {sample_name!r}; "
            f"supported values are {sorted(SUPPORTED_EFT_TREATMENTS)}."
        )
    if sample_metadata.get("isData") is not False:
        raise ValueError(
            "EFT-TREATMENT-E002: explicit EFT treatment is allowed only for MC; "
            f"sample={sample_name!r} treatment={treatment!r}."
        )

    wc_names = sample_metadata.get("WCnames")
    if not isinstance(wc_names, list) or not wc_names:
        raise ValueError(
            "EFT-TREATMENT-E003: sm_only requires a nonempty WCnames list; "
            f"sample={sample_name!r}."
        )
    if any(not isinstance(name, str) or not name for name in wc_names):
        raise ValueError(
            "EFT-TREATMENT-E003: sm_only requires nonempty string WCnames; "
            f"sample={sample_name!r}."
        )
    if len(set(wc_names)) != len(wc_names):
        raise ValueError(
            "EFT-TREATMENT-E003: sm_only requires unique WCnames; "
            f"sample={sample_name!r}."
        )
    return treatment


def project_eft_coefficients_for_treatment(
    eft_coefficients,
    eft_treatment,
    *,
    sample_name="<unknown>",
):
    """Project an explicit EFT source role after native-to-global remapping."""

    if eft_treatment is None:
        return eft_coefficients
    if eft_treatment not in SUPPORTED_EFT_TREATMENTS:
        raise ValueError(
            "EFT-TREATMENT-E001: unsupported runtime eft_treatment "
            f"{eft_treatment!r} for sample {sample_name!r}."
        )
    if eft_coefficients is None:
        raise RuntimeError(
            "EFT-TREATMENT-E004: sm_only requires an EFTfitCoefficients branch; "
            f"sample={sample_name!r}."
        )

    coefficients = np.asarray(eft_coefficients)
    sm_values = evaluate_eft_coefficients_at_sm(coefficients)
    projected = np.zeros_like(coefficients)
    projected[..., 0] = sm_values
    return projected


def prepare_eft_coefficients(
    eft_coefficients,
    native_wc_names,
    global_wc_names,
    eft_treatment,
    *,
    sample_name="<unknown>",
):
    """Remap source coefficients and then apply their explicit treatment."""

    prepared = eft_coefficients
    if prepared is not None and native_wc_names != global_wc_names:
        prepared = efth.remap_coeffs(native_wc_names, global_wc_names, prepared)
    return project_eft_coefficients_for_treatment(
        prepared,
        eft_treatment,
        sample_name=sample_name,
    )


def prepare_event_eft_coefficients(
    events,
    sample_metadata,
    global_wc_names,
    eft_treatment,
    *,
    sample_name="<unknown>",
):
    """Route one source through EFT setup only when its runtime role needs it."""

    eft_coefficients = (
        ak.to_numpy(events["EFTfitCoefficients"])
        if hasattr(events, "EFTfitCoefficients")
        else None
    )
    if eft_coefficients is None and eft_treatment is None:
        return None
    return prepare_eft_coefficients(
        eft_coefficients,
        sample_metadata["WCnames"],
        global_wc_names,
        eft_treatment,
        sample_name=sample_name,
    )


def derive_analysis_enable_toggles(offz_3l_split, tau_h_analysis, fwd_analysis, all_analysis):
    return {
        "enable_offz_blocks": bool(offz_3l_split) or bool(all_analysis),
        "enable_tau_blocks": bool(tau_h_analysis) or bool(all_analysis),
        "enable_fwd_blocks": bool(fwd_analysis) or bool(all_analysis),
    }


def should_apply_fake_tau_sf(tau_run_mode, *, enable_tau_blocks, is_data):
    if not enable_tau_blocks or is_data:
        return False
    if tau_run_mode == "standard":
        return True
    if tau_run_mode == "taufitter":
        return False
    raise ValueError(f"Unknown tau_run_mode '{tau_run_mode}'")


def get_veto_map_input_jets(cleaned_jets, year, is_run3):
    if not is_run3:
        return cleaned_jets

    jet_id_mask = tc_os.run3_nanoV12_ak4puppi_jet_id(
        cleaned_jets,
        year,
        working_point="tight_lepton_veto",
    )
    em_fraction_mask = (cleaned_jets.chEmEF + cleaned_jets.neEmEF) < 0.9
    return cleaned_jets[
        (cleaned_jets.pt > 15.0)
        & jet_id_mask
        & em_fraction_mask
    ]


def resolve_category_dict_names(offz_3l_split, tau_h_analysis, fwd_analysis, all_analysis):
    if all_analysis:
        sr_dict_name = "ALL_CH_LST_SR"
    elif offz_3l_split:
        sr_dict_name = "OFFZ_SPLIT_CH_LST_SR"
    elif tau_h_analysis:
        sr_dict_name = "TAU_CH_LST_SR"
    elif fwd_analysis:
        sr_dict_name = "FWD_CH_LST_SR"
    else:
        sr_dict_name = "TOP22_006_CH_LST_SR"

    cr_dict_name = "TAU_CH_LST_CR" if (tau_h_analysis or all_analysis) else "CH_LST_CR"
    return sr_dict_name, cr_dict_name


def load_category_config(category_config_path=None):
    config_path = (
        topeft_path("channels/ch_lst.json")
        if category_config_path is None
        else category_config_path
    )
    with open(config_path, "r", encoding="utf-8") as ch_json_stream:
        return json.load(ch_json_stream)


class AnalysisProcessor(processor.ProcessorABC):

    @staticmethod
    def _resolve_histogram_names(
        hist_lst,
        *,
        ordered_base_hist_names,
        fill_sumw2_hist,
        selected_sumw2_families=None,
    ):
        """Return ordered collections of requested base and expanded histogram names."""

        available_base_hist_names = set(ordered_base_hist_names)
        sumw2_suffix = "_sumw2"

        if hist_lst is None:
            base_hist_names_ordered = ordered_base_hist_names.copy()
        else:
            base_hist_names_ordered = []
            seen_base_names = set()
            for requested_name in hist_lst:
                base_name = requested_name
                if requested_name.endswith(sumw2_suffix):
                    base_name = requested_name[: -len(sumw2_suffix)]
                if base_name not in available_base_hist_names:
                    raise Exception(
                        f"Error: Cannot specify hist \"{requested_name}\", it is not defined in the processor."
                    )
                if base_name not in seen_base_names:
                    base_hist_names_ordered.append(base_name)
                    seen_base_names.add(base_name)

        expanded_hist_names_ordered = []
        expanded_seen = set()
        for base_name in base_hist_names_ordered:
            if base_name not in expanded_seen:
                expanded_hist_names_ordered.append(base_name)
                expanded_seen.add(base_name)
            if fill_sumw2_hist and (
                selected_sumw2_families is None
                or base_name in selected_sumw2_families
            ):
                sumw2_name = f"{base_name}{sumw2_suffix}"
                if sumw2_name not in expanded_seen:
                    expanded_hist_names_ordered.append(sumw2_name)
                    expanded_seen.add(sumw2_name)

        return base_hist_names_ordered, expanded_hist_names_ordered

    @staticmethod
    def _should_fill_sumw2_histogram(fill_sumw2_hist, *, wgt_fluct):
        """Keep separate *_sumw2 keys, but only fill them for the nominal producer path."""

        return bool(fill_sumw2_hist) and wgt_fluct == "nominal"

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, fill_sumw2_hist=True, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, offZ_split=False, tau_h_analysis=False, fwd_analysis=False, all_analysis=False, useRun3MVA=True, tau_run_mode="standard", sr_category_dict=None, cr_category_dict=None, suppress_forward_eta_stochastic_jer=False, fwd_eta_band_pt_apply="auto", ttgamma_sample_role_policy="split", sumw2_policy=None):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        validated_mode_flags = validate_analysis_mode_flags(
            offZ_split,
            tau_h_analysis,
            fwd_analysis,
            all_analysis,
        )
        self.offZ_3l_split = validated_mode_flags["offz_3l_split"]
        self.tau_h_analysis = validated_mode_flags["tau_h_analysis"]
        self.fwd_analysis = validated_mode_flags["fwd_analysis"]
        self.all_analysis = validated_mode_flags["all_analysis"]
        mode_toggles = derive_analysis_enable_toggles(
            self.offZ_3l_split,
            self.tau_h_analysis,
            self.fwd_analysis,
            self.all_analysis,
        )
        self.enable_offz_blocks = mode_toggles["enable_offz_blocks"]
        self.enable_tau_blocks = mode_toggles["enable_tau_blocks"]
        self.enable_fwd_blocks = mode_toggles["enable_fwd_blocks"]

        if self.all_analysis:
            self._analysis_mode = "all"
        elif self.offZ_3l_split:
            self._analysis_mode = "offz"
        elif self.tau_h_analysis:
            self._analysis_mode = "tau"
        elif self.fwd_analysis:
            self._analysis_mode = "fwd"
        else:
            self._analysis_mode = "default"
        self.sr_category_dict_name, self.cr_category_dict_name = resolve_category_dict_names(
            self.offZ_3l_split,
            self.tau_h_analysis,
            self.fwd_analysis,
            self.all_analysis,
        )
        self.sr_category_dict = (
            copy.deepcopy(sr_category_dict) if sr_category_dict is not None else None
        )
        self.cr_category_dict = (
            copy.deepcopy(cr_category_dict) if cr_category_dict is not None else None
        )

        self.useRun3MVA = useRun3MVA #can be switched to False use the alternative cuts
        self.tau_run_mode = tau_run_mode
        self.suppress_forward_eta_stochastic_jer = suppress_forward_eta_stochastic_jer
        self.fwd_eta_band_pt_apply = fwd_eta_band_pt_apply
        self._ttgamma_sample_role_policy = get_ttgamma_sample_role_policy(
            ttgamma_sample_role_policy
        )
        # self._tau_wp_checked = False

        self._sumw2_policy = sumw2_policy
        self._hist_axis_map = {}
        self._hist_sumw2_axis_mapping = {}
        self._hist_requires_eft = {}
        self.nominal_container_schema_version = NOMINAL_CONTAINER_SCHEMA_VERSION
        self.nominal_container_layout = NOMINAL_CONTAINER_LAYOUT

        ordered_base_hist_names = list(axes_info.keys()) + list(axes_info_2d.keys())
        base_hist_names_ordered, _ = self._resolve_histogram_names(
            hist_lst,
            ordered_base_hist_names=ordered_base_hist_names,
            fill_sumw2_hist=False,
        )
        if self._sumw2_policy is not None:
            if tuple(base_hist_names_ordered) != tuple(
                self._sumw2_policy.runtime_histogram_families
            ):
                raise ValueError(
                    "Resolved sumw2 policy runtime families do not match the processor "
                    "histogram family order."
                )
            selected_sumw2_families = set(
                self._sumw2_policy.selected_families()
            )
        else:
            selected_sumw2_families = (
                set(base_hist_names_ordered) if fill_sumw2_hist else set()
            )
        selected_sumw2_families.difference_update(JVM_ETA_PHI_DIAGNOSTIC_HISTOGRAMS)
        self._selected_sumw2_families = frozenset(selected_sumw2_families)
        self._fill_sumw2_hist = bool(self._selected_sumw2_families)
        (
            base_hist_names_ordered,
            expanded_hist_names_ordered,
        ) = self._resolve_histogram_names(
            hist_lst,
            ordered_base_hist_names=ordered_base_hist_names,
            fill_sumw2_hist=self._fill_sumw2_hist,
            selected_sumw2_families=self._selected_sumw2_families,
        )

        self._base_hist_name_set = set(base_hist_names_ordered)
        self._expanded_hist_name_set = set(expanded_hist_names_ordered)
        self._hist_lst = expanded_hist_names_ordered.copy()

        if samples:
            component_availability = resolve_nominal_component_availability(samples)
        else:
            # Construction-only tests do not have runtime sample metadata. Keep a
            # deterministic scalar fixture and add the EFT sibling only when WCs
            # were explicitly supplied; real runs always resolve from samples.
            component_availability = {
                "scalar": True,
                "eft": bool(wc_names_lst),
            }
        self._nominal_component_availability = component_availability

        sumw2_suffix = "_sumw2"

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)
        syst_axis = hist.axis.StrCategory([], name="systematic", label=r"Systematic Uncertainty", growth=True)
        appl_axis = hist.axis.StrCategory([], name="appl", label=r"AR/SR", growth=True)

        histograms = {}
        def _build_axis(axis_cfg, *, suffix="", label_suffix=""):
            return make_processing_axis(
                axis_cfg,
                name=axis_cfg["name"],
                label=axis_cfg["label"],
                suffix=suffix,
                label_suffix=label_suffix,
            )
        for name, info in axes_info.items():
            sumw2_name = f"{name}{sumw2_suffix}"
            build_base_hist = name in self._base_hist_name_set
            build_sumw2_hist = name in self._selected_sumw2_families
            if not (build_base_hist or build_sumw2_hist):
                continue

            dense_axis = make_processing_axis(
                info, name=name, label=info["label"]
            )
            sumw2_axis = make_processing_axis(
                info,
                name=name,
                label=info["label"],
                suffix="_sumw2",
                label_suffix=" sum of w^2",
            )
            if build_base_hist and component_availability["scalar"]:
                scalar_key = scalar_nominal_key(name)
                histograms[scalar_key] = SparseHist(
                    proc_axis,
                    chan_axis,
                    syst_axis,
                    appl_axis,
                    dense_axis,
                    storage="Double",
                )
                self._hist_axis_map[scalar_key] = [dense_axis.name]
                self._hist_requires_eft[scalar_key] = False
            if build_base_hist and component_availability["eft"]:
                eft_key = eft_nominal_key(name)
                histograms[eft_key] = HistEFT(
                    proc_axis,
                    chan_axis,
                    syst_axis,
                    appl_axis,
                    dense_axis,
                    wc_names=wc_names_lst,
                    label=r"Events",
                )
                self._hist_axis_map[eft_key] = [dense_axis.name]
                self._hist_requires_eft[eft_key] = True
            if build_base_hist:
                self._hist_axis_map[name] = [dense_axis.name]
            if build_sumw2_hist:
                histograms[sumw2_name] = SparseHist(
                    proc_axis,
                    chan_axis,
                    syst_axis,
                    appl_axis,
                    sumw2_axis,
                    storage="Double",
                )
                self._hist_axis_map[sumw2_name] = [sumw2_axis.name]
                self._hist_sumw2_axis_mapping[name] = {sumw2_axis.name: dense_axis.name}
                self._hist_requires_eft[sumw2_name] = False
        for name, axes_cfg in axes_info_2d.items():
            sumw2_name = f"{name}{sumw2_suffix}"
            build_base_hist = name in self._base_hist_name_set
            build_sumw2_hist = name in self._selected_sumw2_families
            if not (build_base_hist or build_sumw2_hist):
                continue

            dense_axes = []
            axis_names = []
            for axis_cfg in axes_cfg["axes"]:
                axis = _build_axis(axis_cfg)
                dense_axes.append(axis)
                axis_names.append(axis.name)
            if build_base_hist:
                histograms[name] = SparseHist(
                    proc_axis,
                    chan_axis,
                    syst_axis,
                    appl_axis,
                    *dense_axes,
                    storage="Double",
                )
                self._hist_axis_map[name] = axis_names
                self._hist_requires_eft[name] = False
            sumw2_axes = []
            sumw2_axis_names = []
            sumw2_axis_mapping = {}
            for axis_cfg, base_axis_name in zip(axes_cfg["axes"], axis_names):
                sumw2_axis = _build_axis(
                    axis_cfg,
                    suffix="_sumw2",
                    label_suffix=" sum of w^2",
                )
                sumw2_axes.append(sumw2_axis)
                sumw2_axis_names.append(sumw2_axis.name)
                sumw2_axis_mapping[sumw2_axis.name] = base_axis_name
            if build_sumw2_hist:
                histograms[sumw2_name] = SparseHist(
                    proc_axis,
                    chan_axis,
                    syst_axis,
                    appl_axis,
                    *sumw2_axes,
                    storage="Double",
                )
                self._hist_axis_map[sumw2_name] = sumw2_axis_names
                self._hist_sumw2_axis_mapping[name] = sumw2_axis_mapping
                self._hist_requires_eft[sumw2_name] = False
        self._accumulator = histograms

        # Ensure the histogram list only tracks objects that actually exist in the
        # accumulator.  Downstream filling logic consults ``self._hist_lst`` to
        # decide whether to touch a given histogram, so stale entries would lead
        # to ``KeyError`` exceptions when the corresponding accumulator key is
        # absent (for example when a filtered ``hist_lst`` omits most
        # histograms).  Restricting the list here keeps the book-keeping
        # consistent with the constructed accumulator contents.
        self._hist_lst = list(self._accumulator.keys())

        # Set the energy threshold to cut on
        self._ecut_threshold = ecut_threshold

        # Set the booleans
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories



    @staticmethod
    def _should_fill_ptz_wtau_channel(lep_chan):
        return (
            (("2lss" in lep_chan) and ("1tau" in lep_chan) and ("onZ" in lep_chan))
            or (lep_chan == "1l_dy_tautau_CR")
        )

    @staticmethod
    def _should_fill_plain_ptz_channel(lep_chan):
        explicit_zll_cr_channels = {
            "2los_CRZ",
            "2lss_CRflip",
            "3l_CR",
        }
        # Diagnostic Z-candidate observable for the SFOS on-Z subset of these
        # selected 2lOS+tau CR events; the categories are not globally on-Z.
        diagnostic_zll_cr_channels = {
            "2los_1tau_Ftau",
            "2los_1tau_Ttau",
            "2los_1tau_0b",
        }
        if lep_chan in explicit_zll_cr_channels:
            return True
        if lep_chan in diagnostic_zll_cr_channels:
            return True
        if ("onZ" in lep_chan) and ("2lss" not in lep_chan):
            return True
        return False

    @staticmethod
    def _should_fill_plain_ptll_channel(lep_chan, allow_offz_split=False):
        return allow_offz_split and lep_chan.startswith(
            (
                "3l_m_offZ_low_",
                "3l_m_offZ_high_",
                "3l_p_offZ_low_",
                "3l_p_offZ_high_",
            )
        )

    def _should_skip_histogram_fill(self, dense_axis_name, ch_name, lep_chan):
        skip_hist = False

        if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & (("CRZ" in ch_name) or ("CRflip" in ch_name))):
            skip_hist = True
        if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & ("0j" in ch_name)):
            skip_hist = True

        # Mode flags are mutually exclusive; mirror the historical loop-local
        # continue/skip behavior by returning a single skip decision.
        if self._analysis_mode == "all":
            if dense_axis_name == "ptz":
                skip_hist = not self._should_fill_plain_ptz_channel(lep_chan)
            if dense_axis_name == "ptll":
                skip_hist = not self._should_fill_plain_ptll_channel(
                    lep_chan, allow_offz_split=True
                )
            # if (("lt" in dense_axis_name) and ("fwd" not in lep_chan)):
            #     skip_hist = True
            if (("ptz_wtau" in dense_axis_name) and not self._should_fill_ptz_wtau_channel(lep_chan)):
                skip_hist = True
        elif self._analysis_mode == "offz":
            if dense_axis_name == "ptz":
                skip_hist = not self._should_fill_plain_ptz_channel(lep_chan)
            if dense_axis_name == "ptll":
                skip_hist = not self._should_fill_plain_ptll_channel(
                    lep_chan, allow_offz_split=True
                )
        elif self._analysis_mode == "tau":
            if dense_axis_name == "ptz":
                skip_hist = not self._should_fill_plain_ptz_channel(lep_chan)
            if dense_axis_name == "ptll":
                skip_hist = True
            if (("ptz_wtau" in dense_axis_name) and not self._should_fill_ptz_wtau_channel(lep_chan)):
                skip_hist = True
        elif self._analysis_mode == "fwd":
            if dense_axis_name == "ptz":
                skip_hist = True
            if dense_axis_name == "ptll":
                skip_hist = True
            # if (("lt" in dense_axis_name) and ("fwd" not in lep_chan)):
            #     skip_hist = True
        else:
            if dense_axis_name == "ptz":
                skip_hist = not self._should_fill_plain_ptz_channel(lep_chan)
            if dense_axis_name == "ptll":
                skip_hist = True

        if ((dense_axis_name in ["o0pt", "b0pt", "bl0pt"]) & ("CR" in ch_name)):
            skip_hist = True

        return skip_hist


    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset_key = events.metadata["dataset"]
        dataset = dataset_key
        sample_metadata = self._samples[dataset_key]
        isEFT   = sample_metadata["WCnames"] != []
        eft_treatment = resolve_eft_treatment(
            sample_metadata,
            sample_name=dataset_key,
        )

        isData             = sample_metadata["isData"]
        histAxisName       = sample_metadata["histAxisName"]
        year               = sample_metadata["year"]
        xsec               = sample_metadata["xsec"]
        sow                = sample_metadata["nSumOfWeights"]
        if isData and isEFT:
            raise ValueError(
                f"Data sample '{dataset_key}' cannot declare WC-dependent content."
            )

        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3
        effective_suppress_forward_eta_stochastic_jer = resolve_forward_eta_stochastic_jer_suppression(
            is_run3,
            self.suppress_forward_eta_stochastic_jer,
        )

        def _log_tau_flag_counts(label, flag_arrays):
            if len(events) == 0:
                return
            counts = []
            for name, array in flag_arrays.items():
                if array is None:
                    continue
                try:
                    mask = ak.fill_none(array > 0, False)
                except TypeError:
                    mask = ak.fill_none(array, False)
                counts.append(f"{name}={int(ak.sum(mask))}")
            if counts:
                print(f"[TauSelectionDebug] {label}: " + ", ".join(counts))

        run_era = None
        if isData:
            if is_run3:
                run_era = self._samples[dataset]["era"]
            else:
                run_era = self._samples[dataset]["path"].split("/")[2].split("-")[0][-1]

        # Get up down weights from input dict
        if (self._do_systematics and not isData):
            if histAxisName in get_te_param("lo_xsec_samples"):
                # We have a LO xsec for these samples, so for these systs we will have e.g. xsec_LO*(N_pass_up/N_gen_nom)
                # Thus these systs will cover the cross section uncty and the acceptance and effeciency and shape
                # So no NLO rate uncty for xsec should be applied in the text data card
                sow_ISRUp          = self._samples[dataset]["nSumOfWeights"]
                sow_ISRDown        = self._samples[dataset]["nSumOfWeights"]
                sow_FSRUp          = self._samples[dataset]["nSumOfWeights"]
                sow_FSRDown        = self._samples[dataset]["nSumOfWeights"]
                sow_renormUp       = self._samples[dataset]["nSumOfWeights"]
                sow_renormDown     = self._samples[dataset]["nSumOfWeights"]
                sow_factUp         = self._samples[dataset]["nSumOfWeights"]
                sow_factDown       = self._samples[dataset]["nSumOfWeights"]
                sow_renormDown_factUp = self._samples[dataset]["nSumOfWeights"]
                sow_renormUp_factDown = self._samples[dataset]["nSumOfWeights"]
            else:
                # Otherwise we have an NLO xsec, so for these systs we will have e.g. xsec_NLO*(N_pass_up/N_gen_up)
                # Thus these systs should only affect acceptance and effeciency and shape
                # The uncty on xsec comes from NLO and is applied as a rate uncty in the text datacard
                sow_ISRUp          = self._samples[dataset]["nSumOfWeights_ISRUp"          ]
                sow_ISRDown        = self._samples[dataset]["nSumOfWeights_ISRDown"        ]
                sow_FSRUp          = self._samples[dataset]["nSumOfWeights_FSRUp"          ]
                sow_FSRDown        = self._samples[dataset]["nSumOfWeights_FSRDown"        ]
                sow_renormUp       = self._samples[dataset]["nSumOfWeights_renormUp"       ]
                sow_renormDown     = self._samples[dataset]["nSumOfWeights_renormDown"     ]
                sow_factUp         = self._samples[dataset]["nSumOfWeights_factUp"         ]
                sow_factDown       = self._samples[dataset]["nSumOfWeights_factDown"       ]
                if is_run3:
                    sow_renormDown_factUp   = self._samples[dataset]["nSumOfWeights_renormDown_factUp"   ]
                    sow_renormUp_factDown = self._samples[dataset]["nSumOfWeights_renormUp_factDown" ]
        else:
            sow_ISRUp          = -1
            sow_ISRDown        = -1
            sow_FSRUp          = -1
            sow_FSRDown        = -1
            sow_renormUp       = -1
            sow_renormDown     = -1
            sow_factUp         = -1
            sow_factDown       = -1
            sow_renormfactDown_factUp   = -1
            sow_renormfactUp_factDown = -1

        datasets = ["Muon", "SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets:
            if dataset.startswith(d):
                dataset = dataset.split('_')[0]

        # Set the sampleType (used for MC matching requirement)
        sampleType = "prompt"
        if isData:
            sampleType = "data"
        elif histAxisName in get_te_param("conv_samples"):
            sampleType = "conversions"
        elif histAxisName in get_te_param("prompt_and_conv_samples"):
            # Just DY (since we care about prompt DY for Z CR, and conv DY for 3l CR)
            sampleType = "prompt_and_conversions"

        # Initialize objects

        met  = get_selected_met(events, year)
        raw_met = get_selected_raw_met(events, year)
        ele  = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet
        pv   = events.PV
        run  = events.run

        if is_run3:
            AttachElectronCorrections(ele, run, year, isData) #need to apply electron energy corrections before calculating conept
            jetsRho = events.Rho["fixedGridRhoFastjetAll"]
            btagAlgo = "btagDeepFlavB" #DeepJet branch
            #btagAlgo = "btagPNetB"    #PNet branch
            leptonSelection = te_os.run3leptonselection(useMVA=self.useRun3MVA, btagger=btagAlgo)
            tauSelection = te_os.run3TauSelection()
        elif is_run2:
            jetsRho = events.fixedGridRhoFastjetAll
            btagAlgo = "btagDeepFlavB"
            leptonSelection = te_os.run2leptonselection(btagger=btagAlgo)
            tauSelection = te_os.run2TauSelection()
        if not btagAlgo in ["btagDeepFlavB", "btagPNetB"]:
            raise ValueError("b-tagging algorithm not recognized!")
        
        te_os.lepJetBTagAdder(ele, btagger=btagAlgo)
        te_os.lepJetBTagAdder(mu, btagger=btagAlgo)

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(met.pt)

        ele["idEmu"] = te_os.ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        mu["pt_raw"] = mu.pt
        mu = AttachMuonMomentumCorrections(
            year,
            mu,
            isData,
            event_numbers=events.event,
            luminosity_blocks=events.luminosityBlock,
        )
        if is_run2:
            ele["pt_raw"] = ele.pt

        if not isData:
            ele["gen_pdgId"] = ak.fill_none(ele.matched_gen.pdgId, 0)
            mu["gen_pdgId"] = ak.fill_none(mu.matched_gen.pdgId, 0)
            ele["genParent_pdgId"] = ak.fill_none(ele.matched_gen.distinctParent.pdgId, 0)
            mu["genParent_pdgId"] = ak.fill_none(mu.matched_gen.distinctParent.pdgId, 0)

        # Get the lumi mask for data
        if year == "2016" or year == "2016APV":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        elif year == "2017":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        elif year == "2018":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        elif year.startswith("2022"):
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_Collisions2022_355100_362760_Golden.txt")
        elif year.startswith("2023"):
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_Collisions2023_366442_370790_Golden.txt")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

        ######### EFT coefficients ##########

        # Extract and prepare EFT coefficients only for sources that carry them
        # or explicitly require the sm_only runtime branch validation.
        eft_coeffs = prepare_event_eft_coefficients(
            events,
            sample_metadata,
            self._wc_names_lst,
            eft_treatment,
            sample_name=dataset_key,
        )
        # Initialize the out object
        hout = self.accumulator

        if self.enable_tau_blocks:
            tau_fo_tag = get_te_param(
                "run2_tau_fo_tag" if is_run2 else "run3_tau_fo_tag"
            )
            tau_T_tag = get_te_param(
                "run2_tau_t_tag" if is_run2 else "run3_tau_t_tag"
            )
            taus = AttachTauEnergyCorrections(
                year, tau, isData, vsJetWP=tau_T_tag
            )
        apply_fake_tau_sf = should_apply_fake_tau_sf(
            self.tau_run_mode,
            enable_tau_blocks=self.enable_tau_blocks,
            is_data=isData,
        )

        # Define the lists of systematics we include
        obj_correction_syst_lst = get_supported_jet_systematics(
            year, isData=isData, era=run_era
        )
        obj_correction_syst_lst.extend(
            get_supported_met_systematics(year, isData=isData, era=run_era)
        )
        obj_correction_syst_lst.extend(
            get_supported_muon_momentum_systematics(year, isData=isData)
        )
        if self.enable_tau_blocks:
            obj_correction_syst_lst.extend(
                get_supported_tau_energy_systematics(year, isData=isData)
            )

        wgt_correction_syst_lst = [
            "lepSF_muonUp","lepSF_muonDown","lepSF_elecUp","lepSF_elecDown",f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown","PreFiringUp","PreFiringDown",f"triggerSF_{year}Up",f"triggerSF_{year}Down", # Exp systs
            "FSRUp","FSRDown","ISRUp","ISRDown","renormUp","renormDown","factUp","factDown", # Theory systs
        ]
        if self.enable_tau_blocks:
            wgt_correction_syst_lst.append("lepSF_taus_realUp")
            wgt_correction_syst_lst.append("lepSF_taus_realDown")
        if apply_fake_tau_sf:
            wgt_correction_syst_lst.append("lepSF_taus_fakeUp")
            wgt_correction_syst_lst.append("lepSF_taus_fakeDown")

        data_syst_lst = [
            "FFUp","FFDown","FFptUp","FFptDown","FFetaUp","FFetaDown",f"FFcloseEl_{year}Up",f"FFcloseEl_{year}Down",f"FFcloseMu_{year}Up",f"FFcloseMu_{year}Down"
        ]

        # print("\n\n\n\n\n\n")
        # print((f"Systematic variations to be applied for objects kinematics: {obj_correction_syst_lst}"))
        # print((f"Systematic variations to be applied for event weights: {wgt_correction_syst_lst}"))
        # print((f"Systematic variations to be applied only for data-driven estimates: {data_syst_lst}"))
        # print("\n\n\n\n\n\n")

        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        if not isData:
            # If this is no an eft sample, get the genWeight
            if eft_coeffs is None:
                genw = events["genWeight"]
            else:
                genw= np.ones_like(events["event"])

            # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
            # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
            lumi = 1000.0*get_tc_param(f"lumi_{year}")
            weights_obj_base.add("norm",(xsec/sow)*genw*lumi)

            if is_run2:
                l1prefiring_args = [events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn]
            elif is_run3:
                l1prefiring_args = [ak.ones_like(events.nom), ak.ones_like(events.nom), ak.ones_like(events.nom)]

            # Attach PS weights (ISR/FSR) and scale weights (renormalization/factorization) and PDF weights
            tc_cor.AttachPSWeights(events) #Run3 ready
            tc_cor.AttachScaleWeights(events) #Run3 ready (with caveat on "nominal")
            #AttachPdfWeights(events) #TODO
            # FSR/ISR weights -- corrections come from AttachPSWeights
            weights_obj_base.add('ISR', events.nom, events.ISRUp*(sow/sow_ISRUp), events.ISRDown*(sow/sow_ISRDown))
            weights_obj_base.add('FSR', events.nom, events.FSRUp*(sow/sow_FSRUp), events.FSRDown*(sow/sow_FSRDown))
            # renorm/fact scale  -- corrections come from AttachScaleWeights
            weights_obj_base.add('renorm', events.nom, events.renormUp*(sow/sow_renormUp), events.renormDown*(sow/sow_renormDown))
            weights_obj_base.add('fact', events.nom, events.factUp*(sow/sow_factUp), events.factDown*(sow/sow_factDown))
            # Prefiring and PU (note prefire weights only available in nanoAODv9 and for Run2)
            weights_obj_base.add('PreFiring', *l1prefiring_args) #Run3 ready
            weights_obj_base.add('PU', tc_cor.GetPUSF((events.Pileup.nTrueInt), year), tc_cor.GetPUSF(events.Pileup.nTrueInt, year, 'up'), tc_cor.GetPUSF(events.Pileup.nTrueInt, year, 'down')) #Run3 ready


        ######### The rest of the processor is inside this loop over systs that affect object kinematics  ###########

        # If we're doing systematics and this isn't data, we will loop over the obj_correction_syst_lst list
        if self._do_systematics and not isData: syst_var_list = ["nominal"] + obj_correction_syst_lst
        # Otherwise loop juse once, for nominal
        else: syst_var_list = ['nominal']

        # Build the Type-1 MET correction from the full NanoAOD Jet
        # collection. The analysis-cleaned jet path below remains separate.
        events_cache = events.caches[0]
        type1_met = None
        if use_type1_met(year):
            type1Jets = jets
            type1Jets = ak.with_field(type1Jets, (1 - type1Jets.rawFactor)*type1Jets.pt, "pt_raw")
            type1Jets = ak.with_field(type1Jets, (1 - type1Jets.rawFactor)*type1Jets.mass, "mass_raw")
            type1Jets = ak.with_field(type1Jets, ak.broadcast_arrays(jetsRho, type1Jets.pt)[0], "rho")
            if not isData:
                type1Jets = ak.with_field(
                    type1Jets,
                    ak.values_astype(ak.fill_none(type1Jets.matched_gen.pt, 0), np.float32),
                    "pt_gen",
                )

            corrT1METJets = get_corr_t1_met_jets(events, year)
            # CorrT1METJet has no per-object rho branch. Broadcast the event-level rho
            # used for Jet JECs to the CorrT1METJet jagged structure because the
            # correctionlib L1/full JEC evaluators can require Rho, e.g. L1FastJet.
            corrT1METJets = ak.with_field(
                corrT1METJets,
                ak.broadcast_arrays(jetsRho, corrT1METJets.rawPt)[0],
                "rho",
            )
            type1_met = ApplyJetCorrections(
                year,
                corr_type='type1_met',
                isData=isData,
                era=run_era,
                run=run,
                suppress_forward_eta_stochastic_jer=effective_suppress_forward_eta_stochastic_jer,
            ).build(
                met,
                raw_met,
                type1Jets,
                corrT1METJets,
                lazy_cache=events_cache,
            )
            del type1Jets
            del corrT1METJets

        # Loop over the list of systematic variations we've constructed
        met_raw=met
        
        for syst_var in syst_var_list:
            # Make a copy of the base weights object, so that each time through the loop we do not double count systs
            # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics
            weights_obj_base_for_kinematic_syst = copy.deepcopy(weights_obj_base)

            #################### Leptons ####################

            mu = ApplyMuonMomentumSystematics(year, mu, syst_var)
            mu["conept"] = leptonSelection.coneptMuon(mu)
            mu["isPres"] = leptonSelection.isPresMuon(mu)
            mu["isLooseM"] = leptonSelection.isLooseMuon(mu)
            mu["isFO"] = leptonSelection.isFOMuon(mu, year)
            mu["isTightLep"] = leptonSelection.tightSelMuon(mu)

            ele["conept"] = leptonSelection.coneptElec(ele)
            ele["isPres"] = leptonSelection.isPresElec(ele)
            ele["isLooseE"] = leptonSelection.isLooseElec(ele)
            ele["isFO"] = leptonSelection.isFOElec(ele, year)
            ele["isTightLep"] = leptonSelection.tightSelElec(ele)

            m_loose = mu[mu.isPres & mu.isLooseM]
            e_loose = ele[ele.isPres & ele.isLooseE]
            l_loose = ak.with_name(
                ak.concatenate([e_loose, m_loose], axis=1),
                "PtEtaPhiMCandidate",
            )
            llpairs = ak.combinations(l_loose, 2, fields=["l0", "l1"])
            min_mll_afas = ak.min((llpairs.l0 + llpairs.l1).mass, axis=-1)

            m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
            e_fo = ele[ele.isPres & ele.isLooseE & ele.isFO]
            if "seediEtaOriX" not in ak.fields(e_fo):
                e_fo["seediEtaOriX"] = ak.zeros_like(e_fo.pt)
            if "seediPhiOriY" not in ak.fields(e_fo):
                e_fo["seediPhiOriY"] = ak.zeros_like(e_fo.pt)

            AttachElectronSF(
                e_fo,
                year=year,
                looseWP="none" if is_run3 else "wpLnoiso",
                useRun3MVA=self.useRun3MVA,
            )
            AttachMuonSF(m_fo, year=year, useRun3MVA=self.useRun3MVA)
            AttachPerLeptonFR(e_fo, flavor="Elec", year=year)
            AttachPerLeptonFR(m_fo, flavor="Muon", year=year)
            m_fo["convVeto"] = ak.ones_like(m_fo.charge)
            m_fo["lostHits"] = ak.zeros_like(m_fo.charge)
            m_fo["seediEtaOriX"] = ak.zeros_like(m_fo.charge)
            m_fo["seediPhiOriY"] = ak.zeros_like(m_fo.charge)
            l_fo = ak.with_name(
                ak.concatenate([e_fo, m_fo], axis=1),
                "PtEtaPhiMCandidate",
            )
            l_fo_conept_sorted = l_fo[
                ak.argsort(l_fo.conept, axis=-1, ascending=False)
            ]
            l_fo_conept_sorted = attach_photon_history_diagnostics(
                events,
                l_fo_conept_sorted,
                None if isData else events.GenPart,
            )
            attach_conversion_overlap_removal_diagnostics(
                events,
                sample_name=events.metadata["dataset"],
                is_data=isData,
                sample_role_policy=self._ttgamma_sample_role_policy,
            )

            #################### Taus ####################

            if self.enable_tau_blocks:
                tau = ApplyTauEnergySystematics(taus, syst_var)

                if is_run2:
                    vs_jet = tau.idDeepTau2017v2p1VSjet
                    vs_e = tau.idDeepTau2017v2p1VSe
                    vs_mu = tau.idDeepTau2017v2p1VSmu
                else:
                    vs_jet = tau.idDeepTau2018v2p5VSjet
                    vs_e = tau.idDeepTau2018v2p5VSe
                    vs_mu = tau.idDeepTau2018v2p5VSmu

                tau["isVLoose"] = tauSelection.isVLooseTau(vs_jet)
                tau["isLoose"] = tauSelection.isLooseTau(vs_jet)
                tau["isMedium"] = tauSelection.isMediumTau(vs_jet)
                tau["iseTight"] = tauSelection.iseTightTau(vs_e)
                tau["ismTight"] = ak.values_astype(
                    tauSelection.ismTightTau(vs_mu), np.int8
                )
                tau["isPresVLoose"] = tauSelection.isPresTau(
                    tau.pt,
                    tau.eta,
                    tau.dxy,
                    tau.dz,
                    vs_jet,
                    vs_e,
                    vs_mu,
                    minpt=20,
                    vsJetWP="VLoose",
                )
                tau["isPresLoose"] = tauSelection.isPresTau(
                    tau.pt,
                    tau.eta,
                    tau.dxy,
                    tau.dz,
                    vs_jet,
                    vs_e,
                    vs_mu,
                    minpt=20,
                    vsJetWP="Loose",
                )
                tau["isPresMedium"] = tauSelection.isPresTau(
                    tau.pt,
                    tau.eta,
                    tau.dxy,
                    tau.dz,
                    vs_jet,
                    vs_e,
                    vs_mu,
                    minpt=20,
                    vsJetWP="Medium",
                )
                tau["isPres"] = tau[f"isPres{tau_fo_tag}"]
                tau["isClean"] = te_os.isClean(tau, l_fo, drmin=0.5)
                tau["isGood"] = tau["isClean"] & tau["isPres"]
                tau = tau[tau.isGood]
                tau["DMflag"] = (
                    (tau.decayMode == 0)
                    | (tau.decayMode == 1)
                    | (tau.decayMode == 10)
                    | (tau.decayMode == 11)
                )
                tau = tau[tau.DMflag]

                tau_fo = tau
                tau_fo_padded = ak.pad_none(tau_fo, 1)
                tau0_fo = tau_fo_padded[:, 0]
                tau_T = tau_fo[tau_fo[f"is{tau_T_tag}"] > 0]
                tau_T_padded = ak.pad_none(tau_T, 1)
                tau0_T = tau_T_padded[:, 0]
                cleaning_taus = tau_T
                nLtau = ak.num(tau_T)

                if self.tau_run_mode == "standard":
                    tau_F_mask = ak.num(tau_fo) == 1
                    tau_L_mask = nLtau == 1
                elif self.tau_run_mode == "taufitter":
                    tau_F_mask = ak.num(tau_fo) >= 1
                    tau_L_mask = nLtau >= 1
                else:
                    raise ValueError(
                        f"Unknown tau_run_mode '{self.tau_run_mode}'"
                    )
                no_tau_mask = nLtau == 0
                tau0 = tau0_T

                if not isData:
                    AttachTauSF(
                        events,
                        tau_T,
                        year=year,
                        vsJetWP=tau_T_tag,
                    )

            #################### Jets ####################

            # Jet cleaning, before any jet selection
            vetos_tocleanjets = ak.with_name(l_fo, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index
            
            if self.enable_tau_blocks:
                cleanedJets["isTauClean"] = te_os.isClean(cleanedJets, cleaning_taus, drmin=0.5)
                cleanedJets = cleanedJets[cleanedJets.isTauClean]

            # Selecting jets and cleaning them
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
            cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
            cleanedJets["rho"] = ak.broadcast_arrays(jetsRho, cleanedJets.pt)[0]

            # Jet energy corrections
            if not isData:
                cleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)

            cleanedJets = ApplyJetCorrections(
                year,
                corr_type='jets',
                isData=isData,
                era=run_era,
                run=run,
                suppress_forward_eta_stochastic_jer=effective_suppress_forward_eta_stochastic_jer,
            ).build(cleanedJets, lazy_cache=events_cache)  #Run3 ready
            cleanedJets = ApplyJetSystematics(year,cleanedJets,syst_var)

            # Jet Veto Maps
            # Removes events that have ANY jet in a specific eta-phi space (not required for Run 2)
            # Zero is passing the veto map, so Run 2 will be assigned an array of length events with all zeros
            veto_map_input_jets = get_veto_map_input_jets(cleanedJets, year, is_run3)
            veto_map_array = ApplyJetVetoMaps(veto_map_input_jets, year) if is_run3 else ak.zeros_like(met.pt)
            veto_map_mask = (veto_map_array == 0)

            if use_type1_met(year):
                met = ApplyMETSystematics(type1_met, syst_var)
            else:
                met = ApplyJetCorrections(year, corr_type='met', isData=isData, era=run_era, run=run).build(met_raw, cleanedJets, lazy_cache=events_cache)
                if is_met_unclustered_systematic(syst_var):
                    met = ApplyMETSystematics(met, syst_var)

            if is_run3:
                jet_id_mask = tc_os.run3_nanoV12_ak4puppi_jet_id(cleanedJets, year, working_point="tight")
                cleanedJets["isGood"] = ((getattr(cleanedJets, jetptname) > 30.) & (abs(cleanedJets.eta) < get_te_param("eta_j_cut")) & jet_id_mask)
            else:
                jet_id_mask = True
                cleanedJets["isGood"] = tc_os.is_tight_jet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=get_te_param("jet_id_cut"))
            cleanedJets["isFwd"] = te_os.is_forward_jet_eta_banded(
                getattr(cleanedJets, jetptname),
                cleanedJets.eta,
                eta_cut=get_te_param("eta_j_cut"),
                baseline_pt_cut=get_te_param("fwd_jet_pt_cut"),
                apply_eta_band_pt=te_os.resolve_fwd_eta_band_pt_apply(is_run3, self.fwd_eta_band_pt_apply),
                eta_band_min=get_te_param("fwd_jet_eta_band_min"),
                eta_band_max=get_te_param("fwd_jet_eta_band_max"),
                eta_band_pt_cut=get_te_param("fwd_jet_eta_band_pt_cut"),
                quality_mask=jet_id_mask,
            )
            goodJets = cleanedJets[cleanedJets.isGood]
            fwdJets  = cleanedJets[cleanedJets.isFwd]

            # Count jets
            njets = ak.num(goodJets)
            nfwdj = ak.num(fwdJets)
            ht = ak.sum(goodJets.pt,axis=-1)
            j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]
            fwd0 = fwdJets[ak.argmax(fwdJets.pt,axis=-1,keepdims=True)]

            if btagAlgo == "btagDeepFlavB":
                btagRef = ""
            elif btagAlgo == "btagPNetB":
                btagRef = "PNet_"

            # Loose DeepJet WP
            loose_tag = "btag_wp_loose_" + btagRef + year.replace("201", "UL1")
            btagwpl = get_tc_param(loose_tag)
            isBtagJetsLoose = (goodJets[btagAlgo] > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])
            # Medium DeepJet WP
            medium_tag = "btag_wp_medium_" + btagRef + year.replace("201", "UL1")
            btagwpm = get_tc_param(medium_tag)
            isBtagJetsMedium = (goodJets[btagAlgo] > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])
            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
            events["minMllAFAS"] = min_mll_afas
            events["l_fo_conept_sorted"] = l_fo_conept_sorted

            # The event selection
            te_es.add1lMaskAndSFs(events, year, isData, sampleType)
            te_es.add2lMaskAndSFs(events, year, isData, sampleType)
            te_es.add3lMaskAndSFs(events, year, isData, sampleType)
            te_es.add4lMaskAndSFs(events, year, isData)
            te_es.addLepCatMasks(events)

            # Convenient to have l0, l1, l2 on hand
            l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
            l0 = l_fo_conept_sorted_padded[:,0]
            l1 = l_fo_conept_sorted_padded[:,1]
            l2 = l_fo_conept_sorted_padded[:,2]


            ######### Event weights that do not depend on the lep cat ##########

            if not isData:
                # Workaround to use UL16APV SFs for UL16 for light jets
                if year == "2016":
                    year_light = "2016APV"
                else:
                    year_light = year

                isBtagJetsLooseNotMedium = (isBtagJetsLoose & isNotBtagJetsMedium)

                light_mask = goodJets.hadronFlavour==0
                bc_mask = goodJets.hadronFlavour>0

                jets_light = goodJets[light_mask]
                jets_bc    = goodJets[bc_mask]

                if is_run2:
                    btagName = "deepJet"
                    suffix_bc = "comb"
                    suffix_light = "incl"
                    btag_method_bc    = f"{btagName}_{suffix_bc}"
                    btag_method_light = f"{btagName}_{suffix_light}"
                elif is_run3:
                    if btagAlgo == "btagDeepFlavB":
                        btagName = "deepJet"
                        suffix_bc = "comb"
                        suffix_light = "light"
                    elif btagAlgo == "btagPNetB":
                        btagName = "particleNet"
                        if year.startswith("2023"):
                            suffix_bc = "tnp"
                            suffix_light = "light"
                        else:
                            suffix_bc = "comb"
                            suffix_light = "light"
                        
                    btag_method_bc    = f"{btagName}_{suffix_bc}"
                    btag_method_light = f"{btagName}_{suffix_light}"
                btag_effM_light = GetBtagEff(jets_light, year, 'medium', btagAlgo)
                btag_effM_bc = GetBtagEff(jets_bc, year, 'medium', btagAlgo)
                print('\n\n\n\n')
                print("btag_effM_light", btag_effM_light)
                print("btag_effM_bc",btag_effM_bc )
                print('\n\n\n\n')
                btag_effL_light = GetBtagEff(jets_light, year, 'loose', btagAlgo)
                btag_effL_bc = GetBtagEff(jets_bc, year, 'loose', btagAlgo)
                btag_sfM_light = tc_cor.btag_sf_eval(jets_light, "M", year_light, btag_method_light, "central")
                btag_sfM_bc    = tc_cor.btag_sf_eval(jets_bc,    "M", year, btag_method_bc, "central")
                btag_sfL_light = tc_cor.btag_sf_eval(jets_light, "L", year_light, btag_method_light, "central")
                btag_sfL_bc    = tc_cor.btag_sf_eval(jets_bc,    "L", year, btag_method_bc, "central")

                pData_light, pMC_light = tc_cor.get_method1a_wgt_doublewp(btag_effM_light, btag_effL_light, btag_sfM_light, btag_sfL_light, isBtagJetsMedium[light_mask], isBtagJetsLooseNotMedium[light_mask], isNotBtagJetsLoose[light_mask])
                btag_w_light = pData_light/pMC_light
                pData_bc, pMC_bc = tc_cor.get_method1a_wgt_doublewp(btag_effM_bc, btag_effL_bc, btag_sfM_bc, btag_sfL_bc, isBtagJetsMedium[bc_mask], isBtagJetsLooseNotMedium[bc_mask], isNotBtagJetsLoose[bc_mask])
                btag_w_bc = pData_bc/pMC_bc
                btag_w = btag_w_light*btag_w_bc

                weights_obj_base_for_kinematic_syst.add("btagSF", btag_w)

                if self._do_systematics and syst_var=='nominal':
                    for b_syst in ["bc_corr","light_corr",f"bc_{year}",f"light_{year}"]:
                        if b_syst.endswith("_corr"):
                            corrtype = "correlated"
                        else:
                            corrtype = "uncorrelated"

                        if b_syst.startswith("light_"):
                            jets_flav = jets_light
                            flav_mask = light_mask
                            sys_year = year_light
                            if is_run2:
                                dJ_tag = "incl"
                            if is_run3:
                                dJ_tag = "light"
                            btag_effM = btag_effM_light
                            btag_effL = btag_effL_light
                            pMC_flav = pMC_light
                            fixed_btag_w = btag_w_bc
                        elif b_syst.startswith("bc_"):
                            jets_flav = jets_bc
                            flav_mask = bc_mask
                            sys_year = year
                            dJ_tag = "comb"
                            btag_effM = btag_effM_bc
                            btag_effL = btag_effL_bc
                            pMC_flav = pMC_bc
                            fixed_btag_w = btag_w_light
                        else:
                            raise ValueError("btag systematics should be divided in flavor (bc or light)!")

                        btag_sfL_up   = tc_cor.btag_sf_eval(jets_flav, "L", sys_year, f"{btagName}_{dJ_tag}", f"up_{corrtype}")
                        btag_sfL_down = tc_cor.btag_sf_eval(jets_flav, "L", sys_year, f"{btagName}_{dJ_tag}", f"down_{corrtype}")
                        btag_sfM_up   = tc_cor.btag_sf_eval(jets_flav, "M", sys_year, f"{btagName}_{dJ_tag}", f"up_{corrtype}")
                        btag_sfM_down = tc_cor.btag_sf_eval(jets_flav, "M", sys_year, f"{btagName}_{dJ_tag}", f"down_{corrtype}")

                        pData_up, pMC_up = tc_cor.get_method1a_wgt_doublewp(btag_effM, btag_effL, btag_sfM_up, btag_sfL_up, isBtagJetsMedium[flav_mask], isBtagJetsLooseNotMedium[flav_mask], isNotBtagJetsLoose[flav_mask])
                        pData_down, pMC_down = tc_cor.get_method1a_wgt_doublewp(btag_effM, btag_effL, btag_sfM_down, btag_sfL_down, isBtagJetsMedium[flav_mask], isBtagJetsLooseNotMedium[flav_mask], isNotBtagJetsLoose[flav_mask])

                        btag_w_up = (pData_up/pMC_flav)
                        btag_w_down = (pData_down/pMC_flav)

                        btag_w_up = fixed_btag_w*btag_w_up/btag_w
                        btag_w_down = fixed_btag_w*btag_w_down/btag_w

                        weights_obj_base_for_kinematic_syst.add(f"btagSF{b_syst}", events.nom, btag_w_up, btag_w_down)

                # Trigger SFs
                GetTriggerSF(year,events,l0,l1)
                weights_obj_base_for_kinematic_syst.add(f"triggerSF_{year}", events.trigger_sf, copy.deepcopy(events.trigger_sfUp), copy.deepcopy(events.trigger_sfDown))            # In principle does not have to be in the lep cat loop

            ######### Event weights that do depend on the lep cat ###########
            select_cat_dict = None
            needs_category_config = (
                (not self._skip_signal_regions and self.sr_category_dict is None)
                or (not self._skip_control_regions and self.cr_category_dict is None)
            )
            if needs_category_config:
                select_cat_dict = load_category_config()

            if not self._skip_signal_regions:
                # If we are not skipping the signal regions, we will import the SR categories
                # This dictionary keeps track of which selections go with which SR categories
                if self.sr_category_dict is not None:
                    import_sr_cat_dict = self.sr_category_dict
                else:
                    import_sr_cat_dict = select_cat_dict[self.sr_category_dict_name]

            if not self._skip_control_regions:
                # If we are not skipping the control regions, we will import the CR categories
                # This dictionary keeps track of which selections go with which CR categories
                if self.cr_category_dict is not None:
                    import_cr_cat_dict = self.cr_category_dict
                else:
                    import_cr_cat_dict = select_cat_dict[self.cr_category_dict_name]
 

            #This list keeps track of the lepton categories
            lep_cats = []
            if not self._skip_signal_regions:
                lep_cats += list(import_sr_cat_dict.keys())
            if not self._skip_control_regions:
                lep_cats += list(import_cr_cat_dict.keys())
            
            # Add the 2l_4t category 

            lep_cats += ["2l_4t"]
            lep_cats_data = [lep_cat for lep_cat in lep_cats if (lep_cat.startswith("2l") and not "os" in lep_cat)]

            weights_dict = {}

            for ch_name in lep_cats:
                # For both data and MC
                weights_dict[ch_name] = copy.deepcopy(weights_obj_base_for_kinematic_syst)
                if ch_name.startswith("1l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_1l, copy.deepcopy(events.fakefactor_1l_up), copy.deepcopy(events.fakefactor_1l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_1l_pt1/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_pt2/events.fakefactor_1l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_1l_be1/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_be2/events.fakefactor_1l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_1l_elclosureup/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_elclosuredown/events.fakefactor_1l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_1l_muclosureup/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_muclosuredown/events.fakefactor_1l))
                if ch_name.startswith("2l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_2l, copy.deepcopy(events.fakefactor_2l_up), copy.deepcopy(events.fakefactor_2l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_2l_pt1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_pt2/events.fakefactor_2l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_2l_be1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_be2/events.fakefactor_2l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_elclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_elclosuredown/events.fakefactor_2l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_muclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_muclosuredown/events.fakefactor_2l))
                elif ch_name.startswith("3l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_3l, copy.deepcopy(events.fakefactor_3l_up), copy.deepcopy(events.fakefactor_3l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_3l_pt1/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_pt2/events.fakefactor_3l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_3l_be1/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_be2/events.fakefactor_3l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_3l_elclosureup/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_elclosuredown/events.fakefactor_3l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_3l_muclosureup/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_muclosuredown/events.fakefactor_3l))

                # For data only
                if isData:
                    if ch_name in lep_cats_data:
                        weights_dict[ch_name].add("fliprate", events.flipfactor_2l)

                # For MC only
                if not isData:
                    if ch_name.startswith("1l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_1l_muon, copy.deepcopy(events.sf_1l_hi_muon), copy.deepcopy(events.sf_1l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_1l_elec, copy.deepcopy(events.sf_1l_hi_elec), copy.deepcopy(events.sf_1l_lo_elec))
                        if self.enable_tau_blocks:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                        if apply_fake_tau_sf:
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("2l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_2l_muon, copy.deepcopy(events.sf_2l_hi_muon), copy.deepcopy(events.sf_2l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_2l_elec, copy.deepcopy(events.sf_2l_hi_elec), copy.deepcopy(events.sf_2l_lo_elec))
                        if self.enable_tau_blocks:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                        if apply_fake_tau_sf:
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("3l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_3l_muon, copy.deepcopy(events.sf_3l_hi_muon), copy.deepcopy(events.sf_3l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_3l_elec, copy.deepcopy(events.sf_3l_hi_elec), copy.deepcopy(events.sf_3l_lo_elec))
                        if self.enable_tau_blocks:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                        if apply_fake_tau_sf:
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("4l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_4l_muon, copy.deepcopy(events.sf_4l_hi_muon), copy.deepcopy(events.sf_4l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_4l_elec, copy.deepcopy(events.sf_4l_hi_elec), copy.deepcopy(events.sf_4l_lo_elec))
                    else:
                        raise Exception(f"Unknown channel name: {ch_name}")

            ######### Masks we need for the selection ##########

            # Get mask for events that have two sf os leps close to z peak
            sfosz_3l_OnZ_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:3],pt_window=10.0)
            sfosz_3l_OffZ_mask = ~sfosz_3l_OnZ_mask
            if self.enable_offz_blocks:
                sfosz_3l_OffZ_low_mask = tc_es.get_off_Z_mask_low(l_fo_conept_sorted_padded[:,0:3],pt_window=0.0)
                sfosz_3l_OffZ_any_mask = tc_es.get_any_sfos_pair(l_fo_conept_sorted_padded[:,0:3])
            sfosz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=10.0)
            sfasz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="as") # Any sign (do not enforce ss or os here)
            if self.enable_tau_blocks:
                tl_zpeak_mask = te_es.lt_Z_mask(l0, l1, tau0)

            # Pass trigger mask
            pass_trg = tc_es.trg_pass_no_overlap(events,isData,dataset,str(year),te_es.dataset_dict_top22006,te_es.exclude_dict_top22006,run_era)

            # b jet masks
            bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
            bmask_exactly0med = (nbtagsm==0) # Used for 3l CR and 2los Z CR
            bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
            bmask_exactly2med = (nbtagsm==2) # Used for CRtt
            bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR
            bmask_atmost2med  = (nbtagsm< 3) # Used to make 2lss mutually exclusive from tttt enriched
            bmask_atleast3med = (nbtagsm>=3) # Used for tttt enriched
            fwdjet_mask       = (nfwdj > 0)  # Used for ttW EWK enriched regions

            # Charge masks
            chargel0_p = ak.fill_none(((l0.charge)>0),False)
            chargel0_m = ak.fill_none(((l0.charge)<0),False)
            charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)
            charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)
            charge3l_p = ak.fill_none(((l0.charge+l1.charge+l2.charge)>0),False)
            charge3l_m = ak.fill_none(((l0.charge+l1.charge+l2.charge)<0),False)

            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint64')
            preselections = PackedSelection(dtype='uint64')
            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)
            preselections.add("is_good_lumi",lumi_mask)

            # Jet veto mask (for Run 3)
            selections.add("jet_veto", veto_map_mask)
            preselections.add("jet_veto", veto_map_mask)

            # 2lss selection
            preselections.add("chargedl0", (chargel0_p | chargel0_m))
            preselections.add("2l_nozeeveto", (events.is2l_nozeeveto & pass_trg))
            preselections.add("2los", charge2l_0)
            preselections.add("2lem", events.is_em)
            preselections.add("2lee", events.is_ee)
            preselections.add("2lmm", events.is_mm)
            preselections.add("2l_onZ_as", sfasz_2l_mask)
            preselections.add("2l_onZ", sfosz_2l_mask)
            preselections.add("bmask_atleast3m", (bmask_atleast3med))
            preselections.add("bmask_atleast1m2l", (bmask_atleast1med_atleast2loose))
            preselections.add("bmask_atmost2m", (bmask_atmost2med))
            preselections.add("fwdjet_mask", (fwdjet_mask))
            preselections.add("~fwdjet_mask", (~fwdjet_mask))
            if self.enable_tau_blocks:
                preselections.add("1l", (events.is1l & pass_trg))
                preselections.add("1tau", (tau_L_mask))
                preselections.add("1Ftau", (tau_F_mask))
                preselections.add("0tau", (no_tau_mask))
                preselections.add("onZ_tau", (tl_zpeak_mask))
                preselections.add("offZ_tau", (~tl_zpeak_mask))
                lt_os_mask = ak.fill_none((l0.charge * tau0.charge) < 0, False)
                lt_vis_mass = ak.fill_none((l0 + tau0).mass, np.inf)
                lt_vis_onZ_mask = ak.fill_none((lt_vis_mass > 60.0) & (lt_vis_mass < 120.0), False)
                preselections.add("lt_os", lt_os_mask)
                preselections.add("lt_vis_onZ", lt_vis_onZ_mask)
                preselections.add("lt_onZ_loose", (tl_zpeak_mask | lt_vis_onZ_mask))
            if self.enable_fwd_blocks:
                preselections.add("2lss_fwd", (events.is2l & pass_trg & fwdjet_mask))

            # 2lss selection
            preselections.add("2lss", (events.is2l & pass_trg))
            preselections.add("2l_p", (chargel0_p))
            preselections.add("2l_m", (chargel0_m))

            # 3l selection
            preselections.add("3l", (events.is3l & pass_trg))
            preselections.add("bmask_exactly0m", (bmask_exactly0med))
            preselections.add("bmask_exactly1m", (bmask_exactly1med))
            preselections.add("bmask_exactly2m", (bmask_exactly2med))
            preselections.add("bmask_atleast2m", (bmask_atleast2med))
            preselections.add("3l_p", (events.is3l & pass_trg & charge3l_p))
            preselections.add("3l_m", (events.is3l & pass_trg & charge3l_m))
            preselections.add("3l_onZ", (sfosz_3l_OnZ_mask))
            preselections.add("3l_offZ", (sfosz_3l_OffZ_mask))

            if self.enable_offz_blocks:
                preselections.add("3l_offZ_low", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_high", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & ~sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_none", (sfosz_3l_OffZ_mask & ~sfosz_3l_OffZ_any_mask))

            # 4l selection
            preselections.add("4l", (events.is4l & pass_trg))

            if not self._skip_signal_regions:
            # If we are not skipping the signal regions, we will fill the selections according to the json specifications
                for lep_cat, lep_cat_dict in import_sr_cat_dict.items():
                    lep_ch_list = lep_cat_dict['lep_chan_lst']
                    chtag = None

                    #looping over each region within the lep category
                    for lep_ch in lep_ch_list:
                        tempmask = None
                        #the first entry of the list is the region name to add in "selections"
                        chtag = lep_ch[0]

                        for chcut in lep_ch[1:]:
                            if not tempmask is None:
                                tempmask = tempmask & preselections.any(chcut)
                            else:
                                tempmask = preselections.any(chcut)
                        selections.add(chtag, tempmask)

            if not self._skip_control_regions:
            # If we are not skipping the control regions, we will fill the selections according to the json specifications
                for lep_cat, lep_cat_dict in import_cr_cat_dict.items():
                    lep_ch_list = lep_cat_dict['lep_chan_lst']
                    chtag = None

                    #looping over each region within the lep category
                    for lep_ch in lep_ch_list:
                        tempmask = None
                        #the first entry of the list is the region name to add in "selections"
                        chtag = lep_ch[0]

                        for chcut in lep_ch[1:]:
                            if not tempmask is None:
                                tempmask = tempmask & preselections.any(chcut)
                            else:
                                tempmask = preselections.any(chcut)
                        selections.add(chtag, tempmask)

            del preselections

            # Lep flavor selection
            selections.add("e",  events.is_e)
            selections.add("m",  events.is_m)
            selections.add("ee",  events.is_ee)
            selections.add("em",  events.is_em)
            selections.add("mm",  events.is_mm)
            selections.add("eee", events.is_eee)
            selections.add("eem", events.is_eem)
            selections.add("emm", events.is_emm)
            selections.add("mmm", events.is_mmm)
            selections.add("llll", (events.is_eeee | events.is_eeem | events.is_eemm | events.is_emmm | events.is_mmmm | events.is_gr4l)) # Not keepting track of these separately

            # Njets selection
            selections.add("exactly_0j", (njets==0))
            selections.add("exactly_1j", (njets==1))
            selections.add("exactly_2j", (njets==2))
            selections.add("exactly_3j", (njets==3))
            selections.add("exactly_4j", (njets==4))
            selections.add("exactly_5j", (njets==5))
            selections.add("exactly_6j", (njets==6))
            selections.add("atleast_1j", (njets>=1))
            selections.add("atleast_3j", (njets>=3))
            selections.add("atleast_4j", (njets>=4))
            selections.add("atleast_5j", (njets>=5))
            selections.add("atleast_6j", (njets>=6))
            selections.add("atleast_7j", (njets>=7))
            selections.add("atleast_0j", (njets>=0))
            selections.add("atmost_2j" , (njets<=2))
            selections.add("atmost_3j" , (njets<=3))

            # AR/SR categories
            selections.add("isSR_2lSS",    ( events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS",    (~events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS_OS", ( events.is2l_SR) & charge2l_0) # Sideband for the charge flip
            selections.add("isSR_2lOS",    ( events.is2l_SR) & charge2l_0)
            selections.add("isAR_2lOS",    (~events.is2l_SR) & charge2l_0)
            if self.enable_tau_blocks:
                selections.add("isSR_1l",    ( events.is1l_SR))
                selections.add("isAR_1l",    (~events.is1l_SR))

            selections.add("isSR_3l",  events.is3l_SR)
            selections.add("isAR_3l", ~events.is3l_SR)
            selections.add("isSR_4l",  events.is4l_SR)


            ######### Variables for the dense axes of the hists ##########

            # Calculate ptbl
            ptbl_bjet = goodJets[(isBtagJetsMedium | isBtagJetsLoose)]
            ptbl_bjet = ptbl_bjet[ak.argmax(ptbl_bjet.pt,axis=-1,keepdims=True)] # Only save hardest b-jet
            ptbl_lep = l_fo_conept_sorted
            ptbl = (ptbl_bjet.nearest(ptbl_lep) + ptbl_bjet).pt

            # Keep the public concepts distinct: ptz is the in-window
            # Z-candidate pT, while ptll is the closest-SFOS dilepton pT used by
            # the 3l off-Z low/high categories.
            ptz = te_es.get_Z_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
            if self.enable_tau_blocks:
                ptz_wtau = te_es.get_Zlt_pt(l0, l1, tau0)

            if self.enable_offz_blocks:
                ptll = te_es.get_ll_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
            # Leading (b+l) pair pt
            bjetsl = goodJets[isBtagJetsLoose][ak.argsort(goodJets[isBtagJetsLoose].pt, axis=-1, ascending=False)]
            bjetsm = goodJets[isBtagJetsMedium][ak.argsort(goodJets[isBtagJetsMedium].pt, axis=-1, ascending=False)]
            bl_pairs = ak.cartesian({"b":bjetsl,"l":l_fo_conept_sorted})
            blpt = (bl_pairs["b"] + bl_pairs["l"]).pt
            bl0pt = ak.flatten(blpt[ak.argmax(blpt,axis=-1,keepdims=True)])

            bjetsl_padded = ak.pad_none(bjetsl, 2)
            b0l = bjetsl_padded[:,0]
            b1l = bjetsl_padded[:,1]
            bjetsm_padded = ak.pad_none(bjetsm, 2)
            b0m = bjetsm_padded[:,0]
            b1m = bjetsm_padded[:,1]

            # Collection of all objects (leptons and jets)
            if self.enable_tau_blocks:
                l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted,goodJets,cleaning_taus], axis=1),"PtEtaPhiMCollection")
            else:
                l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted,goodJets], axis=1),"PtEtaPhiMCollection")

            # Leading object (j or l) pt
            o0pt = ak.max(l_j_collection.pt,axis=-1)

            # Pairs of l+j
            l_j_pairs = ak.combinations(l_j_collection,2,fields=["o0","o1"])
            l_j_pairs_pt = (l_j_pairs.o0 + l_j_pairs.o1).pt
            l_j_pairs_mass = (l_j_pairs.o0 + l_j_pairs.o1).mass
            lj0pt = ak.max(l_j_pairs_pt,axis=-1)

            # LT
            lt = ak.sum(l_fo_conept_sorted_padded.pt, axis=-1) + met.pt

            # Define invariant mass hists
            mll_0_1 = (l0+l1).mass # Invmass for leading two leps

            # ST (but "st" is too hard to search in the code, so call it ljptsum)
            ljptsum = ak.sum(l_j_collection.pt,axis=-1)
            if self._ecut_threshold is not None:
                ecut_mask = (ljptsum<self._ecut_threshold)
                
            # Counts
            counts = np.ones_like(events['event'])
            is_l0_electron = (abs(l0.pdgId)==11)
            is_l1_electron = (abs(l1.pdgId)==11)
            default_l0_seed = ak.zeros_like(l0.pt)
            default_l1_seed = ak.zeros_like(l1.pt)
            l0_seed_etaorx = ak.where(
                is_l0_electron,
                getattr(l0, "seediEtaOriX", default_l0_seed),
                default_l0_seed,
            )
            l0_seed_phiory = ak.where(
                is_l0_electron,
                getattr(l0, "seediPhiOriY", default_l0_seed),
                default_l0_seed,
            )
            l1_seed_etaorx = ak.where(
                is_l1_electron,
                getattr(l1, "seediEtaOriX", default_l1_seed),
                default_l1_seed,
            )
            l1_seed_phiory = ak.where(
                is_l1_electron,
                getattr(l1, "seediPhiOriY", default_l1_seed),
                default_l1_seed,
            )

            # Variables we will loop over when filling hists
            varnames = {}
            varnames["ht"]      = ht
            varnames["met"]     = met.pt
            varnames["ljptsum"] = ljptsum
            varnames["l0conept"]    = l0.conept
            varnames["l0pt"]    = l0.pt_raw
            varnames["l0ptcorr"]= l0.pt
            varnames["l0eta"]   = l0.eta
            varnames["l1conept"]    = l1.conept
            varnames["l1pt"]    = l1.pt_raw
            varnames["l1ptcorr"]= l1.pt
            varnames["l1eta"]   = l1.eta
            varnames["j0pt"]    = ak.flatten(j0.pt)
            varnames["j0eta"]   = ak.flatten(j0.eta)
            varnames["fwd0pt"]  = ak.flatten(fwd0.pt)
            varnames["fwd0eta"] = ak.flatten(fwd0.eta)
            varnames["njets"]   = njets
            varnames["nbtagsl"] = nbtagsl
            varnames["nbtagsm"] = nbtagsm
            varnames["invmass"] = mll_0_1
            varnames["ptbl"]    = ak.flatten(ptbl)
            varnames["ptz"]     = ptz
            if self.enable_offz_blocks:
                varnames["ptll"] = ptll
            varnames["b0pt"]    = ak.flatten(ptbl_bjet.pt)
            varnames["bl0pt"]   = bl0pt
            varnames["o0pt"]    = o0pt
            varnames["lj0pt"]   = lj0pt
            varnames["lt"]      = lt
            varnames["npvs"]    = pv.npvs
            varnames["npvsGood"]= pv.npvsGood
            if is_run3:
                varnames["jet_eta_phi_before_veto"] = veto_map_input_jets
                varnames["jet_eta_phi_after_veto"] = veto_map_input_jets
            lepton0_pt_raw = l0.pt_raw 
            lepton0_abseta = abs(l0.eta) 

            # if not isData:
            #     l0_gen_pdgId = ak.fill_none(l0["gen_pdgId"], -1)
            #     l1_gen_pdgId = ak.fill_none(l1["gen_pdgId"], -1)
            #     l2_gen_pdgId = ak.fill_none(l2["gen_pdgId"], -1)
            #     l0_genParent_pdgId = ak.fill_none(l0["genParent_pdgId"], -1)
            #     l1_genParent_pdgId = ak.fill_none(l1["genParent_pdgId"], -1)
            #     l2_genParent_pdgId = ak.fill_none(l2["genParent_pdgId"], -1)

            #     b0l_hFlav = ak.fill_none(b0l.hadronFlavour, -1) 
            #     b0l_pFlav = ak.fill_none(b0l.partonFlavour, -1)
            #     b1l_hFlav = ak.fill_none(b1l.hadronFlavour, -1) 
            #     b1l_pFlav = ak.fill_none(b1l.partonFlavour, -1)
            #     b0l_genhFlav = ak.fill_none(b0l.matched_gen.hadronFlavour, -1) 
            #     b0l_genpFlav = ak.fill_none(b0l.matched_gen.partonFlavour, -1)
            #     b1l_genhFlav = ak.fill_none(b1l.matched_gen.hadronFlavour, -1) 
            #     b1l_genpFlav = ak.fill_none(b1l.matched_gen.partonFlavour, -1)

            #     b0m_hFlav = ak.fill_none(b0m.hadronFlavour, -1) 
            #     b0m_pFlav = ak.fill_none(b0m.partonFlavour, -1)
            #     b1m_hFlav = ak.fill_none(b1m.hadronFlavour, -1) 
            #     b1m_pFlav = ak.fill_none(b1m.partonFlavour, -1)
            #     b0m_genhFlav = ak.fill_none(b0m.matched_gen.hadronFlavour, -1) 
            #     b0m_genpFlav = ak.fill_none(b0m.matched_gen.partonFlavour, -1)
            #     b1m_genhFlav = ak.fill_none(b1m.matched_gen.hadronFlavour, -1) 
            #     b1m_genpFlav = ak.fill_none(b1m.matched_gen.partonFlavour, -1)

            # else:
            #     l0_gen_pdgId = ak.fill_none(ak.zeros_like(l0.pt), -1)
            #     l1_gen_pdgId = ak.fill_none(ak.zeros_like(l1.pt), -1)
            #     l2_gen_pdgId = ak.fill_none(ak.zeros_like(l2.pt), -1)
            #     l0_genParent_pdgId = ak.fill_none(ak.zeros_like(l0.pt), -1)
            #     l1_genParent_pdgId = ak.fill_none(ak.zeros_like(l1.pt), -1)
            #     l2_genParent_pdgId = ak.fill_none(ak.zeros_like(l2.pt), -1)

            #     b0l_hFlav = ak.fill_none(ak.zeros_like(b0l.pt), -1)
            #     b0l_pFlav = ak.fill_none(ak.zeros_like(b0l.pt), -1)
            #     b1l_hFlav = ak.fill_none(ak.zeros_like(b1l.pt), -1)
            #     b1l_pFlav = ak.fill_none(ak.zeros_like(b1l.pt), -1)
            #     b0l_genhFlav = ak.fill_none(ak.zeros_like(b0l.pt), -1)
            #     b0l_genpFlav = ak.fill_none(ak.zeros_like(b0l.pt), -1)
            #     b1l_genhFlav = ak.fill_none(ak.zeros_like(b1l.pt), -1)
            #     b1l_genpFlav = ak.fill_none(ak.zeros_like(b1l.pt), -1)
                
            #     b0m_hFlav = ak.fill_none(ak.zeros_like(b0m.pt), -1)
            #     b0m_pFlav = ak.fill_none(ak.zeros_like(b0m.pt), -1)
            #     b1m_hFlav = ak.fill_none(ak.zeros_like(b1m.pt), -1)
            #     b1m_pFlav = ak.fill_none(ak.zeros_like(b1m.pt), -1)
            #     b0m_genhFlav = ak.fill_none(ak.zeros_like(b0m.pt), -1)
            #     b0m_genpFlav = ak.fill_none(ak.zeros_like(b0m.pt), -1)
            #     b1m_genhFlav = ak.fill_none(ak.zeros_like(b1m.pt), -1)
            #     b1m_genpFlav = ak.fill_none(ak.zeros_like(b1m.pt), -1)

            # varnames["l0_gen_pdgId"] = l0_gen_pdgId
            # varnames["l1_gen_pdgId"] = l1_gen_pdgId
            # varnames["l2_gen_pdgId"] = l2_gen_pdgId
            # varnames["l0_genParent_pdgId"] = l0_genParent_pdgId
            # varnames["l1_genParent_pdgId"] = l1_genParent_pdgId
            # varnames["l2_genParent_pdgId"] = l2_genParent_pdgId
            
            # varnames["b0l_hFlav"] = b0l_hFlav
            # varnames["b0l_pFlav"] = b0l_pFlav
            # varnames["b1l_hFlav"] = b1l_hFlav
            # varnames["b1l_pFlav"] = b1l_pFlav
            # varnames["b0l_genhFlav"] = b0l_genhFlav
            # varnames["b0l_genpFlav"] = b0l_genpFlav
            # varnames["b1l_genhFlav"] = b1l_genhFlav
            # varnames["b1l_genpFlav"] = b1l_genpFlav
            # varnames["b0m_hFlav"] = b0m_hFlav
            # varnames["b0m_pFlav"] = b0m_pFlav
            # varnames["b1m_hFlav"] = b1m_hFlav
            # varnames["b1m_pFlav"] = b1m_pFlav
            # varnames["b0m_genhFlav"] = b0m_genhFlav
            # varnames["b0m_genpFlav"] = b0m_genpFlav
            # varnames["b1m_genhFlav"] = b1m_genhFlav
            # varnames["b1m_genpFlav"] = b1m_genpFlav
            # varnames["lepton_pt_vs_eta"] = {
            #     "lepton_pt_vs_eta_pt": lepton0_pt_raw,
            #     "lepton_pt_vs_eta_abseta": lepton0_abseta,
            # }
            # varnames["l0_SeedEtaOrX_vs_SeedPhiOrY"] = {
            #     "l0_SeedEtaOrX_vs_SeedPhiOrY_SeedEtaOrX": l0_seed_etaorx,
            #     "l0_SeedEtaOrX_vs_SeedPhiOrY_SeedPhiOrY": l0_seed_phiory,
            # }
            # varnames["l0_eta_vs_phi"] = {
            #     "l0_eta_vs_phi_eta": l0.eta,
            #     "l0_eta_vs_phi_phi": l0.phi,
            # }
            # varnames["l1_SeedEtaOrX_vs_SeedPhiOrY"] = {
            #     "l1_SeedEtaOrX_vs_SeedPhiOrY_SeedEtaOrX": l1_seed_etaorx,
            #     "l1_SeedEtaOrX_vs_SeedPhiOrY_SeedPhiOrY": l1_seed_phiory,
            # }
            # varnames["l1_eta_vs_phi"] = {
            #     "l1_eta_vs_phi_eta": l1.eta,
            #     "l1_eta_vs_phi_phi": l1.phi,
            # }

            if self.enable_tau_blocks:
                varnames["ptz_wtau"] = ptz_wtau
                varnames["tau0Tpt"] = tau0_T.pt
                varnames["tau0Fpt"] = tau0_fo.pt
                pass

            for varname, var in varnames.items():
                if isinstance(var, dict):
                    for subvarname, subvar in var.items():
                        varnames[varname][subvarname] = subvar
                else:
                    varnames[varname] = var

            ########## Fill the histograms ##########
            cat_dict = {}
            if not self._skip_signal_regions:
            # If we are not skipping the signal regions, we will fill the SR categories
                sr_cat_dict = {}    
                for lep_cat in import_sr_cat_dict.keys():
                    sr_cat_dict[lep_cat] = {}
                    for jet_cat in import_sr_cat_dict[lep_cat]["jet_lst"]:
                        jet_mode, jet_threshold, _ = parse_analysis_njet_token(jet_cat)
                        jet_key = f"{jet_mode}_{jet_threshold}j"

                        sr_cat_dict[lep_cat][jet_key] = {}
                        sr_cat_dict[lep_cat][jet_key]["lep_chan_lst"] = []
                        for lep_chan_def in import_sr_cat_dict[lep_cat]["lep_chan_lst"]:
                            sr_cat_dict[lep_cat][jet_key]["lep_chan_lst"].append(lep_chan_def[0])
                        sr_cat_dict[lep_cat][jet_key]["lep_flav_lst"] = import_sr_cat_dict[lep_cat]["lep_flav_lst"]
                        if isData and "appl_lst_data" in import_sr_cat_dict[lep_cat].keys():
                            sr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_sr_cat_dict[lep_cat]["appl_lst"] + import_sr_cat_dict[lep_cat]["appl_lst_data"]
                        else:
                            sr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_sr_cat_dict[lep_cat]["appl_lst"]
                
                cat_dict.update(sr_cat_dict)
                del import_sr_cat_dict

            if not self._skip_control_regions:
            # If we are not skipping the control regions, we will fill the CR categories
                cr_cat_dict = {}
                for lep_cat in import_cr_cat_dict.keys():
                    cr_cat_dict[lep_cat] = {}
                    for jet_cat in import_cr_cat_dict[lep_cat]["jet_lst"]:
                        jettag = None
                        if jet_cat.startswith("="):
                            jettag = "exactly_"
                        elif jet_cat.startswith("<"):
                            jettag = "atmost_"
                        elif jet_cat.startswith(">"):
                            jettag = "atleast_"
                        else:
                            raise RuntimeError(f"jet_cat {jet_cat} in {lep_cat} misses =,<,> !")
                        jet_key = jettag + str(jet_cat).replace("=", "").replace("<", "").replace(">", "") + "j"

                        cr_cat_dict[lep_cat][jet_key] = {}
                        cr_cat_dict[lep_cat][jet_key]["lep_chan_lst"] = []
                        for lep_chan_def in import_cr_cat_dict[lep_cat]["lep_chan_lst"]:
                            cr_cat_dict[lep_cat][jet_key]["lep_chan_lst"].append(lep_chan_def[0])
                        cr_cat_dict[lep_cat][jet_key]["lep_flav_lst"] = import_cr_cat_dict[lep_cat]["lep_flav_lst"]
                        if isData and "appl_lst_data" in import_cr_cat_dict[lep_cat].keys():
                            cr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_cr_cat_dict[lep_cat]["appl_lst"] + import_cr_cat_dict[lep_cat]["appl_lst_data"]
                        else:
                            cr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_cr_cat_dict[lep_cat]["appl_lst"]

                cat_dict.update(cr_cat_dict)
                del import_cr_cat_dict
                
            if (not self._skip_signal_regions and not self._skip_control_regions):
                for k in sr_cat_dict:
                    if k in cr_cat_dict:
                        raise Exception(f"The key {k} is in both CR and SR dictionaries.")

            # Loop over the hists we want to fill
            def _prepare_axis_values(axis_value):
                if isinstance(axis_value, dict):
                    cast_values = {}
                    validity_masks = {}
                    for axis_name, axis_component in axis_value.items():
                        cast_values[axis_name] = ak.values_astype(axis_component, np.float32)
                        validity_masks[axis_name] = ~ak.is_none(axis_component)
                    return cast_values, validity_masks
                cast_value = ak.values_astype(axis_value, np.float32)
                validity_mask = ~ak.is_none(axis_value)
                return cast_value, validity_mask

            for dense_axis_name, dense_axis_vals in varnames.items():
                fill_base_hist = dense_axis_name in self._base_hist_name_set
                companion_axis_mapping = self._hist_sumw2_axis_mapping.get(
                    dense_axis_name
                )
                if self._sumw2_policy is None:
                    target_selected_for_sumw2 = (
                        self._fill_sumw2_hist
                        and dense_axis_name not in JVM_ETA_PHI_DIAGNOSTIC_HISTOGRAMS
                    )
                else:
                    target_selected_for_sumw2 = self._sumw2_policy.selects(
                        dataset_key,
                        histAxisName,
                        dense_axis_name,
                    )
                if target_selected_for_sumw2 and not companion_axis_mapping:
                    raise RuntimeError(
                        "Resolved sumw2 target has no allocated companion for "
                        f"dataset='{dataset_key}', process='{histAxisName}', "
                        f"family='{dense_axis_name}'."
                    )
                fill_sumw2_hist = target_selected_for_sumw2 and bool(
                    companion_axis_mapping
                )
                if not (fill_base_hist or fill_sumw2_hist):
                    continue

                if dense_axis_name in axes_info_2d:
                    nominal_histogram_key = dense_axis_name
                elif isEFT:
                    nominal_histogram_key = eft_nominal_key(dense_axis_name)
                else:
                    nominal_histogram_key = scalar_nominal_key(dense_axis_name)
                if fill_base_hist and nominal_histogram_key not in hout:
                    raise RuntimeError(
                        "Resolved sample metadata requires nominal component "
                        f"'{nominal_histogram_key}', but it was not allocated."
                    )

                # Set up the list of syst wgt variations to loop over
                wgt_var_lst = ["nominal"]
                if self._do_systematics:
                    if not isData:
                        if (syst_var != "nominal"):
                            # In this case, we are dealing with systs that change the kinematics of the objs (e.g. JES)
                            # So we don't want to loop over up/down weight variations here
                            wgt_var_lst = [syst_var]
                        else:
                            # Otherwise we want to loop over the up/down weight variations
                            wgt_var_lst = wgt_var_lst + wgt_correction_syst_lst + data_syst_lst
                    else:
                        # This is data, so we want to loop over just up/down variations relevant for data (i.e. FF up and down)
                        wgt_var_lst = wgt_var_lst + data_syst_lst

                # Loop over the systematics

                for wgt_fluct in wgt_var_lst:
                    # Loop over nlep categories "2l", "3l", "4l"
                    for nlep_cat in cat_dict.keys():
                        # Get the appropriate Weights object for the nlep cat and get the weight to be used when filling the hist
                        # Need to do this inside of nlep cat loop since some wgts depend on lep cat
                        weights_object = weights_dict[nlep_cat]

                        if (wgt_fluct == "nominal") or (wgt_fluct in obj_correction_syst_lst):
                            # In the case of "nominal", or the jet energy systematics, no weight systematic variation is used
                            weight = weights_object.weight(None)
                        else:
                            # Otherwise get the weight from the Weights object
                            if wgt_fluct in weights_object.variations:
                                weight = weights_object.weight(wgt_fluct)
                            else:
                                # Note in this case there is no up/down fluct for this cateogry, so we don't want to fill a hist for it
                                continue

                        # This is a check ot make sure we guard against any unintentional variations being applied to data
                        if self._do_systematics and isData:
                            # Should not have any up/down variations for data in 4l (since we don't estimate the fake rate there)
                            if nlep_cat == "4l":
                                if weights_object.variations != set([]): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{[]}\" but have \"{weights_object.variations}\".")
                            # In all other cases, the up/down variations should correspond to only the ones in the data list
                            else:
                                if weights_object.variations != set(data_syst_lst): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{set(data_syst_lst)}\" but have \"{weights_object.variations}\".")

                        # Get a mask for events that pass any of the njet requiremens in this nlep cat
                        # Useful in cases like njets hist where we don't store njets in a sparse axis
                        njets_any_mask = selections.any(*cat_dict[nlep_cat].keys())

                        # Loop over the njets list for each channel
                        for njet_val in cat_dict[nlep_cat].keys():

                            # Loop over the appropriate AR and SR for this channel
                            for appl in cat_dict[nlep_cat][njet_val]["appl_lst"]:

                                # We don't want or need to fill SR histos with the FF variations
                                if appl.startswith("isSR") and wgt_fluct in data_syst_lst: continue

                                # Loop over the channels in each nlep cat (e.g. "3l_m_offZ_1b")
                                for lep_chan in cat_dict[nlep_cat][njet_val]["lep_chan_lst"]:
                                    # Loop over the lep flavor list for each channel
                                    for lep_flav in cat_dict[nlep_cat][njet_val]["lep_flav_lst"]:
                                        # Construct the hist name
                                        flav_ch = None
                                        njet_ch = None
                                        cuts_lst = [appl,lep_chan]

                                        #Selections applied everywhere
                                        if isData:
                                            cuts_lst.append("is_good_lumi")
                                        is_jvm_eta_phi_diagnostic = (
                                            dense_axis_name in JVM_ETA_PHI_DIAGNOSTIC_HISTOGRAMS
                                        )
                                        if should_include_jet_veto_in_histogram_selection(
                                            dense_axis_name
                                        ):
                                            cuts_lst.append("jet_veto")

                                        if self._split_by_lepton_flavor:
                                            flav_ch = lep_flav
                                            cuts_lst.append(lep_flav)
                                        if dense_axis_name != "njets":
                                            njet_ch = njet_val
                                            cuts_lst.append(njet_val)
                                        ch_name = construct_cat_name(lep_chan,njet_str=njet_ch,flav_str=flav_ch)

                                        # Get the cuts mask for all selections
                                        if dense_axis_name == "njets":
                                            all_cuts_mask = (selections.all(*cuts_lst) & njets_any_mask)
                                        else:
                                            all_cuts_mask = selections.all(*cuts_lst)
                                        all_cuts_mask = (
                                            all_cuts_mask
                                            & events[
                                                "ttgamma_photon_history_"
                                                "pass_conversion_overlap_removal"
                                            ]
                                        )
                                        # Apply the optional cut on energy of the event
                                        if self._ecut_threshold is not None:
                                            all_cuts_mask = (all_cuts_mask & ecut_mask)

                                        if is_jvm_eta_phi_diagnostic:
                                            if not should_fill_jvm_eta_phi_diagnostic(
                                                is_run3,
                                                syst_var,
                                                wgt_fluct,
                                            ):
                                                continue
                                            diagnostic_event_mask = get_jvm_eta_phi_event_mask(
                                                all_cuts_mask,
                                                veto_map_mask,
                                                dense_axis_name,
                                            )
                                            eta_flat, phi_flat, weights_flat = (
                                                flatten_jagged_jet_eta_phi_weights(
                                                    veto_map_input_jets,
                                                    diagnostic_event_mask,
                                                    weight,
                                                )
                                            )
                                            if fill_base_hist:
                                                axis_names = self._hist_axis_map[dense_axis_name]
                                                hout[nominal_histogram_key].fill(
                                                    **{
                                                        axis_names[0]: eta_flat,
                                                        axis_names[1]: phi_flat,
                                                        "channel": ch_name,
                                                        "appl": appl,
                                                        "process": histAxisName,
                                                        "systematic": wgt_fluct,
                                                        "weight": weights_flat,
                                                    }
                                                )
                                            continue

                                        # Weights and eft coeffs
                                        weights_flat = weight[all_cuts_mask]
                                        eft_coeffs_cut = eft_coeffs[all_cuts_mask] if eft_coeffs is not None else None

                                        axis_names = self._hist_axis_map.get(
                                            dense_axis_name,
                                        )
                                        if axis_names is None:
                                            if companion_axis_mapping:
                                                axis_names = list(companion_axis_mapping.values())
                                            else:
                                                axis_names = [dense_axis_name]
                                        sumw2_axis_names = self._hist_axis_map.get(
                                            dense_axis_name+"_sumw2",
                                            [dense_axis_name+"_sumw2"],
                                        )
                                        sumw2_axis_mapping = companion_axis_mapping
                                        if sumw2_axis_mapping is None:
                                            if sumw2_axis_names and axis_names:
                                                sumw2_axis_mapping = {
                                                    sumw2_axis_names[0]: axis_names[0]
                                                }
                                            else:
                                                sumw2_axis_mapping = {}

                                        base_values_cut = None
                                        prepared_axis_vals, axis_validity = _prepare_axis_values(dense_axis_vals)
                                        combined_axis_mask = None
                                        if isinstance(prepared_axis_vals, dict):
                                            values_cut_map = {}
                                            for axis_name in axis_names:
                                                if axis_name not in prepared_axis_vals:
                                                    continue
                                                axis_values = prepared_axis_vals[axis_name][all_cuts_mask]
                                                values_cut_map[axis_name] = axis_values
                                                axis_mask = axis_validity.get(axis_name)
                                                if axis_mask is not None:
                                                    axis_mask_cut = axis_mask[all_cuts_mask]
                                                    combined_axis_mask = (
                                                        axis_mask_cut
                                                        if combined_axis_mask is None
                                                        else (combined_axis_mask & axis_mask_cut)
                                                    )
                                        else:
                                            base_values_cut = prepared_axis_vals[all_cuts_mask]
                                            base_axis_name = axis_names[0] if axis_names else dense_axis_name
                                            values_cut_map = {
                                                base_axis_name: base_values_cut
                                            }
                                            combined_axis_mask = axis_validity[all_cuts_mask]

                                        if combined_axis_mask is not None:
                                            values_cut_map = {
                                                axis_name: axis_values[combined_axis_mask]
                                                for axis_name, axis_values in values_cut_map.items()
                                            }
                                            weights_flat = weights_flat[combined_axis_mask]
                                            if eft_coeffs_cut is not None:
                                                eft_coeffs_cut = eft_coeffs_cut[combined_axis_mask]
                                            if base_values_cut is not None:
                                                base_values_cut = base_values_cut[combined_axis_mask]

                                        fill_nominal_sumw2_hist = self._should_fill_sumw2_histogram(
                                            fill_sumw2_hist,
                                            wgt_fluct=wgt_fluct,
                                        )
                                        sumw2_values_cut_map = {}
                                        if fill_nominal_sumw2_hist:
                                            for sumw2_axis_name, base_axis_name in sumw2_axis_mapping.items():
                                                base_values = values_cut_map.get(base_axis_name)
                                                if (base_values is None) and (base_values_cut is not None):
                                                    base_values = base_values_cut
                                                if base_values is not None:
                                                    sumw2_values_cut_map[sumw2_axis_name] = base_values

                                        # Fill the histos
                                        skip_hist = self._should_skip_histogram_fill(
                                            dense_axis_name,
                                            ch_name,
                                            lep_chan,
                                        )

                                        if skip_hist:
                                            continue

                                        if fill_base_hist:
                                            axes_fill_info_dict = {
                                                **values_cut_map,
                                                "channel"    : ch_name,
                                                "appl"       : appl,
                                                "process"    : histAxisName,
                                                "systematic": wgt_fluct,
                                                "weight"     : weights_flat,
                                            }
                                            if self._hist_requires_eft.get(nominal_histogram_key, False):
                                                axes_fill_info_dict["eft_coeff"] = eft_coeffs_cut
                                            hout[nominal_histogram_key].fill(**axes_fill_info_dict)
                                                                                    
                                        if fill_nominal_sumw2_hist:
                                            # The companion is an SM-only statistical moment.
                                            # EFT factors are evaluated at the SM and folded into
                                            # the scalar contribution before squaring. Filling
                                            # without eft_coeff stores that result as a constant-
                                            # only scalar SparseHist content.
                                            sumw2_fill_info = {
                                                **sumw2_values_cut_map,
                                                "channel"    : ch_name,
                                                "appl"       : appl,
                                                "process"    : histAxisName,
                                                "systematic": wgt_fluct,
                                                "weight"     : calculate_sm_sumw2_weights(
                                                    weights_flat,
                                                    eft_coeffs_cut,
                                                ),
                                            }
                                            hout[dense_axis_name+"_sumw2"].fill(**sumw2_fill_info)

                                        # Do not loop over lep flavors if not self._split_by_lepton_flavor, it's a waste of time and also we'd fill the hists too many times
                                        if not self._split_by_lepton_flavor: break

                            # Do not loop over njets if hist is njets (otherwise we'd fill the hist too many times)
                            if dense_axis_name == "njets":
                                break
        
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)
