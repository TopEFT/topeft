#!/usr/bin/env python
import copy
from collections import OrderedDict
import coffea
import numpy as np
import awkward as ak
import os
import re
import logging

import hist
from topcoffea.modules.histEFT import HistEFT
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask
from typing import Dict, List, Tuple

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.event_selection as tc_es
import topcoffea.modules.object_selection as tc_os
import topcoffea.modules.corrections as tc_cor

from topeft.modules.paths import topeft_path
from topeft.modules.corrections import ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachTauSF, ApplyTES, ApplyTESSystematic, ApplyFESSystematic, AttachPerLeptonFR, ApplyRochesterCorrections, ApplyJetSystematics
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
from topcoffea.modules.get_param_from_jsons import GetParam

logger = logging.getLogger(__name__)
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
        muonSyst='nominal',
        dtype=np.float32,
        rebin=False,
        channel_dict=None,
        golden_json_path=None,
        systematic_variations=None,
        available_systematics=None,
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
            raise ValueError("available_systematics must be provided and cannot be None")
        self._available_systematics = {
            key: tuple(value) for key, value in available_systematics.items()
        }
        self._available_systematics_sets = {
            key: set(values) for key, values in self._available_systematics.items()
        }

        self._golden_json_path = golden_json_path
        if self._sample.get("isData") and not self._golden_json_path:
            raise ValueError("golden_json_path must be provided for data samples")

        if var_info is None:
            raise ValueError("var_info must be provided and cannot be None")

        if hist_keys is None:
            raise ValueError("hist_keys must be provided and cannot be None")

        if not isinstance(hist_keys, dict):
            raise TypeError("hist_keys must be a mapping of variation name to histogram key")

        raw_histogram_key_map = OrderedDict(hist_keys)

        if not raw_histogram_key_map:
            raise ValueError("hist_keys must contain at least one entry")

        histogram_key_map: "OrderedDict[str, Tuple[Tuple[str, ...], ...]]" = OrderedDict()
        for variation_label, key_entries in raw_histogram_key_map.items():
            if isinstance(key_entries, tuple) and len(key_entries) == 5 and not isinstance(key_entries[0], (tuple, list)):
                normalized_entries = (tuple(key_entries),)
            else:
                try:
                    normalized_entries = tuple(tuple(entry) for entry in key_entries)
                except TypeError as exc:
                    raise TypeError(
                        "hist_keys values must be 5-element tuples or an iterable of such tuples"
                    ) from exc
            if not normalized_entries:
                raise ValueError("hist_keys entries must contain at least one histogram key")
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
                    raise ValueError("Histogram keys must refer to the configured sample")

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

                histogram[hist_key_entry] = HistEFT(
                    dense_axis,
                    wc_names=wc_names_lst,
                    label=r"Events",
                )

                self._hist_keys_to_fill.append(hist_key_entry)

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

                    sumw2_key = (
                        f"{self._var}_sumw2",
                        self._channel,
                        key_appl,
                        key_sample,
                        syst_label,
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
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually

        if systematic_variations is None:
            systematic_variations = ()
        else:
            systematic_variations = tuple(systematic_variations)

        self._systematic_variations = systematic_variations
        self._systematic_info = systematic_variations[0] if systematic_variations else None

        if self._systematic_variations and self._systematic_info is None:
            raise ValueError("systematic_variations must contain at least one entry when provided")


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
        ch_name = construct_cat_name(
            lep_chan, njet_str=njet_ch, flav_str=flav_ch
        )
        base_ch_name = construct_cat_name(
            lep_chan, njet_str=njet_ch, flav_str=None
        )
        return ch_name, base_ch_name

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

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        raw_dataset_name = events.metadata["dataset"]
        dataset, trigger_dataset = self._resolve_dataset_names(raw_dataset_name)
        isEFT = self._sample["WCnames"] != []

        isData = self._sample["isData"]
        histAxisName = self._sample["histAxisName"]
        year = self._sample["year"]
        xsec = self._sample["xsec"]
        sow = self._sample["nSumOfWeights"]

        # Build the ordered list of (variation, histogram label) contexts to run.
        variation_contexts = []
        if self._systematic_variations:
            for variation in self._systematic_variations:
                label = self._histogram_label_lookup.get(variation.name)
                if label is None:
                    raise KeyError(
                        f"Missing histogram label for requested variation '{variation.name}'"
                    )
                variation_contexts.append((variation, label))
        else:
            variation_contexts.append((None, "nominal"))

        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3

        run_era = None
        if isData:
            run_era = self._sample["path"].split("/")[2].split("-")[0][-1]

        is_lo_sample = histAxisName in get_te_param("lo_xsec_samples")

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
        met  = events.MET
        ele  = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet

        if is_run3:
            leptonSelection = te_os.run3leptonselection()
            jetsRho = events.Rho["fixedGridRhoFastjetAll"]
        elif is_run2:
            leptonSelection = te_os.run2leptonselection()
            jetsRho = events.fixedGridRhoFastjetAll

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(events.MET.pt)

        ele["idEmu"] = te_os.ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        ele["conept"] = leptonSelection.coneptElec(ele)
        mu["conept"] = leptonSelection.coneptMuon(mu)
        ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)
        if not isData:
            ele["gen_pdgId"] = ak.fill_none(ele.matched_gen.pdgId, 0)
            mu["gen_pdgId"] = ak.fill_none(mu.matched_gen.pdgId, 0)

        # Initialize lumi mask to ``True`` for all events so simulated samples
        # see an identity mask.  Data samples replace it with the configured
        # golden JSON selection below.
        lumi_mask = ak.ones_like(events.run, dtype=bool)
        if isData:
            lumi_mask = LumiMask(self._golden_json_path)(events.run, events.luminosityBlock)

        ######### EFT coefficients ##########

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what wanted
            if self._sample["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._sample["WCnames"], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None
        
        # Initialize the out object
        hout = self.accumulator

        ################### Electron selection ####################

        ele["isPres"] = leptonSelection.isPresElec(ele)
        ele["isLooseE"] = leptonSelection.isLooseElec(ele)
        ele["isFO"] = leptonSelection.isFOElec(ele, year)
        ele["isTightLep"] = leptonSelection.tightSelElec(ele)

        ################### Muon selection ####################

        mu["pt"] = ApplyRochesterCorrections(year, mu, isData) # Run3 ready
        mu["isPres"] = leptonSelection.isPresMuon(mu)
        mu["isLooseM"] = leptonSelection.isLooseMuon(mu)
        mu["isFO"] = leptonSelection.isFOMuon(mu, year)
        mu["isTightLep"]= leptonSelection.tightSelMuon(mu)

        ################### Loose selection ####################

        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = ele[ele.isPres & ele.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        # Compute pair invariant masses, for all flavors all signes
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = ele[ele.isPres & ele.isLooseE & ele.isFO]

        # Attach the lepton SFs to the electron and muons collections
        AttachElectronSF(e_fo,year=year, looseWP="none" if is_run3 else "wpLnoiso") #Run3 ready
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
        AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
        m_fo['convVeto'] = ak.ones_like(m_fo.charge)
        m_fo['lostHits'] = ak.zeros_like(m_fo.charge)
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

        ################### Tau selection ####################

        if self.tau_h_analysis:
            tau["pt"], tau["mass"] = ApplyTES(year, tau, isData)
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

        base_met = copy.deepcopy(met)
        base_ele = copy.deepcopy(ele)
        base_mu = copy.deepcopy(mu)
        base_tau = copy.deepcopy(tau)
        base_jets = copy.deepcopy(jets)
        base_l_loose = copy.deepcopy(l_loose)
        base_l_fo = copy.deepcopy(l_fo)
        base_l_fo_conept_sorted = copy.deepcopy(l_fo_conept_sorted)

        object_systematics = self._available_systematics.get("object", ())
        weight_systematics = self._available_systematics.get("weight", ())
        theory_systematics = self._available_systematics.get("theory", ())
        data_weight_systematics = self._available_systematics.get("data_weight", ())
        data_weight_systematics_set = self._available_systematics_sets.get(
            "data_weight", set()
        )

        ######### Systematics ###########

        events_cache = events.caches[0]

        # print("\n\n\n\n")
        # print("variation_contexts:", [(v.name if v is not None else None, l) for v, l in variation_contexts])
        # print("\n\n\n\n")

        for variation, hist_label in variation_contexts:
            variation_name = variation.name if variation is not None else "nominal"
            variation_base = variation.base if variation is not None else None
            variation_type = getattr(variation, "type", None) if variation is not None else None
            variation_metadata = variation.metadata if variation is not None else {}

            variation_base_str = variation_base or ""
            metadata_lepton_flavor = str(
                variation_metadata.get("lepton_flavor")
                or variation_metadata.get("lepton_type")
                or ""
            ).lower()
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

            print("\n\n\n\n\n")
            print(f"Processing variation '{variation_name}' (type: {variation_type}, base: {variation_base})")
            print("\n\n\n\n\n")

            object_variation = "nominal"
            weight_variations_to_run = []
            requested_data_weight_label = None

            # Restore base physics objects before applying per-variation transforms.
            met = copy.deepcopy(base_met)
            ele = copy.deepcopy(base_ele)
            mu = copy.deepcopy(base_mu)
            tau = copy.deepcopy(base_tau)
            jets = copy.deepcopy(base_jets)
            l_loose = copy.deepcopy(base_l_loose)
            l_fo = copy.deepcopy(base_l_fo)
            l_fo_conept_sorted = copy.deepcopy(base_l_fo_conept_sorted)
            if self.tau_h_analysis:
                cleaning_taus = tau[tau["isLoose"] > 0]
                nLtau = ak.num(tau[tau["isLoose"] > 0])
                tau_padded = ak.pad_none(tau, 1)
                tau0 = tau_padded[:, 0]
            else:
                cleaning_taus = None
                nLtau = None
                tau0 = None

            sow_variation_key_map = {}
            requested_sow_variations: set = set()
            sow_variations = {"nominal": sow}

            if variation is not None and self._systematic_variations and not isData:
                group_mapping = variation.group or {}
                group_key = (variation.base, variation.component, variation.year)
                group_info = group_mapping.get(group_key, {})
                
                print("\n\n\n\n\n")
                print("group_mapping:", group_mapping, "\ngroup_key:", group_key, "\ngroup_info:", group_info)
                print("\n\n\n\n\n") 
                
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

            if self._systematic_variations and not isData:
                for sow_label in requested_sow_variations:
                    if is_lo_sample:
                        sow_variations[sow_label] = sow
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
                        requested_data_weight_label = requested_data_weight_label[: -len(_direction)]
                        break

            # These weights can go outside of the outer syst loop since they do not depend on the
            # reconstructed muon or jet pT. Build a single weights object; the consolidated MC-only
            # block below will register the simulated-sample scale factors sequentially as we go.

            weights_object = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        
            # print("\n\n\n\n")
            # print("Running object systematic:", object_variation)
            # print("Running weight systematics:", weight_variations_to_run)
            # print("\n\n\n\n")

            # In this block we add the pieces that depend on the object kinematics.
            met_raw = met

            #################### Jets ####################

            # Jet cleaning, before any jet selection

            if self.tau_h_analysis:
                vetos_tocleanjets = ak.with_name(ak.concatenate([cleaning_taus, l_fo], axis=1), "PtEtaPhiMCandidate")
            else:
                vetos_tocleanjets = ak.with_name(l_fo, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)]  # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

            # Selecting jets and cleaning them
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor) * cleanedJets.pt
            cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor) * cleanedJets.mass
            cleanedJets["rho"] = ak.broadcast_arrays(jetsRho, cleanedJets.pt)[0]

            # Jet energy corrections
            if not isData:
                cleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
                if self.tau_h_analysis:
                    tau["pt"], tau["mass"] = ApplyTESSystematic(year, tau, isData, object_variation)
                    tau["pt"], tau["mass"] = ApplyFESSystematic(year, tau, isData, object_variation)
                    cleaning_taus = tau[tau["isLoose"] > 0]
                    nLtau = ak.num(tau[tau["isLoose"] > 0])
                    tau_padded = ak.pad_none(tau, 1)
                    tau0 = tau_padded[:, 0]

            events_cache = events.caches[0]
            cleanedJets = ApplyJetCorrections(year, corr_type='jets', isData=isData, era=run_era).build(cleanedJets, lazy_cache=events_cache)  #Run3 ready
            cleanedJets = ApplyJetSystematics(year, cleanedJets, object_variation)
            met = ApplyJetCorrections(year, corr_type='met', isData=isData, era=run_era).build(met_raw, cleanedJets, lazy_cache=events_cache)

            cleanedJets["isGood"] = tc_os.is_tight_jet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=get_te_param("jet_id_cut"))
            cleanedJets["isFwd"] = te_os.isFwdJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=40.)
            goodJets = cleanedJets[cleanedJets.isGood]
            fwdJets = cleanedJets[cleanedJets.isFwd]

            # Count jets
            njets = ak.num(goodJets)
            nfwdj = ak.num(fwdJets)
            if "ht" in self._var_def:
                ht = ak.sum(goodJets.pt, axis=-1)
            if "j0" in self._var_def:
                j0 = goodJets[ak.argmax(goodJets.pt, axis=-1, keepdims=True)]

            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
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

            # Workaround to use UL16APV SFs for UL16 for light jets
            if year == "2016":
                year_light = "2016APV"
            else:
                year_light = year

            # Loose DeepJet WP
            loose_tag = "btag_wp_loose_" + year.replace("201", "UL1")
            btagwpl = get_tc_param(loose_tag)
            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            # Medium DeepJet WP
            medium_tag = "btag_wp_medium_" + year.replace("201", "UL1")
            btagwpm = get_tc_param(medium_tag)
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])

            isBtagJetsLooseNotMedium = (isBtagJetsLoose & isNotBtagJetsMedium)

            trigger_weight_label = f"triggerSF_{year}"

            # Determine the lepton multiplicity category from the requested
            # channel name (e.g. ``2lss_4j`` -> ``2l``).  This is used to know
            # which set of weights to apply when attaching lepton scale factors
            # below.
            nlep_cat = re.match(r"\d+l", self.channel).group(0)
            channel_prefix = nlep_cat[:2]

            default_flavour_mask = ak.values_astype(
                ak.zeros_like(goodJets.pt, highlevel=True), np.bool_
            )

            if not isData:
                has_hadron_flavour = hasattr(goodJets, "hadronFlavour")
                if has_hadron_flavour:
                    light_mask = goodJets.hadronFlavour == 0
                    bc_mask = goodJets.hadronFlavour > 0
                else:
                    logger.warning(
                        "Missing 'hadronFlavour' for MC sample '%s'; defaulting to empty jet flavour masks.",
                        dataset,
                    )
                    light_mask = default_flavour_mask
                    bc_mask = default_flavour_mask

                jets_light = goodJets[light_mask]
                jets_bc = goodJets[bc_mask]

                # Begin consolidated MC-only weight registration.

                # If this is not an EFT sample, use the generator weight; otherwise
                # default to unity.
                if eft_coeffs is None:
                    genw = events["genWeight"]
                else:
                    genw = np.ones_like(events["event"])

                # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples.
                lumi = 1000.0 * get_tc_param(f"lumi_{year}")
                weights_object.add("norm", (xsec / sow) * genw * lumi)

                if is_run2:
                    l1prefiring_args = [
                        events.L1PreFiringWeight.Nom,
                        events.L1PreFiringWeight.Up,
                        events.L1PreFiringWeight.Dn,
                    ]
                elif is_run3:
                    l1prefiring_args = [
                        ak.ones_like(events.nom),
                        ak.ones_like(events.nom),
                        ak.ones_like(events.nom),
                    ]

                have_systematics = bool(self._systematic_variations)
                # Attach renorm/fact scale weights regardless of which variations are being evaluated.
                tc_cor.AttachScaleWeights(events)  # Run3 ready (with caveat on "nominal")

                theory_weight_arguments = apply_theory_weight_variations(
                    events=events,
                    variation=variation,
                    variation_base=variation_base,
                    have_systematics=have_systematics,
                    sow=sow,
                    sow_variations=sow_variations,
                    sow_variation_key_map=sow_variation_key_map,
                    is_lo_sample=is_lo_sample,
                    hist_axis_name=histAxisName,
                    sample=self._sample,
                )

                for label, args in theory_weight_arguments.items():
                    weights_object.add(label, *args)

                # Prefiring and pileup (prefire weights only available in nanoAODv9 for Run 2).
                include_prefiring_vars = have_systematics and variation_base == "prefiring"
                register_weight_variation(
                    weights_object,
                    "PreFiring",
                    l1prefiring_args[0],
                    up=lambda: l1prefiring_args[1],
                    down=lambda: l1prefiring_args[2],
                    active=include_prefiring_vars,
                )

                pu_central = tc_cor.GetPUSF(events.Pileup.nTrueInt, year)
                include_pu_vars = have_systematics and variation_base == "pileup"
                register_weight_variation(
                    weights_object,
                    "PU",
                    pu_central,
                    up=lambda: tc_cor.GetPUSF(events.Pileup.nTrueInt, year, "up"),
                    down=lambda: tc_cor.GetPUSF(events.Pileup.nTrueInt, year, "down"),
                    active=include_pu_vars,
                )

                if has_hadron_flavour:
                    # B-tag efficiencies and central SFs are re-used for central and systematic weights.
                    btag_effM_light = GetBtagEff(jets_light, year, "medium")
                    btag_effM_bc = GetBtagEff(jets_bc, year, "medium")
                    btag_effL_light = GetBtagEff(jets_light, year, "loose")
                    btag_effL_bc = GetBtagEff(jets_bc, year, "loose")
                    btag_sfM_light = tc_cor.btag_sf_eval(jets_light, "M", year_light, "deepJet_incl", "central")
                    btag_sfM_bc = tc_cor.btag_sf_eval(jets_bc, "M", year, "deepJet_comb", "central")
                    btag_sfL_light = tc_cor.btag_sf_eval(jets_light, "L", year_light, "deepJet_incl", "central")
                    btag_sfL_bc = tc_cor.btag_sf_eval(jets_bc, "L", year, "deepJet_comb", "central")

                    pData_light, pMC_light = tc_cor.get_method1a_wgt_doublewp(
                        btag_effM_light,
                        btag_effL_light,
                        btag_sfM_light,
                        btag_sfL_light,
                        isBtagJetsMedium[light_mask],
                        isBtagJetsLooseNotMedium[light_mask],
                        isNotBtagJetsLoose[light_mask],
                    )
                    btag_w_light = pData_light / pMC_light
                    pData_bc, pMC_bc = tc_cor.get_method1a_wgt_doublewp(
                        btag_effM_bc,
                        btag_effL_bc,
                        btag_sfM_bc,
                        btag_sfL_bc,
                        isBtagJetsMedium[bc_mask],
                        isBtagJetsLooseNotMedium[bc_mask],
                        isNotBtagJetsLoose[bc_mask],
                    )
                    btag_w_bc = pData_bc / pMC_bc

                    btag_result = register_btag_sf_weights(
                        jets_light=jets_light,
                        jets_bc=jets_bc,
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
                            "light": light_mask,
                            "bc": bc_mask,
                        },
                        years={"light": year_light, "bc": year},
                        systematic_descriptor={
                            "has_systematics": bool(self._systematic_variations),
                            "object_variation": object_variation,
                            "variation_name": variation_name,
                        },
                    )
                    weights_object.add("btagSF", btag_result.central)

                    if btag_result.variation_label is not None:
                        weights_object.add(
                            btag_result.variation_label,
                            events.nom,
                            btag_result.variation_up,
                            btag_result.variation_down,
                        )
                else:
                    weights_object.add("btagSF", ak.ones_like(events.nom))

                # Trigger SFs are only defined for simulated samples.
                register_trigger_sf_weight(
                    weights_object,
                    year=year,
                    events=events,
                    lepton0=l0,
                    lepton1=l1,
                    label=trigger_weight_label,
                    variation_descriptor={
                        "has_systematics": bool(self._systematic_variations),
                        "variation_base": variation_base,
                        "variation_name": variation_name,
                    },
                    logger_obj=logger,
                )

                if self.tau_h_analysis and not isData:
                    AttachTauSF(events, tau, year=year)

                # Lepton (and optional tau) scale factors depend on the lepton category.
                if channel_prefix == "1l":
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_muon",
                        "sf_1l_muon",
                        "sf_1l_hi_muon",
                        "sf_1l_lo_muon",
                        include_muon_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_elec",
                        "sf_1l_elec",
                        "sf_1l_hi_elec",
                        "sf_1l_lo_elec",
                        include_elec_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    if self.tau_h_analysis:
                        register_lepton_sf_weight(
                            weights_object,
                            events,
                            "lepSF_taus_real",
                            "sf_2l_taus_real",
                            "sf_2l_taus_real_hi",
                            "sf_2l_taus_real_lo",
                            include_tau_real_sf_variations,
                            variation_name=variation_name,
                            logger_obj=logger,
                        )
                        register_lepton_sf_weight(
                            weights_object,
                            events,
                            "lepSF_taus_fake",
                            "sf_2l_taus_fake",
                            "sf_2l_taus_fake_hi",
                            "sf_2l_taus_fake_lo",
                            include_tau_fake_sf_variations,
                            variation_name=variation_name,
                            logger_obj=logger,
                        )
                elif channel_prefix == "2l":
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_muon",
                        "sf_2l_muon",
                        "sf_2l_hi_muon",
                        "sf_2l_lo_muon",
                        include_muon_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_elec",
                        "sf_2l_elec",
                        "sf_2l_hi_elec",
                        "sf_2l_lo_elec",
                        include_elec_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    if self.tau_h_analysis:
                        register_lepton_sf_weight(
                            weights_object,
                            events,
                            "lepSF_taus_real",
                            "sf_2l_taus_real",
                            "sf_2l_taus_real_hi",
                            "sf_2l_taus_real_lo",
                            include_tau_real_sf_variations,
                            variation_name=variation_name,
                            logger_obj=logger,
                        )
                        register_lepton_sf_weight(
                            weights_object,
                            events,
                            "lepSF_taus_fake",
                            "sf_2l_taus_fake",
                            "sf_2l_taus_fake_hi",
                            "sf_2l_taus_fake_lo",
                            include_tau_fake_sf_variations,
                            variation_name=variation_name,
                            logger_obj=logger,
                        )
                elif channel_prefix == "3l":
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_muon",
                        "sf_3l_muon",
                        "sf_3l_hi_muon",
                        "sf_3l_lo_muon",
                        include_muon_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_elec",
                        "sf_3l_elec",
                        "sf_3l_hi_elec",
                        "sf_3l_lo_elec",
                        include_elec_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    if self.tau_h_analysis:
                        register_lepton_sf_weight(
                            weights_object,
                            events,
                            "lepSF_taus_real",
                            "sf_2l_taus_real",
                            "sf_2l_taus_real_hi",
                            "sf_2l_taus_real_lo",
                            include_tau_real_sf_variations,
                            variation_name=variation_name,
                            logger_obj=logger,
                        )
                        register_lepton_sf_weight(
                            weights_object,
                            events,
                            "lepSF_taus_fake",
                            "sf_2l_taus_fake",
                            "sf_2l_taus_fake_hi",
                            "sf_2l_taus_fake_lo",
                            include_tau_fake_sf_variations,
                            variation_name=variation_name,
                            logger_obj=logger,
                        )
                elif channel_prefix == "4l":
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_muon",
                        "sf_4l_muon",
                        "sf_4l_hi_muon",
                        "sf_4l_lo_muon",
                        include_muon_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                    register_lepton_sf_weight(
                        weights_object,
                        events,
                        "lepSF_elec",
                        "sf_4l_elec",
                        "sf_4l_hi_elec",
                        "sf_4l_lo_elec",
                        include_elec_sf_variations,
                        variation_name=variation_name,
                        logger_obj=logger,
                    )
                else:
                    raise Exception(f"Unknown channel name: {nlep_cat}")

            ######### Event weights that do depend on the lep cat ###########

            # Attach the lepton-category specific pieces on top of the
            # previously registered central and kinematic weights.

            if channel_prefix in {"1l", "2l", "3l"}:
                add_fake_factor_weights(
                    weights_object,
                    events,
                    channel_prefix,
                    year,
                    requested_data_weight_label,
                )

            if self._systematic_variations and isData:
                validate_data_weight_variations(
                    weights_object,
                    data_weight_systematics_set,
                    requested_data_weight_label,
                    variation_name,
                )

            # Additional data-only weights
            if isData and channel_prefix == "2l" and ("os" not in self.channel):
                weights_object.add("fliprate", events.flipfactor_2l)

                central_modifiers = getattr(weights_object, "_weights", None).keys()

                if central_modifiers is None or "fliprate" not in set(central_modifiers):
                    raise AssertionError(
                        "The 2l same-sign data branch must register the central 'fliprate' weight."
                    )

            ######### Masks we need for the selection ##########

            # Get mask for events that have two sf os leps close to z peak
            sfosz_3l_OnZ_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:3],pt_window=10.0)
            sfosz_3l_OffZ_mask = ~sfosz_3l_OnZ_mask
            if self.offZ_3l_split:
                sfosz_3l_OffZ_low_mask = tc_es.get_off_Z_mask_low(l_fo_conept_sorted_padded[:,0:3],pt_window=0.0)
                sfosz_3l_OffZ_any_mask = tc_es.get_any_sfos_pair(l_fo_conept_sorted_padded[:,0:3])
            sfosz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=10.0)
            sfasz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="as") # Any sign (do not enforce ss or os here)
            if self.tau_h_analysis:
                tl_zpeak_mask = te_es.lt_Z_mask(l0, l1, tau0, 30.0)

            # Pass trigger mask
            pass_trg = tc_es.trg_pass_no_overlap(
                events,
                isData,
                trigger_dataset,
                str(year),
                te_es.dataset_dict_top22006,
                te_es.exclude_dict_top22006,
            )

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
            if self.tau_h_analysis:
                tau_F_mask = (ak.num(tau[tau["isVLoose"]>0]) >=1)
                tau_L_mask  = (ak.num(tau[tau["isLoose"]>0]) >=1)
                no_tau_mask = (ak.num(tau[tau["isLoose"]>0])==0)


            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint64')
            preselections = PackedSelection(dtype='uint64')
            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)
            preselections.add("is_good_lumi",lumi_mask)

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
            if self.tau_h_analysis:
                preselections.add("1l", (events.is1l & pass_trg))
                preselections.add("1tau", (tau_L_mask))
                preselections.add("1Ftau", (tau_F_mask))
                preselections.add("0tau", (no_tau_mask))
                preselections.add("onZ_tau", (tl_zpeak_mask))
                preselections.add("offZ_tau", (~tl_zpeak_mask))
            if self.fwd_analysis:
                preselections.add("2lss_fwd", (events.is2l & pass_trg & fwdjet_mask))
                preselections.add("2l_fwd_p", (chargel0_p & fwdjet_mask))
                preselections.add("2l_fwd_m", (chargel0_m & fwdjet_mask))

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

            if self.offZ_3l_split:
                preselections.add("3l_offZ_low", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_high", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & ~sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_none", (sfosz_3l_OffZ_mask & ~sfosz_3l_OffZ_any_mask))
            else:
                preselections.add("3l_offZ", (sfosz_3l_OffZ_mask))

            # 4l selection
            preselections.add("4l", (events.is4l & pass_trg))

            # Build the channel mask from the provided channel definition list.
            lep_ch = self._channel_dict["chan_def_lst"]
            tempmask = None
            chtag = lep_ch[0]
            for chcut in lep_ch[1:]:
                tempmask = tempmask & preselections.any(chcut) if tempmask is not None else preselections.any(chcut)
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
            selections.add("atleast_4j", (njets>=4))
            selections.add("atleast_5j", (njets>=5))
            selections.add("atleast_6j", (njets>=6))
            selections.add("atleast_7j", (njets>=7))
            selections.add("atleast_0j", (njets>=0))
            selections.add("atmost_3j" , (njets<=3))

            # AR/SR categories
            selections.add("isSR_2lSS",    ( events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS",    (~events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS_OS", ( events.is2l_SR) & charge2l_0) # Sideband for the charge flip
            selections.add("isSR_2lOS",    ( events.is2l_SR) & charge2l_0)
            selections.add("isAR_2lOS",    (~events.is2l_SR) & charge2l_0)
            if self.tau_h_analysis:
                selections.add("isSR_1l",    ( events.is1l_SR))

            selections.add("isSR_3l",  events.is3l_SR)
            selections.add("isAR_3l", ~events.is3l_SR)
            selections.add("isSR_4l",  events.is4l_SR)

            ######### Variables for the dense axes of the hists ##########

            var_def = self.var_def

            if ("ptbl" in var_def) or ("b0pt" in var_def) or ("bl0pt" in var_def):
                ptbl_bjet = goodJets[(isBtagJetsMedium | isBtagJetsLoose)]
                ptbl_bjet = ptbl_bjet[ak.argmax(ptbl_bjet.pt, axis=-1, keepdims=True)]
                ptbl_lep = l_fo_conept_sorted
                ptbl = (ptbl_bjet.nearest(ptbl_lep) + ptbl_bjet).pt
                ptbl = ak.values_astype(ak.fill_none(ptbl, -1), np.float32)

            if "ptz" in var_def:
                ptz = te_es.get_Z_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
                if self.offZ_3l_split:
                    ptz = te_es.get_ll_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
            if "ptz_wtau" in var_def:
                ptz_wtau = te_es.get_Zlt_pt(l0, l1, tau0)

            if "bl0pt" in var_def:
                bjetsl = goodJets[isBtagJetsLoose][ak.argsort(goodJets[isBtagJetsLoose].pt, axis=-1, ascending=False)]
                bl_pairs = ak.cartesian({"b": bjetsl, "l": l_fo_conept_sorted})
                blpt = (bl_pairs["b"] + bl_pairs["l"]).pt
                bl0pt = ak.flatten(blpt[ak.argmax(blpt, axis=-1, keepdims=True)])

            need_lj_collection = any(t in var_def for t in ["o0pt", "lj0pt", "ljptsum"]) or (self._ecut_threshold is not None)
            if need_lj_collection:
                if self.tau_h_analysis:
                    l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted, goodJets, cleaning_taus], axis=1), "PtEtaPhiMCollection")
                else:
                    l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted, goodJets], axis=1), "PtEtaPhiMCollection")
                if "o0pt" in var_def:
                    o0pt = ak.max(l_j_collection.pt, axis=-1)
                if ("ljptsum" in var_def) or (self._ecut_threshold is not None):
                    ljptsum = ak.sum(l_j_collection.pt, axis=-1)
                if "lj0pt" in var_def:
                    l_j_pairs = ak.combinations(l_j_collection, 2, fields=["o0","o1"])
                    l_j_pairs_pt = (l_j_pairs.o0 + l_j_pairs.o1).pt
                    lj0pt = ak.max(l_j_pairs_pt, axis=-1)

            if "lt" in var_def:
                lt = ak.sum(l_fo_conept_sorted_padded.pt, axis=-1) + met.pt

            if "mll_0_1" in var_def:
                mll_0_1 = (l0 + l1).mass

            if self._ecut_threshold is not None:
                if "ljptsum" not in locals():
                    ljptsum = ak.sum(l_j_collection.pt, axis=-1)
                ecut_mask = (ljptsum < self._ecut_threshold)

            counts = np.ones_like(events['event'])

            ########## Fill the histograms ##########

            dense_axis_name = self._var
            dense_axis_vals = eval(self._var_def, {"ak": ak, "np": np}, locals())

            # Set up the list of systematic weight variations to loop over
            if weight_variations_to_run:
                wgt_var_lst = []
            else:
                wgt_var_lst = ["nominal"]
            for name in weight_variations_to_run:
                if name not in wgt_var_lst:
                    wgt_var_lst.append(name)

            lep_chan = self._channel_dict["chan_def_lst"][0]
            jet_req = self._channel_dict["jet_selection"]
            lep_flav_iter = self._channel_dict["lep_flav_lst"] if self._split_by_lepton_flavor else [None]

            # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            # # print("lep_chan:", lep_chan)
            # # print("jet_req:", jet_req)
            # # print("self._channel_dict:", self._channel_dict)
            # # print("lep_flav_iter:", lep_flav_iter)
            # # print("dense_axis_name:", dense_axis_name)
            # # print("wgt_var_lst:", wgt_var_lst)
            # #print("weights_object.weight:", weights_object.weight())
            # #print(getattr(weights_object, "__dict__"))
            # print("\nweights", getattr(weights_object, "_weights").keys())
            # print("\n",dir(weights_object))
            # print("\nweights_object.variations:", weights_object.variations)
            # print("\n", getattr(weights_object, "_modifiers"))

            # # print("weight_object:", weights_object.weight)
            # # print("hist_variation_label:", hist_variation_label)
            # print("\n\n\n\n\n")

            for wgt_fluct in wgt_var_lst:
                if wgt_fluct == "nominal":
                    weight = weights_object.weight(None)
                elif wgt_fluct in weights_object.variations:
                    weight = weights_object.weight(wgt_fluct)
                else:
                    continue

                # Skip filling SR histograms with data-driven variations
                if self.appregion.startswith("isSR") and wgt_fluct in data_weight_systematics_set:
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

                    print("\n\n\n\n\n\n")
                    print("Filling for channel:", ch_name)
                    print("base channel:", base_ch_name)

                    for cut in cuts_lst:
                        print("  cut:", cut)
                        print("    n passed:", selections.all(cut))

                    all_cuts_mask = selections.all(*cuts_lst)
                    if self._ecut_threshold is not None:
                        all_cuts_mask = (all_cuts_mask & ecut_mask)

                    weights_flat = weight[all_cuts_mask]
                    eft_coeffs_cut = eft_coeffs[all_cuts_mask] if eft_coeffs is not None else None

                    axes_fill_info_dict = {
                        dense_axis_name: dense_axis_vals[all_cuts_mask],
                        "weight": weights_flat,
                        "eft_coeff": eft_coeffs_cut,
                    }

                    histkey = (
                        dense_axis_name,
                        ch_name,
                        self.appregion,
                        dataset,
                        hist_variation_label,
                    )
                
                    if histkey not in hout:
                        fallback_histkey = (
                            dense_axis_name,
                            base_ch_name,
                            self.appregion,
                            dataset,
                            hist_variation_label,
                        )
                        if fallback_histkey not in hout:
                            continue
                        histkey = fallback_histkey
                    
                    hout[histkey].fill(**axes_fill_info_dict)

                    print("\n")
                    print("Filling histkey:", histkey)
                    print("  all_cuts_mask:", all_cuts_mask)
                    print("  with axes_fill_info_dict:", axes_fill_info_dict)
                    print("  dense_axis_vals[all_cuts_mask]", ak.to_list(dense_axis_vals[all_cuts_mask]))
                    #print("  dense_axis_vals", dense_axis_vals)
                    print("\n\n\n\n\n\n\n")

                    axes_fill_info_dict = {
                        dense_axis_name + "_sumw2": dense_axis_vals[all_cuts_mask],
                        "weight": np.square(weights_flat),
                        "eft_coeff": eft_coeffs_cut,
                    }
                    histkey = (
                        dense_axis_name + "_sumw2",
                        base_ch_name,
                        self.appregion,
                        dataset,
                        hist_variation_label,
                    )
                    if histkey not in hout.keys():
                        continue
                    hout[histkey].fill(**axes_fill_info_dict)
        return hout

    def postprocess(self, accumulator):
        return accumulator

