#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
import os
import re

import hist
from topcoffea.modules.histEFT import HistEFT
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.event_selection as tc_es
import topcoffea.modules.object_selection as tc_os
import topcoffea.modules.corrections as tc_cor

from topeft.modules.paths import topeft_path
from topeft.modules.corrections import ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachTauSF, ApplyTES, ApplyTESSystematic, ApplyFESSystematic, AttachPerLeptonFR, ApplyRochesterCorrections, ApplyJetSystematics, GetTriggerSF
import topeft.modules.event_selection as te_es
import topeft.modules.object_selection as te_os
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


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(
        self,
        sample,
        wc_names_lst=[],
        hist_key=None,
        var_info=None,
        ecut_threshold=None,
        do_errors=False,
        do_systematics=False,
        split_by_lepton_flavor=False,
        skip_signal_regions=False,
        skip_control_regions=False,
        muonSyst='nominal',
        dtype=np.float32,
        rebin=False,
        offZ_split=False,
        tau_h_analysis=False,
        fwd_analysis=False,
        channel_dict=None,
        golden_json_path=None,
        systematic_info=None,
        available_systematics=None,
    ):

        self._sample = sample
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self.offZ_3l_split = offZ_split
        self.tau_h_analysis = tau_h_analysis
        self.fwd_analysis = fwd_analysis
        if channel_dict is None:
            raise ValueError("channel_dict must be provided and cannot be None")

        # ``channel_dict`` is expected to be a flat dictionary with keys
        # ``jet_selection``, ``chan_def_lst``, ``lep_flav_lst`` and ``appl_region``.
        # Previous versions of this processor converted this dictionary into a
        # nested structure with several loops in ``process``.  The new logic
        # operates directly on the flat dictionary, so simply store it.
        self._channel_dict = channel_dict
        self._systematic_info = systematic_info
        if available_systematics is None:
            raise ValueError("available_systematics must be provided and cannot be None")
        self._available_systematics = {
            key: tuple(value) for key, value in available_systematics.items()
        }

        histogram = {}

        self._golden_json_path = golden_json_path
        if self._sample.get("isData") and not self._golden_json_path:
            raise ValueError("golden_json_path must be provided for data samples")

        if hist_key is None or var_info is None:
            raise ValueError("hist_key and var_info must be provided and cannot be None")

        var, ch, appl, sample_name, syst = hist_key
        info = var_info

        self._var = var
        self._channel = ch
        self._appregion = appl
        self._syst = syst
        self._var_def = info.get("definition")
        if self._var_def is None:
            raise ValueError(f"No definition provided for variable {var}")

        sumw2_key = (var + "_sumw2", ch, appl, sample_name, syst)

        if not rebin and "variable" in info:
            dense_axis = hist.axis.Variable(
                info["variable"], name=var, label=info["label"]
            )
            sumw2_axis = hist.axis.Variable(
                info["variable"], name=var+"_sumw2", label=info["label"] + " sum of w^2"
            )
        else:
            dense_axis = hist.axis.Regular(
                *info["regular"], name=var, label=info["label"]
            )
            sumw2_axis = hist.axis.Regular(
                *info["regular"], name=var+"_sumw2", label=info["label"] + " sum of w^2"
            )

        histogram[hist_key] = HistEFT(
            dense_axis,
            wc_names=wc_names_lst,
            label=r"Events",
        )
        histogram[sumw2_key] = HistEFT(
            sumw2_axis,
            wc_names=wc_names_lst,
            label=r"Events",
        )

        # Set the list of hists to fill
        self._hist_keys_to_fill = [hist_key, sumw2_key]

        self._accumulator = histogram

        # Set the energy threshold to cut on
        self._ecut_threshold = ecut_threshold

        # Set the booleans
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._do_systematics = do_systematics # Whether to process systematic sample
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories

        if self._do_systematics and self._systematic_info is None:
            raise ValueError("systematic_info must be provided when do_systematics is True")


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

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata["dataset"]
        isEFT   = self._sample["WCnames"] != []

        isData             = self._sample["isData"]
        histAxisName       = self._sample["histAxisName"]
        year               = self._sample["year"]
        xsec               = self._sample["xsec"]
        sow                = self._sample["nSumOfWeights"]

        current_syst = self.syst or "nominal"
        variation_base = self._systematic_info.base if self._systematic_info is not None else None
        variation_type = (
            getattr(self._systematic_info, "type", None)
            if self._systematic_info is not None
            else None
        )

        print("\n\n\n\n\n\n")
        print("current_syst:", current_syst, " variation_base:", variation_base, " variation_type:", variation_type)
        print("\n\n\n\n\n\n")

        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3

        run_era = None
        if isData:
            run_era = self._sample["path"].split("/")[2].split("-")[0][-1]

        # Get up/down sum of weights needed for the current systematic.
        sow_variation_key_map = {}
        requested_sow_variations = set()

        if self._systematic_info is not None and self._do_systematics and not isData:
            group_info = self._systematic_info.group or {}
            if not group_info and self._systematic_info.metadata.get("sum_of_weights"):
                group_info = {
                    self._systematic_info.name: {
                        "sum_of_weights": self._systematic_info.metadata["sum_of_weights"]
                    }
                }
            if group_info:
                requested_sow_variations = set(group_info.keys())
                sow_variation_key_map = {
                    name: info.get("sum_of_weights")
                    for name, info in group_info.items()
                    if info.get("sum_of_weights")
                }

        sow_variations = {"nominal": sow}

        is_lo_sample = histAxisName in get_te_param("lo_xsec_samples")

        if self._do_systematics and not isData:
            for variation in requested_sow_variations:
                if is_lo_sample:
                    sow_variations[variation] = sow
                else:
                    key = sow_variation_key_map.get(variation)
                    if key is not None and key in self._sample:
                        sow_variations[variation] = self._sample[key]

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
            tau["isPres"]  = te_os.isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, tau.idDeepTau2017v2p1VSe, tau.idDeepTau2017v2p1VSmu, minpt=20)
            tau["isClean"] = te_os.isClean(tau, l_fo, drmin=0.3)
            tau["isGood"]  =  tau["isClean"] & tau["isPres"]
            tau = tau[tau.isGood]

            tau['DMflag'] = ((tau.decayMode==0) | (tau.decayMode==1) | (tau.decayMode==10) | (tau.decayMode==11))
            tau = tau[tau['DMflag']]
            tau["isVLoose"]  = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet)
            tau["isLoose"]   = te_os.isLooseTau(tau.idDeepTau2017v2p1VSjet)
            tau["iseTight"]  = te_os.iseTightTau(tau.idDeepTau2017v2p1VSe)
            tau["ismTight"]  = te_os.ismTightTau(tau.idDeepTau2017v2p1VSmu)

            cleaning_taus = tau[tau["isLoose"]>0]
            nLtau  = ak.num(tau[tau["isLoose"]>0] )
            if not isData:
                AttachTauSF(events,tau,year=year)
            tau_padded = ak.pad_none(tau, 1)
            tau0 = tau_padded[:,0]
        else:
            tau["isPres"]  = te_os.isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, tau.idDeepTau2017v2p1VSe, tau.idDeepTau2017v2p1VSmu, minpt=20)
            tau["isClean"] = te_os.isClean(tau, l_loose, drmin=0.3)
            tau["isGood"]  =  tau["isClean"] & tau["isPres"]
            tau = tau[tau.isGood] # use these to clean jets
            tau["isTight"] = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

        ######### Systematics ###########

        # Define the lists of systematics provided by the metadata helper
        object_systematics = tuple(self._available_systematics.get("object", ()))
        weight_systematics = tuple(self._available_systematics.get("weight", ()))
        theory_systematics = tuple(self._available_systematics.get("theory", ()))
        data_weight_systematics = tuple(self._available_systematics.get("data_weight", ()))
        data_weight_systematics_set = set(data_weight_systematics)

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
                genw = np.ones_like(events["event"])

            # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
            # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
            lumi = 1000.0*get_tc_param(f"lumi_{year}")
            weights_obj_base.add("norm",(xsec/sow)*genw*lumi)

            if is_run2:
                l1prefiring_args = [events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn]
            elif is_run3:
                l1prefiring_args = [ak.ones_like(events.nom), ak.ones_like(events.nom), ak.ones_like(events.nom)]

            include_ISR = self._do_systematics and variation_base == "isr"
            include_FSR = self._do_systematics and variation_base == "fsr"
            need_ps_weights = include_ISR or include_FSR
            if need_ps_weights:
                # Attach PS weights (ISR/FSR)
                tc_cor.AttachPSWeights(events)  # Run3 ready

            def get_sow_value(label):
                if label in sow_variations:
                    return sow_variations[label]

                if is_lo_sample:
                    sow_variations[label] = sow
                    return sow

                key = sow_variation_key_map.get(label)
                if key is None:
                    raise KeyError(
                        f"Unsupported sum-of-weights variation '{label}' requested while processing '{current_syst}'"
                    )

                if key not in self._sample:
                    raise KeyError(
                        f"Sample '{histAxisName}' is missing required sum-of-weights entry '{key}' for '{label}' variation"
                    )

                value = self._sample[key]
                sow_variations[label] = value
                return value

            if include_ISR:
                sow_ISRUp = get_sow_value("ISRUp")
                sow_ISRDown = get_sow_value("ISRDown")
                weights_obj_base.add(
                    "ISR",
                    events.nom,
                    events.ISRUp * (sow / sow_ISRUp),
                    events.ISRDown * (sow / sow_ISRDown),
                )
            else:
                weights_obj_base.add("ISR", events.nom)

            if include_FSR:
                sow_FSRUp = get_sow_value("FSRUp")
                sow_FSRDown = get_sow_value("FSRDown")
                weights_obj_base.add(
                    "FSR",
                    events.nom,
                    events.FSRUp * (sow / sow_FSRUp),
                    events.FSRDown * (sow / sow_FSRDown),
                )
            else:
                weights_obj_base.add("FSR", events.nom)

            include_renorm = self._do_systematics and variation_base == "renorm"
            include_fact = self._do_systematics and variation_base == "fact"
            # Attach renorm/fact scale weights regardless of which variations are being evaluated
            tc_cor.AttachScaleWeights(events)  # Run3 ready (with caveat on "nominal")

            if include_renorm:
                sow_renormUp = get_sow_value("renormUp")
                sow_renormDown = get_sow_value("renormDown")
                weights_obj_base.add(
                    "renorm",
                    events.nom,
                    events.renormUp * (sow / sow_renormUp),
                    events.renormDown * (sow / sow_renormDown),
                )
            else:
                weights_obj_base.add("renorm", events.nom)

            if include_fact:
                sow_factUp = get_sow_value("factUp")
                sow_factDown = get_sow_value("factDown")
                weights_obj_base.add(
                    "fact",
                    events.nom,
                    events.factUp * (sow / sow_factUp),
                    events.factDown * (sow / sow_factDown),
                )
            else:
                weights_obj_base.add("fact", events.nom)

            # Prefiring and PU (note prefire weights only available in nanoAODv9 and for Run2)
            include_prefiring_vars = self._do_systematics and variation_base == "prefiring"
            if include_prefiring_vars:
                weights_obj_base.add("PreFiring", *l1prefiring_args)  # Run3 ready
            else:
                weights_obj_base.add("PreFiring", l1prefiring_args[0])

            pu_central = tc_cor.GetPUSF(events.Pileup.nTrueInt, year)
            include_pu_vars = self._do_systematics and variation_base == "pileup"
            if include_pu_vars:
                weights_obj_base.add(
                    "PU",
                    pu_central,
                    tc_cor.GetPUSF(events.Pileup.nTrueInt, year, "up"),
                    tc_cor.GetPUSF(events.Pileup.nTrueInt, year, "down"),
                )  # Run3 ready
            else:
                weights_obj_base.add("PU", pu_central)

        current_variation_name = current_syst
        object_variation = "nominal"
        weight_variations_to_run = ["nominal"]

        if variation_type == "object":
            if current_variation_name not in object_systematics:
                raise ValueError(
                    f"Requested object systematic '{current_variation_name}' is not available in the mapping"
                )
            object_variation = current_variation_name
        elif variation_type in {"weight", "theory", "data_weight"} and current_variation_name != "nominal":
            variation_pool = {
                "weight": weight_systematics,
                "theory": theory_systematics,
                "data_weight": data_weight_systematics,
            }[variation_type]
            if current_variation_name in variation_pool:
                weight_variations_to_run = [current_variation_name]
            else:
                raise ValueError(
                    f"Requested {variation_type} systematic '{current_variation_name}' is not available in the mapping"
                )

        print("\n\n\n\n")
        print("Running object systematic:", object_variation)
        print("Running weight systematics:", weight_variations_to_run)
        print("\n\n\n\n")
        
        # Make a copy of the base weights object.  In this block we add the pieces that depend on the object kinematics.
        met_raw = met
        weights_obj_base_for_kinematic_syst = copy.deepcopy(weights_obj_base)

        trigger_weight_label = f"triggerSF_{year}"

        #################### Jets ####################

        # Jet cleaning, before any jet selection
        #vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
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

            btag_effM_light = GetBtagEff(jets_light, year, 'medium')
            btag_effM_bc = GetBtagEff(jets_bc, year, 'medium')
            btag_effL_light = GetBtagEff(jets_light, year, 'loose')
            btag_effL_bc = GetBtagEff(jets_bc, year, 'loose')
            btag_sfM_light = tc_cor.btag_sf_eval(jets_light, "M",year_light,"deepJet_incl","central")
            btag_sfM_bc    = tc_cor.btag_sf_eval(jets_bc,    "M",year,      "deepJet_comb","central")
            btag_sfL_light = tc_cor.btag_sf_eval(jets_light, "L",year_light,"deepJet_incl","central")
            btag_sfL_bc    = tc_cor.btag_sf_eval(jets_bc,    "L",year,      "deepJet_comb","central")

            pData_light, pMC_light = tc_cor.get_method1a_wgt_doublewp(btag_effM_light, btag_effL_light, btag_sfM_light, btag_sfL_light, isBtagJetsMedium[light_mask], isBtagJetsLooseNotMedium[light_mask], isNotBtagJetsLoose[light_mask])
            btag_w_light = pData_light/pMC_light
            pData_bc, pMC_bc = tc_cor.get_method1a_wgt_doublewp(btag_effM_bc, btag_effL_bc, btag_sfM_bc, btag_sfL_bc, isBtagJetsMedium[bc_mask], isBtagJetsLooseNotMedium[bc_mask], isNotBtagJetsLoose[bc_mask])
            btag_w_bc = pData_bc/pMC_bc
            btag_w = btag_w_light*btag_w_bc
            weights_obj_base_for_kinematic_syst.add("btagSF", btag_w)

            if self._do_systematics and object_variation == "nominal":
                requested_suffix = None
                if current_variation_name and current_variation_name.startswith("btagSF"):
                    requested_suffix = current_variation_name[len("btagSF"):]

                if requested_suffix:
                    directionless_suffix = requested_suffix.rstrip("Up").rstrip("Down")

                    corrtype = (
                        "correlated" if directionless_suffix.endswith("_corr") else "uncorrelated"
                    )

                    if requested_suffix.startswith("light_"):
                        jets_flav = jets_light
                        flav_mask = light_mask
                        sys_year = year_light
                        dJ_tag = "incl"
                        btag_effM = btag_effM_light
                        btag_effL = btag_effL_light
                        pMC_flav = pMC_light
                        fixed_btag_w = btag_w_bc
                    else:
                        jets_flav = jets_bc
                        flav_mask = bc_mask
                        sys_year = year
                        dJ_tag = "comb"
                        btag_effM = btag_effM_bc
                        btag_effL = btag_effL_bc
                        pMC_flav = pMC_bc
                        fixed_btag_w = btag_w_light

                    btag_sfL_up = tc_cor.btag_sf_eval(
                        jets_flav, "L", sys_year, f"deepJet_{dJ_tag}", f"up_{corrtype}"
                    )
                    btag_sfL_down = tc_cor.btag_sf_eval(
                        jets_flav, "L", sys_year, f"deepJet_{dJ_tag}", f"down_{corrtype}"
                    )
                    btag_sfM_up = tc_cor.btag_sf_eval(
                        jets_flav, "M", sys_year, f"deepJet_{dJ_tag}", f"up_{corrtype}"
                    )
                    btag_sfM_down = tc_cor.btag_sf_eval(
                        jets_flav, "M", sys_year, f"deepJet_{dJ_tag}", f"down_{corrtype}"
                    )

                    pData_up, pMC_up = tc_cor.get_method1a_wgt_doublewp(
                        btag_effM,
                        btag_effL,
                        btag_sfM_up,
                        btag_sfL_up,
                        isBtagJetsMedium[flav_mask],
                        isBtagJetsLooseNotMedium[flav_mask],
                        isNotBtagJetsLoose[flav_mask],
                    )

                    pData_down, pMC_down = tc_cor.get_method1a_wgt_doublewp(
                        btag_effM,
                        btag_effL,
                        btag_sfM_down,
                        btag_sfL_down,
                        isBtagJetsMedium[flav_mask],
                        isBtagJetsLooseNotMedium[flav_mask],
                        isNotBtagJetsLoose[flav_mask],
                    )

                    btag_w_up = fixed_btag_w * (pData_up / pMC_flav) / btag_w
                    btag_w_down = fixed_btag_w * (pData_down / pMC_flav) / btag_w

                    variation_label_base = current_variation_name
                    if variation_label_base:
                        for _direction in ("Up", "Down"):
                            if variation_label_base.endswith(_direction):
                                variation_label_base = variation_label_base[: -len(_direction)]
                                break
                    else:
                        variation_label_base = f"btagSF{directionless_suffix}"

                    weights_obj_base_for_kinematic_syst.add(
                        variation_label_base,
                        events.nom,
                        btag_w_up,
                        btag_w_down,
                    )

        # Trigger SFs
        GetTriggerSF(year, events, l0, l1)
        include_trigger_vars = (
            self._do_systematics
            and not isData
            and variation_base == "trigger_sf"
        )
        if include_trigger_vars:
            weights_obj_base_for_kinematic_syst.add(
                trigger_weight_label,
                events.trigger_sf,
                copy.deepcopy(events.trigger_sfUp),
                copy.deepcopy(events.trigger_sfDown),
            )
        else:
            weights_obj_base_for_kinematic_syst.add(
                trigger_weight_label,
                events.trigger_sf,
            )

        ######### Event weights that do depend on the lep cat ###########
        # Determine the lepton multiplicity category from the requested
        # channel name (e.g. ``2lss_4j`` -> ``2l``).  This is used to know
        # which set of weights to apply.
        nlep_cat = re.match(r"\d+l", self.channel).group(0)

        # Start from the base set of weights and attach the
        # lepton-category specific pieces.  The weights object for the
        # requested ``nlep_cat`` is fetched from a dictionary so that this
        # happens once per systematic variation.
        weights_dict = {nlep_cat: copy.deepcopy(weights_obj_base_for_kinematic_syst)}
        weights_object = weights_dict[nlep_cat]

        if nlep_cat.startswith("1l"):
            weights_object.add("FF", events.fakefactor_1l, copy.deepcopy(events.fakefactor_1l_up), copy.deepcopy(events.fakefactor_1l_down))
            weights_object.add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_1l_pt1/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_pt2/events.fakefactor_1l))
            weights_object.add("FFeta", events.nom, copy.deepcopy(events.fakefactor_1l_be1/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_be2/events.fakefactor_1l))
            weights_object.add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_1l_elclosureup/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_elclosuredown/events.fakefactor_1l))
            weights_object.add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_1l_muclosureup/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_muclosuredown/events.fakefactor_1l))
        elif nlep_cat.startswith("2l"):
            weights_object.add("FF", events.fakefactor_2l, copy.deepcopy(events.fakefactor_2l_up), copy.deepcopy(events.fakefactor_2l_down))
            weights_object.add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_2l_pt1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_pt2/events.fakefactor_2l))
            weights_object.add("FFeta", events.nom, copy.deepcopy(events.fakefactor_2l_be1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_be2/events.fakefactor_2l))
            weights_object.add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_elclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_elclosuredown/events.fakefactor_2l))
            weights_object.add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_muclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_muclosuredown/events.fakefactor_2l))
        elif nlep_cat.startswith("3l"):
            weights_object.add("FF", events.fakefactor_3l, copy.deepcopy(events.fakefactor_3l_up), copy.deepcopy(events.fakefactor_3l_down))
            weights_object.add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_3l_pt1/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_pt2/events.fakefactor_3l))
            weights_object.add("FFeta", events.nom, copy.deepcopy(events.fakefactor_3l_be1/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_be2/events.fakefactor_3l))
            weights_object.add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_3l_elclosureup/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_elclosuredown/events.fakefactor_3l))
            weights_object.add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_3l_muclosureup/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_muclosuredown/events.fakefactor_3l))

        # Additional data-only weights
        if isData and nlep_cat.startswith("2l") and ("os" not in self.channel):
            weights_object.add("fliprate", events.flipfactor_2l)

        # MC-only scale factors
        if not isData:
            if nlep_cat.startswith("1l"):
                weights_object.add("lepSF_muon", events.sf_1l_muon, copy.deepcopy(events.sf_1l_hi_muon), copy.deepcopy(events.sf_1l_lo_muon))
                weights_object.add("lepSF_elec", events.sf_1l_elec, copy.deepcopy(events.sf_1l_hi_elec), copy.deepcopy(events.sf_1l_lo_elec))
                if self.tau_h_analysis:
                    weights_object.add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                    weights_object.add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
            elif nlep_cat.startswith("2l"):
                weights_object.add("lepSF_muon", events.sf_2l_muon, copy.deepcopy(events.sf_2l_hi_muon), copy.deepcopy(events.sf_2l_lo_muon))
                weights_object.add("lepSF_elec", events.sf_2l_elec, copy.deepcopy(events.sf_2l_hi_elec), copy.deepcopy(events.sf_2l_lo_elec))
                if self.tau_h_analysis:
                    weights_object.add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                    weights_object.add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
            elif nlep_cat.startswith("3l"):
                weights_object.add("lepSF_muon", events.sf_3l_muon, copy.deepcopy(events.sf_3l_hi_muon), copy.deepcopy(events.sf_3l_lo_muon))
                weights_object.add("lepSF_elec", events.sf_3l_elec, copy.deepcopy(events.sf_3l_hi_elec), copy.deepcopy(events.sf_3l_lo_elec))
                if self.tau_h_analysis:
                    weights_object.add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                    weights_object.add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
            elif nlep_cat.startswith("4l"):
                weights_object.add("lepSF_muon", events.sf_4l_muon, copy.deepcopy(events.sf_4l_hi_muon), copy.deepcopy(events.sf_4l_lo_muon))
                weights_object.add("lepSF_elec", events.sf_4l_elec, copy.deepcopy(events.sf_4l_hi_elec), copy.deepcopy(events.sf_4l_lo_elec))
            else:
                raise Exception(f"Unknown channel name: {nlep_cat}")

        # Ensure that for data we only have the expected systematic
        # variations in the Weights object
        if self._do_systematics and isData:
            expected_vars = set(data_weight_systematics_set)
            if nlep_cat.startswith("2l") and ("os" not in self.channel):
                expected_vars.add("fliprate")
            if weights_object.variations != expected_vars:
                raise Exception(
                    f"Error: Unexpected wgt variations for data! Expected \"{expected_vars}\" but have \"{weights_object.variations}\"."
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
        pass_trg = tc_es.trg_pass_no_overlap(events,isData,dataset,str(year),te_es.dataset_dict_top22006,te_es.exclude_dict_top22006)

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
        if self._do_systematics:
            wgt_var_lst = weight_variations_to_run
        else:
            wgt_var_lst = ["nominal"]

        hist_variation_label = current_variation_name or "nominal"

        lep_chan = self._channel_dict["chan_def_lst"][0]
        jet_req = self._channel_dict["jet_selection"]
        lep_flav_iter = self._channel_dict["lep_flav_lst"] # if self._split_by_lepton_flavor else [None]

        # print("\n\n\n\n\n")
        # print("lep_chan:", lep_chan)
        # print("jet_req:", jet_req)
        # print("self._channel_dict:", self._channel_dict)
        # print("lep_flav_iter:", lep_flav_iter)
        # print("dense_axis_name:", dense_axis_name)
        # print("wgt_var_lst:", wgt_var_lst)
        # print("weights_object.variations:", weights_object.variations)
        # print("weight_object:", weights_object.weight)
        # print("hist_variation_label:", hist_variation_label)
        # print("\n\n\n\n\n")

        for wgt_fluct in wgt_var_lst:
            if wgt_fluct == "nominal" or wgt_fluct == object_variation:
                weight = weights_object.weight(None)
            elif wgt_fluct in weights_object.variations:
                weight = weights_object.weight(wgt_fluct)
            else:
                continue

            # Skip filling SR histograms with data-driven variations
            if self.appregion.startswith("isSR") and wgt_fluct in data_weight_systematics_set:
                continue

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

                ch_name = construct_cat_name(lep_chan, njet_str=njet_ch, flav_str=flav_ch)
                if ch_name != self.channel:
                    continue

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

                # Skip histos that are not defined (or not relevant) to given categories
                if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & (("CRZ" in ch_name) or ("CRflip" in ch_name))):
                    continue
                if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & ("0j" in ch_name)):
                    continue
                if self.offZ_3l_split:
                    if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan) & ("offZ_high" not in lep_chan) & ("offZ_low" not in lep_chan)):
                        continue
                elif self.tau_h_analysis:
                    if (("ptz" in dense_axis_name) and ("onZ" not in lep_chan)):
                        continue
                    if (("ptz" in dense_axis_name) and ("2lss" in lep_chan) and ("ptz_wtau" not in dense_axis_name)):
                        continue
                    if (("ptz_wtau" in dense_axis_name) and (("1tau" not in lep_chan) or ("onZ" not in lep_chan) or ("2lss" not in lep_chan))):
                        continue
                elif self.fwd_analysis:
                    if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)):
                        continue
                    if (("lt" in dense_axis_name) and ("2lss" not in lep_chan)):
                        continue
                else:
                    if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)):
                        continue
                if ((dense_axis_name in ["o0pt","b0pt","bl0pt"]) & ("CR" in ch_name)):
                    continue

                histkey = (
                    dense_axis_name,
                    ch_name,
                    self.appregion,
                    dataset,
                    hist_variation_label,
                )
                
                if histkey not in hout.keys():
                    continue
                hout[histkey].fill(**axes_fill_info_dict)

                axes_fill_info_dict = {
                    dense_axis_name + "_sumw2": dense_axis_vals[all_cuts_mask],
                    "weight": np.square(weights_flat),
                    "eft_coeff": eft_coeffs_cut,
                }
                histkey = (
                    dense_axis_name + "_sumw2",
                    ch_name,
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

