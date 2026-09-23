import os
import shutil
import uproot
import hist
import json
import numpy as np
import boost_histogram as bh
import topcoffea.modules.utils as utils

# Just brute force renaming map
SYST_NAMING_MAP = {
    "run2": {
        # Luminosity
        "lumi": "lumi_13TeV_correlated",
        # PDF rate systematics
        "pdf_scale_gg": "cross_section_pdf_gg",
        "pdf_scale_qg": "cross_section_pdf_gq",
        "pdf_scale_qq": "cross_section_pdf_qqbar",
        # QCD Scale rate systematics --> These are really systematics on the theory xsec normalization, right?
        "qcd_scale_V": "cross_section_qcd_scale_V",
        "qcd_scale_VV": "cross_section_qcd_scale_VV",
        "qcd_scale_VVV": "cross_section_qcd_scale_VVV",
        "qcd_scale_tHq": "cross_section_qcd_scale_tHq",
        "qcd_scale_tWZ": "cross_section_qcd_scale_tWZ",
        "qcd_scale_ttH": "cross_section_qcd_scale_ttH",
        "qcd_scale_ttll": "cross_section_qcd_scale_ttll",
        "qcd_scale_ttlnu": "cross_section_qcd_scale_ttlnu",
        "qcd_scale_tttt": "cross_section_qcd_scale_tttt",
        # JER systematics
        "JER_2016APV": "CMS_res_j_2016preVFP",
        "JER_2016": "CMS_res_j_2016postVFP",
        "JER_2017": "CMS_res_j_2017",
        "JER_2018": "CMS_res_j_2018",
        # JES (ad-hoc) systematics
        "JES_Absolute": "CMS_scale_j_Absolute_13TeV",   # Should this be (un-)correlated with run2/3?
        "JES_BBEC1": "CMS_scale_j_BBEC1_13TeV",   # Should this be (un-)correlated with run2/3?
        "JES_FlavorPureBottom": "CMS_scale_j_FlavorPureBottom",
        "JES_FlavorPureCharm": "CMS_scale_j_FlavorPureCharm",
        "JES_FlavorPureGluon": "CMS_scale_j_FlavorPureGluon",
        "JES_FlavorPureQuark": "CMS_scale_j_FlavorPureQuark",
        "JES_FlavorQCD": "CMS_scale_j_FlavorQCD_13TeV", # Should this be (un-)correlated with run2/3?
        "JES_RelativeBal": "CMS_scale_j_RelativeBal_13TeV",   # Should this be (un-)correlated with run2/3?
        "JES_RelativeSample": "CMS_scale_j_RelativeSample", # Should this be uncorrelated across years, as was done for run3?
        # MET
        "MET_UnclusteredEnergy": "CMS_scale_met_Unclustred_energy",  # Unclear on if should be correlated across years
        # QCD Factorization (shape)
        "fact_Diboson": "QCDscale_fac_VV_ACCEPT",
        "fact_Triboson": "QCDscale_fac_VVV_ACCEPT",
        "fact_convs": "QCDscale_fac_convs_ACCEPT",
        "fact_tWZ": "QCDscale_fac_tWZ_ACCEPT",
        "fact_tHq": "QCDscale_fac_tHq_ACCEPT",
        "fact_tllq": "QCDscale_fac_tllq_ACCEPT",
        "fact_ttH": "QCDscale_fac_ttH_ACCEPT",
        "fact_ttll": "QCDscale_fac_ttll_ACCEPT",
        "fact_ttlnu": "QCDscale_fac_ttlnu_ACCEPT",
        "fact_tttt": "QCDscale_fac_tttt_ACCEPT",
        # QCD Renormalization (shape)
        "renorm_Diboson": "QCDscale_ren_VV_ACCEPT",
        "renorm_Triboson": "QCDscale_ren_VVV_ACCEPT",
        "renorm_convs": "QCDscale_ren_convs_ACCEPT",
        "renorm_tWZ": "QCDscale_ren_tWZ_ACCEPT",
        "renorm_tHq": "QCDscale_ren_tHq_ACCEPT",
        "renorm_tllq": "QCDscale_ren_tllq_ACCEPT",
        "renorm_ttH": "QCDscale_ren_ttH_ACCEPT",
        "renorm_ttll": "QCDscale_ren_ttll_ACCEPT",
        "renorm_ttlnu": "QCDscale_ren_ttlnu_ACCEPT",
        "renorm_tttt": "QCDscale_ren_tttt_ACCEPT",
        # ISR
        "ISR": "ps_isr",
        "ISR_gg": "ps_isr_gg",
        "ISR_qg": "ps_isr_qg",
        "ISR_qq": "ps_isr_qq",
        # FSR
        "FSR": "ps_fsr",
        # b-tag
        "btagSFbc_corr_run2": "CMS_btag_fixedWP_bc_correlated_13TeV", # Should this be (un-)correlated between run2/3?
        "btagSFbc_2016APV": "CMS_btag_fixedWP_bc_uncorrelated_2016preVFP",
        "btagSFbc_2016": "CMS_btag_fixedWP_bc_uncorrelated_2016postVFP",
        "btagSFbc_2017": "CMS_btag_fixedWP_bc_uncorrelated_2017",
        "btagSFbc_2018": "CMS_btag_fixedWP_bc_uncorrelated_2018",
        "btagSFlight_corr_run2": "CMS_btag_fixedWP_light_correlated_13TeV",   # Should this be (un-)correlated between run2/3?
        "btagSFlight_2016APV": "CMS_btag_fixedWP_light_uncorrelated_2016preVFP",
        "btagSFlight_2016": "CMS_btag_fixedWP_light_uncorrelated_2016postVFP",
        "btagSFlight_2017": "CMS_btag_fixedWP_light_uncorrelated_2017",
        "btagSFlight_2018": "CMS_btag_fixedWP_light_uncorrelated_2018",
        # PU
        "PU": "CMS_pileup",
        # L1 Prefiring
        "PreFiring": "CMS_l1_ecal_prefiring",
        # Taus
        "TES_run2": "CMS_scale_t_13TeV",
        "FES_run2": "CMS_fake_t_DeepTau2017v2p1",   # We don't distinguish between VSe and VSmu corrections
        "lepSF_taus_fake_run2": "CMS_TOP26006_eff_fake_t_13TeV",
        "lepSF_taus_real_run2": "CMS_TOP26006_eff_real_t_13TeV",
        # lepSF
        "lepSF_elec_run2": "CMS_eff_e_13TeV",   # To check
        "lepSF_muon_run2": "CMS_eff_m_13TeV",   # To check
        # triggerSF
        "triggerSF_2016APV": "CMS_TOP26006_eff_trigger_2016preVFP",
        "triggerSF_2016": "CMS_TOP26006_eff_trigger_2016postVFP",
        "triggerSF_2017": "CMS_TOP26006_eff_trigger_2017",
        "triggerSF_2018": "CMS_TOP26006_eff_trigger_2018",
        # Fakes
        "FF_run2": "CMS_TOP26006_FF_13TeV",
        "FFeta_run2": "CMS_TOP26006_FFeta_13TeV",
        "FFpt_run2": "CMS_TOP26006_FFpt_13TeV",
        "FFcloseEl_2016": "CMS_TOP26006_FFcloseEl_2016",
        "FFcloseEl_2017": "CMS_TOP26006_FFcloseEl_2017",
        "FFcloseEl_2018": "CMS_TOP26006_FFcloseEl_2018",
        "FFcloseMu_2016": "CMS_TOP26006_FFcloseMu_2016",
        "FFcloseMu_2017": "CMS_TOP26006_FFcloseMu_2017",
        "FFcloseMu_2018": "CMS_TOP26006_FFcloseMu_2018",
        # Charge flips
        "charge_flips": "CMS_TOP26006_charge_flips_13TeV",
        # Other
        "diboson_njets": "CMS_TOP26006_njets_VV",
        "missing_parton": "CMS_TOP26006_missing_parton",
    },
    "run3": {
        # Luminosity
        "lumi_run3": "lumi_13p6TeV_2223",
        # PDF rate systematics
        "pdf_scale_gg": "cross_section_pdf_gg",
        "pdf_scale_qg": "cross_section_pdf_gq",
        "pdf_scale_qq": "cross_section_pdf_qqbar",
        # QCD Scale rate systematics --> These are really systematics on the theory xsec normalization, right?
        "qcd_scale_V": "cross_section_qcd_scale_V",
        "qcd_scale_VV": "cross_section_qcd_scale_VV",
        "qcd_scale_VVV": "cross_section_qcd_scale_VVV",
        "qcd_scale_tHq": "cross_section_qcd_scale_tHq",
        "qcd_scale_tWZ": "cross_section_qcd_scale_tWZ",
        "qcd_scale_ttH": "cross_section_qcd_scale_ttH",
        "qcd_scale_ttll": "cross_section_qcd_scale_ttll",
        "qcd_scale_ttlnu": "cross_section_qcd_scale_ttlnu",
        "qcd_scale_tttt": "cross_section_qcd_scale_tttt",
        # JER systematics
        "JER_2022": "CMS_res_j_2022",
        "JER_2022EE": "CMS_res_j_2022EE",
        "JER_2023": "CMS_res_j_2023",
        "JER_2023BPix": "CMS_res_j_2023BPix",
        # JES (reduced) systematics
        "JES_Regrouped_Absolute": "CMS_scale_j_Absolute_13p6TeV",   # Should this be (un-)correlated with run2/3?
        "JES_Regrouped_Absolute_2022": "CMS_scale_j_Absolute_2022",
        "JES_Regrouped_Absolute_2022EE": "CMS_scale_j_Absolute_2022EE",
        "JES_Regrouped_Absolute_2023": "CMS_scale_j_Absolute_2023",
        "JES_Regrouped_Absolute_2023BPix": "CMS_scale_j_Absolute_2023BPix",
        "JES_Regrouped_BBEC1": "CMS_scale_j_BBEC1_13p6TeV", # Should this be (un-)correlated with run2/3?
        "JES_Regrouped_BBEC1_2022": "CMS_scale_j_BBEC1_2022",
        "JES_Regrouped_BBEC1_2022EE": "CMS_scale_j_BBEC1_2022EE",
        "JES_Regrouped_BBEC1_2023": "CMS_scale_j_BBEC1_2023",
        "JES_Regrouped_BBEC1_2023BPix": "CMS_scale_j_BBEC1_2023BPix",
        "JES_Regrouped_EC2": "CMS_scale_j_EC2",
        "JES_Regrouped_EC2_2022": "CMS_scale_j_EC2_2022",
        "JES_Regrouped_EC2_2022EE": "CMS_scale_j_EC2_2022EE",
        "JES_Regrouped_EC2_2023": "CMS_scale_j_EC2_2023",
        "JES_Regrouped_EC2_2023BPix": "CMS_scale_j_EC2_2023BPix",
        "JES_Regrouped_FlavorQCD": "CMS_scale_j_FlavorQCD_13p6TeV", # Should this be (un-)correlated with run2/3?
        "JES_Regrouped_HF": "CMS_scale_j_HF",
        "JES_Regrouped_HF_2022": "CMS_scale_j_HF_2022",
        "JES_Regrouped_HF_2022EE": "CMS_scale_j_HF_2022EE",
        "JES_Regrouped_HF_2023": "CMS_scale_j_HF_2023",
        "JES_Regrouped_HF_2023BPix": "CMS_scale_j_HF_2023BPix",
        "JES_Regrouped_RelativeBal": "CMS_scale_j_RelativeBal_13p6TeV", # Should this be (un-)correlated with run2/3?
        "JES_Regrouped_RelativeSample_2022": "CMS_scale_j_RelativeSample_2022",
        "JES_Regrouped_RelativeSample_2022EE": "CMS_scale_j_RelativeSample_2022EE",
        "JES_Regrouped_RelativeSample_2023": "CMS_scale_j_RelativeSample_2023",
        "JES_Regrouped_RelativeSample_2023BPix": "CMS_scale_j_RelativeSample_2023BPix",
        # MET
        "MET_UnclusteredEnergy": "CMS_scale_met_Unclustred_energy",  # Unclear on if should be correlated across years
        # Muon systematics
        "MuonResolution": "CMS_res_m_13p6TeV",
        "MuonScale": "CMS_scale_m_13p6TeV",
        # QCD Factorization (shape)
        "fact_Diboson": "QCDscale_fac_VV_ACCEPT",
        "fact_Triboson": "QCDscale_fac_VVV_ACCEPT",
        "fact_convs": "QCDscale_fac_convs_ACCEPT",
        "fact_tWZ": "QCDscale_fac_tWZ_ACCEPT",
        "fact_tHq": "QCDscale_fac_tHq_ACCEPT",
        "fact_tllq": "QCDscale_fac_tllq_ACCEPT",
        "fact_ttH": "QCDscale_fac_ttH_ACCEPT",
        "fact_ttll": "QCDscale_fac_ttll_ACCEPT",
        "fact_ttlnu": "QCDscale_fac_ttlnu_ACCEPT",
        "fact_tttt": "QCDscale_fac_tttt_ACCEPT",
        # QCD Renormalization (shape)
        "renorm_Diboson": "QCDscale_ren_VV_ACCEPT",
        "renorm_Triboson": "QCDscale_ren_VVV_ACCEPT",
        "renorm_convs": "QCDscale_ren_convs_ACCEPT",
        "renorm_tWZ": "QCDscale_ren_tWZ_ACCEPT",
        "renorm_tHq": "QCDscale_ren_tHq_ACCEPT",
        "renorm_tllq": "QCDscale_ren_tllq_ACCEPT",
        "renorm_ttH": "QCDscale_ren_ttH_ACCEPT",
        "renorm_ttll": "QCDscale_ren_ttll_ACCEPT",
        "renorm_ttlnu": "QCDscale_ren_ttlnu_ACCEPT",
        "renorm_tttt": "QCDscale_ren_tttt_ACCEPT",
        # ISR
        "ISR": "ps_isr",
        "ISR_gg": "ps_isr_gg",
        "ISR_qg": "ps_isr_qg",
        "ISR_qq": "ps_isr_qq",
        # FSR
        "FSR": "ps_fsr",
        # b-tag
        "btagSFbc_corr_run3": "CMS_btag_fixedWP_bc_correlated_13p6TeV", # Should this be (un-)correlated between run2/3?
        "btagSFbc_2022": "CMS_btag_fixedWP_bc_uncorrelated_2022",
        "btagSFbc_2022EE": "CMS_btag_fixedWP_bc_uncorrelated_2022EE",
        "btagSFbc_2023": "CMS_btag_fixedWP_bc_uncorrelated_2023",
        "btagSFbc_2023BPix": "CMS_btag_fixedWP_bc_uncorrelated_2023BPix",
        "btagSFlight_corr_run3": "CMS_btag_fixedWP_light_correlated_13p6TeV",   # Should this be (un-)correlated between run2/3?
        "btagSFlight_2022": "CMS_btag_fixedWP_light_uncorrelated_2022",
        "btagSFlight_2022EE": "CMS_btag_fixedWP_light_uncorrelated_2022EE",
        "btagSFlight_2023": "CMS_btag_fixedWP_light_uncorrelated_2023",
        "btagSFlight_2023BPix": "CMS_btag_fixedWP_light_uncorrelated_2023BPix",
        # PU
        "PU": "CMS_pileup",
        # L1 Prefiring
        "PreFiring": "CMS_l1_ecal_prefiring",   # Exists in Run3, but variations are just set to nominal
        # Taus
        "TES_run3": "CMS_scale_t_13p6TeV",
        "FES_run3": "CMS_fake_t_DeepTau2018v2p5",   # We don't distinguish between VSe and VSmu corrections?
        "lepSF_taus_fake_run3": "CMS_TOP26006_eff_fake_t_13p6TeV",
        "lepSF_taus_real_run3": "CMS_TOP26006_eff_real_t_13p6TeV",
        # lepSF
        "lepSF_elec_run3": "CMS_eff_e_13p6TeV", # To check
        "lepSF_muon_run3": "CMS_eff_m_13p6TeV", # To check
        # triggerSF
        "triggerSF_2022": "CMS_TOP26006_eff_trigger_2022",
        "triggerSF_2022EE": "CMS_TOP26006_eff_trigger_2022EE",
        "triggerSF_2023": "CMS_TOP26006_eff_trigger_2023",
        "triggerSF_2023BPix": "CMS_TOP26006_eff_trigger_2023BPix",
        # Fakes
        "FF_run3": "CMS_TOP26006_FF_13p6TeV",
        "FFeta_run3": "CMS_TOP26006_FFeta_13p6TeV",
        "FFpt_run3": "CMS_TOP26006_FFpt_13p6TeV",
        "FFcloseEl_2022": "CMS_TOP26006_FFcloseEl_2022",
        "FFcloseEl_2023": "CMS_TOP26006_FFcloseEl_2023",
        "FFcloseMu_2022": "CMS_TOP26006_FFcloseMu_2022",
        "FFcloseMu_2023": "CMS_TOP26006_FFcloseMu_2023",
        # Charge flips
        "charge_flips_run3": "CMS_TOP26006_charge_flips_13p6TeV",
        # Other
        "diboson_njets": "CMS_TOP26006_njets_VV",
        "missing_parton": "CMS_TOP26006_missing_parton",
    },
}

def create_directory(out_dir,clean=None):
    if ".." in out_dir:
        raise RuntimeError("Do not use '..' in the output directory path, it can confuse the call to os.makedirs()")
    if not os.path.exists(out_dir):
        print(f"Creating directory: {out_dir}")
        os.makedirs(out_dir)
    if clean:
        print(f"Cleaning {out_dir}")
        utils.clean_dir(out_dir,targets=clean,dry_run=False)

def parse_datacard(fpath):
    shape_systs = []
    rate_systs = []
    # Maybe not a good idea, but we're going to store all the text in memory so we can easily write it back later
    lines = []
    with open(fpath) as f:
        # We only care about the systematic naming, so we only need to keep the lines corresponding to systematics
        for i,l in enumerate(f.readlines()):
            l = l.strip()
            lines.append(l)
            if len(l) == 0:
                continue
            elif l[0] == "#" or l[0] == "-":
                continue
            words = l.split()
            if len(words) >= 2:
                if words[1] == "shape":
                    shape_systs.append((i,words[0]))
                elif words[1] == "lnN":
                    rate_systs.append((i,words[0]))
    return lines, shape_systs, rate_systs

def remap_name(old_name,run):
    pieces = old_name.split("_")
    if len(pieces) == 1:
        raise RuntimeError(f"Unexpected histogram name encountered: {old_name}")

    if f"{pieces[0]}_{pieces[1]}" == "charge_flips":
        process = "_".join(pieces[:3])
        pieces = pieces[3:]
    else:
        process = "_".join(pieces[:2])
        pieces = pieces[2:]

    if len(pieces) == 0:
        # This is either a nominal histogram or the observed data histogram, don't need to do any renaming
        return old_name

    if not process.endswith("_sm"):
        # All of our histograms should be of the form: {process_name}_sm_{systematic} + Up or Down
        raise RuntimeError(f"Unexpected histogram name encountered: {old_name}")
    # Note: This will include the Up/Down string in the name for template histograms
    syst_name = "_".join(pieces)
    direction = ""
    if syst_name.endswith("Up"):
        direction = "Up"
        syst_name = syst_name.removesuffix("Up")
    elif syst_name.endswith("Down"):
        direction = "Down"
        syst_name = syst_name.removesuffix("Down")
    else:
        raise RuntimeError(f"The histogram {old_name} has a systematic without an up/down variation: {syst_name}")
    if not syst_name in SYST_NAMING_MAP[run]:
        raise RuntimeError(f"{syst_name} not found in our naming map!")
    new_syst_name = SYST_NAMING_MAP[run][syst_name]
    new_name = f"{process}_{new_syst_name}{direction}"

    return new_name

def validate_histograms(old_file,new_file,run,verbose=False):
    INDENT = " "*4
    threshold = 1e-6
    print(f"{INDENT}Validating {new_file}")
    with uproot.open(old_file) as fold:
        with uproot.open(new_file) as fnew:
            for _,old_hist in fold.items():
                old_name = old_hist.name
                new_name = remap_name(old_name,run)
                new_hist = fnew[new_name]

                old_vals = old_hist.values()
                new_vals = new_hist.values()

                if verbose:
                    print(f"{INDENT}Comparing: {old_name} <--> {new_name}")

                if len(old_vals) != len(new_vals):
                    raise RuntimeError(f"Difference detected comparing {old_name} <-> {new_name}:\n\t: {old_vals}\n\t{new_vals}")
                for i,(old,new) in enumerate(zip(old_vals,new_vals)):
                    diff = abs(new - old)
                    if verbose:
                        print(f"{INDENT*2}bin {i:>2}: {old:>8.4f} -- {new:>8.4f} -- diff: {diff:>6.2f}")
                    if diff > threshold:
                        raise RuntimeError(f"Difference detected comparing {old_name} <-> {new_name}:\n\told: {old_vals}\n\tnew:{new_vals}")

                old_vars = old_hist.variances()
                new_vars = new_hist.variances()
                if len(old_vars) != len(new_vars):
                    raise RuntimeError(f"Difference detected comparing {old_name} <-> {new_name}:\n\t: {old_vars}\n\t{new_vars}")
                for i,(old,new) in enumerate(zip(old_vars,new_vars)):
                    diff = abs(new - old)
                    if verbose:
                        print(f"{INDENT*2}bin {i:>2}: {old:>8.4f} -- {new:>8.4f} -- diff: {diff:>6.2f}")
                    if diff > threshold:
                        raise RuntimeError(f"Difference detected comparing {old_name} <-> {new_name}:\n\told: {old_vars}\n\tnew:{new_vars}")

def remake_histogram(root_histogram,run,verbose=False):
    '''
        Description: We extract the bin contents and bin variances from the ROOT histogram as numpy
            arrays. We then use those arrays to create a new Hist.hist object from scratch with the
            appropriate systematic name

        Note: This needs to take care of the edge cases where we process a ROOT histogram, but don't
            want to rename it, e.g. one of the nominal histograms or the observed data histogram

        root_histogram: A TH1D root histogram obtained from a datacard template root file
        run: A string that should either be "run2" or "run3", so we know which mapping to use

        Returns: A hist.Hist object created in the same manner as the datacard_tools code, with the
            histogram renamed to use the CMS systematic naming conventions
    '''
    old_name = root_histogram.name
    new_name = remap_name(old_name,run)
    if old_name == new_name:
        # This is either a nominal histogram or the observed data histogram, don't need to do any renaming
        if verbose:
            print(f"\tNo change to {old_name}")
        return root_histogram.name, root_histogram
    # Now let's get the bin contents and sumw2. All histograms should have a sumw2, but for most it will just be 0
    values = root_histogram.values()
    sumw2  = root_histogram.variances()
    nbins = len(values)
    if verbose:
        print(f"\t{old_name} --> {new_name}")
    # We now re-create the histogram in the exact same manner as done in the datacard_tools code
    new_hist = hist.Hist(hist.axis.Regular(nbins,0,nbins,name=new_name),storage=bh.storage.Weight())
    new_hist[...] = np.stack([values,sumw2],axis=-1)

    return new_name,new_hist

def process_templates(templates,run,out_dir):
    run_dir = os.path.join(out_dir,run)
    create_directory(run_dir,clean=[r".*\.root"])
    nfiles = len(templates)
    for i,old_fpath in enumerate(templates):
        fname = old_fpath.rsplit("/",1)[1]
        print(f"[{i+1:>3}/{nfiles}] Checking {fname}",flush=True)
        old_category = fname.removesuffix(".root")
        # new_category = f"{old_category}_{run}"
        # new_category = f"{run}_{old_category}"
        new_category = old_category.replace("ttx_multileptons-",f"ttx_multileptons-{run}_")
        remade_histograms = []
        with uproot.open(old_fpath) as f:
            for k,th1d in f.items():
                new_name,new_hist = remake_histogram(th1d,run,verbose=False)
                remade_histograms.append((new_name,new_hist))
        new_fpath = f"{run_dir}/{new_category}.root"
        with uproot.recreate(new_fpath) as f_new:
            for name,hist in remade_histograms:
                f_new[name] = hist
        validate_histograms(old_file=old_fpath,new_file=new_fpath,run=run,verbose=True)

def process_cards(cards,run,out_dir):
    # When replacing the string names, the columns for the nuisance parameters will be offset, but shouldn't affect combine
    run_dir = os.path.join(out_dir,run)
    create_directory(run_dir,clean=[r".*\.txt"])
    nfiles = len(cards)
    for i,fpath in enumerate(cards):
        fname = fpath.rsplit("/",1)[1]
        print(f"[{i+1:>3}/{nfiles}] Checking {fname}")
        old_category = fname.removesuffix(".txt")
        # new_category = f"{old_category}_{run}"
        # new_category = f"{run}_{old_category}"
        new_category = old_category.replace("ttx_multileptons-",f"ttx_multileptons-{run}_")
        lines,shapes,rates = parse_datacard(fpath)
        # Get the max width of new names, to at least make sure columns stay aligned
        max_width = 0
        for ln,old in shapes:
            new = SYST_NAMING_MAP[run][old]
            max_width = max(max_width,len(new))
        for ln,old in rates:
            new = SYST_NAMING_MAP[run][old]
            max_width = max(max_width,len(new))

        for j,(ln,old) in enumerate(shapes):
            new = SYST_NAMING_MAP[run][old]
            idx = lines[ln].find(" shape") + 1
            lines[ln] = lines[ln].replace(f"{old:<{idx}}",f"{new:<{max_width+1}}")
        for j,(ln,old) in enumerate(rates):
            new = SYST_NAMING_MAP[run][old]
            idx = lines[ln].find(" lnN") + 1
            lines[ln] = lines[ln].replace(f"{old:<{idx}}",f"{new:<{max_width+1}}")
        new_fpath = f"{run_dir}/{new_category}.txt"
        with open(new_fpath,"w") as f:
            for l in lines:
                # We need to update the name of the template root file in the text datacard
                if l.find(old_category) > 0:
                    # A little verbose, but at least keeps things clear
                    old_template_location = f"{old_category}.root"
                    new_template_location = f"{run}/{new_category}.root"
                    f.write(f"{l.replace(old_template_location,new_template_location)}\n")
                else:
                    f.write(f"{l}\n")

# Note: It is VERY important that the run2 and run3 scalings.json files are specified in the correct
#   order, since this function just shifts the channel numbers of the json file for the 'r3_loc'
#   variable as that's the assumed ordering when we combine the datacards
# Note: This is potentially quite fragile, as it relies on the scalings.json to already have the
#   correctly renamed channel names that you get from running the combineCards script and I'm not
#   sure how to ensure this gets enforced
def combine_scalings(r2_loc,r3_loc,out_dir):
    fp_r2 = open(f"{r2_loc}/scalings.json",'r')
    fp_r3 = open(f"{r3_loc}/scalings.json",'r')

    jsn_r2 = json.load(fp_r2)
    jsn_r3 = json.load(fp_r3)

    fp_r2.close()
    fp_r3.close()

    if len(jsn_r2) // 6 != 129:
        raise RuntimeError(f"{r2_loc} does not have the expected number of channels")
    elif len(jsn_r3) // 6 != 129:
        raise RuntimeError(f"{r3_loc} does not have the expected number of channels")

    # Shift the channel numbers for the run3 categories
    for i,d in enumerate(jsn_r3):
        shifted = int(d['channel'][2:]) + 129
        jsn_r3[i]['channel'] = f"ch{shifted}"
        # Note: Since we're appending a dictionary here, this is shallow copy! Modifying jsn_r3 later
        #   will also change the corresponding contents of jsn_r2
        jsn_r2.append(jsn_r3[i])

    fpath = f"{out_dir}/scalings.json"
    with open(fpath,'w') as f:
        json.dump(jsn_r2,f,indent=4)

def main():
    user = os.environ["USER"]
    out_dir = f"/tmpscratch/users/{user}/TOP-26-006-renamed-systematics/testing"

    run2_cards_dir = "/tmpscratch/users/apiccine/TOP-26-006-final-datacards/run2/"
    run3_cards_dir = "/tmpscratch/users/apiccine/TOP-26-006-final-datacards/run3/"

    run2_cards = utils.get_files(run2_cards_dir,match_files=["^ttx_multileptons.*txt$"])
    run3_cards = utils.get_files(run3_cards_dir,match_files=["^ttx_multileptons.*txt$"])

    run2_templates = utils.get_files(run2_cards_dir,match_files=["^ttx_multileptons.*root$"])
    run3_templates = utils.get_files(run3_cards_dir,match_files=["^ttx_multileptons.*root$"])

    process_templates(templates=run2_templates,run="run2",out_dir=out_dir)
    process_templates(templates=run3_templates,run="run3",out_dir=out_dir)

    process_cards(run2_cards,run="run2",out_dir=out_dir)
    process_cards(run3_cards,run="run3",out_dir=out_dir)

    run2_other_files = utils.get_files(run2_cards_dir,match_files=["selectedWCs.txt",".*json$"])
    run3_other_files = utils.get_files(run3_cards_dir,match_files=["selectedWCs.txt",".*json$"])

    combine_scalings(run2_cards_dir,run3_cards_dir,out_dir)

    print("Done!")

main()