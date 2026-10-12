import os
import shutil
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import uproot
import hist
import json
import numpy as np
import boost_histogram as bh
import topcoffea.modules.utils as utils
from topeft.modules.datacard_tools import cross_run_mva_identities
from analysis.topeft_run2.build_fit_input_package import _source_contract, _validate_source_members

accepted_mapping_commit = "013d9de2b53ad2f0eb5cfae5e47e5d796b07fe1f"

stale_run2_tau_apv_bases = frozenset({
    "CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5",
    "CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3",
    "CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4",
    "CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8",
    "CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2",
    "CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7",
    "CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3",
    "CMS_scale_t_DeepTau2017v2p1_DM0_genTau",
    "CMS_scale_t_DeepTau2017v2p1_DM1_genTau",
    "CMS_scale_t_DeepTau2017v2p1_DM10_genTau",
    "CMS_scale_t_DeepTau2017v2p1_DM11_genTau",
    "CMS_scale_t_DeepTau2017v2p1_DM0_genElectron",
    "CMS_scale_t_DeepTau2017v2p1_DM1_genElectron",
})

# Final nuisance names for TOP-26-006 cards.
SYST_NAMING_MAP = {
    "run2": {
        # Luminosity
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
        "MET_UnclusteredEnergy": "CMS_scale_met_Unclustred_energy_13TeV",
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
        "PU": "CMS_pileup_13TeV",
        # L1 Prefiring
        "PreFiring": "CMS_l1_ecal_prefiring",
        # Taus
        "lepSF_taus_fake_run2": "CMS_TOP26006_eff_fake_t_13TeV",
        # lepSF
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
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2016': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2016',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2017': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2017',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2018': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2018',
        'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_alleras_Run2': 'CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_alleras_13TeV',
        'CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2016': 'CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2017': 'CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2018': 'CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2018',
        'CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2016': 'CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2017': 'CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2018': 'CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2018',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2016': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2017': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2018': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2018',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2016': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2017': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2018': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2018',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2016': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2017': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2018': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2018',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2016': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2017': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2018': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2018',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2016': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2016',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2017': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2017',
        'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2018': 'CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2018',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2016': 'CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2016',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2017': 'CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2017',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2018': 'CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2018',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genMuon_Run2': 'CMS_scale_t_DeepTau2017v2p1_DM0_genMuon_13TeV',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016': 'CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2017': 'CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2017',
        'CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2018': 'CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2018',
        'CMS_scale_t_DeepTau2017v2p1_DM10_genMuon_Run2': 'CMS_scale_t_DeepTau2017v2p1_DM10_genMuon_13TeV',
        'CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2016': 'CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2016',
        'CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2017': 'CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2017',
        'CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2018': 'CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2018',
        'CMS_scale_t_DeepTau2017v2p1_DM11_genMuon_Run2': 'CMS_scale_t_DeepTau2017v2p1_DM11_genMuon_13TeV',
        'CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2016': 'CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2016',
        'CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2017': 'CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2017',
        'CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2018': 'CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2018',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2016': 'CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2016',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2017': 'CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2017',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2018': 'CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2018',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genMuon_Run2': 'CMS_scale_t_DeepTau2017v2p1_DM1_genMuon_13TeV',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2016': 'CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2016',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2017': 'CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2017',
        'CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2018': 'CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2018',
        'lepSF_elec_non_mva': 'lepSF_elec_non_mva_13TeV',
        'lepSF_muon_non_mva': 'lepSF_muon_non_mva_13TeV',
        'lepSF_elec_mva': 'lepSF_elec_mva',
        'lepSF_muon_mva': 'lepSF_muon_mva',
        'lumi_13TeV_15161718_l': 'lumi_13TeV_15161718_l',
        'lumi_13TeV_151617_l': 'lumi_13TeV_151617_l',
        'lumi_13TeV_1516_l': 'lumi_13TeV_1516_l',
    },
    "run3": {
        # Luminosity
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
        "MET_UnclusteredEnergy": "CMS_scale_met_Unclustred_energy_13p6TeV",
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
        "PU": "CMS_pileup_13p6TeV",
        # L1 Prefiring
        "PreFiring": "CMS_l1_ecal_prefiring",   # Exists in Run3, but variations are just set to nominal
        # Taus
        "lepSF_taus_fake_run3": "CMS_TOP26006_eff_fake_t_13p6TeV",
        # lepSF
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
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022EE': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022EE',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023BPix': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023BPix',
        'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_Run3': 'CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_13p6TeV',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022EE': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022EE',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023BPix': 'CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023BPix',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2022': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2023': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2022': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2023': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2023': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2022': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2023': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2023',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2022': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2022',
        'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2023': 'CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2022': 'CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2023': 'CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genMuon_Run3': 'CMS_scale_t_DeepTau2018v2p5_DM0_genMuon_13p6TeV',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022': 'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022EE': 'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022EE',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023': 'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023BPix': 'CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023BPix',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2022': 'CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2023': 'CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genMuon_Run3': 'CMS_scale_t_DeepTau2018v2p5_DM10_genMuon_13p6TeV',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022': 'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022EE': 'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022EE',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023': 'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023BPix': 'CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023BPix',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2022': 'CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2023': 'CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genMuon_Run3': 'CMS_scale_t_DeepTau2018v2p5_DM11_genMuon_13p6TeV',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022': 'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022EE': 'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022EE',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023': 'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023BPix': 'CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023BPix',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2022': 'CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2023': 'CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genMuon_Run3': 'CMS_scale_t_DeepTau2018v2p5_DM1_genMuon_13p6TeV',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022': 'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022EE': 'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022EE',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023': 'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023',
        'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023BPix': 'CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023BPix',
        'lepSF_elec_non_mva': 'lepSF_elec_non_mva_13p6TeV',
        'lepSF_muon_non_mva': 'lepSF_muon_non_mva_13p6TeV',
        'lepSF_elec_mva': 'lepSF_elec_mva',
        'lepSF_muon_mva': 'lepSF_muon_mva',
        'lumi_1': 'lumi_1',
        'lumi_2': 'lumi_2',
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

def resolve_nuisance_name(source_name, run):
    if run == "run2" and source_name.endswith("_2016APV"):
        source_base = source_name.removesuffix("_2016APV")
        if source_base in stale_run2_tau_apv_bases:
            raise ValueError(
                f"Stale pre-correlation tau nuisance {source_name!r}; "
                "regenerate the Run-2 datacards with native year correlation."
            )
        if (
            source_base.startswith("CMS_eff_t_DeepTau2017v2p1_VSjet_dm_")
            and f"{source_base}_2016" in SYST_NAMING_MAP["run2"]
        ):
            return source_name
    try:
        return SYST_NAMING_MAP[run][source_name]
    except KeyError as exc:
        if run in SYST_NAMING_MAP and source_name in (
            f"{base}_{'13TeV' if run == 'run2' else '13p6TeV'}"
            for base in cross_run_mva_identities
        ):
            return source_name
        if run in SYST_NAMING_MAP and source_name in (
            SYST_NAMING_MAP[run][base]
            for base in (
                "lepSF_elec_mva", "lepSF_muon_mva",
                "lepSF_elec_non_mva", "lepSF_muon_non_mva",
            )
        ):
            return source_name
        raise RuntimeError(
            f"{source_name} not found in the {run} naming map!"
        ) from exc


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
    new_syst_name = resolve_nuisance_name(syst_name, run)
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
    # Reject stale tau APV inputs before cleaning or writing output.
    for old_fpath in templates:
        with uproot.open(old_fpath) as source_file:
            for _, root_histogram in source_file.items():
                remap_name(root_histogram.name, run)
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
    for fpath in cards:
        _, shapes, rates = parse_datacard(fpath)
        for _, source_name in shapes + rates:
            resolve_nuisance_name(source_name, run)
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
            new = resolve_nuisance_name(old, run)
            max_width = max(max_width,len(new))
        for ln,old in rates:
            new = resolve_nuisance_name(old, run)
            max_width = max(max_width,len(new))

        for j,(ln,old) in enumerate(shapes):
            new = resolve_nuisance_name(old, run)
            idx = lines[ln].find(" shape") + 1
            lines[ln] = lines[ln].replace(f"{old:<{idx}}",f"{new:<{max_width+1}}")
        for j,(ln,old) in enumerate(rates):
            new = resolve_nuisance_name(old, run)
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

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register_identity(source_name, run, source_to_target, target_to_source):
    target_name = resolve_nuisance_name(source_name, run)
    source_key = (run, source_name)
    previous_target = source_to_target.setdefault(source_key, target_name)
    if previous_target != target_name:
        raise ValueError(f"Nuisance split: {source_key}: {previous_target} / {target_name}")
    previous_source = target_to_source.setdefault(target_name, source_name)
    if previous_source != source_name:
        raise ValueError(
            f"Nuisance collision: {previous_source} and {source_name} -> {target_name}"
        )
    return target_name


def _root_key_identity(key):
    if key == "data_obs" or key.endswith("_sm"):
        return None
    pieces = key.split("_")
    if len(pieces) < 3:
        raise ValueError(f"Unexpected ROOT key: {key}")
    process_end = 3 if key.startswith("charge_flips_") else 2
    process = "_".join(pieces[:process_end])
    variation = "_".join(pieces[process_end:])
    if not process.endswith("_sm") or not variation.endswith(("Up", "Down")):
        raise ValueError(f"Unexpected ROOT systematic key: {key}")
    direction = "Up" if variation.endswith("Up") else "Down"
    return process, variation.removesuffix(direction), direction


def _read_package_inputs(input_package, source_rows):
    cards_dir = input_package / "cards"
    ordered_path = input_package / "ordered_card_inputs.txt"
    if not cards_dir.is_dir() or not ordered_path.is_file():
        raise ValueError("Input package requires cards/ and ordered_card_inputs.txt")
    ordered = ordered_path.read_text().splitlines()
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("Ordered card inputs must be nonempty and unique")
    if set(ordered) != {f"cards/{path.name}" for path in cards_dir.glob("*.txt")}:
        raise ValueError("Ordered card inputs do not match package TXT cards")
    pairs = []
    root_names = set()
    for relative_card, row in zip(ordered, source_rows):
        card_relative = Path(relative_card)
        if (len(card_relative.parts) != 2 or card_relative.parts[0] != "cards"
                or card_relative.suffix != ".txt"):
            raise ValueError(f"Unsafe ordered card path: {relative_card}")
        card_path = input_package / card_relative
        if not card_path.is_file() or card_path.is_symlink():
            raise ValueError(f"Missing or linked card: {relative_card}")
        shapes = [line.split() for line in card_path.read_text().splitlines()
                  if line.split() and line.split()[0] == "shapes"]
        shape_roots = {words[3] for words in shapes if len(words) >= 4}
        if not shapes or len(shape_roots) != 1:
            raise ValueError(f"Expected one shapes ROOT reference in {relative_card}")
        root_name = next(iter(shape_roots))
        if Path(root_name).name != root_name or not root_name.endswith(".root"):
            raise ValueError(f"Unsafe shapes ROOT reference: {root_name}")
        root_path = cards_dir / root_name
        if not root_path.is_file() or root_path.is_symlink():
            raise ValueError(f"Missing or linked shapes ROOT file: {root_name}")
        if root_name in root_names:
            raise ValueError(f"Shapes ROOT file used by multiple cards: {root_name}")
        root_names.add(root_name)
        run = row["era"]
        if run not in SYST_NAMING_MAP:
            raise ValueError(f"Unsupported card era: {run}")
        pairs.append((card_path, root_path, run))
    if root_names != {path.name for path in cards_dir.glob("*.root")}:
        raise ValueError("Shapes references do not match package ROOT files")
    return ordered_path, ordered, pairs


def _preflight_package(pairs):
    source_to_target = {}
    target_to_source = {}
    output_names = set()
    observed_mva_topologies = {base: set() for base in cross_run_mva_identities}

    def record_mva_identity(target_name):
        for base in cross_run_mva_identities:
            if target_name == base:
                observed_mva_topologies[base].add("cross-run")
            elif target_name in (f"{base}_13TeV", f"{base}_13p6TeV"):
                observed_mva_topologies[base].add("per-run")

    for card_path, root_path, run in pairs:
        for path in (card_path, root_path):
            if path.name in output_names:
                raise ValueError(f"Duplicate output member: {path.name}")
            output_names.add(path.name)
        _, shapes, rates = parse_datacard(card_path)
        for _, source_name in shapes + rates:
            target_name = _register_identity(source_name, run, source_to_target, target_to_source)
            record_mva_identity(target_name)
        key_targets = {}
        with uproot.open(root_path) as source_file:
            for key in source_file.keys(cycle=False):
                identity = _root_key_identity(key)
                if identity is None:
                    continue
                process, source_base, direction = identity
                target_base = _register_identity(
                    source_base, run, source_to_target, target_to_source
                )
                record_mva_identity(target_base)
                target_key = (process, target_base, direction)
                previous = key_targets.setdefault(target_key, key)
                if previous != key:
                    raise ValueError(f"ROOT key collision: {previous} and {key}")
    for base, topologies in observed_mva_topologies.items():
        if len(topologies) > 1:
            raise ValueError(f"Mixed lepton MVA correlation identities for {base}: {sorted(topologies)}")
    return source_to_target


def _write_standardized_card(card_path, root_path, run, output_path):
    lines = []
    for line in card_path.read_text().splitlines(keepends=True):
        words = line.split()
        if words and words[0] == "shapes":
            words[3] = root_path.name
            line = " ".join(words) + "\n"
        elif len(words) >= 2 and words[1] in ("shape", "lnN"):
            target_name = resolve_nuisance_name(words[0], run)
            line = line.replace(words[0], target_name, 1)
        lines.append(line)
    output_path.write_text("".join(lines))


def standardize_package(input_package, output_package):
    input_package = Path(input_package).resolve()
    output_package = Path(output_package)
    if output_package.exists() or output_package.is_symlink():
        raise FileExistsError(f"Output package already exists: {output_package}")
    if output_package.resolve().is_relative_to(input_package):
        raise ValueError("Output package must be outside the input package")
    _validate_source_members(input_package)
    source_rows, _, _ = _source_contract(input_package)
    ordered_path, ordered, pairs = _read_package_inputs(input_package, source_rows)
    source_manifest = json.loads((input_package / "combined_mapping_manifest.json").read_text())
    source_provenance_path = input_package / "package-provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text())
    if (source_manifest.get("package_root") != str(input_package)
            or source_provenance.get("package_root") != str(input_package)):
        raise ValueError("Input metadata package root does not identify the input package")
    if len(source_manifest["rows"]) != len(pairs):
        raise ValueError("Input mapping and card population differ")
    if source_manifest["rows"] != source_rows:
        raise ValueError("Input mapping contains unsupported fields")
    _preflight_package(pairs)
    output_package.mkdir(parents=True, exist_ok=False)
    output_cards = output_package / "cards"
    output_cards.mkdir()
    for card_path, root_path, run in pairs:
        destination = output_cards / root_path.name
        with uproot.open(root_path) as source_file, uproot.recreate(destination) as output_file:
            for _, root_histogram in source_file.items(cycle=False):
                name, histogram = remake_histogram(root_histogram, run)
                output_file[name] = histogram
        _write_standardized_card(card_path, root_path, run, output_cards / card_path.name)
    shutil.copyfile(ordered_path, output_package / "ordered_card_inputs.txt")
    shutil.copyfile(input_package / "scalings.json", output_package / "scalings.json")
    source_manifest["package_root"] = str(output_package.resolve())
    (output_package / "combined_mapping_manifest.json").write_text(
        json.dumps(source_manifest, sort_keys=True, indent=2) + "\n"
    )
    (output_package / "README.md").write_text(
        "# Standardized datacard package\n\n"
        "Drop-in datacard-package input for the existing pre-workspace Wynona/lane_09 intake.\n"
        "Cards contain standardized nuisance names and corresponding ROOT keys.\n"
        "EFT scaling and fit-configuration semantics are unchanged.\n"
        "This package is not a combined card, workspace, or fit result.\n"
        "Workspace and fit qualification remain lane_09-owned.\n"
    )
    member_inventory = [
        {"path": path.relative_to(output_package).as_posix(), "sha256": _sha256(path)}
        for path in sorted(output_package.rglob("*")) if path.is_file()
    ]
    provenance = dict(source_provenance)
    provenance.update({
        "package_root": str(output_package.resolve()),
        "package_version": source_provenance["package_version"] + ".standardized",
        "package_date": datetime.now(timezone.utc).strftime("%y%m%d"),
        "manifest_sha256": _sha256(output_package / "combined_mapping_manifest.json"),
        "ordered_card_inputs_sha256": _sha256(output_package / "ordered_card_inputs.txt"),
        "scalings_sha256": _sha256(output_package / "scalings.json"),
        "packaged_txt_count": len(pairs),
        "packaged_root_count": len(pairs),
        "standardization": {
            "source_package_root": str(input_package),
            "source_package_provenance_sha256": _sha256(source_provenance_path),
            "accepted_mapping_commit": accepted_mapping_commit,
            "standardizer_file_sha256": _sha256(Path(__file__)),
            "member_inventory_scope": "all package files except package-provenance.json",
            "member_inventory": member_inventory,
        },
    })
    (output_package / "package-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description="Standardize a certified datacard package")
    parser.add_argument("--input-package", required=True, type=Path)
    parser.add_argument("--output-package", required=True, type=Path)
    arguments = parser.parse_args()
    standardize_package(arguments.input_package, arguments.output_package)

if __name__ == "__main__":
    main()
