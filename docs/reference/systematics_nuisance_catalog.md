# Systematics nuisance catalog

**Generated file: do not edit manually.** Canonical scientific and software authorities remain `topeft/modules/datacard_tools.py`, `topeft/params/rate_systs_run2.json`, `topeft/params/rate_systs_run3.json`, and `analysis/topeft_run2/standardize_systematics.py`. Regenerate with `python tools/generate_systematics_nuisance_catalog.py --write`; verify freshness with `python tools/generate_systematics_nuisance_catalog.py --check`. DatacardMaker policy and standardizer mapping are separate layers.

## Physicist quick index

| Mechanism | Run/domain | Entries | Controls | Canonical authority |
| --- | --- | --- | --- | --- |
| Year correlation | Run 2 / Run 3 | 37 | Groups source periods | DatacardMaker.syst_year_corr |
| Run decorrelation | Run 2 / Run 3 | 16 | Resolves final base by run | DatacardMaker.run_decorrelate; resolve_shape_nuisance_identity |
| Lepton MVA correlation | Run 2 / Run 3 | 4 | Selects per-run or shared MVA identity | resolve_shape_nuisance_identity |
| Fully specified Tau precedence | Both runs | 4 | Preserves already specified nuisance bases | FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES; resolve_shape_nuisance_identity |
| Process shape decorrelation | Both runs | 11 | Matches processes and group tokens | DatacardMaker.syst_shape_decorrelate |
| Luminosity process exclusions | Run 2 / Run 3 | 6 | Excludes logical processes from luminosity templates | run2_luminosity_excluded_processes |
| Luminosity shapes | Run 2 | 12 | Component factors by period | run2_luminosity_shape_factors |
| Rate uncertainties | Run 2 | 15 | Configured process values | rate_systs_run2.json |
| Rate correlation tags | Run 2 | 18 | Configured process-family tags | rate_systs_run2.json |
| Diboson njets | Run 2 | 7 | Configured jet-bin factors | rate_systs_run2.json |
| Standardizer map | Run 2 | 170 | Final identity rename or explicit identity mapping | SYST_NAMING_MAP |
| Luminosity shapes | Run 3 | 8 | Component factors by period | run3_luminosity_shape_factors |
| Rate uncertainties | Run 3 | 15 | Configured process values | rate_systs_run3.json |
| Rate correlation tags | Run 3 | 18 | Configured process-family tags | rate_systs_run3.json |
| Diboson njets | Run 3 | 7 | Configured jet-bin factors | rate_systs_run3.json |
| Standardizer map | Run 3 | 242 | Final identity rename or explicit identity mapping | SYST_NAMING_MAP |

## DatacardMaker year-correlation groups

Each row is one normalized group; no particular textual year label is guaranteed to survive card construction.

| Source nuisance base | Period group | Mechanism/authority |
| --- | --- | --- |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3 | 2016, 2016APV | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8 | 2022, 2022EE | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8 | 2023, 2023BPix | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2 | 2022, 2022EE | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2 | 2023, 2023BPix | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4 | 2022, 2022EE | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4 | 2023, 2023BPix | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7 | 2022, 2022EE | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7 | 2023, 2023BPix | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4 | 2022, 2022EE | syst_year_corr |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4 | 2023, 2023BPix | syst_year_corr |
| CMS_scale_t_DeepTau2017v2p1_DM0_genElectron | 2016, 2016APV | syst_year_corr |
| CMS_scale_t_DeepTau2017v2p1_DM0_genTau | 2016, 2016APV | syst_year_corr |
| CMS_scale_t_DeepTau2017v2p1_DM10_genTau | 2016, 2016APV | syst_year_corr |
| CMS_scale_t_DeepTau2017v2p1_DM11_genTau | 2016, 2016APV | syst_year_corr |
| CMS_scale_t_DeepTau2017v2p1_DM1_genElectron | 2016, 2016APV | syst_year_corr |
| CMS_scale_t_DeepTau2017v2p1_DM1_genTau | 2016, 2016APV | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM0_genElectron | 2022, 2022EE | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM0_genElectron | 2023, 2023BPix | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM10_genElectron | 2022, 2022EE | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM10_genElectron | 2023, 2023BPix | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM11_genElectron | 2022, 2022EE | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM11_genElectron | 2023, 2023BPix | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM1_genElectron | 2022, 2022EE | syst_year_corr |
| CMS_scale_t_DeepTau2018v2p5_DM1_genElectron | 2023, 2023BPix | syst_year_corr |
| FFcloseEl | 2016, 2016APV | syst_year_corr |
| FFcloseEl | 2022, 2022EE | syst_year_corr |
| FFcloseEl | 2023, 2023BPix | syst_year_corr |
| FFcloseMu | 2016, 2016APV | syst_year_corr |
| FFcloseMu | 2022, 2022EE | syst_year_corr |
| FFcloseMu | 2023, 2023BPix | syst_year_corr |

## DatacardMaker run-domain decorrelation

| Source base | Run 2 final base | Run 3 final base |
| --- | --- | --- |
| FES | FES_run2 | FES_run3 |
| FF | FF_run2 | FF_run3 |
| FFeta | FFeta_run2 | FFeta_run3 |
| FFpt | FFpt_run2 | FFpt_run3 |
| TES | TES_run2 | TES_run3 |
| btagSFbc_corr | btagSFbc_corr_run2 | btagSFbc_corr_run3 |
| btagSFlight_corr | btagSFlight_corr_run2 | btagSFlight_corr_run3 |
| charge_flips | charge_flips_run2 | charge_flips_run3 |
| lepSF_elec | lepSF_elec_run2 | lepSF_elec_run3 |
| lepSF_elec_mva | lepSF_elec_mva_13TeV | lepSF_elec_mva_13p6TeV |
| lepSF_elec_non_mva | lepSF_elec_non_mva_13TeV | lepSF_elec_non_mva_13p6TeV |
| lepSF_muon | lepSF_muon_run2 | lepSF_muon_run3 |
| lepSF_muon_mva | lepSF_muon_mva_13TeV | lepSF_muon_mva_13p6TeV |
| lepSF_muon_non_mva | lepSF_muon_non_mva_13TeV | lepSF_muon_non_mva_13p6TeV |
| lepSF_taus_fake | lepSF_taus_fake_run2 | lepSF_taus_fake_run3 |
| lepSF_taus_real | lepSF_taus_real_run2 | lepSF_taus_real_run3 |

## Selectable electron and muon MVA correlation

`per-run` is the default. `cross-run` shares each flavor's nuisance across Run 2 and Run 3; electron and muon remain separate. DatacardMaker selects the identity before TXT and ROOT writing. The original TXT/ROOT and standardized TXT/ROOT use the same MVA identities: unsuffixed names are shared, while run-suffixed names are per-run. The standardizer infers this from source identities and rejects mixed topology within either flavor.

| Mode | Source base | Run 2 final base | Run 3 final base |
| --- | --- | --- | --- |
| cross-run | lepSF_elec_mva | lepSF_elec_mva | lepSF_elec_mva |
| cross-run | lepSF_muon_mva | lepSF_muon_mva | lepSF_muon_mva |
| per-run | lepSF_elec_mva | lepSF_elec_mva_13TeV | lepSF_elec_mva_13p6TeV |
| per-run | lepSF_muon_mva | lepSF_muon_mva_13TeV | lepSF_muon_mva_13p6TeV |

## Fully specified Tau identity precedence

A nuisance base beginning with a maintained prefix is already fully specified. `resolve_shape_nuisance_identity` preserves that base before the generic `run_decorrelate` suffix path.

| Maintained prefix | Final identity rule |
| --- | --- |
| CMS_eff_t_DeepTau | preserve base; bypass generic run suffix |
| CMS_fake_t_DeepTau | preserve base; bypass generic run suffix |
| CMS_scale_t_DeepTau | preserve base; bypass generic run suffix |
| lepSF_taus_fake_run | preserve base; bypass generic run suffix |

## Process-conditioned shape decorrelation

An empty group token appends the matched process name; unmatched processes retain the original systematic identity.

| Systematic base | Process match pattern | Group token | Final identity rule |
| --- | --- | --- | --- |
| ISR | Diboson | qq | ISR_qq |
| ISR | Triboson | qq | ISR_qq |
| ISR | convs | gg | ISR_gg |
| ISR | tHq | qg | ISR_qg |
| ISR | tllq | qq | ISR_qq |
| ISR | ttH | gg | ISR_gg |
| ISR | ttll | gg | ISR_gg |
| ISR | ttlnu | qq | ISR_qq |
| ISR | tttt | gg | ISR_gg |
| fact | .* |  | fact_<matched process> |
| renorm | .* |  | renorm_<matched process> |

## Luminosity shape nuisances

The maintained luminosity components are not applied to these logical process identities. The exclusion authority is shared by the Run 2 and Run 3 template paths.

| Excluded logical process | Applies to |
| --- | --- |
| charge_flips | all maintained Run 2 and Run 3 luminosity shape components |
| data | all maintained Run 2 and Run 3 luminosity shape components |
| data_obs | all maintained Run 2 and Run 3 luminosity shape components |
| fakes | all maintained Run 2 and Run 3 luminosity shape components |
| flips | all maintained Run 2 and Run 3 luminosity shape components |
| nonprompt | all maintained Run 2 and Run 3 luminosity shape components |

### Run 2 luminosity components

| Nuisance identity | Period | Factor |
| --- | --- | --- |
| lumi_13TeV_15161718_l | UL16 | 1.0035 |
| lumi_13TeV_15161718_l | UL16APV | 1.0035 |
| lumi_13TeV_15161718_l | UL17 | 1.0061 |
| lumi_13TeV_15161718_l | UL18 | 1.0084 |
| lumi_13TeV_151617_l | UL16 | 1.0004 |
| lumi_13TeV_151617_l | UL16APV | 1.0004 |
| lumi_13TeV_151617_l | UL17 | 1.0055 |
| lumi_13TeV_151617_l | UL18 | 1.0 |
| lumi_13TeV_1516_l | UL16 | 1.0118 |
| lumi_13TeV_1516_l | UL16APV | 1.0118 |
| lumi_13TeV_1516_l | UL17 | 1.0 |
| lumi_13TeV_1516_l | UL18 | 1.0 |

### Run 3 luminosity components

| Nuisance identity | Period | Factor |
| --- | --- | --- |
| lumi_1 | 2022 | 1.0138 |
| lumi_1 | 2022EE | 1.0138 |
| lumi_1 | 2023 | 1.0017 |
| lumi_1 | 2023BPix | 1.0017 |
| lumi_2 | 2022 | 1.0 |
| lumi_2 | 2022EE | 1.0 |
| lumi_2 | 2023 | 1.0127 |
| lumi_2 | 2023BPix | 1.0127 |

## Configured rate nuisances

### Run 2 rate uncertainties

| Source/config family | Process | Correlation tag | Resolved DatacardMaker nuisance identity | Uncertainty |
| --- | --- | --- | --- | --- |
| charge_flips | charge_flips_sm |  | charge_flips | 1.3 |
| pdf_scale | Diboson | qq | pdf_scale_qq | 1.014 |
| pdf_scale | tHq | qg | pdf_scale_qg | 1.051 |
| pdf_scale | tllq | qq | pdf_scale_qq | 1.027 |
| pdf_scale | ttH | gg | pdf_scale_gg | 1.024 |
| pdf_scale | tttt | gg | pdf_scale_gg | 0.9312/1.0688 |
| qcd_scale | Diboson | VV | qcd_scale_VV | 1.023 |
| qcd_scale | Triboson | VVV | qcd_scale_VVV | 1.5 |
| qcd_scale | tHq | tHq | qcd_scale_tHq | 0.851/1.065 |
| qcd_scale | tWZ | tWZ | qcd_scale_tWZ | 1.5 |
| qcd_scale | tllq | V | qcd_scale_V | 1.02 |
| qcd_scale | ttH | ttH | qcd_scale_ttH | 0.929/1.076 |
| qcd_scale | ttll | ttll | qcd_scale_ttll | 0.898/1.117 |
| qcd_scale | ttlnu | ttlnu | qcd_scale_ttlnu | 0.89/1.10 |
| qcd_scale | tttt | tttt | qcd_scale_tttt | 0.8863/1.0359 |

### Run 2 correlation tags

| Process | Nuisance family | Tag |
| --- | --- | --- |
| Diboson | pdf_scale | qq |
| Diboson | qcd_scale | VV |
| Triboson | pdf_scale | qq |
| Triboson | qcd_scale | VVV |
| convs | pdf_scale | gg |
| tHq | pdf_scale | qg |
| tHq | qcd_scale | tHq |
| tWZ | qcd_scale | tWZ |
| tllq | pdf_scale | qq |
| tllq | qcd_scale | V |
| ttH | pdf_scale | gg |
| ttH | qcd_scale | ttH |
| ttll | pdf_scale | gg |
| ttll | qcd_scale | ttll |
| ttlnu | pdf_scale | qq |
| ttlnu | qcd_scale | ttlnu |
| tttt | pdf_scale | gg |
| tttt | qcd_scale | tttt |

### Run 2 diboson_njets

| Process | Jet count | Factor |
| --- | --- | --- |
| Diboson | 1 | 1.0 |
| Diboson | 2 | 1.2 |
| Diboson | 3 | 1.48 |
| Diboson | 4 | 1.77 |
| Diboson | 5 | 2.05 |
| Diboson | 6 | 2.34 |
| Diboson | 7 | 2.62 |

### Run 3 rate uncertainties

| Source/config family | Process | Correlation tag | Resolved DatacardMaker nuisance identity | Uncertainty |
| --- | --- | --- | --- | --- |
| charge_flips_run3 | charge_flips_sm |  | charge_flips_run3 | 1.3 |
| pdf_scale | Diboson | qq | pdf_scale_qq | 1.014 |
| pdf_scale | tHq | qg | pdf_scale_qg | 1.051 |
| pdf_scale | tllq | qq | pdf_scale_qq | 1.027 |
| pdf_scale | ttH | gg | pdf_scale_gg | 1.024 |
| pdf_scale | tttt | gg | pdf_scale_gg | 0.9312/1.0688 |
| qcd_scale | Diboson | VV | qcd_scale_VV | 1.023 |
| qcd_scale | Triboson | VVV | qcd_scale_VVV | 1.5 |
| qcd_scale | tHq | tHq | qcd_scale_tHq | 0.851/1.065 |
| qcd_scale | tWZ | tWZ | qcd_scale_tWZ | 1.5 |
| qcd_scale | tllq | V | qcd_scale_V | 1.02 |
| qcd_scale | ttH | ttH | qcd_scale_ttH | 0.929/1.076 |
| qcd_scale | ttll | ttll | qcd_scale_ttll | 0.898/1.117 |
| qcd_scale | ttlnu | ttlnu | qcd_scale_ttlnu | 0.89/1.10 |
| qcd_scale | tttt | tttt | qcd_scale_tttt | 0.8863/1.0359 |

### Run 3 correlation tags

| Process | Nuisance family | Tag |
| --- | --- | --- |
| Diboson | pdf_scale | qq |
| Diboson | qcd_scale | VV |
| Triboson | pdf_scale | qq |
| Triboson | qcd_scale | VVV |
| convs | pdf_scale | gg |
| tHq | pdf_scale | qg |
| tHq | qcd_scale | tHq |
| tWZ | qcd_scale | tWZ |
| tllq | pdf_scale | qq |
| tllq | qcd_scale | V |
| ttH | pdf_scale | gg |
| ttH | qcd_scale | ttH |
| ttll | pdf_scale | gg |
| ttll | qcd_scale | ttll |
| ttlnu | pdf_scale | qq |
| ttlnu | qcd_scale | ttlnu |
| tttt | pdf_scale | gg |
| tttt | qcd_scale | tttt |

### Run 3 diboson_njets

| Process | Jet count | Factor |
| --- | --- | --- |
| Diboson | 1 | 1.0 |
| Diboson | 2 | 0.86 |
| Diboson | 3 | 1.43 |
| Diboson | 4 | 1.43 |
| Diboson | 5 | 1.72 |
| Diboson | 6 | 2.0 |
| Diboson | 7 | 2.29 |

### Missing-parton rate nuisance

Static nuisance identity: `missing_parton`. DatacardMaker selects the run-era payload and owns the card value; the ROOT payload owns channel-dependent values. Those dynamic values are not in this catalog.

| Run | Default payload authority |
| --- | --- |
| run2 | data/missing_parton/missing_parton_run2.root |
| run3 | data/missing_parton/missing_parton_run3.root |

## Complete standardizer mapping

### Run 2 SYST_NAMING_MAP

| Input final nuisance identity | Standardized nuisance identity | Renamed |
| --- | --- | --- |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM0_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM10_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM11_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat1_DM1_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM0_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM10_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM11_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_stat2_DM1_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM0_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM11_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2016 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2016 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2017 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2017 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2018 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM1_2018 | no |
| CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_alleras_Run2 | CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_alleras_13TeV | yes |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2016 | CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2017 | CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2018 | CMS_fake_t_DeepTau2017v2p1_VSe_abseta0to1p5_2018 | no |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2016 | CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2017 | CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2018 | CMS_fake_t_DeepTau2017v2p1_VSe_abseta1p5to2p3_2018 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2016 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2017 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2018 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p4to0p8_2018 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2016 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2017 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2018 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0p8to1p2_2018 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2016 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2017 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2018 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta0to0p4_2018 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2016 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2017 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2018 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p2to1p7_2018 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2016 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2016 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2017 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2017 | no |
| CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2018 | CMS_fake_t_DeepTau2017v2p1_VSmu_abseta1p7to2p3_2018 | no |
| CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2016 | CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2016 | no |
| CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2017 | CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2017 | no |
| CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2018 | CMS_scale_t_DeepTau2017v2p1_DM0_genElectron_2018 | no |
| CMS_scale_t_DeepTau2017v2p1_DM0_genMuon_Run2 | CMS_scale_t_DeepTau2017v2p1_DM0_genMuon_13TeV | yes |
| CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016 | CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016 | no |
| CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2017 | CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2017 | no |
| CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2018 | CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2018 | no |
| CMS_scale_t_DeepTau2017v2p1_DM10_genMuon_Run2 | CMS_scale_t_DeepTau2017v2p1_DM10_genMuon_13TeV | yes |
| CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2016 | CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2016 | no |
| CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2017 | CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2017 | no |
| CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2018 | CMS_scale_t_DeepTau2017v2p1_DM10_genTau_2018 | no |
| CMS_scale_t_DeepTau2017v2p1_DM11_genMuon_Run2 | CMS_scale_t_DeepTau2017v2p1_DM11_genMuon_13TeV | yes |
| CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2016 | CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2016 | no |
| CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2017 | CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2017 | no |
| CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2018 | CMS_scale_t_DeepTau2017v2p1_DM11_genTau_2018 | no |
| CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2016 | CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2016 | no |
| CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2017 | CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2017 | no |
| CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2018 | CMS_scale_t_DeepTau2017v2p1_DM1_genElectron_2018 | no |
| CMS_scale_t_DeepTau2017v2p1_DM1_genMuon_Run2 | CMS_scale_t_DeepTau2017v2p1_DM1_genMuon_13TeV | yes |
| CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2016 | CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2016 | no |
| CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2017 | CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2017 | no |
| CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2018 | CMS_scale_t_DeepTau2017v2p1_DM1_genTau_2018 | no |
| FF_run2 | CMS_TOP26006_FF_13TeV | yes |
| FFcloseEl_2016 | CMS_TOP26006_FFcloseEl_2016 | yes |
| FFcloseEl_2017 | CMS_TOP26006_FFcloseEl_2017 | yes |
| FFcloseEl_2018 | CMS_TOP26006_FFcloseEl_2018 | yes |
| FFcloseMu_2016 | CMS_TOP26006_FFcloseMu_2016 | yes |
| FFcloseMu_2017 | CMS_TOP26006_FFcloseMu_2017 | yes |
| FFcloseMu_2018 | CMS_TOP26006_FFcloseMu_2018 | yes |
| FFeta_run2 | CMS_TOP26006_FFeta_13TeV | yes |
| FFpt_run2 | CMS_TOP26006_FFpt_13TeV | yes |
| FSR | ps_fsr | yes |
| ISR | ps_isr | yes |
| ISR_gg | ps_isr_gg | yes |
| ISR_qg | ps_isr_qg | yes |
| ISR_qq | ps_isr_qq | yes |
| JER_2016 | CMS_res_j_2016postVFP | yes |
| JER_2016APV | CMS_res_j_2016preVFP | yes |
| JER_2017 | CMS_res_j_2017 | yes |
| JER_2018 | CMS_res_j_2018 | yes |
| JES_Absolute | CMS_scale_j_Absolute_13TeV | yes |
| JES_BBEC1 | CMS_scale_j_BBEC1_13TeV | yes |
| JES_FlavorPureBottom | CMS_scale_j_FlavorPureBottom | yes |
| JES_FlavorPureCharm | CMS_scale_j_FlavorPureCharm | yes |
| JES_FlavorPureGluon | CMS_scale_j_FlavorPureGluon | yes |
| JES_FlavorPureQuark | CMS_scale_j_FlavorPureQuark | yes |
| JES_FlavorQCD | CMS_scale_j_FlavorQCD_13TeV | yes |
| JES_RelativeBal | CMS_scale_j_RelativeBal_13TeV | yes |
| JES_RelativeSample | CMS_scale_j_RelativeSample | yes |
| MET_UnclusteredEnergy | CMS_scale_met_Unclustred_energy_13TeV | yes |
| PU | CMS_pileup_13TeV | yes |
| PreFiring | CMS_l1_ecal_prefiring | yes |
| btagSFbc_2016 | CMS_btag_fixedWP_bc_uncorrelated_2016postVFP | yes |
| btagSFbc_2016APV | CMS_btag_fixedWP_bc_uncorrelated_2016preVFP | yes |
| btagSFbc_2017 | CMS_btag_fixedWP_bc_uncorrelated_2017 | yes |
| btagSFbc_2018 | CMS_btag_fixedWP_bc_uncorrelated_2018 | yes |
| btagSFbc_corr_run2 | CMS_btag_fixedWP_bc_correlated_13TeV | yes |
| btagSFlight_2016 | CMS_btag_fixedWP_light_uncorrelated_2016postVFP | yes |
| btagSFlight_2016APV | CMS_btag_fixedWP_light_uncorrelated_2016preVFP | yes |
| btagSFlight_2017 | CMS_btag_fixedWP_light_uncorrelated_2017 | yes |
| btagSFlight_2018 | CMS_btag_fixedWP_light_uncorrelated_2018 | yes |
| btagSFlight_corr_run2 | CMS_btag_fixedWP_light_correlated_13TeV | yes |
| charge_flips | CMS_TOP26006_charge_flips_13TeV | yes |
| diboson_njets | CMS_TOP26006_njets_VV | yes |
| fact_Diboson | QCDscale_fac_VV_ACCEPT | yes |
| fact_Triboson | QCDscale_fac_VVV_ACCEPT | yes |
| fact_convs | QCDscale_fac_convs_ACCEPT | yes |
| fact_tHq | QCDscale_fac_tHq_ACCEPT | yes |
| fact_tWZ | QCDscale_fac_tWZ_ACCEPT | yes |
| fact_tllq | QCDscale_fac_tllq_ACCEPT | yes |
| fact_ttH | QCDscale_fac_ttH_ACCEPT | yes |
| fact_ttll | QCDscale_fac_ttll_ACCEPT | yes |
| fact_ttlnu | QCDscale_fac_ttlnu_ACCEPT | yes |
| fact_tttt | QCDscale_fac_tttt_ACCEPT | yes |
| lepSF_elec_mva | lepSF_elec_mva | no |
| lepSF_elec_non_mva | lepSF_elec_non_mva_13TeV | yes |
| lepSF_muon_mva | lepSF_muon_mva | no |
| lepSF_muon_non_mva | lepSF_muon_non_mva_13TeV | yes |
| lepSF_taus_fake_run2 | CMS_TOP26006_eff_fake_t_13TeV | yes |
| lumi_13TeV_15161718_l | lumi_13TeV_15161718_l | no |
| lumi_13TeV_151617_l | lumi_13TeV_151617_l | no |
| lumi_13TeV_1516_l | lumi_13TeV_1516_l | no |
| missing_parton | CMS_TOP26006_missing_parton | yes |
| pdf_scale_gg | cross_section_pdf_gg | yes |
| pdf_scale_qg | cross_section_pdf_gq | yes |
| pdf_scale_qq | cross_section_pdf_qqbar | yes |
| qcd_scale_V | cross_section_qcd_scale_V | yes |
| qcd_scale_VV | cross_section_qcd_scale_VV | yes |
| qcd_scale_VVV | cross_section_qcd_scale_VVV | yes |
| qcd_scale_tHq | cross_section_qcd_scale_tHq | yes |
| qcd_scale_tWZ | cross_section_qcd_scale_tWZ | yes |
| qcd_scale_ttH | cross_section_qcd_scale_ttH | yes |
| qcd_scale_ttll | cross_section_qcd_scale_ttll | yes |
| qcd_scale_ttlnu | cross_section_qcd_scale_ttlnu | yes |
| qcd_scale_tttt | cross_section_qcd_scale_tttt | yes |
| renorm_Diboson | QCDscale_ren_VV_ACCEPT | yes |
| renorm_Triboson | QCDscale_ren_VVV_ACCEPT | yes |
| renorm_convs | QCDscale_ren_convs_ACCEPT | yes |
| renorm_tHq | QCDscale_ren_tHq_ACCEPT | yes |
| renorm_tWZ | QCDscale_ren_tWZ_ACCEPT | yes |
| renorm_tllq | QCDscale_ren_tllq_ACCEPT | yes |
| renorm_ttH | QCDscale_ren_ttH_ACCEPT | yes |
| renorm_ttll | QCDscale_ren_ttll_ACCEPT | yes |
| renorm_ttlnu | QCDscale_ren_ttlnu_ACCEPT | yes |
| renorm_tttt | QCDscale_ren_tttt_ACCEPT | yes |
| triggerSF_2016 | CMS_TOP26006_eff_trigger_2016postVFP | yes |
| triggerSF_2016APV | CMS_TOP26006_eff_trigger_2016preVFP | yes |
| triggerSF_2017 | CMS_TOP26006_eff_trigger_2017 | yes |
| triggerSF_2018 | CMS_TOP26006_eff_trigger_2018 | yes |

### Run 3 SYST_NAMING_MAP

| Input final nuisance identity | Standardized nuisance identity | Renamed |
| --- | --- | --- |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM10_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM11_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM1_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM0_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM10_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM1_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM0_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM11_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022EE | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2022EE | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023 | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023BPix | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM1_2023BPix | no |
| CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_Run3 | CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_13p6TeV | yes |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p46to1p56_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta1p56to2p5_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta0to1p46_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p46to1p56_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM10_abseta1p56to2p5_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta0to1p46_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p46to1p56_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM11_abseta1p56to2p5_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta0to1p46_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p46to1p56_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022 | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022EE | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2022EE | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023 | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023BPix | CMS_fake_t_DeepTau2018v2p5_VSe_DM1_abseta1p56to2p5_2023BPix | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2022 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2023 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p4to0p8_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2022 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2023 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0p8to1p2_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2023 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2022 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2023 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p2to1p7_2023 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2022 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2022 | no |
| CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2023 | CMS_fake_t_DeepTau2018v2p5_VSmu_abseta1p7to2p4_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2022 | CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2023 | CMS_scale_t_DeepTau2018v2p5_DM0_genElectron_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM0_genMuon_Run3 | CMS_scale_t_DeepTau2018v2p5_DM0_genMuon_13p6TeV | yes |
| CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022 | CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022EE | CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2022EE | no |
| CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023 | CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023BPix | CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2023BPix | no |
| CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2022 | CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2023 | CMS_scale_t_DeepTau2018v2p5_DM10_genElectron_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM10_genMuon_Run3 | CMS_scale_t_DeepTau2018v2p5_DM10_genMuon_13p6TeV | yes |
| CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022 | CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022EE | CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2022EE | no |
| CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023 | CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023BPix | CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2023BPix | no |
| CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2022 | CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2023 | CMS_scale_t_DeepTau2018v2p5_DM11_genElectron_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM11_genMuon_Run3 | CMS_scale_t_DeepTau2018v2p5_DM11_genMuon_13p6TeV | yes |
| CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022 | CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022EE | CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2022EE | no |
| CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023 | CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023BPix | CMS_scale_t_DeepTau2018v2p5_DM11_genTau_2023BPix | no |
| CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2022 | CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2023 | CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM1_genMuon_Run3 | CMS_scale_t_DeepTau2018v2p5_DM1_genMuon_13p6TeV | yes |
| CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022 | CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022 | no |
| CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022EE | CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022EE | no |
| CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023 | CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023 | no |
| CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023BPix | CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2023BPix | no |
| FF_run3 | CMS_TOP26006_FF_13p6TeV | yes |
| FFcloseEl_2022 | CMS_TOP26006_FFcloseEl_2022 | yes |
| FFcloseEl_2023 | CMS_TOP26006_FFcloseEl_2023 | yes |
| FFcloseMu_2022 | CMS_TOP26006_FFcloseMu_2022 | yes |
| FFcloseMu_2023 | CMS_TOP26006_FFcloseMu_2023 | yes |
| FFeta_run3 | CMS_TOP26006_FFeta_13p6TeV | yes |
| FFpt_run3 | CMS_TOP26006_FFpt_13p6TeV | yes |
| FSR | ps_fsr | yes |
| ISR | ps_isr | yes |
| ISR_gg | ps_isr_gg | yes |
| ISR_qg | ps_isr_qg | yes |
| ISR_qq | ps_isr_qq | yes |
| JER_2022 | CMS_res_j_2022 | yes |
| JER_2022EE | CMS_res_j_2022EE | yes |
| JER_2023 | CMS_res_j_2023 | yes |
| JER_2023BPix | CMS_res_j_2023BPix | yes |
| JES_Regrouped_Absolute | CMS_scale_j_Absolute_13p6TeV | yes |
| JES_Regrouped_Absolute_2022 | CMS_scale_j_Absolute_2022 | yes |
| JES_Regrouped_Absolute_2022EE | CMS_scale_j_Absolute_2022EE | yes |
| JES_Regrouped_Absolute_2023 | CMS_scale_j_Absolute_2023 | yes |
| JES_Regrouped_Absolute_2023BPix | CMS_scale_j_Absolute_2023BPix | yes |
| JES_Regrouped_BBEC1 | CMS_scale_j_BBEC1_13p6TeV | yes |
| JES_Regrouped_BBEC1_2022 | CMS_scale_j_BBEC1_2022 | yes |
| JES_Regrouped_BBEC1_2022EE | CMS_scale_j_BBEC1_2022EE | yes |
| JES_Regrouped_BBEC1_2023 | CMS_scale_j_BBEC1_2023 | yes |
| JES_Regrouped_BBEC1_2023BPix | CMS_scale_j_BBEC1_2023BPix | yes |
| JES_Regrouped_EC2 | CMS_scale_j_EC2 | yes |
| JES_Regrouped_EC2_2022 | CMS_scale_j_EC2_2022 | yes |
| JES_Regrouped_EC2_2022EE | CMS_scale_j_EC2_2022EE | yes |
| JES_Regrouped_EC2_2023 | CMS_scale_j_EC2_2023 | yes |
| JES_Regrouped_EC2_2023BPix | CMS_scale_j_EC2_2023BPix | yes |
| JES_Regrouped_FlavorQCD | CMS_scale_j_FlavorQCD_13p6TeV | yes |
| JES_Regrouped_HF | CMS_scale_j_HF | yes |
| JES_Regrouped_HF_2022 | CMS_scale_j_HF_2022 | yes |
| JES_Regrouped_HF_2022EE | CMS_scale_j_HF_2022EE | yes |
| JES_Regrouped_HF_2023 | CMS_scale_j_HF_2023 | yes |
| JES_Regrouped_HF_2023BPix | CMS_scale_j_HF_2023BPix | yes |
| JES_Regrouped_RelativeBal | CMS_scale_j_RelativeBal_13p6TeV | yes |
| JES_Regrouped_RelativeSample_2022 | CMS_scale_j_RelativeSample_2022 | yes |
| JES_Regrouped_RelativeSample_2022EE | CMS_scale_j_RelativeSample_2022EE | yes |
| JES_Regrouped_RelativeSample_2023 | CMS_scale_j_RelativeSample_2023 | yes |
| JES_Regrouped_RelativeSample_2023BPix | CMS_scale_j_RelativeSample_2023BPix | yes |
| MET_UnclusteredEnergy | CMS_scale_met_Unclustred_energy_13p6TeV | yes |
| MuonResolution | CMS_res_m_13p6TeV | yes |
| MuonScale | CMS_scale_m_13p6TeV | yes |
| PU | CMS_pileup_13p6TeV | yes |
| PreFiring | CMS_l1_ecal_prefiring | yes |
| btagSFbc_2022 | CMS_btag_fixedWP_bc_uncorrelated_2022 | yes |
| btagSFbc_2022EE | CMS_btag_fixedWP_bc_uncorrelated_2022EE | yes |
| btagSFbc_2023 | CMS_btag_fixedWP_bc_uncorrelated_2023 | yes |
| btagSFbc_2023BPix | CMS_btag_fixedWP_bc_uncorrelated_2023BPix | yes |
| btagSFbc_corr_run3 | CMS_btag_fixedWP_bc_correlated_13p6TeV | yes |
| btagSFlight_2022 | CMS_btag_fixedWP_light_uncorrelated_2022 | yes |
| btagSFlight_2022EE | CMS_btag_fixedWP_light_uncorrelated_2022EE | yes |
| btagSFlight_2023 | CMS_btag_fixedWP_light_uncorrelated_2023 | yes |
| btagSFlight_2023BPix | CMS_btag_fixedWP_light_uncorrelated_2023BPix | yes |
| btagSFlight_corr_run3 | CMS_btag_fixedWP_light_correlated_13p6TeV | yes |
| charge_flips_run3 | CMS_TOP26006_charge_flips_13p6TeV | yes |
| diboson_njets | CMS_TOP26006_njets_VV | yes |
| fact_Diboson | QCDscale_fac_VV_ACCEPT | yes |
| fact_Triboson | QCDscale_fac_VVV_ACCEPT | yes |
| fact_convs | QCDscale_fac_convs_ACCEPT | yes |
| fact_tHq | QCDscale_fac_tHq_ACCEPT | yes |
| fact_tWZ | QCDscale_fac_tWZ_ACCEPT | yes |
| fact_tllq | QCDscale_fac_tllq_ACCEPT | yes |
| fact_ttH | QCDscale_fac_ttH_ACCEPT | yes |
| fact_ttll | QCDscale_fac_ttll_ACCEPT | yes |
| fact_ttlnu | QCDscale_fac_ttlnu_ACCEPT | yes |
| fact_tttt | QCDscale_fac_tttt_ACCEPT | yes |
| lepSF_elec_mva | lepSF_elec_mva | no |
| lepSF_elec_non_mva | lepSF_elec_non_mva_13p6TeV | yes |
| lepSF_muon_mva | lepSF_muon_mva | no |
| lepSF_muon_non_mva | lepSF_muon_non_mva_13p6TeV | yes |
| lepSF_taus_fake_run3 | CMS_TOP26006_eff_fake_t_13p6TeV | yes |
| lumi_1 | lumi_1 | no |
| lumi_2 | lumi_2 | no |
| missing_parton | CMS_TOP26006_missing_parton | yes |
| pdf_scale_gg | cross_section_pdf_gg | yes |
| pdf_scale_qg | cross_section_pdf_gq | yes |
| pdf_scale_qq | cross_section_pdf_qqbar | yes |
| qcd_scale_V | cross_section_qcd_scale_V | yes |
| qcd_scale_VV | cross_section_qcd_scale_VV | yes |
| qcd_scale_VVV | cross_section_qcd_scale_VVV | yes |
| qcd_scale_tHq | cross_section_qcd_scale_tHq | yes |
| qcd_scale_tWZ | cross_section_qcd_scale_tWZ | yes |
| qcd_scale_ttH | cross_section_qcd_scale_ttH | yes |
| qcd_scale_ttll | cross_section_qcd_scale_ttll | yes |
| qcd_scale_ttlnu | cross_section_qcd_scale_ttlnu | yes |
| qcd_scale_tttt | cross_section_qcd_scale_tttt | yes |
| renorm_Diboson | QCDscale_ren_VV_ACCEPT | yes |
| renorm_Triboson | QCDscale_ren_VVV_ACCEPT | yes |
| renorm_convs | QCDscale_ren_convs_ACCEPT | yes |
| renorm_tHq | QCDscale_ren_tHq_ACCEPT | yes |
| renorm_tWZ | QCDscale_ren_tWZ_ACCEPT | yes |
| renorm_tllq | QCDscale_ren_tllq_ACCEPT | yes |
| renorm_ttH | QCDscale_ren_ttH_ACCEPT | yes |
| renorm_ttll | QCDscale_ren_ttll_ACCEPT | yes |
| renorm_ttlnu | QCDscale_ren_ttlnu_ACCEPT | yes |
| renorm_tttt | QCDscale_ren_tttt_ACCEPT | yes |
| triggerSF_2022 | CMS_TOP26006_eff_trigger_2022 | yes |
| triggerSF_2022EE | CMS_TOP26006_eff_trigger_2022EE | yes |
| triggerSF_2023 | CMS_TOP26006_eff_trigger_2023 | yes |
| triggerSF_2023BPix | CMS_TOP26006_eff_trigger_2023BPix | yes |

## Scope boundary

This generated catalog is exhaustive for the current active static nuisance policy authorities rendered here and for `SYST_NAMING_MAP`. Nuisance labels discovered only from concrete input histograms or artifacts are not presented as a universal static inventory. Alternative nuisance representations intentionally not written to current datacards are outside the active inventory. Actual package and card nuisance populations are artifacts, not source registries. Standardization does not decide scientific correlation topology.
