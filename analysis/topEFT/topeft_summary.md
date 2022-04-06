## Summary of topeft processor

This document summarizes `topeft.py`, focusing on parts where we access info from the `events` object, or <span style="color:green">put</span> new info into `events`, or access additional external files (e.g. txt, csv, root)

* [L117](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L117): Get dataset name from `events`, accesses 1 column from `events.metadata`
* [L151-156](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L151-L156): Get the physics objects we care about from the `events` object, accesses 5 columns (E.g.: `mu = events.Muon`, note that the rest of the code mainly uses these copies (Question: Are they actually copies?), e.g. from here forward we use `mu`, not `events.Muon`)
* [L158-162](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L158-L162): Calculate variables to be used in objects section, putting these into the `e` and `mu` objects (but this does not touch the `events` object)
* [L173](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L173): Access 1 column from `events` (`events.luminosityBlock`) to get a mask that specifies "good” data taking conditions
   * Note this uses an external txt file (~50K)
   * In principle this mask should be mostly `True` (`False` entries indicate something went wrong during data collection)
* [L179](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L179): Access 1 column to get EFT coefficients (`events["EFTfitCoefficients"]`)
* [L190-193](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L190-L193): Object selection for electrons
    * I.e. make make masks to specify electrons we want to keep), this uses the `e` object (not the `events.Electron` object)
    * Note: These are object level cuts, not event level cuts (e.g. they are masts for the columns of `e` (which is itself a column of `events`), not of `events`. This set of masks (pre selection, loose, fakable, tight) are increasingly tight (were the tightest cuts mask a good fraction of the electron objects, but note we cannot apply this mask right now, as we still need to make use of some of the looser selections).
* [L214-234](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L214-L234): Get weights for some of the scale factors and systematics
    * From the `events` object, we access 10 columns (events, genWeight, nominal/up/down weights for "L1PreFiringWeight" and "Pileup", also "LHEScaleWeight" and "PSWeight")
    * This step also puts 9 new columns into the events object
    * Then access 5 of these new columns
    * Not this uses external root files from `data/pileup` (~160K)
* [L243-802](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L243-L802): The rest of the processor is inside of a for loop over some systematics (so everything after this is repeated multiple times when running with systematics):
    * [L245-247](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L245-L247): Apply muon pt corrections
      * Note this uses an external txt files (~8M)
    * [L249-252](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L249-L252): Muon object selection (similar comments as electron object selection above). 
    * [L260](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L260): Put in 1 column into `events` for an invariant mass cut we use later on
    * [L266-272](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L266-L272): Calculate lepton scale factors and fake rates (does not use `events`)
      * Note this accesses values from external root and json files (~700K)
    * [L275-276](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L275-L276): Build collection of leptons we care about ("fakable leptons") from selected e and mu objects
    * [L278-283](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L278-L283): Tau selection (note we do not actually use taus)
    * [L287-343](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L287-L343): Jet selection: Access 2 columns from `events` (`events.caches[0]`, `events.fixedGridRhoFastjetAll`)
      * Note this uses info from external txt files (~700K)
      * Note that similar to the lepton selection, this is an object level selection (not an event level selection)
    * [L349-350](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L349-L350): Next put 2 columns into `events` (the collection of leptons we care about (`l_fo_conept_sorted`) and number of jets `njets`, though we do not actually use `events.njets`) 
    * [L353-355](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L353-L355): Event selection: construct masks to keep track of which events pass
      * This accesses 3 columns from `events` 3 times (two of these we put in ourselves), and puts 19 columns (the masks related to the selections) into `events`
      * Note that as opposed to the selection mentioned above (which were all object level selections) these selections are event level selections (so the `False` values in these masks correspond to events that do not pass the given event selection)
      * These selections are quite selective, where only a small fraction fo the entries in the masks should be `True` (depending of course on the input data and whether or not it has been skimmed)
    * [L356](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L356): Build masks for keeping track of the flavors of the leptons, accesses 1 column (which was put in ourselves), and puts 13 columns into events
    * [L375-378](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L375-L378): Get systematics for btag jets
      * Note this uses external csv and pkl files (~1.2M)
    * [L394](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L394): Calculate trigger scale factors: Access 7 columns from `events` (all of which we put in ourselves), and puts in 3 columns
      * Note this uses external pkl files (~18K)
    * [L395-418](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L395-L418): More systematics and scale factors: Access 19 columns (all of these we put into `events` ourselves)
    * [L429](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L429): Construct masks for the trigger selection: Accesses 2 columns (`events.HLT` and (unnecessarily) `events.MET.pt`)
    * [L451-515](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L451-L515): Using all of the information calculated above, construct the masks we will use for the final selection for each category: Access 20 columns from `events` (all of which are columns we have added ourselves)
      * Note: This is the part of the code can in principle change frequently (essentially whenever we decide to change the categories we include in the analysis, the masks we construct here are likely to change)  
    * [L518-576](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L518-L576): Calculate the variables we are interested in (i.e. get the variables that will constitute the dense axes of our histograms)
      * Note: This part of the code can change very frequently
    * [L579-689](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L579-L689): Construct the event selection categories
      * This is the part of the code where we map out which masks go with which categories
      * Note: This part of the code can change frequently
    * [L705-801](https://github.com/TopEFT/topcoffea/blob/3ba04eb74314f3a5ad10e2727522a386ebec3bca/analysis/topEFT/topeft.py#L705-L801): Finally we loop through categories, apply masks, fill histograms


### Summary of the sizes of the external files
The external files are located in `topcoffea/data`, and the total of all external files is O(10M).
```
1.2M	btagSF
430K	fromTTH
47K	goldenJsons
570K	JEC
115K	JER
278K	leptonSF
8.3M	MuonScale
160K	pileup
26K	scaleFactors # Not used?
18K	triggerSF
```

### Summary of the frequency with which the code changes

During development (i.e. while writing the processor) the entire processor can be subject to fairly frequent changes. However, in principle, once the object selection, event selection, and corrections are all put into place, the processor should remain fairly stable up to ~L451. On L451 onward, we define the categories and variables we are interested in studying, and this part of the code may continue to change frequently as we pursue new ideas for the analysis.
