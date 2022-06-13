import pickle
import gzip
import topcoffea.modules.HistEFT
import numpy as np
import uproot
import hist
import os
import re
import json
import time

from coffea.hist import StringBin, Cat, Bin

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth

PRECISION = 3   # Decimal point precision in the text datacard output

def prune_axis(h,axis,to_keep):
    to_remove = [x.name for x in h.identifiers(axis) if x.name not in to_keep]
    return h.remove(to_remove,axis)

def to_hist(arr,name):
    nbins = len(arr) - 2 # The passed in array already includes under/overflow bins
    h = hist.Hist(hist.axis.Regular(nbins,0,nbins,name=name))
    h[:] = arr[1:-1]    # Assign the bin values
    h[-1] += arr[-1]    # Add in the overflow bin to the right most bin content
    return h

class RateSystematic():
    def __init__(self,name,**kwargs):
        self.all = kwargs.pop("all",False)      # If true, this syst applies to all processes
        if self.all:
            try:
                self.all_unc = kwargs.pop("unc")
            except KeyError:
                msg = "Missing 'unc' argument. Must specify an uncertainty when using the 'all' option"
                raise KeyError(msg)
        self.name = name
        self.corrs = {}  # keys are the name of processes and values are the corresponding unc.

    def has_process(self,p):
        return self.all or (p in self.corrs)

    def add_process(self,p,v=None):
        if self.all:
            raise KeyError("Can't add a correlated process for systematic defined with the 'all' option")
        self.corrs[p] = v

    # TODO: This needs to be given a better name
    def get_process(self,p):
        if self.all:
            return self.all_unc
        if self.has_process(p):
            return self.corrs[p]
        else:
            # This is the case for a systematic that doesn't apply to the specified process
            return '-'

class DatacardMaker():
    # TODO:
    #   We are abusing the grouping mechanism to also handle renaming samples, but might want to
    #   separate into two distinct actions to make things easier to follow for the reader
    # Note:
    #   Care must be taken with regards to the underscores, due to 'nonprompt', 'data', and 'flips'
    GROUP = {
        "Diboson_": [
            "WZTo3LNu_",
            "WWTo2L2Nu_",
            "ZZTo4L_",
        ],
        "Triboson_": [
            "WWW_",
            "WWZ_",
            "WZZ_",
            "ZZZ_"
        ],
        "fakes": ["nonprompt"],
        "charge_flips_": ["flips"],
        "data_obs": ["data"],

        "ttH_": ["ttHJet_"],
        "ttll_": ["ttllJet_"],
        "ttlnu_": ["ttlnuJet_"],
    }

    # Controls how we rebin the dense axis of the corresponding distribution
    BINNING = {
        # TODO: njets re-binning still not correctly implemented
        "njets": {
            "2l": [4,5,6,7],
            "3l": [2,3,4,5],
            "4l": [2,3,4],    
        },

        "ptbl":    [0,100,200,400],
        "ht":      [0,300,500,800],
        "ljptsum": [0,400,600,1000],
        "ptz":     [0,200,300,400,500],
        "o0pt":    [0,100,200,400],
        "bl0pt":   [0,100,200,400],
        "l0pt":    [0,50,100,200],
        "lj0pt":   [0,150,250,500]
    }

    YEARS = ["UL16","UL16APV","UL17","UL18"]

    FNAME_TEMPLATE = "TESTING_ttx_multileptons-{cat}.{ext}"

    SIGNALS = set(["ttH","tllq","ttll","ttlnu","tHq","tttt"])

    @classmethod
    def get_year(cls,s):
        for yr in cls.YEARS:
            if s.endswith(yr): return yr
        return None

    @classmethod
    def strip_fluctuation(cls,s):
        return s.replace("Down","").replace("Up","")

    @classmethod
    def is_signal(cls,s):
        s = cls.get_process(s)
        return (s in cls.SIGNALS)

    @classmethod
    def is_per_year_systematic(cls,s):
        end_chks = [
            "_2016APVUp","_2016Up","_2017Up","_2018Up",
            "_2016APVDown","_2016Down","_2017Down","_2018Down",
        ]
        return any([s.endswith(x) for x in end_chks])

    # Checks if the string corresponds to the name of an EFT process term after decomposition
    @classmethod
    def is_eft_term(cls,s):
        chks = ["_lin_","_quad_"]
        return any([x in s for x in chks])

    # Strips off the year designation from a process name, can also be used for decomposed terms
    @classmethod
    def get_process(cls,s):
        for yr in cls.YEARS:
            if s.endswith(yr):
                s = s.replace(yr,"")
        if cls.is_eft_term(s):
            # For now we can simply split on first underscore, if signal process names get underscores
            #   will need to update this to be smarter
            s = s.split("_",1)[0]
        if "_" in s:
            s = s.rsplit("_",1)[0]
        return s

    # Returns the njet and bjet multiplicities corresponding to the bin name in (j,b) order
    # TODO: I don't like the naming
    @classmethod
    def get_jet_mults(cls,s):
        # Group 1 matches 'njet_bjet' and Group 2 matches 'bjet_njet' Group 3 matches '_njet'
        rgx = re.compile(r"(_[2-7]j_[1-2]b)|(_[1-2]b_[2-7]j)|(_[2-7]j$)")

        m = rgx.search(s)
        if m.group(1) and m.group(2) is None and m.group(3) is None:
            # The order is '_Nj_Mb'
            _,j,b = m.group(1).split("_")
        elif m.group(1) is None and m.group(2) and m.group(3) is None:
            # The order is '_Nb_Mj'
            _,b,j = m.group(2).split("_")
        elif m.group(1) is None and m.group(2) is None and m.group(3):
            # This occurs when the string ends in '_Mj' and doesn't have a bjet multiplicity
            b = None
            j = m.group(3).replace("_","")
        else:
            raise ValueError(f"Unable to find rgx match in string {s}")
        j = int(j.replace("j",""))
        if b is not None:
            b = int(b.replace("b",""))
        return (j,b)

    # Same thing but for lepton multiplicity
    @classmethod
    def get_lep_mult(cls,s):
        if s.startswith("2lss_"):
            return 2
        elif s.startswith("3l_"):
            return 3
        elif s.startswith("4l_"):
            return 4
        else:
            raise ValueError(f"Unable to determine lepton multiplicity from string {s}")

    @classmethod
    def get_processes_by_years(cls,h):
        r = {}
        for x in h.identifiers("sample"):
            p = cls.get_process(x.name)
            if p not in r:
                r[p] = []
            r[p].append(x.name)
        return r

    def __init__(self,pkl_path,**kwargs):
        self.year            = kwargs.pop("single_year","")
        self.do_sm           = kwargs.pop("do_sm",False)
        self.do_nuisance     = kwargs.pop("do_nuisance",False)
        self.var_lst         = kwargs.pop("var_lst",[])
        self.coeffs          = kwargs.pop("wcs",[])
        self.use_real_data   = kwargs.pop("unblind",False)
        self.verbose         = kwargs.pop("verbose",True)

        rate_syst_path = kwargs.pop("rate_systs_path","json/rate_systs.json")
        lumi_json_path = kwargs.pop("lumi_json_path","json/lumi.json")
        miss_part_path = kwargs.pop("missing_parton_path","data/missing_parton/missing_parton.root")

        # TODO: Need to find a better name for this variable
        self.rate_systs = self.load_systematics(rate_syst_path,miss_part_path)

        self.lumi = {}
        with open(topcoffea_path(lumi_json_path)) as f:
            jf = json.load(f)
            for yr,lm in jf.items():
                yr = yr.replace("20","UL")
                self.lumi[yr] = 1000*lm


        # Samples to be excluded from the datacard
        self.ignore = [
            "DYJetsToLL", "DY10to50", "DY50",
            "ST_antitop_t-channel", "ST_top_s-channel", "ST_top_t-channel", "tbarW", "tW",
            "TTJets", "TTTo2L2Nu", "TTToSemiLeptonic",
            "WJetsToLNu",

            "TTGJets",

            # "data","flips","nonprompt",
            # "flips","nonprompt",
            # "data",

            # "tttt","ttlnuJet","ttHJet", "tllq", "tHq",    # Keeps ttll
            # "tttt","ttlnuJet","ttHJet", "tllq",
            # "tttt","ttlnuJet","ttHJet",

            # "tttt","ttlnuJet","tllq","ttllJet","tHq", # Keeps ttH
            # "tttt","tllq","ttllJet","tHq","ttHJet", # Keeps ttlnu
        ]

        self.tolerance = 1e-4#1e-5
        self.hists = None

        tic = time.time()
        self.read(pkl_path)
        dt = time.time() - tic
        print(f"Total Read+Prune Time: {dt:.2f} s")

    def read(self,fpath):
        print(f"Opening: {fpath}")
        tic = time.time()
        self.hists = pickle.load(gzip.open(fpath))
        dt = time.time() - tic
        print(f"Pkl Open Time: {dt:.2f} s")

        for km_dist,h in self.hists.items():
            if len(h.values()) == 0: continue
            if self.var_lst and not km_dist in self.var_lst: continue
            print(f"Loading: {km_dist}")
            # Remove samples that we don't include in the datacard
            to_remove = []
            for x in h.identifiers("sample"):
                p = self.get_process(x.name)
                if p in self.ignore:
                    to_remove.append(x.name)
            h = h.remove(to_remove,"sample")

            if not self.do_nuisance:
                # Remove all shape systematics
                h = prune_axis(h,"systematic",["nominal"])

            if km_dist != "njets":
                edge_arr = self.BINNING[km_dist] + [h.axis(km_dist).edges()[-1]]
                h = h.rebin(km_dist,Bin(km_dist,h.axis(km_dist).label,edge_arr))
            else:
                # TODO: Still need to handle this case properly
                pass

            # Scale the histograms to intg. luminosity based on years
            scale_map = {}
            for x in h.identifiers("sample"):
                yr = self.get_year(x.name)
                proc = self.get_process(x.name)
                if proc == "data":
                    scale_map[x.name] = 1
                else:
                    scale_map[x.name] = self.lumi[yr]
            h.scale(scale_map,axis="sample")

            # Remove 'central', 'private', '_4F' text from sample names
            grp_map = {}
            for x in h.identifiers("sample"):
                new_name = x.name.replace("private","").replace("central","").replace("_4F","")
                grp_map[new_name] = x.name
            h = h.group("sample",Cat("sample","sample"),grp_map)

            h = self.group_processes(h)
            h = self.correlate_years(h)

            num_systs = len(h.identifiers("systematic"))
            print(f"Num. Systematics: {num_systs}")

            self.hists[km_dist] = h

    def channels(self,km_dist):
        return [x.name for x in self.hists[km_dist].identifiers("channel")]

    # Parse out the correlated and decorrelated systematics from rate_systs.json and missing_parton.root files
    # TODO: Can be a static member function
    def load_systematics(self,rs_fpath,mp_fpath):
        rate_systs = {}
        if not self.do_nuisance:
            return rate_systs
        fpath = topcoffea_path(rs_fpath)
        print(f"Opening: {fpath}")
        with open(fpath) as f:
            rates_json = json.load(f)
        for k1,v1 in rates_json["rate_uncertainties"].items():
            # k1 will be the name of a rate systematic, like 'lumi' or 'pdf_scale'
            if isinstance(v1,dict):
                # This is a correlated rate systematic
                syst_name = f"{k1}"
                new_syst = RateSystematic(syst_name)
                for k2,v2 in v1.items():
                    # k2 will be the name of process, like 'charge_flips' or 'ttH' or 'Diboson'
                    p = self.get_process(k2)
                    new_syst.add_process(p,v2)
                rate_systs[k1] = new_syst
            else:
                # The systematic gets applied to everything
                syst_name = f"{k1}"
                new_syst = RateSystematic(syst_name,all=True,unc=v1)
                rate_systs[k1] = new_syst

        # Certain rate systematics are only correlated between subsets of processes
        to_remove = set()
        for p,corr_systs in rates_json["correlations"].items():
            # 'p' is the name of a process and 'corr_systs' is a dictionary defining which rate systematic
            #   needs to be decorrelated into a specific sub-group, e.g. for ttH: pdf_scale -> pdf_scale_gg
            for syst,grp in corr_systs.items():
                # 'syst' should be the name of a systematic already defined in the 'rate_systs' dictionary
                #   and 'grp' is the string we are going to differentiate it from the other variants of 'syst'
                to_remove.add(syst)
                syst_name = f"{syst}_{grp}"
                if not syst_name in rate_systs:
                    rate_systs[syst_name] = RateSystematic(syst_name)
                if not rate_systs[syst].has_process(p):
                    print(f"Warning: No process {p} found for {syst} systematic")
                    continue
                unc = rate_systs[syst].get_process(p)
                rate_systs[syst_name].add_process(p,unc)
        # Now lets remove the original systematics which we decorrelated into sub-groups
        for syst in to_remove:
            rate_systs.pop(syst)

        # Note: The 'jet_scale' and 'missing_parton' uncertainties are a bit special, the values we
        #   store in their corresponding RateSystematic objects will be dictionaries that encode the
        #   uncertainty split by njets

        # Now deal with the 'jet_scale' systematic for Dibosons
        syst_name = "jet_scale"
        new_syst = RateSystematic(syst_name)
        for p,per_jet_uncs in rates_json["jet_scale"].items():
            new_syst.add_process(p,per_jet_uncs)
        rate_systs[syst_name] = new_syst

        # Finally, deal with the missing_parton systematic
        # TODO: This feels pretty hardcoded, but not sure there's any way around it
        branch_key = "tllq"
        syst_name = "missing_parton"
        new_syst = RateSystematic(syst_name)

        fpath = topcoffea_path(mp_fpath)
        print(f"Opening: {fpath}")
        with uproot.open(fpath) as f:
            d = {}
            for k in f.keys():
                k = k.replace(";1","")
                d[k] = f[f"{k}/{branch_key}"].array()
            new_syst.add_process("tllq",d)
            new_syst.add_process("tHq",d)
        rate_systs[syst_name] = new_syst

        return rate_systs

    # Group same type processes together
    # TODO: Can be a static member function
    def group_processes(self,h):
        # TODO: This needs work to be less convoluted...
        all_procs = set(x.name for x in h.identifiers("sample"))
        grp_map = {}
        for grp_name,to_grp in self.GROUP.items():
            for yr in self.YEARS:
                new_name = f"{grp_name}{yr}"
                lst = []
                for x in to_grp:
                    old_name = f"{x}{yr}"
                    if old_name in all_procs:
                        lst.append(old_name)
                        all_procs.remove(old_name)
                # Note: Some samples only exist in certain channels (e.g. flips), so we need to
                #   skip them when they don't appear in the identifiers list
                if len(lst):
                    grp_map[new_name] = lst
        # Include back in everything that wasn't specified by the initial groupings
        for x in all_procs:
            grp_map[x] = [x]
        h = h.group("sample",Cat("sample","sample"),grp_map)
        return h

    # Combine stuff over years
    # TODO: Can be a static member function
    def correlate_years(self,h):
        if not self.do_nuisance:
            # Only sum over the years, don't mess with nuisance stuff
            grp_map = {}
            for x in h.identifiers("sample"):
                p = self.get_process(x.name)
                if p not in grp_map:
                    grp_map[p] = []
                grp_map[p].append(x.name)
            h = h.group("sample",Cat("sample","sample"),grp_map)
            return h
        # This requires some fancy footwork to make work
        print("Correlating years")

        # First we need to get a histogram with only the nominal contributions
        h_nom = prune_axis(h,"systematic",["nominal"])

        by_years = self.get_processes_by_years(h_nom)
        # Next we need to create a group mapping for each sample that combines all but one of the years
        grp_map = {}
        for p,p_yrs in by_years.items():
            for p_yr in p_yrs:
                # The list is all OTHER proc+year combos
                grp_map[p_yr] = [x for x in p_yrs if x != p_yr]
        h_nom = h_nom.group("sample",Cat("sample","sample"),grp_map)

        # Now for the tricky part
        for sp_key in h._sumw.keys():
            syst = ""
            nom_key = []
            for i,sp_field in enumerate(h.fields[:-1]):  # This gives us the ordering of the sparse axes for the sparse key
                if sp_field == "systematic":
                    nom_key.append(StringBin("nominal"))
                    syst = sp_key[i].name
                else:
                    nom_key.append(sp_key[i])
            nom_key = tuple(nom_key)
            if syst == "nominal":
                continue
            if not self.is_per_year_systematic(syst):
                continue
            # Remember, h_nom only has 1 "systematic" bin called "nominal", which is the sum of the
            #   the OTHER proc_years, e.g. ttH_UL16 nominal is really ttH_UL16APV+ttH_UL17+ttH_UL18
            #   so when we add it to the per-year systematic, it has the proper decorrelated yield
            nom_arr = h_nom._sumw[nom_key]
            h._sumw[sp_key] += nom_arr

        # Finally sum over years, since the per-year systematics only appear in a corresponding
        #   "sample year", the grouping for those systematics just adds itself with nothing from
        #   the other sample years
        grp_map = {}
        for x in h.identifiers("sample"):
            p = self.get_process(x.name)
            if p not in grp_map:
                grp_map[p] = []
            grp_map[p].append(x.name)
        h = h.group("sample",Cat("sample","sample"),grp_map)

        return h

    def get_selected_wcs(self,km_dist):
        print("Selecting WCs")
        tic = time.time()
        h = self.hists[km_dist].integrate("systematic",["nominal"])

        procs = [x.name for x in h.identifiers("sample")]
        selected_wcs = {p: set() for p in procs}

        wcs = ["sm"] + h._wcnames

        # This maps a WC to a list whose elements are the indices of the coefficient array of the
        #   HistEFT that involve that particular WC
        # NOTE: Building up the index array MUST match exactly with how the HistEFT coeff array is
        #       constructed/computed [1], otherwise the index array that gets computed won't pick
        #       out the correct coeff array indices for the corresponding WC!
        # [1] https://github.com/TopEFT/topcoffea/blob/3bef686fead216183ebb6dfb464e67629cfe75f5/topcoffea/modules/eft_helper.py#L32-L36
        wc_to_terms = {}
        index = 0
        for i in range(len(wcs)):
            wc1 = wcs[i]
            wc_to_terms[wc1] = set()
            for j in range(i+1):
                wc2 = wcs[j]
                wc_to_terms[wc1].add(index)
                wc_to_terms[wc2].add(index)
                index += 1

        # Convert the set to a sorted np.array
        for wc in wcs:
            wc_to_terms[wc] = np.array(sorted(wc_to_terms[wc]))

        for p in procs:
            if not self.is_signal(p):
                continue
            p_hist = h.integrate("sample",[p])
            for wc,idx_arr in wc_to_terms.items():
                if len(self.coeffs) and not wc in self.coeffs:
                    continue
                if wc == "sm":
                    continue
                if wc == "ctlTi" and p == "tttt":
                    continue
                for (ch,),arr in p_hist._sumw.items():
                    # Ignore nanflow,underflow, and overflow bins
                    sl_arr = arr[2:-1]
                    # Here we replace any SM terms that are too small with a large dummy value
                    sm_norm = np.where(sl_arr[:,0] < 1e-5,999,sl_arr[:,0])
                    # Normalize each sub-array by corresponding SM term
                    n_arr = (sl_arr.T / sm_norm).T
                    # This will contain only the coefficients which involve the given WC
                    wc_terms = np.abs(n_arr[:,idx_arr])
                    if np.any(wc_terms > self.tolerance):
                        selected_wcs[p].add(wc)
                        break
        dt = time.time() - tic
        print(f"WC Selection Time: {dt:.2f} s")

        return selected_wcs

    def analyze(self,km_dist,ch,selected_wcs):
        if not km_dist in self.hists:
            print(f"[ERROR] Unknown kinematic distribution: {km_dist}")
            return None
        elif StringBin(ch) not in self.hists[km_dist].identifiers("channel"):
            print(f"[ERROR] Unknown channel {ch}")
            return None

        print(f"Analyzing {km_dist} in {ch}")

        bin_str = f"bin_{ch}_{km_dist}"
        col_width = len(bin_str)
        syst_width = 0

        if km_dist != "njets":
            num_j,num_b = self.get_jet_mults(ch)
        else:
            num_j,num_b = 0,0
        num_l = self.get_lep_mult(ch)
        if num_l == 2 or num_l == 4:
            num_b = 2

        outf_root_name = self.FNAME_TEMPLATE.format(cat=ch,ext="root")

        h = self.hists[km_dist]
        ch_hist = h.integrate("channel",[ch])
        data_obs = np.zeros(ch_hist._dense_shape[0] - 1) # '_dense_shape' includes the nanflow bin

        print(f"Generating root file: {outf_root_name}")
        tic = time.time()
        num_h = 0
        all_shapes = set()
        text_card_info = {}
        with uproot.recreate(outf_root_name) as f:
            for p,wcs in selected_wcs.items():
                proc_hist = ch_hist.integrate("sample",[p])
                if self.verbose:
                    print(f"Decomposing {ch}-{p}")
                decomposed_templates = self.decompose(proc_hist,wcs)
                # Note: This feels like a messy way of picking out the data_obs info
                if p == "data":
                    data_sm = decomposed_templates.pop("sm")
                    if self.use_real_data:
                        if len(data_sm) != 1:
                            raise RuntimeError("obs data has unexpected number of sparse bins")
                        elif sum(data_obs) != 0:
                            raise RuntimeError("filling obs data more than once!")
                        for sp_key,arr in data_sm.items():
                            data_obs += arr
                for base,v in decomposed_templates.items():
                    proc_name = f"{p}_{base}"
                    col_width = max(len(proc_name),col_width)
                    text_card_info[proc_name] = {
                        "shapes": set(),
                        "rate": -1
                    }
                    # There should be only 1 sparse axis at this point, the systematics axis
                    for sp_key,arr in v.items():
                        syst = sp_key[0]
                        sum_arr = sum(arr)
                        if syst == "nominal" and base == "sm":
                            if self.verbose:
                                print(f"\t{proc_name:<12}: {sum_arr:.4f} {arr}")
                            if not self.use_real_data:
                                # Create asimov dataset
                                data_obs += arr
                        if syst == "nominal":
                            hist_name = f"{proc_name}"
                            text_card_info[proc_name]["rate"] = sum_arr
                        else:
                            hist_name = f"{proc_name}_{syst}"
                            all_shapes.add(syst)
                            text_card_info[proc_name]["shapes"].add(syst)
                            syst_width = max(len(syst),syst_width)
                        f[hist_name] = to_hist(arr,hist_name)
                        num_h += 1
                    if km_dist == "njets":
                        # We need to handle certain systematics differently when looking at njets procs
                        if p == "Diboson":
                            # Handle the 'jet_scale' uncertainty
                            # syst = "jet_scale"
                            # hist_name = f"{proc_name}_{syst}"
                            # syst_kappa = self.rate_systs[syst].get_process(p)[str(num_j)]
                            # if syst_kappa == "-":
                            #     raise ValueError(f"The kappa value for {syst} is missing!")
                            pass

                        if p == "tllq" or p == "tHq":
                            # Handle the 'missing_parton' uncertainty
                            pass
            f["data_obs"] = to_hist(data_obs,"data_obs")

        line_break = "##----------------------------------\n"
        left_width = len(line_break) + 2
        left_width = max(syst_width+len("shape")+1,left_width)

        outf_card_name = self.FNAME_TEMPLATE.format(cat=ch,ext="txt")
        print(f"Generating text file: {outf_card_name}")
        with open(outf_card_name,"w") as f:
            f.write(f"shapes *        * {outf_root_name} $PROCESS $PROCESS_$SYSTEMATIC\n")
            f.write(line_break)
            f.write(f"bin         {bin_str}\n")
            f.write(f"observation {sum(data_obs):.{PRECISION}f}\n")
            f.write(line_break)
            f.write(line_break)

            # proc_order = sorted(text_card_info.keys())
            proc_order = text_card_info.keys()

            # Bin row
            row = [f"{'bin':<{left_width}}"]    # Python string formatting is pretty great!
            for p in proc_order:
                row.append(f"{bin_str:>{col_width}}")
            row = " ".join(row) + "\n"
            f.write(row)

            # 1st process row
            row = [f"{'process':<{left_width}}"]
            for p in proc_order:
                row.append(f"{p:>{col_width}}")
            row = " ".join(row) + "\n"
            f.write(row)

            # 2nd process row
            row = [f"{'process':<{left_width}}"]
            bkgd_count =  1
            sgnl_count = -1
            for p in proc_order:
                if any([x in p for x in self.SIGNALS]): # Check for if the process is signal or not
                    row.append(f"{sgnl_count:>{col_width}}")
                    sgnl_count += -1
                else:
                    row.append(f"{bkgd_count:>{col_width}}")
                    bkgd_count += 1
            row = " ".join(row) + "\n"
            f.write(row)

            # Rate row
            row = [f"{'rate':<{left_width}}"]
            for p in proc_order:
                r = text_card_info[p]["rate"]
                row.append(f"{r:>{col_width}.{PRECISION}f}") # Do not challenge me on Python string formatting!
            row = " ".join(row) + "\n"
            f.write(row)
            f.write(line_break)

            # Shape systematics rows
            for syst in sorted(all_shapes):
                left_text = f"{syst:<{syst_width}} shape"
                row = [f"{left_text:<{left_width}}"]
                for p in proc_order:
                    if syst in text_card_info[p]["shapes"]:
                        row.append(f"{'1':>{col_width}}")
                    else:
                        row.append(f"{'-':>{col_width}}")
                row = " ".join(row) + "\n"
                f.write(row)

            # Rate systematics rows
            for k,rate_syst in self.rate_systs.items():
                syst_name = rate_syst.name
                left_text = f"{syst_name:<{syst_width}} lnN"
                if km_dist == "njets" and (syst_name == "jet_scale" or syst_name == "missing_parton"):
                    # These systematics are only treated as rate systs for njets distribution
                    continue
                row = [f"{left_text:<{left_width}}"]
                for p in proc_order:
                    proc_name = self.get_process(p) # Strips off any "_sm" or "_lin_*" junk
                    v = rate_syst.get_process(proc_name)
                    # Need to handle certain systematics in a special way
                    if syst_name == "jet_scale":
                        v = rate_syst.get_process(proc_name)
                        if isinstance(v,dict):
                            v = v[str(num_j)]
                    elif syst_name == "missing_parton":
                        v = rate_syst.get_process(proc_name)
                        # First strip off any njet and/or bjet labels
                        ch_key = ch.replace(f"_{num_j}j","").replace(f"_{num_b}b","")
                        # Now construct the category key, matching names in the missing_parton file to the current category
                        if num_l == 2:
                            njet_offset = 4
                            ch_key = f"{ch_key}_{num_b}b"
                        elif num_l == 3:
                            njet_offset = 2
                            if "_onZ" in ch:
                                ch_key = f"{num_l}l_sfz_{num_b}b"
                            elif "_p_offZ" in ch:
                                ch_key = f"{num_l}l{num_b}b_p"
                            elif "_m_offZ" in ch:
                                ch_key = f"{num_l}l{num_b}b_m"
                            else:
                                raise ValueError(f"Unable to match {ch} for {syst_name} rate systematic")
                        elif num_l == 4:
                            njet_offset = 2
                            ch_key = f"{ch_key}_{num_b}b"
                        else:
                            raise ValueError(f"Unable to match {ch} for {syst_name} rate systematic")
                        # The bins in the missing_parton root files start indexing from 0
                        bin_idx = num_j - njet_offset
                        if isinstance(v,dict):
                            v = v[ch_key][bin_idx]
                            v = f"{v:.{PRECISION}f}"
                    row.append(f"{v:>{col_width}}")
                row = " ".join(row) + "\n"
                f.write(row)
        dt = time.time() - tic
        print(f"File Write Time: {dt:.2f} s")
        print(f"Total Hists Written: {num_h}")

    # TODO: Can be a static member function
    def decompose(self,h,wcs):
        '''
            Note: All other WCs are assumed set to 0
            sm piece:    set(c1=0.0)
            lin piece:   set(c1=1.0)
            mixed piece: set(c1=1.0,c2=1.0)
            quad piece:  0.5*[set(c1=2.0) - 2*set(c1=1.0) + set(sm)]
        '''

        tic = time.time()
        h.set_sm()
        sm = h.values(overflow='all')

        # Note: The keys of this dictionary are a pretty contrived, but are useful later on
        r = {}
        r["sm"] = sm
        terms = 1
        for n1,wc1 in enumerate(wcs):
            h.set_wilson_coefficients(**{wc1: 1.0})
            tmp_lin_1 = h.values(overflow='all')
            h.set_wilson_coefficients(**{wc1: 2.0})
            tmp_lin_2 = h.values(overflow='all')

            lin_name = f"lin_{wc1}"
            quad_name = f"quad_{wc1}"

            terms += 2

            r[lin_name] = tmp_lin_1
            r[quad_name] = {}
            for sparse_key in h._sumw.keys():
                tup = tuple(x.name for x in sparse_key)
                r[quad_name][tup] = 0.5*(tmp_lin_2[tup] - 2*tmp_lin_1[tup] + sm[tup])

            for n2,wc2 in enumerate(wcs):
                if n1 >= n2: continue
                mixed_name = f"quad_mixed_{wc1}_{wc2}"
                h.set_wilson_coefficients(**{wc1:1.0,wc2:1.0})
                r[mixed_name] = h.values(overflow='all')
                terms += 1

        toc = time.time()
        dt = toc - tic
        if self.verbose:
            print(f"\tDecompose Time: {dt:.2f} s")
            print(f"\tTotal terms: {terms}")

        return r

if __name__ == '__main__':
    fpath = topcoffea_path("../analysis/topEFT/histos/may18_fullRun2_withSys_anatest08_np.pkl.gz")

    tic = time.time()
    dc = DatacardMaker(fpath)

    km_dist = "lj0pt"
    chans = ["2lss_m_4j","2lss_4t_m_4j"]
    # km_dist = "njets"
    # chans = ["2lss_m","2lss_4t_m"]

    target_selected = {
        "tHq": ["ctp", "cptb", "cQq13", "cbW", "cpQ3", "ctW", "cQq83", "ctG"],
        "tllq": ["cpt", "cptb", "cQlMi", "cQl3i", "ctlTi", "ctli", "cQq13", "cbW", "cpQM", "cpQ3", "ctei", "cQei", "ctW", "ctlSi", "cQq83", "ctZ", "ctG"],
        "ttH": ["cpt", "ctp", "cptb", "cQq81", "cQq11", "ctq8", "ctq1", "cQq13", "cbW", "cpQM", "cpQ3", "ctW", "cQq83", "ctZ", "ctG"],
        "ttll": ["cpt", "cptb", "cQlMi", "cQq81", "cQq11", "cQl3i", "ctq8", "ctlTi", "ctq1", "ctli", "cQq13", "cbW", "cpQM", "cpQ3", "ctei", "cQei", "ctW", "ctlSi", "cQq83", "ctZ", "ctG"],
        "ttlnu": ["cpt", "ctp", "cQlMi", "cQq81", "cQq11", "cQl3i", "ctq8", "ctlTi", "ctq1", "ctli", "cQq13", "cpQM", "cpQ3", "ctW", "ctlSi", "cQq83", "ctZ", "ctG"],
        "tttt": ["cpt", "ctp", "cptb", "cQq81", "cQq11", "ctq8", "ctq1", "cQq13", "cbW", "cpQM", "cpQ3", "ctW", "cQq83", "ctZ", "ctG", "ctt1", "cQt1", "cQt8", "cQQ1"]
    }

    selected_wcs = dc.get_selected_wcs(km_dist)
    for p,tar_wcs in target_selected.items():
        if p not in selected_wcs:
            print(f"Skipping {p} for selected WC comparison")
            continue
        sel_wcs = selected_wcs[p]
        print(f"old {p:>5}: {sorted(tar_wcs)}")
        print(f"new {p:>5}: {sorted(sel_wcs)}")
        miss_old = set(tar_wcs).difference(sel_wcs)
        miss_new = sel_wcs.difference(set(tar_wcs))

        print(f"Missing from old: {sorted(miss_old)}")
        print(f"Missing from new: {sorted(miss_new)}")
        print("-"*50)

    for cat in dc.channels(km_dist):
        if not cat in chans:
            continue
        r = dc.analyze(km_dist,cat,selected_wcs)
    dt = time.time() - tic
    print(f"Total Time: {dt:.2f} s")

    wc_to_terms = {}
    h = dc.hists[km_dist]
    wcs = ["sm"] + h._wcnames

    index = 0
    for i in range(len(wcs)):
        wc1 = wcs[i]
        wc_to_terms[wc1] = set()
        for j in range(i+1):
            wc2 = wcs[j]
            wc_to_terms[wc1].add(index)
            wc_to_terms[wc2].add(index)
            index += 1

    for wc in wcs:
        terms = sorted(wc_to_terms[wc])
        s1 = ", ".join([f"{x:>3d}" for x in terms[:6]])
        s2 = terms[-1]
        print(f"{wc:>5}: [{s1}, ... , {s2:>3d} ]")
