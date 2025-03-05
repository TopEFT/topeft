import pickle
import gzip
import numpy as np
import boost_histogram as bh
import uproot
import hist
import os
import re
import json
import time

from collections import defaultdict

from topcoffea.modules.utils import regex_match
from topeft.modules.paths import topeft_path
from topeft.modules.axes import info as axes_info
from topeft.modules.compatibility import add_sumw2_stub


PRECISION = 6   # Decimal point precision in the text datacard output

def to_hist(arr,name,zero_wgts=False):
    """
        Converts a numpy array into a hist.Hist object suitable for being written to a root file by
        uproot. If 'zero_wgts' is true, then the resulting histogram will be created with bin errors
        set to 0 (instead of left unset)
    """
    # NOTE:
    #   If we don't instantiate a new np.array here, then clipped will store a reference to the
    #   sub-array arr and when we modify clipped, it will propagate back to arr as well!
    clipped = []
    for i in range(2):  # first entry is sum(weight), second entry is sum(weight^2)
        if arr[i] is not None:
            clipped.append(np.array(arr[i][1:]))  # Strip off the underoverflow bin
        else:
            clipped[i] = None

    nbins = len(clipped[0])
    h = hist.Hist(hist.axis.Regular(nbins,0,nbins,name=name),storage=bh.storage.Weight())
    if zero_wgts:
        h[...] = np.stack([clipped[0],np.zeros_like(clipped[0])],axis=-1) # Set the bin errors all to 0
    else:
        h[...] = np.stack([clipped[0], clipped[1]],axis=-1)
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
    # Returns the corresponding unc. (i.e. kappa values) that have been associated with a particular process
    # Note: The return value should be as a string
    def get_process(self,p):
        if self.all:
            return self.all_unc
        if self.has_process(p):
            return self.corrs[p]
        else:
            # This is the case for a systematic that doesn't apply to the specified process
            return '-'

class JetScale(RateSystematic):
    def __init__(self,name,**kwargs):
        super().__init__(name,**kwargs)

        self.symmeterize = True     # whether or not we attempt to make the up/down shifts equal in absolute terms
        self.min_lo = 0.01          # For large kappa values, do not let the symmeterization go negative

    # Override the base implementation to handle the different dict structure
    # Note: The return value should be as a string
    def get_process(self,p,j):
        j = str(j)
        if self.all:
            unc_hi = self.all_unc[j]
            if self.symmeterize:
                unc_lo = max(self.min_lo,2 - unc_hi)
                return f"{unc_lo:.{PRECISION}f}/{unc_hi:.{PRECISION}f}"
            else:
                return f"{unc_hi:.{PRECISION}f}"
        if self.has_process(p):
            unc_hi = self.corrs[p][j]
            if self.symmeterize:
                unc_lo = max(self.min_lo,2 - unc_hi)
                return f"{unc_lo:.{PRECISION}f}/{unc_hi:.{PRECISION}f}"
            else:
                return f"{unc_hi:.{PRECISION}f}"
        else:
            return '-'

class MissingParton(RateSystematic):
    # Maps channel name from pkl file to hist name in missing_parton.root file
    CH_MAP = {
        "2lss_4t_m": "2lss_4t_m_2b",
        "2lss_4t_p": "2lss_4t_p_2b",
        "2lss_m": "2lss_m_2b",
        "2lss_p": "2lss_p_2b",
        "2lss_fwd_m": "2lss_fwd_m_2b",
        "2lss_fwd_p": "2lss_fwd_p_2b",
        "3l_onZ_1b": "3l_sfz_1b",
        "3l_onZ_2b": "3l_sfz_2b",
        "3l_p_offZ_1b": "3l1b_p",
        "3l_m_offZ_1b": "3l1b_m",
        "3l_p_offZ_2b": "3l2b_p",
        "3l_m_offZ_2b": "3l2b_m",
        "4l_2b": "4l",
        "2los_ph_CR_sf_Zg": "2lss_m_2b",  #CAUTION: temporary
        "2los_sf_ph": "2lss_m_2b", #CAUTION: temporary
        "2los_of_ph": "2lss_m_2b", #CAUTION: temporary
    }

    def __init__(self,name,**kwargs):
        super().__init__(name,**kwargs)

    # Override the base implementation to handle the different dict structure
    def get_process(self,p,ch,l,j,b):
        pass

class DatacardMaker():
    # TODO:
    #   We are abusing the grouping mechanism to also handle renaming processes, but might want to
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
            "ZZZ_",
        ],
        "tWZ": ["TWZToLL_"],
        #"convs": ["TTGamma_"],
        "fakes": ["nonprompt"],
        "fakePh": ["nonpromptPh"],
        "charge_flips_": ["flips"],
        "data_obs": ["data"],

        "ttH_": ["ttHJet_"],
        "ttll_": [
            "ttllJet_",
            "TTZToLL_M1to10_",
            "TTToSemiLeptonic_",
            "TTTo2L2Nu_",
        ],
        "ttlnu_": ["ttlnuJet_"],
        "ttA_": ["TTGamma_Dilept_","TTGamma_"],
    }

    # Controls how we rebin the dense axis of the corresponding distribution
    BINNING = {}
    for name, value in axes_info.items():
        if "variable" in value:
            BINNING[name] = value["variable"]

    YEARS = ["UL16","UL16APV","UL17","UL18"]

    SYST_YEARS = ["2016","2016APV","2017","2018"]

    FNAME_TEMPLATE = "ttx_multileptons-{cat}_{kmvar}.{ext}"
    # FNAME_TEMPLATE = "TESTING_ttx_multileptons-{cat}.{ext}"

    SIGNALS = set(["ttH","tllq","ttll","ttlnu","tHq","tttt","ttA"])

    @classmethod
    def get_year(cls,s):
        """
            Attempt to return the year of the process or systematic string
        """
        for yr in cls.YEARS:
            if s.endswith(yr): return yr
        for yr in cls.SYST_YEARS:
            if s.endswith(yr+"Up"): return yr
            if s.endswith(yr+"Down"): return yr
        return None

    @classmethod
    def strip_fluctuation(cls,s):
        return s.replace("Down","").replace("Up","")

    @classmethod
    def strip_year(cls,s):
        for yr in cls.YEARS:
            s = s.replace(yr,"")
        for yr in cls.SYST_YEARS:
            s = s.replace(f"_{yr}","")  # Note the underscore
        return s

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

    @classmethod
    def is_eft_term(cls,s):
        """ Check if string corresponds an EFT process term after decomposition."""
        chks = ["_lin_","_quad_"]
        return any([x in s for x in chks])

    @classmethod
    def get_process(cls,s):
        """ Strips off the year designation from a process name, can also be used for decomposed terms."""
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

    # TODO: I don't like the naming
    @classmethod
    def get_jet_mults(cls,s):
        """
            Returns the njet and bjet multiplicities based on the string passed to it in (j,b) order.
            For the regular expression, group 1 matches 'njet_bjet', group 2 matches 'bjet_njet'
            group 3 matches '_njet'.
        """
        rgx = re.compile(r"(_[2-7]j_[1-2]b)|(_[1-2]b_[2-7]j)|(_[1-7]j$)")

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

    @classmethod
    def get_lep_mult(cls,s):
        """ Returns the lepton multiplicity based on the string passed to it."""
        if s.startswith("2los_"):
            return 2.1 # Hack because photon 2los starts at 1 jet
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
        """
            Reads the 'process' sparse axis of a histogram and returns a dictionary that maps stripped
            process names to the list of sparse axis categories it came from.
        """
        r = defaultdict(lambda: [])
        for x in h.axes["process"]:
            p = cls.get_process(x)
            r[p].append(cls.get_process(x))
        return r

    def __init__(self,pkl_path,**kwargs):
        self.year_lst        = kwargs.pop("year_lst",[])
        self.do_sm           = kwargs.pop("do_sm",False)
        self.do_nuisance     = kwargs.pop("do_nuisance",False)
        self.drop_syst       = kwargs.pop("drop_syst",[])
        self.out_dir         = kwargs.pop("out_dir",".")
        self.var_lst         = kwargs.pop("var_lst",[])
        self.do_mc_stat      = kwargs.pop("do_mc_stat",False)
        self.coeffs          = kwargs.pop("wcs",[])
        self.use_real_data   = kwargs.pop("unblind",False)
        self.verbose         = kwargs.pop("verbose",True)
        self.use_AAC          = kwargs.pop("use_AAC",False)
        self.wc_scalings     = kwargs.pop("wc_scalings",[])
        self.scalings        = []

        # get wc ranges from json
        with open(topeft_path("params/wc_ranges.json"), "r") as wc_ranges_json:
            self.wc_ranges = json.load(wc_ranges_json)

        if self.year_lst:
            for yr in self.year_lst:
                if not yr in self.YEARS:
                    raise ValueError(f"Invalid year choice '{yr}', should be empty if running over all years or one of: {self.YEARS}")

        rate_syst_path = kwargs.pop("rate_systs_path","params/rate_systs.json")
        miss_part_path = kwargs.pop("missing_parton_path","data/missing_parton/missing_parton.root")

        # TODO: Need to find a better name for this variable
        self.rate_systs = self.load_systematics(rate_syst_path,miss_part_path)

        # Samples to be excluded from the datacard, should correspond to names before group_processes is run
        self.ignore = [
            "DYJetsToLL", "DY10to50", "DY50",
            "ST_antitop_t-channel", "ST_top_s-channel", "ST_top_t-channel", "tbarW", "tW",
            "TTJets",
            "WJetsToLNu",
            "TTGJets",  # This is the old low stats convs process, new one should be TTGamma
            #"TTGamma_central",

            # "TTGamma",
            # "WWTo2L2Nu","ZZTo4L",#"WZTo3LNu",
            # "WWW","WWW_4F","WWZ_4F","WWZ","WZZ","ZZZ",
            # "flips","nonprompt",
            # "tttt","ttlnuJet","tllq","tHq","ttHJet",
            # "TTTo2L2Nu", "TTToSemiLeptonic",
            # "data",
        ]

        if not self.use_real_data:
            # Since we're just going to generate Asimov data, this lets us drop the real data histograms
            #   from the histograms for a minor speed-up
            self.ignore.append("data")

        extra_ignore = kwargs.pop("ignore",[])

        # For now, we leave this as a hardcoded thing, a bit tedious but it works
        # Note: If not explicitly listed, it is assumed that all years should be uncorrelated
        # Note: It is important to list the correlations for ALL years in which it is relevant, so
        #       for example, if a systematic is correlated in 2016, 2016APV, and 2017, there needs
        #       to be an entry for all three years and the list corresponding to each entry needs to
        #       be consistent (i.e. contain the other correlated years) across all three entries
        # Note: As a final note, the actual systematic that appears in the datacards will be just one
        #       of the set to be correlated. So for example, if a systematic is correlated over 2016
        #       and 2016APV, then either the 2016 or 2016APV version will appear in the datacard, but
        #       not both. Typically, the one that remains will be the 2016 version as that's the one
        #       that gets handled first in the loop, but it would be different if we processed things
        #       in a different order
        self.syst_year_corr = {
            # Example of correlated for only 2016 and 2016APV
            "FFcloseEl": {"2016": ["2016APV"], "2016APV": ["2016"]},
            "FFcloseMu": {"2016": ["2016APV"], "2016APV": ["2016"]},

            # Example of correlated over 2016, 2016APV, 2017, and 2018
            # Note: This is not correct for the analysis, but just serves as an example
            # "FFcloseEl": {"2016": ["2016APV","2017","2018"], "2016APV": ["2016","2017","2018"], "2017": ["2016","2016APV","2018"], "2018": ["2016","2016APV","2017"]},
            # "FFcloseMu": {"2016": ["2016APV","2017","2018"], "2016APV": ["2016","2017","2018"], "2017": ["2016","2016APV","2018"], "2018": ["2016","2016APV","2017"]},
        }

        # Defines which systematics should be decorrelated in the self.analysis() step. Each key
        #   should match (exactly) a particular systematic. The list for each systematic specifies
        #   which processes should remain remain correlated or not.
        # Note: A given process should appear AT MOST once in the "matches" list for a given systematic
        #       grouping. If a process has an associated systematic, but doesn't match any of the
        #       groups, then it will retain its original systematic name (i.e. all unmatched
        #       processes will remain correlated).
        # Note: For the special case where the group name is an empty string the systematic will
        #       instead have the matched process' name appended to it, meaning that all matched
        #       processes will be decorrelated!
        # Note: Since the decorrelation happens during the self.analysis() step, the matched names
        #       should correspond to the renamed/re-grouped processes, e.g. use "Diboson" instead of
        #       "ZZ","WZ","WW".
        self.syst_shape_decorrelate = {
            "ISR": [
                {
                    "matches": ["ttH","ttll","tttt","ttA"],
                    "group": "gg",
                },
                {
                    "matches": ["ttlnu","tllq","Diboson","Triboson"],
                    "group": "qq",
                },
                {
                    "matches": ["tHq"],
                    "group": "qg"
                }
            ],
            "renorm": [{
                "matches": [".*"],
                "group": "",
            }],
            "fact": [{
                "matches": [".*"],
                "group": "",
            }]
        }

        self.syst_to_skip = {
            "ttA": "charge_flips"
        }

        if extra_ignore:
            print(f"Adding processes to ignore: {extra_ignore}")
        self.ignore.extend(extra_ignore)

        self.tolerance = 1e-4
        self.hists = None

        tic = time.time()
        self.read(pkl_path)
        dt = time.time() - tic
        print(f"Total Read+Prune Time: {dt:.2f} s")

        print (f"Saving output to {os.path.realpath(self.out_dir)}")

    def read(self,fpath):
        """
            Input should be a file path to a pkl file containing histograms produced by the topeft.py
            processor. The histograms are extracted and then pre-processed to remove / group / scale
            various sparse axes categories.
        """
        print(f"Opening: {fpath}")
        tic = time.time()
        self.hists = pickle.load(gzip.open(fpath))
        dt = time.time() - tic
        print(f"Pkl Open Time: {dt:.2f} s")

        for km_dist, h in self.hists.items():
            if h.empty():
                continue
            km_dist_missing = km_dist not in self.var_lst
            km_base_dist_missing = km_dist.replace('_sumw2','') not in self.var_lst
            if self.var_lst and km_dist_missing and km_base_dist_missing:
                continue
            print(f"Loading: {km_dist}")
            # Remove processes that we don't include in the datacard
            to_remove = []
            for x in h.axes["process"]:
                p = self.get_process(x)
                if p in self.ignore:
                    if self.verbose:
                        print(f"Skipping (ignored): {x}")
                    to_remove.append(x)
                    continue
                if self.year_lst:
                    yr = self.get_year(x)
                    if yr not in self.year_lst:
                        if self.verbose:
                            print(f"Skipping (year): {x}")
                        to_remove.append(x)
                        continue
            h = h.remove("process", to_remove)

            if not self.do_nuisance:
                # Remove all shape systematics
                h.prune("systematic", ["nominal"])

            if self.drop_syst:
                to_drop = set()
                for syst in self.drop_syst:
                    if syst.endswith("Up"):
                        to_drop.add(syst)
                    elif syst.endswith("Down"):
                        to_drop.add(syst)
                    else:
                        to_drop.add(f"{syst}Up")
                        to_drop.add(f"{syst}Down")
                for x in to_drop:
                    print(f"Removing systematic: {x}")
                h = h.remove("systematic", list(to_drop))

            if h.should_rebin() and km_dist != "njets":
                edge_arr = self.BINNING[km_dist] + [list(h.axes[km_dist].edges())[-1]]
                h = h.rebin(km_dist, hist.axis.Variable(edge_arr, km_dist, h.axes[km_dist].label))
            else:
                # TODO: Still need to handle njets case properly
                pass

            # Remove 'central', 'private', '_4F' text from process names
            grp_map = {}
            for x in h.axes["process"]:
                new_name = (
                    x.replace("private", "").replace("central", "").replace("_4F", "")
                )
                grp_map[new_name] = x
            h = h.group("process", grp_map)

            h = self.group_processes(h)
            h = self.correlate_years(h)

            num_systs = len(h.axes["systematic"])
            print(f"Num. Systematics: {num_systs}")

            self.hists[km_dist] = h

    def channels(self, km_dist):
        return list(self.hists[km_dist].axes["channel"])

    def processes(self, km_dist):
        return list(self.hists[km_dist].axes["process"])

    # TODO: Can be a static member function
    def load_systematics(self,rs_fpath,mp_fpath):
        """
            Parse out the correlated and decorrelated systematics from rate_systs.json and
            missing_parton.root files.
        """
        rate_systs = {}
        if not self.do_nuisance:
            return rate_systs
        fpath = topeft_path(rs_fpath)
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

        # Note: The 'diboson_njets' and 'missing_parton' uncertainties are a bit special, the values we
        #   store in their corresponding RateSystematic objects will be dictionaries that encode the
        #   uncertainty split by njets

        # Now deal with the 'diboson_njets' systematic for Dibosons
        syst_name = "diboson_njets"
        # new_syst = RateSystematic(syst_name)
        new_syst = JetScale(syst_name)
        for p,per_jet_uncs in rates_json["diboson_njets"].items():
            new_syst.add_process(p,per_jet_uncs)
        rate_systs[syst_name] = new_syst

        # Finally, deal with the missing_parton systematic
        # TODO: This feels pretty hardcoded, but not sure there's any way around it
        branch_key = "tllq"
        syst_name = "missing_parton"
        new_syst = RateSystematic(syst_name)

        fpath = topeft_path(mp_fpath)
        print(f"Opening: {fpath}")
        with uproot.open(fpath) as f:
            d = {}
            for k in f.keys():
                k = k.replace(";1","")
                # Note: Values in the ROOT file are computed as the fraction of the rate needed to
                #   reach agreement, so need to add 1 to get the corresponding kapaa value
                d[k] = f[f"{k}/{branch_key}"].array() + 1
            new_syst.add_process("tllq",d)
            new_syst.add_process("tHq",d)
        rate_systs[syst_name] = new_syst

        return rate_systs

    # TODO: Can be a static member function
    def group_processes(self,h):
        """
            Groups together certain processes from the 'process' axis. We also abuse this method to
            rename specific process categories. Both of which are determined by the GROUP static data
            member.
        """
        # TODO: This needs work to be less convoluted...
        all_procs = set(h.axes["process"])
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
                # Note: Some processes only exist in certain channels (e.g. flips), so we need to
                #   skip them when they don't appear in the identifiers list
                if len(lst):
                    grp_map[new_name] = lst
        # Include back in everything that wasn't specified by the initial groupings
        for x in all_procs:
            grp_map[x] = [x]
        h = h.group("process", grp_map)
        return h

    # TODO: Can be a static member function
    def correlate_years(self,h):
        """
            Merges together different run years, taking care to treat year-specific systematics as
            uncorrelated from one another
        """
        if not self.do_nuisance:
            # Only sum over the years, don't mess with nuisance stuff
            grp_map = defaultdict(lambda: [])
            for x in h.axes["process"]:
                p = self.get_process(x)
                grp_map[p].append(x)
            h = h.group("process", grp_map)
            return h
        # This requires some fancy footwork to make work
        print("Correlating years")

        # Need to figure out which years are actually present in the histogram
        unique_proc_years = set()
        for x in h.axes["process"]:
            yr = self.get_year(x)
            unique_proc_years.add(yr)

        already_correlated = set()  # Keeps track of which systematics have already been correlated
        for sp_key in h.categorical_keys:
            proc = sp_key.process
            syst = sp_key.systematic
            proc_year = self.get_year(proc)
            syst_year = self.get_year(syst)
            if syst_year is None:
                # This ensures that the systematic in question is a per-year systematic
                continue
            if syst in already_correlated:
                if self.verbose:
                    print(f"Skipping {syst} as it was already correlated in a previous year")
                continue
            syst_base = self.strip_fluctuation(syst)
            syst_base = self.strip_year(syst_base)
            corr_keys = []
            for p_yr, s_yr in zip(self.YEARS, self.SYST_YEARS):
                if p_yr not in unique_proc_years:
                    # The histogram file was generated by running over a subset of the years or we are
                    # only making cards for a certain year
                    continue
                if p_yr == proc_year:
                    # We never add self to self
                    continue
                if syst_base in self.syst_year_corr and s_yr in self.syst_year_corr[syst_base] and syst_year in self.syst_year_corr[syst_base][s_yr]:
                    # The systematic for this year needs to be correlated
                    syst_key = syst.replace(syst_year, s_yr)
                    already_correlated.add(syst_key)
                else:
                    # The systematic for this year needs to be uncorrelated
                    syst_key = "nominal"
                proc_key = proc.replace(proc_year, p_yr)

                # Construct the sparse key
                corr_key = sp_key._asdict()
                corr_key["process"] = proc_key
                corr_key["systematic"] = syst_key
                corr_key = type(sp_key)(**corr_key)
                corr_keys.append(corr_key)

            for k in corr_keys:
                h[sp_key] += h[k]

            sp_tup = tuple(sp_key)
            if self.verbose:
                print(f"{tuple(sp_tup)} -- {' + '.join(map(lambda k: str(tuple(k)), corr_keys))}")

        # Finally sum over years, since the per-year systematics only appear in a corresponding
        #   "process year", the grouping for those systematics just adds itself with nothing from
        #   the other process years
        grp_map = defaultdict(lambda: [])
        for x in h.axes["process"]:
            p = self.get_process(x)
            grp_map[p].append(x)
        h = h.group("process", grp_map)

        # Remove the categories which were already correlated together so as to not double count
        if already_correlated:
            for k in already_correlated:
                if self.verbose: print(f"Removing: {k}")
            h = h.remove("systematic", list(already_correlated))

        return h

    def get_selected_wcs(self,km_dist,ch_lst=[]):
        """
            For each process, iterates over every channel and every bin checking the EFT parameterization
            coefficients for if they have a significant impact or not relative to the SM contribution. If
            any term from any channel+bin is determined to be significant, the WC is selected, otherwise
            it is excluded for that process and won't be included in the EFT decomposition
        """
        tic = time.time()
        h = self.hists[km_dist].integrate("systematic",["nominal"])
        if ch_lst:
            # Only select from a subset of channels
            if self.verbose:
                print(f"Selecting WCs from subset of channels: {ch_lst}")
            h.prune("channel", ch_lst)

        procs = list(h.axes["process"])
        selected_wcs = {p: set() for p in procs}

        wcs = ["sm"] + h.wc_names

        # This maps a WC to a list whose elements are the indices of the coefficient array of the
        #   HistEFT that involve that particular WC
        # NOTE: Building up the index array MUST match exactly with how the HistEFT coeff array is
        #       constructed/computed [1], otherwise the index array that gets computed won't pick
        #       out the correct coeff array indices for the corresponding WC!
        # [1] https://github.com/TopEFT/topcoffea/blob/3bef686fead216183ebb6dfb464e67629cfe75f5/topcoffea/modules/eft_helper.py#L32-L36
        wc_to_terms = {}
        start_index = 1
        index = start_index
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
            p_hist = h.integrate("process",[p])
            for wc,idx_arr in wc_to_terms.items():
                if len(self.coeffs) and not wc in self.coeffs:
                    continue
                if wc == "sm":
                    continue
                if wc == "ctlTi" and p == "tttt":
                    continue
                for sp_key, arr in p_hist.view(as_dict=True, flow=True).items():
                    # Ignore underflow, and overflow bins
                    sl_arr = arr[1:-1]
                    # Here we replace any SM terms that are too small with a large dummy value
                    sm_norm = np.where(sl_arr[:,start_index] < 1e-5,999,sl_arr[:,start_index])
                    # Normalize each sub-array by corresponding SM term
                    n_arr = (sl_arr.T / sm_norm).T
                    # This will contain only the coefficients which involve the given WC
                    wc_terms = np.abs(n_arr[:,idx_arr])
                    if np.any(wc_terms > self.tolerance):
                        selected_wcs[p].add(wc)
                        break
        if self.verbose:
            dt = time.time() - tic
            print(f"WC Selection Time: {dt:.2f} s")
        return selected_wcs

    def make_scalings_json(self,scalings_json,ch,km_dist,p,wc_names,scalings):
        scalings = scalings.tolist()
        scalings_json.append(
            {
                "channel": ch + "_" + str(km_dist),
                "process": p + "_sm",  # NOTE: needs to be in the datacard
                "parameters": ["cSM[1]"] + [
                    self.format_wc(wcname) for wcname in wc_names
                ],
                "scaling":
                    scalings[1:], # exclude underflow bin
            }
        )
        return scalings_json

    def format_wc(self,wcname):
        lo, hi = self.wc_ranges[wcname]
        return "%s[0,%.1f,%.1f]" % (wcname, lo, hi)

    def analyze(self,km_dist,ch,selected_wcs, crop_negative_bins, wcs_dict):
        """ Handles the EFT decomposition and the actual writing of the ROOT and text datacard files."""
        if not km_dist in self.hists:
            print(f"[ERROR] Unknown kinematic distribution: {km_dist}")
            return None
        elif ch not in self.hists[km_dist].axes["channel"]:
            print(f"[ERROR] Unknown channel {ch}")
            return None

        print(f"Analyzing {km_dist} in {ch}")

        bin_str = f"bin_{ch}_{km_dist}"
        col_width = max(PRECISION*2+5,len(bin_str))
        syst_width = 0

        if km_dist != "njets":
            num_j,num_b = self.get_jet_mults(ch)
        else:
            num_j,num_b = 0,0
        num_l = self.get_lep_mult(ch)
        if num_l == 2 or num_l == 4:
            num_b = 2
        if num_l == 2.1:  #corresponds to the photon specific channels
            num_b = 1

        outf_root_name = self.FNAME_TEMPLATE.format(cat=ch,kmvar=km_dist,ext="root")

        h = self.hists[km_dist]
        h_sumw2 = None
        if f"{km_dist}_sumw2" in self.hists:
            h_sumw2 = self.hists[km_dist+"_sumw2"]
        else:
            msg = "No sumw2 histogram found! Setting errors to 0"
            print(msg)
        ch_hist = h.integrate("channel",[ch])
        ch_sumw2 = h_sumw2 if h_sumw2 is None else h_sumw2.integrate("channel",[ch])
        data_obs = np.zeros((2, ch_hist.dense_axis.extent))

        print(f"Generating root file: {outf_root_name}")
        tic = time.time()
        num_h = 0
        all_shapes = set()
        text_card_info = {}
        outf_root_name = os.path.join(self.out_dir,outf_root_name)
        with uproot.recreate(outf_root_name) as f:
            for p,wcs in selected_wcs.items():
                # TODO This is a hack for now, track this upstream
                if 'charge_flip' in p and '2l' not in ch:
                    continue
                if 'charge_flip' in p and '2los' in ch:
                    continue
                # TODO This is a hack for now, track this upstream
                if 'fakes' in p and '4l' in ch:
                    continue
                if 'fakePh' in p and '_ph' not in ch:
                    continue
                proc_hist = ch_hist.integrate("process",[p])
                proc_sumw2 = ch_sumw2 if ch_sumw2 is None else ch_sumw2.integrate("process",[p])
                if self.verbose:
                    print(f"Decomposing {ch}-{p}")
                decomposed_templates = self.decompose(proc_hist,proc_sumw2,wcs)
                is_eft = self.is_signal(p)
                # Note: This feels like a messy way of picking out the data_obs info
                if p == "data":
                    data_sm = decomposed_templates.pop("sm")
                    if self.use_real_data:
                        if len(data_sm) != 1:
                            raise RuntimeError("obs data has unexpected number of sparse bins")
                        elif sum(data_obs[0]) != 0:
                            raise RuntimeError("filling obs data more than once!")
                        for sp_key,arr in data_sm.items():
                            data_obs += arr
                if not self.use_AAC:
                    decomposed_templates = {k: v for k, v in decomposed_templates.items() if k == 'sm'}
                for base,v in decomposed_templates.items():
                    proc_name = f"{p}_{base}"
                    col_width = max(len(proc_name),col_width)
                    text_card_info[proc_name] = {
                        "shapes": set(),
                        "rate": -1
                    }
                    # There should be only 1 sparse axis at this point, the systematics axis
                    check_zero_arr0 = False
                    check_zero_arr1 = False
                    for sp_key,arr in v.items():
                        syst = sp_key[0]
                        if crop_negative_bins:
                            negative_bin_mask = np.where( arr[0] < 0) # see where bins are negative
                            arr[0][negative_bin_mask] = np.zeros_like( arr[0][negative_bin_mask] )  # set those to zero
                            if arr[1] is not None:
                                arr[1][negative_bin_mask] = np.zeros_like( arr[1][negative_bin_mask] )  # if there's a sumw2 defined, that one's set to zero as well. Otherwise we will get 0 +/- something, which is compatible with negative


                        syst = sp_key.systematic
                        if syst =="nominal":  # check systematics error for fake factors
                            if sum(arr[0]) == 0:
                                check_zero_arr0 = True
                            if sum(arr[1]) == 0:
                                check_zero_arr1 = True
                        if "FF" in syst:
                            if check_zero_arr0 and sum(arr[0]) != 0:
                                raise Warning("Systematics Error arr[0]:Zero values in 'nominal' but non-zero in '%s'" % (syst))
                            if check_zero_arr1 and sum(arr[1]) != 0:
                                raise Warning("Systematics Error arr[1]:Zero values in 'nominal' but non-zero in '%s'" % (syst))

                        #CAUTION: Not sure if we want this to be permanent. Employing this hack to handle bins with 0 sum
                        if not check_zero_arr0 and sum(arr[0]) == 0:
                            arr[0] = np.ones_like(arr[0]) * 0.01
                            print("Systematics Error arr[0]:Zero values in '%s' but non-zero in 'nominal'" % (syst))
                        if not check_zero_arr1 and sum(arr[1]) == 0:
                            arr[1] = np.ones_like(arr[1]) * 0.01
                            print("Systematics Error arr[1]:Zero values in '%s' but non-zero in 'nominal'" % (syst))

                        sum_arr = sum(arr[0])
                        if sum_arr == 0: continue #TODO find a more elegant solution
                        if syst == "nominal" and base == "sm":
                            if self.verbose:
                                print(f"\t{proc_name:<12}: {sum_arr:.4f} {arr[0]}")
                            if not self.use_real_data:
                                # Create asimov dataset
                                vals = wcs_dict # set wcs to certain values from command line
                                decomposed_templates_Asimov = self.decompose(proc_hist,proc_sumw2,wcs,vals)
                                data_sm = decomposed_templates_Asimov.pop("sm")
                                data_obs += data_sm[sp_key]
                        if syst == "nominal":
                            hist_name = f"{proc_name}"
                            text_card_info[proc_name]["rate"] = sum_arr
                        else:
                            hist_name = f"{proc_name}_{syst}"
                            # Systematics in the text datacard don't have the Up/Down postfix
                            syst_base = syst.replace("Up","").replace("Down","")
                            if syst_base in self.syst_shape_decorrelate:
                                # We want to split this systematic to be uncorrelated between certain
                                #   processes, so we modify the systematic name to make combine treat
                                #   them as separate systematics. Also, we use 'p' instead of 'proc_name'
                                #   for renaming since we want the decomposed EFT terms for a particular
                                #   process to share the same nuisance parameter
                                matched = []
                                for r in self.syst_shape_decorrelate[syst_base]:
                                    if regex_match([p],r["matches"]):
                                        # The matched process should have this systematic put into a new group
                                        matched.append(r["group"])
                                if len(matched) == 0:
                                    # No matches found, so keep the original systematic name
                                    split_syst = syst_base
                                elif len(matched) == 1:
                                    # Found a match, so decorrelate the process from non-matched processes
                                    group = matched[0]
                                    split_syst = f"{syst_base}_{group}"
                                    if group == "":
                                        # In the special case that the group is an empty string,
                                        #   decorrelate ALL matched processes
                                        split_syst = f"{syst_base}_{p}"
                                else:
                                    # We shouldn't have more than one match for a given systematic
                                    raise RuntimeError(f"Unable to decorrelate shape systematic {syst_base} for {p}. Multiple group matches found: {matched}")
                                hist_name = hist_name.replace(syst_base,split_syst)
                                all_shapes.add(split_syst)
                                text_card_info[proc_name]["shapes"].add(split_syst)
                                if base == "sm" and self.verbose:
                                    print(f"\tDecorrelate {p} for {syst_base} into {split_syst} ({syst.replace(syst_base,'')})")
                            else:
                                all_shapes.add(syst_base)
                                text_card_info[proc_name]["shapes"].add(syst_base)
                            syst_width = max(len(syst),syst_width)
                        zero_out_sumw2 = p != "fakes" and "close" not in p # Zero out sumw2 for all proc but fakes, so that we only do auto stats for fakes
                        f[hist_name] = to_hist(arr,hist_name,zero_wgts=zero_out_sumw2)

                        num_h += 1
                    if km_dist == "njets":
                        # We need to handle certain systematics differently when looking at njets procs
                        if p == "Diboson":
                            # Handle the 'diboson_njets' uncertainty
                            # syst = "diboson_njets"
                            # hist_name = f"{proc_name}_{syst}"
                            # syst_kappa = self.rate_systs[syst].get_process(p)[str(num_j)]
                            # if syst_kappa == "-":
                            #     raise ValueError(f"The kappa value for {syst} is missing!")
                            pass

                        if p == "tllq" or p == "tHq":
                            # Handle the 'missing_parton' uncertainty
                            pass
                # obtain the scalings for scalings.json file
                if p in self.SIGNALS:
                    if self.wc_scalings:
                        scalings = h[{'channel':ch,'process':p,'systematic':'nominal'}].make_scaling(self.wc_scalings)
                        self.scalings_json = self.make_scalings_json(self.scalings,ch,km_dist,p,self.wc_scalings,scalings)
                    else:
                        scalings = h[{'channel':ch,'process':p,'systematic':'nominal'}].make_scaling()
                        self.scalings_json = self.make_scalings_json(self.scalings,ch,km_dist,p,h.wc_names,scalings)
            f["data_obs"] = to_hist(data_obs,"data_obs")

        line_break = "##----------------------------------\n"
        left_width = len(line_break) + 2
        left_width = max(syst_width+len("shape")+1,left_width)

        outf_card_name = self.FNAME_TEMPLATE.format(cat=ch,kmvar=km_dist,ext="txt")
        print(f"Generating text file: {outf_card_name}")
        outf_card_name = os.path.join(self.out_dir,outf_card_name)
        with open(outf_card_name,"w") as f:
            f.write(f"shapes *        * {os.path.split(outf_root_name)[1]} $PROCESS $PROCESS_$SYSTEMATIC\n")
            f.write(line_break)
            f.write(f"bin         {bin_str}\n")
            f.write(f"observation {sum(data_obs[0]):.{PRECISION}f}\n")
            f.write(line_break)
            f.write(line_break)

            # Note: This list is what controls the columns in the text datacard, if a process appears
            #       in this list it should NEVER be skipped in any of the following for loops.
            # proc_order = sorted(text_card_info.keys())
            proc_order = [k for k in text_card_info.keys() if text_card_info[k]["rate"] != -1]  # rate = -1 only happens when there's no syst histograms (e.g. flips in 3l/4l)

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
                if r < 0:
                    print(f"Process {p} has negative total rate: {r:.3f} -> setting to 0 in text card")
                    r = 0
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
                if km_dist == "njets" and (syst_name == "diboson_njets" or syst_name == "missing_parton"):
                    # These systematics are only treated as rate systs for njets distribution
                    continue
                row = [f"{left_text:<{left_width}}"]
                for p in proc_order:
                    proc_name = self.get_process(p) # Strips off any "_sm" or "_lin_*" junk
                    # Need to handle certain systematics in a special way
                    if syst_name == "diboson_njets":
                        v = rate_syst.get_process(proc_name,num_j)
                        # v = rate_syst.get_process(proc_name)
                        # if isinstance(v,dict):
                        #     v = v[str(num_j)]
                    elif syst_name == "missing_parton":
                        v = rate_syst.get_process(proc_name)
                        # First strip off any njet and/or bjet labels
                        ch_key = ch.replace(f"_{num_j}j","").replace(f"_{num_b}b","")
                        # Now construct the category key, matching names in the missing_parton file to the current category
                        if num_l == 2:
                            njet_offset = 4
                            ch_key = f"{ch_key}_{num_b}b"
                        elif num_l == 2.1:
                            njet_offset = 1
                            ch_key = "2lss_m_2b"   #CAUTION: Temporary! Fix this later!
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
                        elif num_l == 2.1:
                            njet_offset = 1
                            ch_key = f"{ch_key}_{num_b}b"
                        else:
                            raise ValueError(f"Unable to match {ch} for {syst_name} rate systematic")
                        # The bins in the missing_parton root files start indexing from 0
                        bin_idx = num_j - njet_offset
                        if isinstance(v,dict):
                            unc_hi = v[ch_key][bin_idx]     # Attempt to symmeterize
                            unc_lo = max(0.01,2 - unc_hi)   # Clip unc_lo to not go negative
                            # v = f"{v:.{PRECISION}f}"
                            v = f"{unc_lo:.{PRECISION}f}/{unc_hi:.{PRECISION}f}"
                        elif v != "-":
                            raise ValueError(f"The missing_parton systematic isn't a dictionary (ch={ch}): {v}")
                    else:
                        v = rate_syst.get_process(proc_name)
                    row.append(f"{v:>{col_width}}")
                row = " ".join(row) + "\n"
                f.write(row)

            if self.do_mc_stat:
                f.write("* autoMCStats 10\n")
            else:
                f.write("* autoMCStats -1\n")

        outf_json_name = self.FNAME_TEMPLATE.format(cat=ch,kmvar=km_dist,ext="json")
        with open(os.path.join(self.out_dir,f"{outf_json_name}"),"w") as f:
            print('making', os.path.join(self.out_dir,f"{outf_json_name}"))
            json.dump(self.scalings_json, f, indent=4)

        dt = time.time() - tic
        print(f"File Write Time: {dt:.2f} s")
        print(f"Total Hists Written: {num_h}")

    # TODO: Can be a static member function

    def decompose(self,h,sumw2,wcs,vals={}):
        """
            Decomposes the EFT quadratic parameterization coefficients into combinations that result
            in non-negative coefficient terms.

            Note: All other WCs are assumed set to 0
            sm piece:    set(c1=0.0)
            lin piece:   set(c1=1.0)
            mixed piece: set(c1=1.0,c2=1.0)
            quad piece:  0.5*[set(c1=2.0) - 2*set(c1=1.0) + set(sm)]
        """
        tic = time.time()

        sm = h.eval({})
        if sumw2 is None:
            print("No sumw2 histogram found! Setting errors to 0")
        sm_w2 = sumw2.eval(vals) if sumw2 is not None else 0
        sm = add_sumw2_stub(sm,sm_w2)

        # Note: The keys of this dictionary are a pretty contrived, but are useful later on
        r = {}
        r["sm"] = sm
        terms = 1
        for n1, wc1 in enumerate(wcs):
            tmp_lin_1 = h.eval({wc1: 1.0})
            tmp_lin_2 = h.eval({wc1: 2.0})

            tmp_lin_1 = add_sumw2_stub(tmp_lin_1)
            tmp_lin_2 = add_sumw2_stub(tmp_lin_2)

            lin_name = f"lin_{wc1}"
            quad_name = f"quad_{wc1}"

            terms += 2

            r[lin_name] = tmp_lin_1
            r[quad_name] = {}
            for sp_key in h.categorical_keys:
                r[quad_name][sp_key] = []
                for i in range(2):
                    r[quad_name][sp_key].append(
                        0.5 * (
                            tmp_lin_2[sp_key][i] - 2 * tmp_lin_1[sp_key][i] + sm[sp_key][i]
                        )
                    )

            for n2, wc2 in enumerate(wcs):
                if n1 >= n2:
                    continue
                mixed_name = f"quad_mixed_{wc1}_{wc2}"
                mixed = h.eval({wc1: 1.0, wc2: 1.0})
                mixed = add_sumw2_stub(mixed)
                r[mixed_name] = mixed
                terms += 1

        toc = time.time()
        dt = toc - tic
        if self.verbose:
            print(f"\tDecompose Time: {dt:.2f} s")
            print(f"\tTotal terms: {terms}")

        return r

if __name__ == '__main__':
    fpath = topeft_path("../analysis/topEFT/histos/may18_fullRun2_withSys_anatest08_np.pkl.gz")

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
        r = dc.analyze(km_dist,cat,selected_wcs, True)
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
