# This script makes a plot of the energy scale Lambda
#   - The WC limits (for 1TeV) are hardcoded in this script
#   - The script converts these to Lambda for various assumptions for c
#   - Plots the resulting Lambda for each WC

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.patches as mpatches
import copy

import mplhep as hep

WC_LST = [
    "cQq13", "cQq83", "cQq11", "ctq1", "cQq81", "ctq8",
    "ctt1", "cQQ1", "cQt8", "cQt1",
    "ctW", "ctZ", "ctp", "cpQM", "ctG", "cbW", "cpQ3", "cptb", "cpt",
    "cQl3i", "cQlMi", "cQei", "ctli", "ctei", "ctlSi", "ctlTi"
]

WC_FORMAT_DICT = {
    "cQq13"  : "$c_{\mathrm{Qq}}^{31}$",
    "cQq83"  : "$c_{\mathrm{Qq}}^{38}$",
    "cQq11"  : "$c_{\mathrm{Qq}}^{11}$",
    "ctq1"   : "$c_{\mathrm{tq}}^{1}$" ,
    "cQq81"  : "$c_{\mathrm{Qq}}^{18}$",
    "ctq8"   : "$c_{\mathrm{tq}}^{8}$" ,
    "ctt1"   : "$c_{\mathrm{tt}}^{1}$" ,
    "cQQ1"   : "$c_{\mathrm{QQ}}^{1}$" ,
    "cQt8"   : "$c_{\mathrm{Qt}}^{8}$" ,
    "cQt1"   : "$c_{\mathrm{Qt}}^{1}$" ,
    "ctW"    : "$c_{\mathrm{tW}}$"  ,
    "ctZ"    : "$c_{\mathrm{tZ}}$"  ,
    "ctp"    : "$c_{\mathrm{t} \\varphi}$"  ,
    "cpQM"   : "$c_{\\varphi \mathrm{Q}}^{-}$" ,
    "ctG"    : "$c_{\mathrm{tG}}$"  ,
    "cbW"    : "$c_{\mathrm{bW}}$"  ,
    "cpQ3"   : "$c_{\\varphi \mathrm{Q}}^{3}$" ,
    "cptb"   : "$c_{\\varphi \mathrm{t b}}$" ,
    "cpt"    : "$c_{\\varphi \mathrm{t}}$"  ,
    "cQl3i"  : "$c_{\mathrm{Q\ell}}^{3(\ell)}$" ,
    "cQlMi"  : "$c_{\mathrm{Q\ell}}^{-(\ell)}$" ,
    "cQei"   : "$c_{\mathrm{Qe}}^{(\ell)}$"  ,
    "ctli"   : "$c_{\mathrm{t\ell}}^{(\ell)}$"  ,
    "ctei"   : "$c_{\mathrm{te}}^{(\ell)}$"  ,
    "ctlSi"  : "$c_{\mathrm{t}}^{S(\ell)}$" ,
    "ctlTi"  : "$c_{\mathrm{t}}^{T(\ell)}$" ,
}

################### From TOP-22-006 ###################

TOP22006_LIMS_DICT = {
    "at25v01_2sig_mtm1_prof" : {'ctlTi': [-0.4399112969726758, 0.4399352089667389], 'ctq1': [-0.33363625958117343, 0.3266537776544366], 'ctq8': [-0.8535602498909929, 0.5646116848748659], 'cQq83': [-0.3177526423503681, 0.31429053818017594], 'cQQ1': [-2.11241839215093, 2.3802695238175224], 'cQt1': [-1.9705816779137306, 1.8481936752873787], 'cQt8': [-3.663586260220296, 4.1057428747688345], 'ctli': [-2.202088741189655, 2.3540560446906666], 'cQq81': [-0.8949989204313149, 0.5894190534712986], 'cQlMi': [-2.0019541163166634, 2.465062026758357], 'cbW': [-1.4764518408817633, 1.4754046891973944], 'cpQ3': [-2.34940892156014, 2.3947324888408406], 'ctei': [-2.1081028747917596, 2.457037850248103], 'ctlSi': [-3.0039505253409997, 3.004157596275258], 'ctW': [-0.9099733692495261, 0.8379410321905573], 'cpQM': [-5.941104572965038, 11.823042940378432], 'cQei': [-2.218377103774046, 2.281220243185001], 'ctZ': [-1.1360884100084296, 1.0889590936454137], 'cQl3i': [-3.2434954379884284, 3.1146248326594135], 'ctG': [-0.41143896261600693, 0.3718138508184272], 'cQq13': [-0.135989922505322, 0.13968135607926457], 'cQq11': [-0.31947355260628074, 0.32188747563795633], 'cptb': [-5.672328808263534, 5.684002782249502], 'ctt1': [-1.0923878017635278, 1.1733880326110746], 'ctp': [-6.598568258842153, 28.953778714164894], 'cpt': [-10.234475907261464, 10.09719928451009]},
    "at25v01_2sig_mtm1_froz" : {'ctlTi': [-0.41957628129495045, 0.4195856318325092], 'ctq1': [-0.315427920178287, 0.2953710731003175], 'ctq8': [-0.8088543757215535, 0.47938600833938194], 'cQq83': [-0.29349205924937055, 0.2976921378275847], 'cQQ1': [-1.995944869862657, 2.3445675211685852], 'cQt1': [-1.9509370471213292, 1.7273483723338348], 'cQt8': [-3.4557837498589294, 4.004248872358771], 'ctli': [-2.0867098261144235, 2.2606097215785206], 'cQq81': [-0.8512200731649805, 0.4999864632777593], 'cQlMi': [-1.8913311067660024, 2.366745622293502], 'cbW': [-1.3619516382873902, 1.360509482217129], 'cpQ3': [-2.193284115651667, 2.211741583440906], 'ctei': [-1.9943140291600607, 2.3779791574624527], 'ctlSi': [-2.8908379298971316, 2.890795802306167], 'ctW': [-0.6878206337019525, 0.6278801815768214], 'cpQM': [-2.7125885747309857, 2.519659032306932], 'cQei': [-2.1109275312251676, 2.191000002250593], 'ctZ': [-0.834787841198943, 0.8465355570045555], 'cQl3i': [-3.0542803689361047, 2.9990590764420095], 'ctG': [-0.34849427423738766, 0.2561329364931658], 'cQq13': [-0.1283199856711555, 0.12790949129126972], 'cQq11': [-0.28943488576666104, 0.3018247481355339], 'cptb': [-5.313839663436239, 5.317987027365856], 'ctt1': [-1.0198083636445434, 1.1567167601747368], 'ctp': [-4.650705791718632, 5.863235434104037], 'cpt': [-3.8601036098697024, 3.4756138873321265]},
    "at25v01_2sig_obs_prof"  : {'ctlTi': [-0.3722636669423329, 0.37347056680560714], 'ctq1': [-0.21446140175256925, 0.20680415417909467], 'ctq8': [-0.676170100223615, 0.24972065602392593], 'cQq83': [-0.17187094723666652, 0.16375766417549023], 'cQQ1': [-2.56481573701492, 2.8422585233403797], 'cQt1': [-2.3350361905692867, 2.2736270067278483], 'cQt8': [-4.365141154256117, 4.968989095874007], 'ctli': [-1.8013396217743463, 2.1081406078020755], 'cQq81': [-0.6838379951469792, 0.2188263259635688], 'cQlMi': [-1.5799724006004248, 2.279228772202121], 'cbW': [-0.7591037212908646, 0.7615309124765759], 'cpQ3': [-0.8392028626455179, 1.9983391080221484], 'ctei': [-1.7791921439721134, 2.212912659099416], 'ctlSi': [-2.6042229476128123, 2.615744210840162], 'ctW': [-0.5456257244037721, 0.455257729361443], 'cpQM': [-6.055032334979707, 8.115162379446765], 'cQei': [-1.9052566984221664, 1.9558230470953744], 'ctZ': [-0.7124687843352048, 0.6358178731504084], 'cQl3i': [-2.8359159133072174, 2.5477415034091804], 'ctG': [-0.27550079799575083, 0.23840943594185215], 'cQq13': [-0.07606418938836267, 0.07070017998681018], 'cQq11': [-0.19213098576818102, 0.19097412421388965], 'cptb': [-3.2542127643942904, 3.262844132416997], 'ctt1': [-1.3311739155313131, 1.377943345402796], 'ctp': [-8.851568394757551, 2.745922410763371], 'cpt': [-10.523963296034836, 7.872030177071878]},
    "at25v01_2sig_obs_froz"  : {'ctlTi': [-0.4040215857066777, 0.40396222476094523], 'ctq1': [-0.22262743431117013, 0.20310406121836383], 'ctq8': [-0.6807265439525944, 0.24221086643323614], 'cQq83': [-0.16629383461219321, 0.1627817283847941], 'cQQ1': [-2.5745663075983014, 2.8869839992276454], 'cQt1': [-2.412025509605445, 2.2235580169152214], 'cQt8': [-4.446644137402875, 4.955270735650078], 'ctli': [-2.01775347579095, 2.1998573823298946], 'cQq81': [-0.6663006695724595, 0.21491306860042325], 'cQlMi': [-1.8001583041122582, 2.3304201097068886], 'cbW': [-0.7456260388575396, 0.7481797057595229], 'cpQ3': [-0.8468934615297244, 1.8878828571079087], 'ctei': [-1.9123324404273196, 2.3867983036874483], 'ctlSi': [-2.797151186813185, 2.8022675852658128], 'ctW': [-0.4663371770133328, 0.41316085091574584], 'cpQM': [-2.677691614019029, 2.939216584435089], 'cQei': [-2.0351448737291924, 2.1195938473551994], 'ctZ': [-0.5775934287041611, 0.5892004027967093], 'cQl3i': [-2.6860390588002483, 2.584035285640636], 'ctG': [-0.21671248755061537, 0.24808231320898286], 'cQq13': [-0.07529046662488932, 0.06852338499677613], 'cQq11': [-0.18896014600113764, 0.19538139434535176], 'cptb': [-3.1359164297785993, 3.1812640378921104], 'ctt1': [-1.3072145978406586, 1.4251094555715154], 'ctp': [-7.540624040223664, 2.106356111778846], 'cpt': [-4.934487704753807, 3.1796777520405395]},
}


################### Plotting ###################

# Make the summary comparison plot
def make_plot(wc_lst,range_dict_a,range_dict_b=None,save_name="summary_lims_comp",tag_a="DataA",tab_b="DataB",tag_b=None,xlog=False):

    y_min = 0
    y_max = len(wc_lst)

    style_a = "-"
    style_b = "-"
    width_a = 8 
    width_b = 8

    clr_lst_a = ["dimgrey","darkgrey","lightgrey"]
    clr_lst_b = ["mediumblue","royalblue","lightsteelblue"]

    plt.figure(figsize = (5,10)) 
    if xlog: plt.xscale('log')
    plt.ylim(y_min+0.5, y_max+0.5+1.3) # If leaving room for legend

    plt.xlabel("$\Lambda$ = $\sqrt{\mathrm{WC} \; / \; 2\sigma \, \mathrm{limit}}$ [TeV]",loc="right")

    # Figure out where to put each row
    y_lst = []
    for i,wc in enumerate(wc_lst):

        # Get the y coordinate
        if range_dict_b is not None: y_offset = 0.2
        else: y_offset = 0
        y = i+1
        y_lst.append(y)
        y_a = y + y_offset
        y_b = y - y_offset

        # Plot the values from the a dataset
        if wc in range_dict_a.keys():
            x_a = range_dict_a[wc]
            plt.plot([x_a[0],x_a[1]], [y_a,y_a], clr_lst_a[0], linestyle=style_a, linewidth=width_a, zorder=100)
            if len(x_a)>=4: plt.plot([x_a[2],x_a[3]], [y_a,y_a], clr_lst_a[1], linestyle=style_a, linewidth=width_a, zorder=99) # If there is a second range, plot that too
            if len(x_a)>=6: plt.plot([x_a[4],x_a[5]], [y_a,y_a], clr_lst_a[2], linestyle=style_a, linewidth=width_a, zorder=98) # If there is a third range, plot that too

        # Plot the values from the b dataset if we have them
        if range_dict_b is not None:
            if wc in range_dict_b.keys():
                x_b = range_dict_b[wc]
                plt.plot([x_b[0],x_b[1]], [y_b,y_b], clr_lst_b[0], linestyle=style_b, linewidth=width_b, zorder=100)
                if len(x_b)>=4: plt.plot([x_b[2],x_b[3]], [y_b,y_b], clr_lst_b[1], linestyle=style_b, linewidth=width_b, zorder=99) # If there is a second range, plot that too
                if len(x_b)>=6: plt.plot([x_b[4],x_b[5]], [y_b,y_b], clr_lst_b[2], linestyle=style_b, linewidth=width_b, zorder=98) # If there is a third range, plot that too


    # Make the legend
    #tag_lst = ["$c=0.01$", "$c=1$", "c=$(4\pi)^2$"]
    tag_lst = ["$\mathrm{WC}=0.01$", "$\mathrm{WC}=1$", "$\mathrm{WC}=(4\pi)^2$"]
    patch_a0 = mpatches.Patch(color=clr_lst_a[0], label=tag_lst[0])
    patch_a1 = mpatches.Patch(color=clr_lst_a[1], label=tag_lst[1])
    patch_a2 = mpatches.Patch(color=clr_lst_a[2], label=tag_lst[2])
    plt.legend([patch_a0,patch_a1,patch_a2], tag_lst, loc='upper center', prop={'size': 9.5},ncol=3,framealpha=0)

    # Label the y axis with the WC names and put in the grid lines
    wc_lst_formatted = []
    for wc in wc_lst: wc_lst_formatted.append(WC_FORMAT_DICT[wc])
    plt.yticks(y_lst,wc_lst_formatted)
    plt.axvline(x=0,color="k",linestyle="-",linewidth=1,zorder=5)
    plt.grid(linestyle="--",zorder=-10)

    # CMS labels
    hep.cms.label(data=True,label="Supplementary",lumi="138")

    plt.savefig(save_name+".pdf",format="pdf")
    plt.savefig(save_name+".png",format="png")
    plt.show()
    return plt


################### Manipulate limit dicts ###################

# Given a limit dict {"c":[val_lo,val_hi]}, return {"c":val} where val is abs of the the farthest from 0 val
def get_extreme_val_from_range(limit_dict_wc_range):
    limit_dict_extreme = {}
    for wc in limit_dict_wc_range.keys():
        # Find the most extreme value in the range (+ or -)
        maxval = 0
        for val in limit_dict_wc_range[wc]:
            if abs(val) > maxval: maxval = abs(val)
        limit_dict_extreme[wc] = maxval
    return limit_dict_extreme

# Takes {"wc":v} and a value (x) for the wc, returns {"wc": sqrt(x/v)}
#   - Note that v = c/Lambda^2
#   - Usually we think of v as the WC value c, but that's assuming Lambda=1TeV
#   - So we can instead solve for Lambda and plug in something interesting for c
#   - So we have: Lambda=sqrt(c/v), might be interestint to consider eg c=1, or c=4pi^2
def get_lambda_dict(in_dict,c):
    out_dict = {}
    for k,v in in_dict.items():
        out_dict[k] = np.sqrt(c/v)
    return out_dict

# Take a dict and if the keys are smefit keys, return new dict with keys swapped out for dim6top keys
def smefit_to_dim6top(in_dict):
    out_dict = {}
    for k, v in in_dict.items():
        if k in SMEFIT_TO_DIM6TOP_NAME_MAP.keys():
            out_dict[SMEFIT_TO_DIM6TOP_NAME_MAP[k]] = copy.deepcopy(v)
        else:
            out_dict[k] = v
    return out_dict


################### Convenience functions ###################

def get_lambda_dict_wrapper(ranges_dict):

    # Very very specific function
    # Takes a set of dictionaries, combines into one in the format that the plotter likes
    # E.g. takes [{c:v1},{c:v2},{c:v3}] and returns {c:[0,v1,0,v2,0,v3]}
    # Assumes all in dicts have the exact same keys
    def construct_dict_for_plotter(lim_dict_lst):
        out_dict = {}
        wc_lst = list(lim_dict_lst[0].keys())
        for wc in wc_lst:
            out_dict[wc] = []
            for d in lim_dict_lst:
                out_dict[wc].append(0)
                out_dict[wc].append(d[wc])
        return out_dict

    # Get the Lambda dicts for a few values for c
    limit_dict_extreme = get_extreme_val_from_range(ranges_dict)
    lambda_dict_c0p1  = get_lambda_dict(limit_dict_extreme,0.01)
    lambda_dict_c1    = get_lambda_dict(limit_dict_extreme,1.0)
    lambda_dict_c4pi2 = get_lambda_dict(limit_dict_extreme,16.0*np.pi**2)
    
    # This is in a ver specific format so that the plotter will plot each of the ranges on the same line
    lambda_dict = construct_dict_for_plotter([lambda_dict_c0p1,lambda_dict_c1,lambda_dict_c4pi2])

    return lambda_dict



################### Main ###################

def main():

    lambda_dict_top22006 = get_lambda_dict_wrapper(TOP22006_LIMS_DICT["at25v01_2sig_obs_prof"])
    for k,v in lambda_dict_top22006.items():
        print(k,v)

    make_plot(
        wc_lst=WC_LST,
        range_dict_a = lambda_dict_top22006,
        tag_a = "TOP-21-006 2$\sigma$ profiled asimov",
        save_name="lambda_lims_b",
        xlog = True,
    )

main()
