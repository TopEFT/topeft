'''
This script computes the msising parton rate
It requires the central (tZq) and private (tllq) samples exist in
`histos/central_sm/` and `histos/private_sm/` respectively
To create these, run the datacard maker (tllq `with` systematics, tZq without)
'''

import numpy as np
import matplotlib.pyplot as plt
import uproot
import matplotlib as mpl
mpl.use('Agg')
import mplhep as hep
import math
from topeft.modules.comp_datacard import strip
import re

from topeft.modules.paths import topeft_path
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))

files = ['2lss_4t_m', '2lss_4t_p', '2lss_fwd_m', '2lss_fwd_p', '2lss_m', '2lss_p', '3l_m_offZ_1b', '3l_m_offZ_2b', '3l_onZ_1b', '3l_onZ_2b', '3l_p_offZ_1b', '3l_p_offZ_2b', '4l']
files = ['2lss_fwd_m', '2lss_fwd_p']
files_diff = ['2lss_4t_m_4j_2b', '2lss_4t_m_5j_2b', '2lss_4t_m_6j_2b', '2lss_4t_m_7j_2b', '2lss_4t_p_4j_2b', '2lss_4t_p_5j_2b', '2lss_4t_p_6j_2b', '2lss_4t_p_7j_2b', '2lss_m_4j_2b', '2lss_m_5j_2b', '2lss_m_6j_2b', '2lss_m_7j_2b', '2lss_p_4j_2b', '2lss_p_5j_2b', '2lss_p_6j_2b', '2lss_p_7j_2b', '3l_m_offZ_1b_2j', '3l_m_offZ_1b_3j', '3l_m_offZ_1b_4j', '3l_m_offZ_1b_5j', '3l_m_offZ_2b_2j', '3l_m_offZ_2b_3j', '3l_m_offZ_2b_4j', '3l_m_offZ_2b_5j', '3l_onZ_1b_2j', '3l_onZ_1b_3j', '3l_onZ_1b_4j', '3l_onZ_1b_5j', '3l_onZ_2b_2j', '3l_onZ_2b_3j', '3l_onZ_2b_4j', '3l_onZ_2b_5j', '3l_p_offZ_1b_2j', '3l_p_offZ_1b_3j', '3l_p_offZ_1b_4j', '3l_p_offZ_1b_5j', '3l_p_offZ_2b_2j', '3l_p_offZ_2b_3j', '3l_p_offZ_2b_4j', '3l_p_offZ_2b_5j', '4l_2j_2b', '4l_3j_2b', '4l_4j_2b']
files = ['3l_onZ_1b_2j', '3l_onZ_1b_3j', '3l_onZ_1b_4j', '3l_onZ_1b_5j', '3l_onZ_2b_2j', '3l_onZ_2b_3j', '3l_onZ_2b_4j', '3l_onZ_2b_5j']

#files_diff = ['2lss_4t_m_4j', '2lss_4t_m_5j', '2lss_4t_m_6j', '2lss_4t_m_7j', '2lss_4t_p_4j', '2lss_4t_p_5j', '2lss_4t_p_6j', '2lss_4t_p_7j', '2lss_m_4j', '2lss_m_5j', '2lss_m_6j', '2lss_m_7j', '2lss_p_4j', '2lss_p_5j', '2lss_p_6j', '2lss_p_7j', '3l_1tau_1b_2j', '3l_1tau_1b_3j', '3l_1tau_1b_4j', '3l_1tau_1b_5j','3l_m_offZ_none_1b_2j', '3l_m_offZ_none_1b_3j', '3l_m_offZ_none_1b_4j', '3l_m_offZ_none_1b_5j', '3l_m_offZ_none_2b_2j', '3l_m_offZ_none_2b_3j', '3l_m_offZ_none_2b_4j', '3l_m_offZ_none_2b_5j', '3l_onZ_1b_2j', '3l_onZ_1b_3j', '3l_onZ_1b_4j', '3l_onZ_1b_5j', '3l_onZ_2b_2j', '3l_onZ_2b_3j', '3l_onZ_2b_4j', '3l_onZ_2b_5j', '3l_p_offZ_1b_2j', '3l_p_offZ_1b_3j', '3l_p_offZ_1b_4j', '3l_p_offZ_1b_5j', '3l_p_offZ_2b_2j', '3l_p_offZ_2b_3j', '3l_p_offZ_2b_4j', '3l_p_offZ_2b_5j', '4l_2j_2b', '4l_3j_2b', '4l_4j_2b']

files_diff = ['2los_onZ_1tau', '2lss_4t_m', '2lss_4t_p', '2lss_fwd_m', '2lss_fwd_p', '2lss_m_1tau_offZ', '2lss_m_1tau_onZ', '2lss_m', '2lss_p_1tau_offZ', '2lss_p_1tau_onZ', '2lss_p', '3l_1tau_1b', '3l_1tau_2b', '3l_m_offZ_1b_fwd', '3l_m_offZ_2b_fwd', '3l_m_offZ_high_1b', '3l_m_offZ_high_2b', '3l_m_offZ_low_1b', '3l_m_offZ_low_2b', '3l_m_offZ_none_1b', '3l_m_offZ_none_2b', '3l_onZ_1b_fwd', '3l_onZ_1b', '3l_onZ_2b_fwd', '3l_onZ_2b', '3l_p_offZ_1b_fwd', '3l_p_offZ_2b_fwd', '3l_p_offZ_high_1b', '3l_p_offZ_high_2b', '3l_p_offZ_low_1b', '3l_p_offZ_low_2b', '3l_p_offZ_none_1b', '3l_p_offZ_none_2b', '4l']

#files_ptz = ['3l_m_offZ_low_1b_2j', '3l_m_offZ_low_1b_3j', '3l_m_offZ_low_1b_4j', '3l_m_offZ_low_1b_5j', '3l_m_offZ_low_2b_2j', '3l_m_offZ_low_2b_3j', '3l_m_offZ_low_2b_4j', '3l_m_offZ_low_2b_5j', '3l_m_offZ_high_1b_2j', '3l_m_offZ_high_1b_3j', '3l_m_offZ_high_1b_4j', '3l_m_offZ_high_1b_5j', '3l_m_offZ_high_2b_2j', '3l_m_offZ_high_2b_3j', '3l_m_offZ_high_2b_4j', '3l_m_offZ_high_2b_5j']
files_ptz_wtau = ['2lss_m_1tau_onZ_3j', '2lss_m_1tau_onZ_4j', '2lss_m_1tau_onZ_5j', '2lss_m_1tau_onZ_6j', '2lss_p_1tau_onZ_3j', '2lss_p_1tau_onZ_4j', '2lss_p_1tau_onZ_5j','2lss_p_1tau_onZ_6j']

def matches_process(proc_name, process):

    aliases = {
        'tllq': [
            'tllq',
            'tZq',
            'TZQB-Zto2L-4FS_MLL-30'
        ],

        'tZq': [
            'tZq',
            'TZQB-Zto2L-4FS_MLL-30'
            ]
    }

    targets = aliases.get(process, [process])

    return any(t in proc_name for t in targets)

def get_hists(fname, path, process):
    fin = uproot.open('parton_datacards/Run2/'+path+'/ttx_multileptons-'+fname+'.root')
    card = strip('parton_datacards/Run2/'+path+'/ttx_multileptons-'+fname+'.txt')
    #print("card", card)
    sm = [k.split(';')[0] for k in fin.keys() if 'sm' in k]

    nom = {}; up = {}; down = {}

    #nom = {proc.strip(';1'): fin[proc].values() for proc in fin if 'sm;' in proc and (process in proc or process.replace('ll','Z') in proc)}
    nom = {
        proc.replace(";1", ""): fin[proc].values()
        for proc in fin
        if (
            'sm;' in proc
            and matches_process(proc, process)
        )
    } 
    for val in nom.values():
        val = [x if not math.isinf(x) else 0 for x in val]

#    up = {proc.strip('Up;1'): fin[proc].to_numpy()[0] for proc in fin if 'sm' in proc and ('Up;' in proc or 'flat' in proc) and 'fakes' not in proc}
#    down = {proc.strip('Down;1'): fin[proc].to_numpy()[0] for proc in fin if 'sm' in proc and ('Down;' in proc or 'flat' in proc) and 'fakes' not in proc}
    up = {
        proc.replace("Up;1", ""): fin[proc].to_numpy()[0]
        for proc in fin
        if (
            'sm' in proc
            and ('Up;' in proc or 'flat' in proc)
            and 'fakes' not in proc
            and matches_process(proc, process)
        )
    }
    
    down = {
        proc.replace("Down;1", ""): fin[proc].to_numpy()[0]
        for proc in fin
        if (
            'sm' in proc
            and ('Down;' in proc or 'flat' in proc)
            and 'fakes' not in proc
            and matches_process(proc, process)
        )
    }

    if len(nom) == 0:
        print("NO MATCH FOUND")
        print("fname =", fname)
        print("path =", path)
        print("process =", process)
        print("available keys =", fin.keys())
        return None, {}, None, None, None

    total = np.array([v for v in nom.values()])[0]

    systs = [0,0]
    err = [np.zeros_like(total), np.zeros_like(total)]
    total_systs = [
        fin[k].to_numpy()[0]
        for k in fin.keys()
        if (
            'sm' in k
            and ('Up' in k or 'Down' in k)
            and 'fakes' not in k
            and matches_process(k, process)
        )
    ]

    # Handle shape systematics
    if len(total_systs) > 0:
        systs[0] = [k.split(';')[0] for k in fin.keys() if 'sm' in k and 'Up' in k and 'fakes' not in k]
        systs[1] = [k.split(';')[0] for k in fin.keys() if 'sm' in k and 'Down' in k and 'fakes' not in k]
        systs = [k for k in zip(systs[0], systs[1])]
        for syst in total_systs:
            mask = syst - total > 0
            shift = (syst - total)
            err[0][~mask] = np.sqrt(np.square(err[1][~mask]) + np.square(-shift[~mask]))
            err[1][mask] = np.sqrt(np.square(err[0][mask]) + np.square(shift[mask]))

    # Handle flat rate systematics
    flat_systs = zip(card[0], card[0].values(), *card[1])
    for c in flat_systs:
        proc = c[0]
        rate = c[1]
        vals = c[2:]
        if 'sm' not in proc: continue
        if 'tllq' not in proc: continue
        for val in vals:
            s = [0,0]
            if '-' in val: continue
            if '/' in val:
                s[0] = 1 - float(val.split('/')[0])
                s[1] = float(val.split('/')[1]) - 1
            else:
                s[0] = float(val) - 1
                s[1] = float(val) - 1
            err[0] = np.sqrt(np.square(err[0]) + np.square(total*s[0]))
            err[1] = np.sqrt(np.square(err[1]) + np.square(total*s[1]))

    hist_key = list(nom.keys())[0]
    bins = fin[hist_key + ';1'].axis().edges()

    return total, nom, err, bins, [proc.split('_sm')[0]for proc in fin if 'sm;' in proc]

if __name__ == '__main__':
    import argparse
    import datetime
    import os
    from topcoffea.scripts.make_html import make_html

    parser = argparse.ArgumentParser(description='You can select which file to run over')
    parser.add_argument('--years',          default=[], action='extend', nargs='+', help = 'Specify a list of years')
    parser.add_argument('--time', '-t',     action='store_true', help = 'Append time to dir')
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument('--var',            default='njets', help = 'Specify variable to run over')

    args = parser.parse_args()
    years    = args.years
    var      = args.var
    if var == 'njets':
        files = files_diff
    if var == 'ptz':
        files = files_ptz
    if var == 'ptz_wtau':
        files = files_ptz_wtau
    if len(years)==0: years = ['2016APV', '2016', '2017', '2018']
    #if len(years)==0: years = ['2022', '2022EE', '2023', '2023BPix']
    lumi = {}
    for year in years:
        lumi[year] = get_tc_param(f"lumi_{year}")
    print(f'Running over: {", ".join(list(lumi.keys()))} (%0.3g fb^-1)' % sum(lumi.values()))

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = save_dir_path+"/missing_parton/"+var
    if args.time:
        outdir_name = outdir_name+'_'+timestamp_tag
    outdir_name += '/'
    if not os.path.exists(outdir_name):
        os.mkdir(outdir_name)
    else:
        print(f'Overwriting contents in {outdir_name}\nUse the `-t` flag to make unique directories')
    save_dir_path = os.path.join(args.output_path,outdir_name)

    fout = topeft_path('data/missing_parton/missing_parton_test.root')
    print("saving output in data/missing_parton/missing_parton_test.root")
    if var == 'njets' or var == 'ptz_wtau':
        if not os.path.exists(fout):
            fout = uproot.create(fout)
        else:
            fout = uproot.update(fout)
    else:
        fout = 'topcoffea/data/missing_parton/missing_parton_test.root'
        fout = uproot.open(fout)

    rename = {'tllq': 'tZq', 'ttZ': 'ttll', 'ttW': 'ttlnu'} #Used to rename things like ttZ to ttll and ttHnobb to ttH
    rename = {} #Used to rename things like ttZ to ttll and ttHnobb to ttH
    for proc in ['tllq']:
        for fname in files:
            fname += '_' + var
            total_private, nom_private, err, bins, label = get_hists(fname, 'private_tllq', proc)
            print("proc,", proc)
            rproc = rename[proc] if proc in rename else proc
            total_central, nom_central, _, _, _ = get_hists(fname, 'central_tZq', 'tZq')
            if total_private is None or total_central is None:
                print(f"Skipping empty category: {fname}")
                continue

            hep.style.use("CMS")
            fig,ax = plt.subplots(figsize=(8, 6))
            hep.histplot(total_private, bins=bins, stack=False, label='Priavte LO', ax=ax, sort='yield')
            hep.histplot(total_central, bins=bins, stack=False, label='Central NLO', ax=ax, sort='yield')
            # Keep track of negative sign (since abs is requried to in sqrt)
            err_low  = total_private - err[0]
            err_high = total_private + err[1]
            plt.fill_between(bins, np.append(err_low, 0.), np.append(err_high, 0.), step='post', facecolor='none', edgecolor='lightgray', label='Other syst.', hatch='///')
            parton = np.zeros_like(total_private)
            pos = total_private >= total_central
            neg = total_private < total_central
            for n in range(len(total_private)):
                #sign = total_central[n] / np.abs(total_central[n])
                if total_central[n] != 0:
                    sign = total_central[n] / abs(total_central[n])
                else:
                    sign = 1

                # total_private - sqrt(err_low^2 + parton^2) = total_central
                if total_private[n] >= total_central[n]:
                    if err_low[n]<total_central[n]: parton[n] = 0 # Error larger than central value
                    else: parton[n] = np.sqrt(np.square(total_private[n] - total_central[n]) - np.square(err[0][n]))
                # total_private + sqrt(err_low^2 + parton^2) = total_central
                else:
                    if err_high[n]>total_central[n]: parton[n] = 0 # Error larger than central value
                    else: parton[n] = np.sqrt(np.square(total_private[n] - total_central[n]) - np.square(err[1][n]))
            #if var == 'njets': fout[fname.replace('njets', '2b')] = {proc : np.nan_to_num(parton/total_private, 0)}
            if var == 'njets' or var == 'ptz_wtau':

                lep_bin = re.sub('_'+var, '', fname)

                vals = np.nan_to_num(parton / total_private, 0)
                merge_rules = [
                    ("m_1tau",     6),  # 2lss 1tau merging 6j+
                    ("p_1tau",     6),
                    ("2los",       3),
                    ("onZ_1b_fwd", 4),
                    ("onZ_2b_fwd", 4),
                    ("offZ_2b_fwd",4),
                    ("4l",         4),
                    ("3l",         5),  # avoid conflicting with 3l fwd
                ]
                
                for pattern, idx in merge_rules:
                    if pattern in lep_bin:
                        denom = np.sum(total_private[idx:])
                        merged = 0.0 if denom == 0 else np.sum(parton[idx:]) / denom
                        vals = np.concatenate([vals[:idx], [merged]])
                        break
                
                fout[lep_bin] = {
                    proc: vals
                }

            else:                
                lep_bin = re.sub(f'_{var}$', '', lep_bin)
                
                # remove jet bin like _2j, _3j, _4j, ...
                lep_bin = re.sub(r'_\dj$', '', lep_bin)
                
                # add fixed b-category for 2lss regions
                if '2lss' in lep_bin:
                    lep_bin += '_2b'

                offset = -4 if '3l' not in fname else -2
                jet_bin = int(re.findall('\dj', fname)[0][:-1])
                parton = np.array(fout[lep_bin]['tllq'].array())[jet_bin + offset] * total_private
            sign = np.ones_like(parton)
            err_low  = total_private - np.sqrt(np.square(err[0]) + np.square(parton))
            err_high = total_private + np.sqrt(np.square(err[1]) + np.square(parton))
            # Correct for cases where parton > err_low (negative)
            for n,_ in enumerate(sign):
                if np.square(err_low[n]) - np.square(parton[n]) < 0 or err_low[n] < 0: sign[n] = -1
            plt.fill_between(bins, np.append(err_low, 0), np.append(err_high, 0), step='post', facecolor='none', edgecolor='lightgray', label='Total syst.', hatch='\\\\\\') # append 0 to pad plots (matplotlib plots up to but not including the last bin)
            np.seterr(invalid='ignore')
            maxbin = np.max(np.max([total_private,total_private+np.max(err, axis=0)+parton], axis=0))*2
            if np.isnan(maxbin) or np.isinf(maxbin): maxbin = 1
            plt.ylim([0, maxbin])
            hep.cms.label(lumi='%0.3g'%sum(lumi.values()))
            plt.ylabel('Predicted yield')
            ax.legend(loc='upper right', fontsize='xx-small', ncol=2)
            if var == 'njets':
                plt.xlabel('$N_{jets}$')
            plt.xlabel(var)
            plt.show()
            plt.tight_layout()
            plt.savefig(f'{outdir_name}/{fname}.png')
            print(f'save {fname}.png in {outdir_name}')
            plt.savefig(f'{outdir_name}/{fname}.pdf')
            print(f'save {fname}.pdf in {outdir_name}')
            plt.close('all')

    # Make an index.html file if saving to web area
    if "www" in outdir_name:
        make_html(save_dir_path, 400, 300)
    fout.close()
