'''
This script computes the msising parton rate
It requires the central (tZq) and private (tllq) samples exist in
`histos/central_sm/` and `histos/private_sm/` respectively
To create these, run the datacard maker (tllq `with` systematics, tZq without)
'''
import numpy as np
import matplotlib.pyplot as plt
import uproot
import mplhep as hep
import math
import json
from topeft.modules.comp_datacard import strip
import re

files = ['2lss_m_2b',  '2lss_p_2b',  '2lss_4t_m_2b', '2lss_4t_p_2b', '3l1b_m',  '3l1b_p',  '3l2b_m',  '3l2b_p',  '3l_sfz_1b',  '3l_sfz_2b',  '4l_2b']
files_diff = ['2lss_4t_m_4j_2b', '2lss_4t_m_5j_2b', '2lss_4t_m_6j_2b', '2lss_4t_m_7j_2b', '2lss_4t_p_4j_2b', '2lss_4t_p_5j_2b', '2lss_4t_p_6j_2b', '2lss_4t_p_7j_2b', '2lss_m_4j_2b', '2lss_m_5j_2b', '2lss_m_6j_2b', '2lss_m_7j_2b', '2lss_p_4j_2b', '2lss_p_5j_2b', '2lss_p_6j_2b', '2lss_p_7j_2b', '3l_m_offZ_1b_2j', '3l_m_offZ_1b_3j', '3l_m_offZ_1b_4j', '3l_m_offZ_1b_5j', '3l_m_offZ_2b_2j', '3l_m_offZ_2b_3j', '3l_m_offZ_2b_4j', '3l_m_offZ_2b_5j', '3l_onZ_1b_2j', '3l_onZ_1b_3j', '3l_onZ_1b_4j', '3l_onZ_1b_5j', '3l_onZ_2b_2j', '3l_onZ_2b_3j', '3l_onZ_2b_4j', '3l_onZ_2b_5j', '3l_p_offZ_1b_2j', '3l_p_offZ_1b_3j', '3l_p_offZ_1b_4j', '3l_p_offZ_1b_5j', '3l_p_offZ_2b_2j', '3l_p_offZ_2b_3j', '3l_p_offZ_2b_4j', '3l_p_offZ_2b_5j', '4l_2j_2b', '4l_3j_2b', '4l_4j_2b']
files_ptz = ['3l_onZ_1b_2j', '3l_onZ_1b_3j', '3l_onZ_1b_4j', '3l_onZ_1b_5j', '3l_onZ_2b_2j', '3l_onZ_2b_3j', '3l_onZ_2b_4j', '3l_onZ_2b_5j']

def get_hists(fname, path, process):
    fin = uproot.open('histos/'+path+'/ttx_multileptons-'+fname+'.root')
    card = strip('histos/'+path+'/ttx_multileptons-'+fname+'.txt')
    sm = [k.split(';')[0] for k in fin.keys() if 'sm' in k]

    nom = {}; up = {}; down = {}

    nom = {proc.strip(';1'): fin[proc].values() for proc in fin if 'sm;' in proc and (process in proc or process.replace('ll','Z') in proc)}
    for val in nom.values():
        val = [x if not math.isinf(x) else 0 for x in val]

    up = {proc.strip('Up;1'): fin[proc].to_numpy()[0] for proc in fin if 'sm' in proc and ('Up;' in proc or 'flat' in proc) and 'fakes' not in proc}
    down = {proc.strip('Down;1'): fin[proc].to_numpy()[0] for proc in fin if 'sm' in proc and ('Down;' in proc or 'flat' in proc) and 'fakes' not in proc}
    total = np.array([v for v in nom.values()])[0]

    systs = [0,0]
    err = [np.zeros_like(total), np.zeros_like(total)]
    total_systs = [fin[k].to_numpy()[0] for k in fin.keys() if 'sm' in k and ('Up' in k or 'Down' in k) and 'fakes' not in k]

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

    bins = fin[process+'_sm'].axis().edges()
    return total, nom, err, bins, [proc.split('_sm')[0]for proc in fin if 'sm;' in proc]

if __name__ == '__main__':
    import argparse
    import datetime
    import os
    from topcoffea.scripts.make_html import make_html

    parser = argparse.ArgumentParser(description='You can select which file to run over')
    parser.add_argument('--lumiJson', '-l', default='topcoffea/json/lumi.json'    , help = 'Lumi json file')
    parser.add_argument('--years',          default=[], action='extend', nargs='+', help = 'Specify a list of years')
    parser.add_argument('--time', '-t',     action='store_true', help = 'Append time to dir')
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument('--var',            default='njets', help = 'Specify variable to run over')

    args = parser.parse_args()
    lumiJson = args.lumiJson
    years    = args.years
    var      = args.var
    if var != 'njets':
        files = files_diff
    if var == 'ptz':
        files = files_ptz
    if len(years)==0: years = ['2016APV', '2016', '2017', '2018']
    with open(lumiJson) as jf:
        lumi = json.load(jf)
        lumi = lumi
        lumi = {year : lumi for year,lumi in lumi.items() if year in years}
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

    fout = 'histos/missing_parton.root'
    if var == 'njets':
        if not os.path.exists(fout):
            fout = uproot.create(fout)
        else:
            fout = uproot.update(fout)
    else:
        fout = 'topcoffea/data/missing_parton/missing_parton.root'
        fout = uproot.open(fout)

    rename = {'tllq': 'tZq', 'ttZ': 'ttll', 'ttW': 'ttlnu'} #Used to rename things like ttZ to ttll and ttHnobb to ttH
    for proc in ['tllq']:
        for fname in files:
            if var != 'njets': fname += '_' + var
            total_private, nom_private, err, bins, label = get_hists(fname, 'private_sm', proc)
            rproc = rename[proc] if proc in rename else proc
            total_central, nom_central, _, _, _ = get_hists(fname, 'central_sm', rproc)
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
                sign = total_central[n] / np.abs(total_central[n])
                # total_private - sqrt(err_low^2 + parton^2) = total_central
                if total_private[n] >= total_central[n]:
                    if err_low[n]<total_central[n]: parton[n] = 0 # Error larger than central value
                    else: parton[n] = np.sqrt(np.square(total_private[n] - total_central[n]) - np.square(err[0][n]))
                # total_private + sqrt(err_low^2 + parton^2) = total_central
                else:
                    if err_high[n]>total_central[n]: parton[n] = 0 # Error larger than central value
                    else: parton[n] = np.sqrt(np.square(total_private[n] - total_central[n]) - np.square(err[1][n]))
            if var == 'njets': fout[fname] = {proc : parton/total_private}
            else:
                lep_bin = re.sub('_'+var, '', fname)
                lep_bin = re.sub('_\wj', '', lep_bin)
                if 'offZ' in lep_bin:
                    lep_bin = re.sub('_offZ', '', lep_bin)
                    lep_bin = lep_bin.split('_')
                    lep_bin = lep_bin[0] + lep_bin[-1] + '_' + lep_bin[1]
                if 'onZ' in lep_bin:
                    lep_bin = re.sub('onZ', 'sfz', lep_bin)
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
            plt.savefig(f'{outdir_name}/{fname}.pdf')
            plt.close('all')

    # Make an index.html file if saving to web area
    if "www" in outdir_name:
        make_html(save_dir_path, 400, 300)
    fout.close()
