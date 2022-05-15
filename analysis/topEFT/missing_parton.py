import numpy as np
import matplotlib.pyplot as plt
import uproot
import mplhep as hep
import math
import json
from topcoffea.modules.comp_datacard import strip
from coffea import hist
import re

files = ['2lss_m_2b',  '2lss_p_2b',  '2lss_4t_m_2b', '2lss_4t_p_2b', '3l1b_m',  '3l1b_p',  '3l2b_m',  '3l2b_p',  '3l_sfz_1b',  '3l_sfz_2b',  '4l_2b']
files_diff = ['2lss_4t_m_4j_2b', '2lss_4t_m_5j_2b', '2lss_4t_m_6j_2b', '2lss_4t_m_7j_2b', '2lss_4t_p_4j_2b', '2lss_4t_p_5j_2b', '2lss_4t_p_6j_2b', '2lss_4t_p_7j_2b', '2lss_m_4j_2b', '2lss_m_5j_2b', '2lss_m_6j_2b', '2lss_m_7j_2b', '2lss_p_4j_2b', '2lss_p_5j_2b', '2lss_p_6j_2b', '2lss_p_7j_2b', '3l_m_offZ_1b_2j', '3l_m_offZ_1b_3j', '3l_m_offZ_1b_4j', '3l_m_offZ_1b_5j', '3l_m_offZ_2b_2j', '3l_m_offZ_2b_3j', '3l_m_offZ_2b_4j', '3l_m_offZ_2b_5j', '3l_onZ_1b_2j', '3l_onZ_1b_3j', '3l_onZ_1b_4j', '3l_onZ_1b_5j', '3l_onZ_2b_2j', '3l_onZ_2b_3j', '3l_onZ_2b_4j', '3l_onZ_2b_5j', '3l_p_offZ_1b_2j', '3l_p_offZ_1b_3j', '3l_p_offZ_1b_4j', '3l_p_offZ_1b_5j', '3l_p_offZ_2b_2j', '3l_p_offZ_2b_3j', '3l_p_offZ_2b_4j', '3l_p_offZ_2b_5j', '4l_2j_2b', '4l_3j_2b', '4l_4j_2b']

def get_hists(fname, path, process):
    fin = uproot.open('histos/'+path+'/ttx_multileptons-'+fname+'.root')
    card = strip('histos/'+path+'/ttx_multileptons-'+fname+'.txt')
    sm = [k.split(';')[0] for k in fin.keys() if 'sm' in k]
  
    nom = {}; up = {}; down = {}
        
    nom = {proc.strip(';1'): fin[proc].values(flow=True)[1:] for proc in fin if 'sm;' in proc and (process in proc or process.replace('ll','Z') in proc)}
    for val in nom.values():
        val = [x if not math.isinf(x) else 0 for x in val]
   
    up = {proc.strip('Up;1'): fin[proc].to_numpy()[0] for proc in fin if 'sm' in proc and ('Up;' in proc or 'flat' in proc)}
    down = {proc.strip('Down;1'): fin[proc].to_numpy()[0] for proc in fin if 'sm' in proc and ('Down;' in proc or 'flat' in proc)}
    #syst_names = list(set([k.split('_')[-1] for k in up]))
    total = np.array([v for v in nom.values()]).sum(0)

    systs = [0,0]
    err = [np.zeros_like(total), np.zeros_like(total)]

    # Handle shape systematics
    if len(up) > 0:
        systs[0] = [k.split(';')[0] for k in fin.keys() if 'sm' in k and 'Up' in k]
        systs[1] = [k.split(';')[0] for k in fin.keys() if 'sm' in k and 'Down' in k]
        total_systs = [[fin[k[0]+';1'].to_numpy()[0].sum(), fin[k[1]+';1'].to_numpy()[0].sum()] for k in systs]
        up_shift   = [abs(s[0] - total) for s in total_systs]
        down_shift = [abs(s[1] - total) for s in total_systs]
        err = [np.sqrt(np.sum(np.square(up_shift), axis=0)), np.sqrt(np.sum(np.square(down_shift), axis=0))]

    # Handle flat rate systematics
    flat_systs = ([k,r,v] for c in card[1] for k,r,v in zip(fin.keys(), total, c))
    for proc,rate,val in flat_systs:
        if 'sm' not in proc: continue
        s = [0,0]
        if '-' in val: continue
        if '/' in val:
            s[0] = 1. - float(val.split('/')[0])
            s[1] = 1. - float(val.split('/')[1])
        else:
            s[0] = 1. - float(val)
            s[1] = 1. - float(val)
        err[0] = np.sqrt(np.square(err[0]) + np.square(rate*s[0]))
        err[1] = np.sqrt(np.square(err[1]) + np.square(rate*s[1]))
  
    bins = fin[process+'_sm'].axis().edges(flow=True)[1:]
    bins[-1] = bins[-2] + 1.
    return total, nom, err, bins, [proc.split('_sm')[0]for proc in fin if 'sm;' in proc]

if __name__ == '__main__':
    import argparse
    import datetime
    import os
    from topcoffea.plotter.make_html import make_html

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
    if not os.path.exists(fout):
        fout = uproot.create(fout)
    else:
        fout = uproot.update(fout)

    rename = {'tllq': 'tZq', 'ttZ': 'ttll', 'ttW': 'ttlnu'} #Used to rename things like ttZ to ttll and ttHnobb to ttH
    for proc in ['tllq']:#, 'tHq']:
        for fname in files:
            if var != 'njets': fname += '_' + var
            total_private, nom_private, err, bins, label = get_hists(fname, 'private_sm', proc)
            rproc = rename[proc] if proc in rename else proc
            total_central, nom_central, _, _, _ = get_hists(fname, 'central_sm', rproc)
            hep.style.use("CMS")
            fig,ax = plt.subplots(figsize=(8, 6))
            hep.histplot(total_private, bins=bins, stack=False, label='Priavte LO', ax=ax, sort='yield')#, histtype='fill')
            hep.histplot(total_central, bins=bins, stack=False, label='Central NLO', ax=ax, sort='yield')#, histtype='fill')
            # Keep track of negative sign (since abs is requried to in sqrt)
            sign = [(np.square(total_private)-np.square(err[0])) / np.abs(np.square(total_private)-np.square(err[0])), (np.square(total_private)-np.square(err[1])) / np.abs(np.square(total_private)-np.square(err[1]))]
            err_low  = np.min([sign[0]*np.sqrt(np.abs(np.square(total_private)-np.square(err[0]))), sign[1]*np.sqrt(np.abs(np.square(total_private)-np.square(err[1])))], axis=0)
            err_high = np.max([np.sqrt(np.square(total_private)+np.square(err[0])),np.sqrt(np.square(total_private)+np.square(err[1]))], axis=0)
            plt.fill_between(bins[:-1], err_low, err_high, step='post', facecolor='none', edgecolor='lightgray', label='Other syst.', hatch='///')
            parton = np.zeros_like(total_private)
            pos = total_private >= total_central
            neg = total_private < total_central
            for n in range(len(total_private)):
                if total_private[n] >= total_central[n]:
                    if err_low[n]<total_central[n]: parton[n] = 0 # Error larger than central value
                    else: parton[n] = np.sqrt(np.abs(np.square(err_low[n]) - np.square(total_central[n])))
                else:
                    if err_high[n]>total_central[n]: parton[n] = 0 # Error larger than central value
                    else: parton[n] = np.sqrt(np.abs(np.square(total_central[n]) - np.square(err_high[n])))
            fout[fname] = {proc : parton}
            sign = err_low / np.abs(err_low)
            plt.fill_between(bins[:-1], sign*np.sqrt(np.abs(np.square(err_low)-np.square(parton))), np.sqrt(np.square(err_high)+np.square(parton)), step='post', facecolor='none', edgecolor='lightgray', label='Total syst.', hatch='\\\\\\')
            np.seterr(invalid='ignore')
            plt.ylim([0, np.max(np.max([total_private,total_private+np.max(err, axis=0)+parton], axis=0))*2])
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
