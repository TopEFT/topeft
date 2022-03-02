import numpy as np
import matplotlib.pyplot as plt
import uproot
import mplhep as hep
import math
import uproot3
import json
from topcoffea.modules.comp_datacard import strip

files = ['2lss_m_2b',  '2lss_p_2b',  '2lss_4t_m_2b', '2lss_4t_p_2b', '3l1b_m',  '3l1b_p',  '3l2b_m',  '3l2b_p',  '3l_sfz_1b',  '3l_sfz_2b',  '4l_2b']

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

    systs = []
    err = [np.ones_like(total), np.ones_like(total)]

    # Handle shape systematics
    if len(up) > 0:
        systs[0] = [k.split(';')[0] for k in fin.keys() if 'sm' in k and 'Up' in k]
        systs[1] = [k.split(';')[0] for k in fin.keys() if 'sm' in k and 'Down' in k]
        up_shift   = [abs(s[0] - total) for s in total_systs]
        down_shift = [abs(s[1] - total) for s in total_systs]
        err = [np.sqrt(np.sum(np.square(up_shift), axis=0)), np.sqrt(np.sum(np.square(down_shift), axis=0))]

    # Handle flat rate systematics
    flat_systs = ([k,r,v] for c in card[1] for k,r,v in zip(fin.keys(), total, c))
    for proc,rate,val in flat_systs:
        if 'sm' not in proc: continue
        s = [1,1]
        if '-' in val: continue
        if '/' in val:
            s[0] = float(val.split('/')[0])
            s[1] = float(val.split('/')[1])
        else:
            s[0] = 1. + float(val)
            s[1] = 1. + float(val)
        err[0] = np.sqrt(np.square(err[0]) + rate*s[0])
        err[1] = np.sqrt(np.square(err[1]) + rate*s[1])
  
    bins = fin['ttH_sm'].axis().edges(flow=True)[1:]
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
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")

    args = parser.parse_args()
    lumiJson = args.lumiJson
    years    = args.years
    if len(years)==0: years = ['2016APV', '2016', '2017', '2018']
    with open(lumiJson) as jf:
        lumi = json.load(jf)
        lumi = lumi
        lumi = {year : lumi for year,lumi in lumi.items() if year in years}
    print(f'Running over: {", ".join(list(lumi.keys()))} (%0.3g fb^-1)' % sum(lumi.values()))

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = save_dir_path+"/missing_parton_"+timestamp_tag
    os.mkdir(outdir_name)
    save_dir_path = os.path.join(args.output_path,outdir_name)

    fout = uproot.recreate('histos/missing_parton.root')
    for proc in ['tllq']:#, 'tHq']:
        for fname in files:
            #bins = fin['ttH_sm'].to_numpy()[1]
            total_private, nom_private, err, bins, label = get_hists(fname, 'private_sm', proc)
            total_central, nom_central, _, _, _ = get_hists(fname, 'central_sm', proc)
            hep.style.use("CMS")
            fig,ax = plt.subplots(figsize=(8, 6))
            hep.histplot(total_private, bins=bins, stack=False, label='Priavte LO', ax=ax, sort='yield')#, histtype='fill')
            hep.histplot(total_central, bins=bins, stack=False, label='Central NLO', ax=ax, sort='yield')#, histtype='fill')
            err_low  = np.min([total_private-err[0],total_private-err[1]], axis=0)
            err_high = np.max([total_private+err[0],total_private+err[1]], axis=0)
            plt.fill_between(bins[:-1], err_low, err_high, step='post', facecolor='none', edgecolor='lightgray', label='Other syst.', hatch='///')
            parton = np.zeros_like(total_private)
            pos = total_private >= total_central
            neg = total_private < total_central
            if np.any(pos):
                parton[pos] = np.max([total_central - np.abs(err_low), np.zeros_like(total_private)], axis=0)[pos]
            if np.any(neg):
                parton[neg] = np.max([total_central - np.abs(err_high), np.zeros_like(total_private)], axis=0)[neg]
            fout[fname] = {proc : parton}
            hep.histplot(err_high+parton/2, bins=bins, ax=ax, yerr=parton/2, histtype='errorbar', label="Mis. parton syst.", color='r', capsize=4)
            hep.histplot(err_low-parton/2, bins=bins, ax=ax, yerr=parton/2, histtype='errorbar', color='r', capsize=4)
            plt.fill_between(bins[:-1], err_low-parton, err_high+parton, step='post', facecolor='none', edgecolor='lightgray', label='Total syst.', hatch='\\\\\\')
            plt.ylim([0, np.max(np.max([total_private,total_private+np.max(err, axis=0)+parton], axis=0))*2])
            plt.xlabel('$N_{jets}$')
            plt.ylabel('Predicted yield')
            hep.cms.label(lumi='%0.3g'%sum(lumi.values()))
            ax.legend(loc='upper right', fontsize='xx-small', ncol=2)
            plt.show()
            plt.tight_layout()
            plt.savefig(save_dir_path+'/'+fname+'.png')
            plt.savefig(save_dir_path+'/'+fname+'.pdf')

    # Make an index.html file if saving to web area
    if "www" in save_dir_path:
        make_html(save_dir_path)
    fout.close()
