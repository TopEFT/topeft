'''
This script plots weights produced by `gen_hist_eventweights_processor.py`
Example:
python gen_hist_eventweights_plotter.py 2022_tllq_NewStPt4.pkl.gz /users/byates2/afs/www/EFT/tllq_NewStPt4_Run3/weights/weights.pdf
'''
import pickle
#from coffea import hist
import hist
from topcoffea.modules.histEFT import HistEFT
import gzip
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
#from coffea.hist import Bin
from topeft.modules import axes
BINNING = {k: v['variable'] for k,v in axes.info.items() if 'variable' in v}

#Load hists from pickle file created by TopCoffea
hists={}

parser = argparse.ArgumentParser(description='You can select which file to run over')
parser.add_argument('fin'   , default='analysis/topEFT/histos/mar03_central17_pdf_np.pkl.gz' , help = 'File to run over')
parser.add_argument('output'   , default='/users/byates2/afs/www/EFT/tllq_NewStPt4_Run3/weights/' , help = 'Output path')
args  = parser.parse_args()
fin   = args.fin

#hin = pickle.load(gzip.open(fin))
#for k in hin.keys():
#  if k in hists: hists[k]+=hin[k]
#  else:               hists[k]=hin[k]
with gzip.open(fin) as fin:
  hin = pickle.load(fin)
  for k in hin.keys():
    if isinstance(hin[k], dict):
        continue
    if k in hists: hists[k]+=hin[k]
    else:               hists[k]=hin[k]

for h_name in hists:
    ls = '-'
    if 'coeff' in h_name: continue
    if 'efth' in h_name: continue
    if 'SM' in h_name and False:
        label = 'SM'
    elif 'neg' in h_name:
        ls = '--'
    elif 'abs' in h_name:
        ls = '-.'
    if 'pt' in h_name:
        label = 'EFT' + h_name.split('_')[1]
    else:
        label = h_name.split('_')[1]
    hists[h_name].plot1d(label=label, yerr=False, ls=ls, flow='show')
    #(hists[h_name]/np.sum(hists['weights_SMabs_log'].values(flow=True))).plot1d(label=label, yerr=False, ls=ls, flow='show')
    #plt.gca().set_ylabel('log(weights) / sum(SMpos)')
    if 'coeff' in h_name: continue
    #if 'coeff' in h_name: hists[h_name].plot1d(label=label, yerr=False)#, flow='show', ls='--')
    elif 'efth' in h_name: continue
    #elif 'efth' in h_name: hists[h_name].plot1d(label=label, yerr=False, flow='show', ls='-.')
    #else: hists[h_name].plot1d(label=label, yerr=False)#, flow='show')

plt.legend(ncol=3)
plt.gca().set_yscale('log')
plt.gca().set_xlabel('log(event weights)')
plt.tight_layout()
plt.savefig(f'{args.output}/weights.pdf')
plt.savefig(f'{args.output}/weights.png')
#plt.savefig(args.output.replace('.pdf', '.png'))
