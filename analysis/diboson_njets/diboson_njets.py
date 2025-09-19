#!/usr/bin/env python
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import hist
import argparse
import pickle
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description='You can select which file to run over')
parser.add_argument('fin'   , default='histos/diboson_njets.pkl.gz' , help = 'Variable to run over')
args  = parser.parse_args()
fin   = args.fin

hists = {}
with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
      if k in hists: hists[k]+=hin[k]
      else:               hists[k]=hin[k]

h = hists['njets']
#h_data = h.integrate('sample',  [proc for proc in h.axis('sample').identifiers() if 'data' in proc.name]).integrate('channel',  [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic',  'nominal').integrate('appl',  'isSR_3l').to_hist()
#h_diboson = h.integrate('sample', [proc for proc in h.axis('sample').identifiers() if any(p == proc.name[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])]).integrate('channel', [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic', 'nominal').integrate('appl', 'isSR_3l').to_hist()
#h_bkg = h.integrate('sample', [proc for proc in h.axis('sample').identifiers() if 'data' not in proc.name and not any(p == proc.name[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])]).integrate('channel', [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic', 'nominal').integrate('appl', 'isSR_3l').to_hist()
h_data = h[{'process': [proc for proc in h.axes['process'] if 'data' in proc], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal', 'appl': 'isSR_3l'}][{'process': sum, 'channel': sum}]
h_diboson = h[{'process': [proc for proc in h.axes['process'] if any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal', 'appl': 'isSR_3l'}][{'process': sum, 'channel': sum}]
h_bkg = h[{'process': [proc for proc in h.axes['process'] if 'data' not in proc and not any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal', 'appl': 'isSR_3l'}][{'process': sum, 'channel': sum}]

 
h_nodi = h_data - h_bkg
print(h_diboson.axes['njets'].edges[:-1])
print(((h_data - h_bkg).eval({})[()] / h_diboson.eval({})[()])[1:-1])
print(((h_data - h_bkg).eval({})[()] / h_diboson.eval({})[()])[3:8])
data = (h_data - h_bkg).eval({})[()]
tot_data = np.sum(data)
yerr = np.nan_to_num(np.sqrt(1/h_data.eval({})[()] + 1/h_bkg.eval({})[()]), nan=0)
yerr = np.nan_to_num(np.sqrt(1/h_data.eval({})[()]), nan=0)
ratio = (h_data - h_bkg).eval({})[()] / h_diboson.eval({})[()]
bins=h_data.axes['njets'].edges
print(yerr)
hep.style.use("CMS")
hep.histplot((ratio)[3:-4], bins=h_diboson.axes['njets'].edges[2:-3], yerr=yerr[3:-4], histtype='errorbar', label='$N_{jets}$', color='tab:orange', capsize=4)
#hep.histplot(((h_data - h_bkg).eval({})[()] / h_diboson.eval({})[()])[3:8], bins=h_diboson.axes['njets'].edges[3:7], histtype='errorbar', label='$N_{jets}$', color='tab:orange', capsize=4)
print(bins, ratio)
print('fitting', bins[2:-4], ratio[3:-4])
fits = np.polyfit(bins[2:-4], ratio[3:-4], 1)
print('fits', bins[2:-4], np.polyval(fits, bins)[3:-4])
print('fitting', bins[1:-3], ratio[2:-3], [1, 1], yerr[2:-3])
popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], bins[1:-3], ratio[2:-3], [1, 1], yerr[2:-3])
print('fitting', np.arange(2,8), ratio[2:-4], [1, 1], yerr[2:-4])
popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], np.arange(2,8), ratio[2:-4], [1, 1], yerr[2:-4])
print(popt)
print('fits', np.arange(2,8), np.polyval(popt, np.arange(2,8)))
plt.plot(np.arange(2,8), np.polyval(popt, np.arange(2,8)))
plt.xlim([2,7])
plt.ylim([1,3.5])
plt.xlabel(r'$N_{jets}$', loc='right')
plt.ylabel(r'$\frac{data\;-\;non-diboson}{diboson}$')
plt.savefig('diboson.pdf')
plt.savefig('diboson.png')
#plt.show()
