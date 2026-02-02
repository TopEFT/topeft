#!/usr/bin/env python
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import argparse
import pickle
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description='You can select which file to run over')
parser.add_argument('fin'   , default='' , help = 'Variable to run over')
args  = parser.parse_args()
fin   = args.fin

hists = {}
with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
        if k in hists: hists[k]+=hin[k]
        else:  
            hists[k]=hin[k]
h = hists['njets']
h_sumw2 = hists['njets_sumw2']
#h_data = h.integrate('sample',  [proc for proc in h.axis('sample').identifiers() if 'data' in proc.name]).integrate('channel',  [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic',  'nominal').integrate('appl',  'isSR_3l').to_hist()
#h_diboson = h.integrate('sample', [proc for proc in h.axis('sample').identifiers() if any(p == proc.name[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])]).integrate('channel', [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic', 'nominal').integrate('appl', 'isSR_3l').to_hist()
#h_bkg = h.integrate('sample', [proc for proc in h.axis('sample').identifiers() if 'data' not in proc.name and not any(p == proc.name[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])]).integrate('channel', [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic', 'nominal').integrate('appl', 'isSR_3l').to_hist()
#h_data = h[{'process': [proc for proc in h.axes['process'] if 'data' in proc], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
h_diboson = h[{'process': [proc for proc in h.axes['process'] if any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
h_diboson2 = h_sumw2[{'process': [proc for proc in h.axes['process'] if any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
#h_bkg = h[{'process': [proc for proc in h.axes['process'] if 'data' not in proc and not any(p == proc[:4] for p  in ['WWTo', 'WZTo', 'ZZTo']) and not any (p == proc[:6] for p in ['WLLJJ_']) and not any (p == proc[:3] for p in ['TWZ'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
#h_bkg2 = h_sumw2[{'process': [proc for proc in h.axes['process'] if 'data' not in proc and not any (p == proc[:4] for p  in ['WWTo', 'WZTo', 'ZZTo']) and not any (p == proc[:6] for p in ['WLLJJ_']) and not any (p == proc[:3] for p in ['TWZ'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
h_data = h[{'process': [proc for proc in h.axes['process'] if 'data' in proc], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
#h_diboson = h[{'process': [proc for proc in h.axes['process'] if any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
#h_diboson2 = h_sumw2[{'process': [proc for proc in h.axes['process'] if any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
h_bkg = h[{'process': [proc for proc in h.axes['process'] if 'data' not in proc and not any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
h_bkg2 = h_sumw2[{'process': [proc for proc in h.axes['process'] if 'data' not in proc and not any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if '3l_CR' in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
#for proc in h.axes['process']:
#    print('proc: ', proc)
#    if  any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo']):
#        print('diboson')
#    if 'data' not in proc and not any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo']) and not any (p == proc[:6] for p in ['WLLJJ_']) and not any (p == proc[:3] for p in ['TWZ']):
#        print('bkg')
#    if 'data' in proc:
#            print('data')
data = list(h_data.eval({}).values())[0].flatten()
print('h_data', data)
bkg = list(h_bkg.eval({}).values())[0].flatten()
print('h_bkg', bkg)
diboson = list(h_diboson.eval({}).values())[0].flatten()
print('h_diboson', diboson)
bkg2 = list(h_bkg2.eval({}).values())[0].flatten()
print('h_bkg2', bkg2[3:8])
diboson2 = list(h_diboson2.eval({}).values())[0].flatten()
print('h_diboson2', diboson2[3:8])
h_nodi = h_data - h_bkg
bins=h_diboson.axes['njets'].edges

# --- Difference of data and background
data_minus_bkg = data - bkg
ediboson = np.sqrt(diboson2)/(diboson)
ebkg = np.sqrt(bkg2)/(bkg2)
#print('ediboson', ediboson)
# --- Ratios
ratio = data_minus_bkg / diboson

# --- Slice examples
print(ratio[1:-1])
print(ratio[3:8])

# --- Total data
tot_data = np.sum(data_minus_bkg)

# --- Uncertainties (avoid divide-by-zero with np.nan_to_num)
yerr = np.nan_to_num(np.sqrt(1/data + 1/bkg), nan=0)
yerr2 = np.nan_to_num(np.sqrt((1/data)), nan=0)
yerr3 = np.nan_to_num(ratio * np.sqrt(((np.sqrt(data)+np.sqrt(ebkg))**2)/((data-bkg)**2)+(np.sqrt(ediboson)/diboson)**2), nan=0)
# --- Print uncertainties
print('yerr3: ', yerr3)
#h_nodi = h_data - h_bkg
#bins=h_diboson.axes['njets'].edges
#print(((h_data - h_bkg).values()/ h_diboson.values())[1:-1])
#print(((h_data - h_bkg).values() / h_diboson.values())[3:8])
#data = (h_data - h_bkg).values()
#tot_data = np.sum(data)
#yerr = np.nan_to_num(np.sqrt(1/h_data.eval({}) + 1/h_bkg.eval({})), nan=0)
#yerr = np.nan_to_num(np.sqrt(1/h_data.eval({})), nan=0)
#ratio = (h_data - h_bkg).eval({}) / h_diboson.eval({})
#print(yerr)
hep.style.use("CMS")
#hep.histplot((ratio)[3:-4], bins=h_diboson.axes['njets'].edges[2:-3], yerr=yerr[3:-4], histtype='errorbar', label='$N_{jets}$', color='tab:orange', capsize=4)
#hep.histplot(((h_data - h_bkg).eval({})[()] / h_diboson.eval({})[()])[3:8], bins=h_diboson.axes['njets'].edges[3:7], histtype='errorbar', label='$N_{jets}$', color='tab:orange', capsize=4)
#print('fitting', bins[2:-4], ratio[3:-4])
#fits = np.polyfit(bins[2:-4], ratio[3:-4], 1)
#print('fits', bins[2:-4], np.polyval(fits, bins)[3:-4])
#print('fitting', bins[1:-3], ratio[2:-3], [1, 1], yerr[2:-3])
#popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], bins[1:-3], ratio[2:-3], [1, 1], yerr[2:-3])
#print('fitting', np.arange(2,8), ratio[2:8], [1, 1], yerr[2:8])
#print('x: ', np.arange(2,8), 'y: ', ratio[2:8])
#popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], np.arange(2,8), ratio[2:8], [1, 1], yerr[2:8])
#print(popt)
#print('fits', np.arange(2,8), np.polyval(popt, np.arange(2,8)))
#plt.plot(np.arange(2,8), np.polyval(popt, np.arange(2,8)))
#print('ratio', ratio)
#print('ratio2', ratio[1:8])
#plt.plot(np.arange(2,8), ratio[2:8], marker='o', linestyle='none')
print('fitting', np.arange(2,8), ratio[2:-4], [1, 1], yerr3[2:-4])
print('x: ', np.arange(2,8), 'y: ', ratio[2:8])
print('bins', bins)
print('ratio', ratio)
print('ratio[2:-4]', ratio[2:-4])
popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], np.arange(2,6), ratio[2:6], [1, 1], yerr3[2:6])
#popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], bins[1:-3], ratio[2:-4], [1, 1], yerr[2:-4])
print(popt)
print('fits', np.arange(2,6), np.polyval(popt, np.arange(2,6)))
plt.plot(np.arange(2,8), np.polyval(popt, np.arange(2,8)))
print('ratio', ratio)
print('ratio2', ratio[1:8])
plt.plot(np.arange(2,8), ratio[2:8], marker='o', linestyle='none')

plt.xlim([2,8])
plt.ylim([0.5,3.5])
plt.xlabel(r'$N_{jets}$', loc='right')
plt.ylabel(r'$\frac{data\;-\;non-diboson}{diboson}$')
plt.savefig('diboson.pdf')
plt.savefig('diboson.png')
#plt.show()
