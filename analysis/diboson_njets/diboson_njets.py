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
parser.add_argument('fin'   , default='histos/diboson_njets.pkl.gz' , help = 'Variable to run over')
args  = parser.parse_args()
fin   = args.fin

hists = {}
with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
        if k in hists: 
            hists[k]+=hin[k]
        else:
            hists[k]=hin[k]
h = hists['njets']
print("HIST: ", h)
#h_data = h.integrate('sample',  [proc for proc in h.axis('sample').identifiers() if 'data' in proc.name]).integrate('channel',  [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic',  'nominal').integrate('appl',  'isSR_3l').to_hist()
#h_diboson = h.integrate('sample', [proc for proc in h.axis('sample').identifiers() if any(p == proc.name[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])]).integrate('channel', [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic', 'nominal').integrate('appl', 'isSR_3l').to_hist()
#h_bkg = h.integrate('sample', [proc for proc in h.axis('sample').identifiers() if 'data' not in proc.name and not any(p == proc.name[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])]).integrate('channel', [chan for chan in h.axis('channel').identifiers() if '3l_CR' in chan.name]).integrate('systematic', 'nominal').integrate('appl', 'isSR_3l').to_hist()
h_data = h[{'process': [proc for proc in h.axes['process'] if 'data' in proc] , 'channel': [chan for chan in h.axes['channel'] if ('3l' and 'CR') in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
print("H_DATA: ", h_data)
h_diboson = h[{'process': [proc for proc in h.axes['process'] if any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if ('3l' and 'CR') in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]
h_bkg = h[{'process': [proc for proc in h.axes['process'] if 'data' not in proc and not any(p == proc[:4] for p in ['WWTo', 'WZTo', 'ZZTo'])], 'channel': [chan for chan in h.axes['channel'] if ('3l' and 'CR') in chan], 'systematic': 'nominal'}][{'process': sum, 'channel': sum}]


# Subtract background
h_nodi = h_data - h_bkg

# Compute data arrays
data = h_nodi.values()
tot_data = np.sum(data)
yerr = np.nan_to_num(np.sqrt(1 / h_data.values() + 1 / h_bkg.values()), nan=0)
# Optional: only data uncertainty
yerr_data_only = np.nan_to_num(np.sqrt(1 / h_data.values()), nan=0)

# Compute ratio: (data - non-diboson) / diboson
ratio = h_nodi.values() / h_diboson.values()

# Bin edges and bin centers
edges = h_data.axes['njets'].edges
centers = 0.5 * (edges[:-1] + edges[1:])

# Print info
print("Bin edges:", edges[:-1])
print("Full ratio slice [1:-1]:", ratio[1:-1])
print("Ratio slice [3:8]:", ratio[3:8])
print("Y errors:", yerr)

# Select range for plotting/fitting
sel = slice(3, -4)
sel_ratio = ratio[sel]
sel_err = yerr[sel]
sel_centers = centers[sel]
sel_edges = edges[3:-4+1]
# Plot using bin centers with error bars
hep.style.use("CMS")
plt.errorbar(
    sel_centers, sel_ratio.flatten(),
    yerr=sel_err.flatten(),
    fmt='o',
    color='tab:orange',
    capsize=4,
    label='$N_{jets}$'
)

# Linear fit using polyfit for visualization
fits = np.polyfit(sel_centers, sel_ratio, 1)
print("Polyfit coefficients:", fits)
plt.plot(sel_centers, np.polyval(fits, sel_centers), linestyle='--', color='k', label='polyfit')

# Linear fit using curve_fit with uncertainties
def linear(x, m, b):
    return m * x + b

sel_ratio = sel_ratio.flatten()
sel_err   = sel_err.flatten()

popt, pcov = curve_fit(
    linear, sel_centers, sel_ratio,
    p0=[1, 1],
    sigma=sel_err,
    absolute_sigma=True
)
sel_centers = np.asarray(sel_centers).flatten()
sel_ratio   = np.asarray(sel_ratio).flatten()
sel_err     = np.asarray(sel_err).flatten()

print("Shapes:", sel_centers.shape, sel_ratio.shape, sel_err.shape)

plt.errorbar(
    sel_centers, sel_ratio,
    yerr=sel_err,
    fmt='o',
    color='tab:orange',
    capsize=4,
    label='$N_{jets}$'
)

plt.xlabel("Njets")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True)

plt.savefig("output.pdf")
plt.savefig("output.png")
plt.close()


# plt.show()



# plt.show()
#hep.histplot((ratio)[3:-4], bins=h_diboson.axes['njets'].edges[2:-3], yerr=yerr[3:-4], histtype='errorbar', label='$N_{jets}$', color='tab:orange', capsize=4)

#hep.histplot(((h_data - h_bkg).eval({})[()] / h_diboson.eval({})[()])[3:8], bins=h_diboson.axes['njets'].edges[3:7], histtype='errorbar', label='$N_{jets}$', color='tab:orange', capsize=4)
#print(bins, ratio)
#print('fitting', bins[2:-4], ratio[3:-4])
#fits = np.polyfit(bins[2:-4], ratio[3:-4], 1)
#print('fits', bins[2:-4], np.polyval(fits, bins)[3:-4])
#print('fitting', bins[1:-3], ratio[2:-3], [1, 1], yerr[2:-3])
#popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], bins[1:-3], ratio[2:-3], [1, 1], yerr[2:-3])
#print('fitting', np.arange(2,8), ratio[2:-4], [1, 1], yerr[2:-4])
#popt, pcov = curve_fit(lambda x, *p : p[0] * x + p[1], np.arange(2,8), ratio[2:-4], [1, 1], yerr[2:-4])
#print(popt)
#print('fits', np.arange(2,8), np.polyval(popt, np.arange(2,8)))
#plt.plot(np.arange(2,8), np.polyval(popt, np.arange(2,8)))
#plt.xlim([2,7])
#plt.ylim([1,3.5])
#plt.xlabel(r'$N_{jets}$', loc='right')
#plt.ylabel(r'$\frac{data\;-\;non-diboson}{diboson}$')
#plt.savefig('diboson.pdf')
#plt.savefig('diboson.png')
#plt.show()
