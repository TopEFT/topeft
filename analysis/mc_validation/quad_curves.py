'''
This script produces the quadratic curves based on the SumOfWeights (inclusive cross-section).
As a minimum it takes in the SumOfWeights pkl file and a path to save the plots.
The path defaults to `~/afs/www/EFT/` which is the web are on `glados`.
Modify the script if you're using another system like `lxplus`.

Starting Points:
You can also pass the json file for the process which may have optional fields.
If you have the starting point used for the process, add it to the json file as
`
"StPt": {
    "WC1": st_pt,
    "WC2": st_pt,
    ...
    "WCn": stpt
},
`
then the script will draw the orange dots at the starting point.
Otherwise it draws them at zero

MadGraph dedicated point validation:
If you have the MadGraph cross-section for dedicated points, add them to the json file as
`
  "mg_weights": {
    "cHQ3": {"-15": 0.332, "-5": 0.299, "-1": 0.30, "1": 0.300, "5": 0.3296},
    ...
  },
`
then the script will draw the red stars and fit a quadratic to them as well.


Example:
python quad_curves.py ../topeft_run2/histos/2022_ttlnuJet_Run3.pkl.gz --json ../../input_samples/sample_jsons/signal_samples/private_UL/2022_ttlnuJet_nanoGEN.json --dout ttlnu_NewStPt_Run3
'''
import os
import pickle
from coffea import hist
import gzip
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from topcoffea.scripts.make_html import make_html

#Load hists from pickle file created by TopCoffea
hists={}

parser = argparse.ArgumentParser(description='You can select which file to run over')
parser.add_argument('fin'      , default='analysis/topEFT/histos/mar03_central17_pdf_np.pkl.gz' , help = 'Variable to run over')
parser.add_argument('--dout'   , default='ttgamma' , help = 'Where to save')
parser.add_argument('--save-all', action='store_true' , help = 'Where to save')
parser.add_argument('--json', default='', help = 'Json file(s) containing files and metadata')
parser.add_argument("--fixed"  , action="extend", nargs="+", help="Fixed WC versions (sum of weights pkl file)" )
parser.add_argument("--fixed-wcs"  , action="extend", nargs="+", help="Fixed WC values (`wc:val`)" )
parser.add_argument("--do-grad"   , action="store_true", help="Do full gradient (defaults to single derivative for each WC)" )
args  = parser.parse_args()
fin   = args.fin
assert args.fixed is None or len(args.fixed) == len(args.fixed_wcs), print(f'{len(args.fixed)} must match {len(args.fixed_wcs)}!')

with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
      if k in hists: hists[k]+=hin[k]
      else:               hists[k]=hin[k]

fixed_points = []
if args.fixed is not None:
    for ifin,fin in enumerate(args.fixed):
        print(f'Loading {ifin} {fin}')
        with gzip.open(fin) as fin:
            hin = pickle.load(fin)
            fixed_points.append(hin['SumOfWeights'])
    print('Done loading fixed points')
    

sow = hists['SumOfWeights']
wcs = sow._wc_names
#wcs = ['ctW', 'ctZ']
step = 0.01 # Small step size
wc_start = np.random.normal(0, .1, len(wcs)) # Pick some random values
wc_pt = dict(zip(wcs, wc_start))
wc_start = wc_pt
sm = sow[{'process': sum}].eval(wc_pt)[()][1]
sm = sow[{'process': sum}].eval({})[()][1]

for ifixed,fixed in enumerate(fixed_points):
    print(args.fixed_wcs[ifixed], fixed_points[ifixed][{'process': sum}].eval({})[()], sm, sow[{'process': sum}].eval({'cHbox': 0.81}))
    fixed_points[ifixed] = (args.fixed_wcs[ifixed], fixed_points[ifixed][{'process': sum}].eval({})[()][1] / sm)
    print(fixed_points[ifixed])

wc_vals = {}

wc_best = wc_pt

for key in wc_start:
    wc_vals[key] = [0]
wc_pt = {wc: 0 for wc in wcs} # Start at SM
#wc_pt = wc_start

wc_start = wc_pt
#wc_pt = dict(zip(wcs, [0]*len(wcs)))
print(wc_pt)
# Gradient "ascent" in 26D
old = wc_pt # This restarts each WC to 0 to just compute the derivative, not the gradient
for n in range(1000):
    val = sow[{'process': sum}].eval(wc_pt)[()][1]
    # 10% of SM?
    #if np.abs(val - 54688) < 1e-3:
    if np.abs(val / sm - 1.1) < 1e-2:
        print("Reached 10%", val/sm, f'in {n} steps')
        wc_best = wc_pt
        break
    #grad = np.array([])
    grad = np.empty(len(wc_pt))
    if not args.do_grad:
        old = wc_pt # This restarts each WC to 0 to just compute the derivative, not the gradient
    # Compute partial derivatives
    i = 0
    for wc,wc_val in wc_pt.items():
        tmp = old
        prev_val = val
        h_val = val
        '''
        if val/sm > 1.1:
            h_val = wc_val - step
        else:
            h_val = wc_val + step
        '''
        h_val = wc_val + step
        wc_pt[wc] = h_val
        tmp[wc] = h_val
        val = sow[{'process': sum}].eval(tmp)[()][1]
        # Approximate d(xsec)/d(wc) = slope
        # Shift WC value by slope * step
        '''
        if args.do_grad:
            grad = np.append(grad, (step*(1.1 - val/sm) / h_val + wc_val))
            #if n % 200 == 0:
            #    print(f'{wc=}, {val/sm=}, {h_val=}, {wc_val=}')
            #    print('Will step by', step*(1.1 - val/sm) / h_val + wc_val)
        else:
            grad = np.append(grad, (1.1 - val/sm) / h_val + wc_val)
        '''
        #grad = np.append(grad, step*(1.1 - val/sm) / h_val + wc_val)
        #FIXME grad = np.append(grad, (1.1 - val/sm) / h_val + wc_val)
        grad[i] = step*(1.1 - val/sm) / h_val + wc_val
        i += 1
        #grad = np.append(grad, (54688 - val/sm) / h_val * step + wc_val)
        #print(wc, tmp, val, grad[-1], val/sm, (1.1 - val/sm) / h_val + wc_val)
        #print(wc, grad, (1.1 - val/sm) / h_val * step + wc_val)
        #grad = np.append(grad, (1.1 - val/sm) / h_val * step + wc_val)
        if args.do_grad:
            old = tmp # Update so the other WCs aren't replaced with 0
            #if n % 200 == 0: print(old)
    wc_pt = dict(zip(wcs, grad))
    # Save current WC values for later processing
    for wc,wc_val in wc_pt.items():
        wc_vals[wc].append(wc_val)

# Set insensitive WCs to inf
#print('delta', {wc: np.abs(wc_best[wc] - wc_start[wc]) for wc in wc_best})
for wc in wc_best:
    if np.abs(wc_best[wc] - wc_start[wc]) < 1e-3 or (np.abs(wc_best[wc]) > 100):
        print('Dropping', wc)
        wc_best[wc] = np.inf
#print(f'{wc_best=}')
# Drop insensitive WCs
for wc in wc_best:
    if wc_best[wc] == np.inf:
        wc_vals.pop(wc)
        wc_best[wc] = 0

good_wcs = wc_vals.keys()
# Redo WC values
wc_vals = {wc: [] for wc in wc_vals}
wc_pt = {wc: -2 for wc in wc_vals}
val = sow[{'process': sum}].eval({})[()][1]
# Probably overkill if not computing yields here
for n in range(100):
    for wc,wc_val in wc_pt.items():
        h_val = wc_val + 0.1
        prev = val
        wc_pt[wc] = np.real(h_val)
        wc_vals[wc].append(h_val)
    #for wc in wc_vals:
        #val = sow[{'process': sum}].eval(wc_pt)[()][1]
        #wc_vals[wc].append(wc_pt[wc])
wc_1d = {}

# Make 1D plots
user = os.getlogin()
os.makedirs(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/EFT/{args.dout}/', exist_ok=True)
#wc_start = {"ctW": 1.580000, "cpQM": 62.660000, "ctq1": 1.190000, "cQq81": 2.430000, "ctZ": 2.560000, "cQq83": 2.780000, "ctG": 0.310000, "ctq8": 2.020000, "cQq13": 1.340000, "cQq11": 1.350000, "cpt": 32.930000}
#wc_start = {"ctW": 3.160000, "ctq1": 2.380000, "cQq81": 4.860000, "ctZ": 5.120000, "cQq83": 5.560000, "ctG": 0.620000, "ctq8": 4.040000, "cQq13": 2.680000, "cQq11": 2.700000}
# tttt
#wc_start = {"cbGRe": 23.360000, "ctj1": 0.640000, "cQj31": 0.750000, "ctGRe": 1.810000, "ctj8": 0.880000, "ctHRe": 11.050000, "cQj11": 0.760000, "ctu1": 0.870000, "cHtbRe": 52.960000, "cQj18": 1.280000, "clj1": 100.000000, "cleQt1Re22": 100.000000, "ctu8": 1.430000, "cQQ1": 0.800000, "cQt1": 0.860000, "cQj38": 1.320000, "ctb8": 69.420000, "ctd8": 1.990000, "cQt8": 1.660000, "cleQt3Re11": 100.000000, "ctd1": 1.170000, "cleQt3Re33": 100.000000, "cbWRe": 17.450000, "cHbox": 7.430000, "cld": 100.000000, "cleQt3Re22": 100.000000, "cQu1": 0.890000, "cQe": 100.000000, "cQd1": 1.160000, "cHt": 3.220000, "ctWRe": 2.980000, "clu": 100.000000, "cQd8": 1.790000, "cleQt1Re33": 100.000000, "cleQt1Re11": 100.000000, "ctt": 0.650000, "cHQ3": 2.960000, "cQb8": 69.850000, "cHQ1": 10.400000, "ctBRe": 3.720000, "cQu8": 1.290000, "cte": 100.000000, "ctl": 100.000000, "cQl1": 100.000000, "cbBRe": 100.000000, "cQl3": 100.000000}
#wc_start = {"cbGRe": 16.730000, "ctj1": -0.670000, "cQj31": 0.770000, "ctGRe": -0.190000, "ctj8": 1.140000, "ctHRe": -5.150000, "cQj11": -0.770000, "ctu1": -0.870000, "cHtbRe": 25.840000, "cQj18": 1.400000, "clj1": 100.000000, "cleQt1Re22": 100.000000, "ctu8": 1.790000, "cQQ1": -0.260000, "cQt1": 0.480000, "cQj38": 1.900000, "ctb8": 46.520000, "ctd8": 2.620000, "cQt8": -1.060000, "cleQt3Re11": 100.000000, "ctd1": -1.160000, "cleQt3Re33": 100.000000, "cbWRe": 13.780000, "cHbox": 5.190000, "cld": 100.000000, "cleQt3Re22": 100.000000, "cQu1": -0.900000, "cQe": 100.000000, "cQd1": -1.160000, "cHt": 1.990000, "ctWRe": 2.260000, "clu": 100.000000, "cQd8": 2.240000, "cleQt1Re33": 100.000000, "cleQt1Re11": 100.000000, "ctt": -0.280000 , "cHQ3": 1.970000, "cQb8": 44.360000, "cHQ1": -2.010000, "ctBRe": 2.670000, "cQu8": 1.650000, "cte": 100.000000, "ctl": 100.000000, "cQl1": 100.000000, "cbBRe": 64.600000, "cQl3": 100.000000}
mg_weights = {}
sm_xsec = 1
if args.json != '':
    with open(args.json) as fin:
        print(f'Loading {args.json}')
        j = json.load(fin)
        if 'StPt' in j:
            wc_start = j['StPt']
            print(f'Using {wc_start=}')
        if 'mg_weights' in j:
            mg_weights = j['mg_weights']
            if 'sm_xsec' in j:
                sm_xsec = j['sm_xsec']
            print(mg_weights)
wc_bad = []
if args.save_all:
    os.makedirs(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/EFT/{args.dout}/bad/', exist_ok=True)
for wc in wc_vals:
    has_mg = False
    vals = wc_vals[wc]
    vmin = np.min(wc_vals[wc])
    vmin = np.min([np.min(wc_vals[wc]), 0.9])
    vmax = np.max(wc_vals[wc])
    vmax = np.max([np.max(wc_vals[wc]), 6])
    vmin,vmax = [-6,6]
    yields = []
    # Compute yields along saved points
    for val in vals:
        yields.append(sow[{'process': sum}].eval({wc: val})[()][1]/sm)
        if val < vmin: vmin = val
        if val > vmax: vmax = val
    #if np.abs(np.max(yields[-1]) - 1) < 1e-2:
    #    wc_best.pop(wc)
    #    print(f'Removing {wc}')
    #    if args.save_all:
    #        wc_bad.append(wc)
    #    else:
    #        continue
    # Fit 1D curve
    poly = np.polyfit(vals, yields, 2)
    roots = np.roots(poly - np.array([0,0,1.1]))
    if np.any(np.iscomplex(roots)):# or ((np.abs(poly[0]) < 1e-3) and (np.abs(poly[1]) < 1e-3)):
        print('Skipping', wc, poly, '(complex roots)')
        if not args.save_all:
            continue
        if wc in wc_best:
            wc_best.pop(wc)
        if wc not in wc_bad:
            wc_bad.append(wc)
    # Make smooth plot
    if wc not in wc_start:
        wc_start[wc] = 100.
    vmin = min(min(np.min(roots)*1.1, -6), wc_start[wc])
    vmax = max(max(np.max(roots)*1.1, 6), wc_start[wc])
    polymin = min(np.polyval(poly, np.linspace(vmin,vmax,100)))
    if vmin < 100 and vmax > 100:
        print('Skipping', wc, poly, f'10% * SM @ {wc} > 100')
        if not args.save_all:
            continue
        if wc in wc_best:
            wc_best.pop(wc)
        if wc not in wc_bad:
            wc_bad.append(wc)

    plt.plot(np.linspace(vmin,vmax,100), np.polyval(poly, np.linspace(vmin,vmax,100)), label=f'1D curve\n{np.round(poly[0], 2)} * {wc}^2 + {np.round(poly[1], 2)} * {wc} + {np.round(poly[2], 2)}')
    # 10% larger than SM
    plt.axhline(1.1, linestyle='--', color='b')
    #plt.plot(vals, yields, 'o', color='r')
    #plt.plot(vals[::10], yields[::10], 'o', color='r')
    wc_1d[wc] = roots
    for iroot,root in enumerate(roots):
        best = sow[{'process': sum}].eval({wc: root})[()][1]
        if iroot==0:
            plt.plot(root, best/sm, color='k', marker='o', label='$\sigma=1.1\sigma_{SM}$')
        else:
            plt.plot(root, best/sm, color='k', marker='o')
    #best = sow[{'process': sum}].eval(wc_best)[()][1]
    #plt.plot(wc_best[wc], best/sm, color='k', marker='o', label='26D $\sigma=1.1\sigma_{SM}$')
    # Plot starting point
    plt.plot(wc_start[wc], sow[{'process': sum}].eval({wc: wc_start[wc]})[()][1]/sm, '*', label='Starting point')
    if 'mg_weights' in hists:
        plt.plot([-1, 0, 1], hists['mg_weights'][wc]/hists['mg_weights'][wc][1], marker='*', ls='None', markersize=10, color='red')
    if mg_weights:
        mgv = []
        mgw = []
        #for mg_wc in mg_weights:
        #    if wc == mg_wc.split('_')[0]:
        #        if has_mg: plt.plot(mg_weights[mg_wc][0], mg_weights[mg_wc][1]/sm_xsec, marker='*', ls='None', markersize=10, color='red')
        #        else: plt.plot(mg_weights[mg_wc][0], mg_weights[mg_wc][1]/sm_xsec, marker='*', ls='None', markersize=10, color='red', label='MadGraph')
        #        #if has_mg: plt.plot(mg_weights[mg_wc][0], mg_weights[mg_wc][1]/mg_weights[mg_wc][2], marker='*', ls='None', markersize=10, color='red')
        #        #else: plt.plot(mg_weights[mg_wc][0], mg_weights[mg_wc][1]/mg_weights[mg_wc][2], marker='*', ls='None', markersize=10, color='red', label='MadGraph')
        #        has_mg = True
        #        mgv.append(mg_weights[mg_wc][0])
        #        #mgw.append(mg_weights[mg_wc][1]/mg_weights[mg_wc][2])
        #        mgw.append(mg_weights[mg_wc][1]/sm_xsec)
        if wc in mg_weights:
            sm_wc_xsec = sm_xsec
            if '0' in mg_weights[wc]:
                sm_wc_xsec = mg_weights[wc]['0']
            for wgt in mg_weights[wc]:
                #if float(wgt) == 0: continue
                if has_mg: plt.plot(float(wgt), mg_weights[wc][wgt]/sm_wc_xsec, marker='*', ls='None', markersize=10, color='red')
                else: plt.plot(float(wgt), mg_weights[wc][wgt]/sm_wc_xsec, marker='*', ls='None', markersize=10, color='red', label='MadGraph')
                #if has_mg: plt.plot(wgt, mg_weights[wc][wgt]/wgt[2], marker='*', ls='None', markersize=10, color='red')
                #else: plt.plot(wgt, mg_weights[wc][wgt]/wgt[2], marker='*', ls='None', markersize=10, color='red', label='MadGraph')
                has_mg = True
                mgv.append(float(wgt))
                #mgw.append(mg_weights[wc][wgt]/wgt[2])
                mgw.append(mg_weights[wc][wgt]/sm_wc_xsec)
        '''
        if wc == 'ctt':
            mgpoly = np.polyfit(mgv, mgw, 2)
            print(np.roots(mgpoly - np.array([0,0,1.1])))
        '''
        if len(mgv) > 0 and len(mgv) == len(mgw):
            plt.plot(np.linspace(vmin,vmax,100), np.polyval(np.polyfit(mgv, mgw, 2), np.linspace(vmin,vmax,100)), color='r', ls='--')
    if fixed_points:
        for fixed in fixed_points:
            fix_wc, fix_value = fixed[0].split('=')
            if wc != fix_wc:
                continue
            plt.plot(float(fix_value), fixed[1], marker='*', ls='None', markersize=10, color='red')
    plt.grid()
    #plt.xlim(vmin*1.1,vmax*1.1)
    #plt.xlim(-1, 1)
    #plt.ylim(np.min(yields)-0.1,1.6)
    #plt.ylim(np.min(yields)-0.1,max(1.5, sow[{'process': sum}].eval({wc: wc_start[wc]})[()][1]/sm * 1.1))
    plt.ylim(min(np.min(yields), polymin)-0.2,max(1.6, sow[{'process': sum}].eval({wc: wc_start[wc]})[()][1]/sm * 1.1))
    #plt.ylim(np.min(yields)-0.8,max(1.5, sow[{'process': sum}].eval({wc: wc_start[wc]})[()][1]/sm * 1.1))
    #if 'box' in wc: plt.ylim(0, 1.5)
    plt.xlabel(wc)
    plt.ylabel('$\sigma_{EFT} / \sigma_{SM}$')
    plt.legend()
    if wc in wc_best: plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}//www/EFT/{args.dout}/{wc}_quad.png')
    else: plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/EFT/{args.dout}/bad/{wc}_quad.png')
    #print(wc, vals, yields)
    plt.close()
#print(wc_best)
print("All roots:", {k:wc_1d[k] for k in wcs if k in wc_1d})
'''
for wc in wc_1d:
    if np.iscomplex(wc_1d[wc][0]):
        print(wc, wc_1d[wc], len(wc_1d[wc]))
        wc_1d[wc][0] = 100
        print(wc, wc_1d[wc])
    if np.iscomplex(wc_1d[wc][1]):
        print(wc, wc_1d[wc])
        wc_1d[wc][1] = 100
        print(wc, wc_1d[wc])
'''
print("Smallest root:", {k:(100.0 if np.any(np.iscomplex(wc_1d[k])) else wc_1d[k][np.argmin(np.abs(wc_1d[k]))]) for k in wcs if k in wc_1d})
#print({k:(np.max(wc_1d[k]) if k in wc_1d else 100) for k in wcs})
for wc in wc_1d:
    if np.any(np.iscomplex(wc_1d[wc])) or np.min(np.abs(wc_1d[wc]) > 100):
        wc_1d[wc] = [100.0, 100.0]
print({k:np.round(wc_1d[k][np.argmin(np.abs(wc_1d[k]))], 2) for k in wcs if k in wc_1d})
#print({k:(100.0 if np.any(np.iscomplex(wc_1d[k]) | (np.min(np.abs(wc_1d[k]) > 100)) else np.round(wc_1d[k][np.argmin(np.abs(wc_1d[k]))], 2)) for k in wcs if k in wc_1d})
#print({k:np.round(np.(wc_1d[k]), 2) for k in wcs if k in wc_1d})
print('Good wcs', [wc for wc in wc_1d])

# 2D plots
wcs = ['ctW', 'ctZ']
wcs = ['ctZ']
wc_pt = {wc: -2 for wc in wcs}
wc_vals = {wc: [] for wc in wcs}
grads = {(wc1,wc2): [] for iwc,wc1 in enumerate(wcs) for wc2 in wcs[iwc:] if wc1 != wc2}
yields = {(wc1,wc2): [] for iwc,wc1 in enumerate(wcs) for wc2 in wcs[iwc:] if wc1 != wc2}
num = 10
r = 10
for iwc,wc1 in enumerate(wcs):
    for wc2 in wcs[iwc+1:]:
        for v1 in np.linspace(-r, r, num+1):
            tmp = wc_pt
            prev = val
            tmp[wc1] = v1
            val1 = sow[{'process': sum}].eval(tmp)[()][1]
            for v2 in np.linspace(-r, r, num+1):
                tmp = wc_pt
                tmp[wc2] = v2
                val2 = sow[{'process': sum}].eval(tmp)[()][1]
                tmp[wc1] = v1
                tmp[wc2] = v2
                val = sow[{'process': sum}].eval(tmp)[()][1]
                grads[(wc1,wc2)].append(((val1 - prev) / (2*r/num), (val2-prev) / (2*r/num)))
                wc_vals[wc1].append(v1)
                wc_vals[wc2].append(v2)
                wc_pt[wc1] = v1
                wc_pt[wc2] = v2
                val = sow[{'process': sum}].eval(wc_pt)[()][1]
                yields[(wc1,wc2)].append(val/sm)
for wc1,wc2 in yields:
    color = np.linspace(-1, 5, len(yields[(wc1,wc2)]))
    grad1 = np.array([g[0] for g in grads[(wc1,wc2)]])
    grad2 = np.array([g[1] for g in grads[(wc1,wc2)]])
    grad1 /= np.sqrt(np.square(grad1) + np.square(grad2))
    grad2 /= np.sqrt(np.square(grad1) + np.square(grad2))
    #grad1 *= step
    #grad2 *= step
    c=plt.quiver(wc_vals[wc1], wc_vals[wc2], grad1, grad2, color, cmap=plt.cm.jet)
    plt.colorbar(c, cmap=plt.cm.jet)
    plt.xlabel(wc1)
    plt.ylabel(wc2)
    plt.savefig(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/EFT/{args.dout}/{wc1}_{wc2}_quad.png')

make_html(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/EFT/{args.dout}/')
if args.save_all: make_html(f'/afs/crc.nd.edu/user/{user[0]}/{user}/www/EFT/{args.dout}/bad/')
