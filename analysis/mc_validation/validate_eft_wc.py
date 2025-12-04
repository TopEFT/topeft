'''
This script runs over a collection of outputs for a particular process
and WC and checks if each gridpack produced consistent reweighting.

wc: the WC to run over
proc: the process to run over
tag: the tag used when making the gridpack
wc_tag: an optional tag used when making the nanoGEN samples (defaults to blank/None if not given)
tolerance: adjust the threshold for agreement
debug: print out more messages

This script first looks for the customize cards in `InputCards`.
If it can't find that folder, it will try to extract it from the gridpacks.
To run properly, either ensure you have the gridpacks on the same servery you'll run from,
or simply copy the customize cards with a strucutre like:
`InputCards/ttllNuNuJetNoHiggs/ttllNuNuJetNoHiggs_Run3Dim6TopWithTOP22006AxisScan_run0_customizecards.dat`

Example:
`python validate_eft_wc.py --proc ttH --tag Run3Dim6TopWithTOP22006AxisScan --wc-tag 7pts_500 --wc ctp`
'''
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from topcoffea.modules.utils import get_hist_from_pkl
from topcoffea.scripts.make_html import make_html

parser = argparse.ArgumentParser(description='You can select which file to run over')
parser.add_argument('--tolerance', '-t' , default='3' , help = 'Tolerance level (e.g. 3 = 1e-3)')
parser.add_argument('--wc',   required=True, help = 'WC to use')
parser.add_argument('--proc', required=True, help = 'Process to use')
parser.add_argument('--tag',  required=True, help = 'Gridpack tag (example: Run3With52WCsSMEFTsimTopMasslessTOP22006AxisScan)')
parser.add_argument('--wc-tag',  default=None, help = 'Gridpack wc-tag (example: Run3With52WCsSMEFTsimTopMasslessTOP22006AxisScan)')
parser.add_argument('--debug', '-v', action="store_true", help = 'Debug printouts')

if len(sys.argv) < 7:
    parser.print_help(sys.stderr)
    exit(1)
args  = parser.parse_args()

TOLERANCE = int(args.tolerance)

process_map = {
    'tllq': 'tllq4fNoSchanWNoHiggs0p',
    'ttlnu': 'ttlnuJet',
    'tHq': 'tHq4f',
    'ttH': 'ttHJet',
    'ttll': 'ttllNuNuJetNoHiggs',
    'tttt': 'tttt',
    'tWZ': 'tWZll',
    'tWZll': 'tWZll',
}

def get_sow(fname, run='run0'):
    hname = f'{fname}_{run}.pkl.gz'
    if 'run0' in run:
        hname = f'{fname}.pkl.gz'
    if not os.path.exists(hname):
        print(f'WARNING: {hname} does not exist! Skipping!')
        return None
    hists = get_hist_from_pkl(hname)
    return hists['SumOfWeights']


def get_orig_xsec(fname, run='run0'):
    hname = f'{fname}_{run}.pkl.gz'
    if 'run0' in run:
        hname = f'{fname}.pkl.gz'
    if not os.path.exists(hname):
        print(f'WARNING: {hname} does not exist! Skipping!')
        return None
    hists = get_hist_from_pkl(hname)
    if 'originalXWGTUP' in hists:
        if args.debug: print(f'Returning for {run}: {hists["originalXWGTUP"]=}')
        return hists['originalXWGTUP']
    else:
        print(f'WARNING! originalXWGTUP not found in {fname}')
        return 0


def get_start(path, fname, run='run0'):
    #/hadoop/store/user/byates2/ttgamma_dilep_ttgamma_run0_slc7_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz
    #gridpack = f'{path}/{fname}_{run}_slc7_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz'
    card_path = 'InputCards'
    card_name = f'{fname}_{run}_customizecards.dat'
    card = f'{card_path}/{process_map[args.proc]}/{card_name}'
    if not os.path.exists(card):
        tcard = f'{card_path}/{card_name}'
        gridpack = f'{path}/{fname}_{run}_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz'
        if not os.path.exists(gridpack):
            print(f'WARNING: Gridpack {gridpack} not found!')
            return None
        print(f'Untaring {card} from {gridpack}')
        os.system(f'tar xf {gridpack} {tcard}')
        #os.system(f'ls -lrth InputCards')
        os.system(f'mkdir -p InputCards/{process_map[args.proc]}')
        os.system(f'mv {tcard} {card}')
    with open(card) as fin:
        lines = [x.strip() for x in fin.readlines()]
    lines = (l.replace('set param_card ', '') for l in lines)
    wcs = (l.split() for l in lines if l[0] == 'c')
    wcs = ((l[0], float(l[1])) for l in wcs)
    return dict(wcs)


def get_json(fname, run='run0'):
    #../../input_samples/sample_jsons/signal_samples/private_UL/UL16_TTGamma_Dilept_run0.json
    jname = f'../../input_samples/sample_jsons/signal_samples/private_UL/2022_{fname}_{run}.json'
    if 'run0' in run:
        jname = f'../../input_samples/sample_jsons/signal_samples/private_UL/2022_{fname}.json'
    if not os.path.exists(jname):
        print(f'WARNING: {jname} does not exist! Skipping!')
        return None
    with open(jname) as fin:
        j = json.load(fin)
    return j


def get_tasks(j):
    return len(j['files'])


def rename_wcs(wc_pt, wcs):
    tmp = {}
    if wc_pt is None: return tmp
    for wc, pt in wc_pt.items():
        if wc in wcs:
            tmp[wc] = pt
        elif wc[:-1] + 'i' in wcs:
            tmp[wc[:-1]+'i'] = pt
            print(tmp)
        else:
            raise Exception(f'Not sure how to handle {wc}!')
    return tmp


def get_xsec(hist, wc_pt={}, sow=1, debug=False):
    num = np.sum(hist[{'process': sum}].eval(wc_pt)[()])
    if debug: print(f'{num=} / {sow=}')
    return np.sum(hist[{'process': sum}].eval(wc_pt)[()]) / sow


def get_xsec_error(hist, wc_pt={}, sow=1, debug=False):
    num = np.sum(hist[{'process': sum}].eval(wc_pt)[()])
    var = np.sum(hist[{'process': sum}].as_hist(wc_pt).variances()[()])
    return get_xsec(hist, wc_pt, sow) * np.sqrt(sow) / sow # at the starting point the weights are equal, so error = xsec * sqrt(n_tasks) / n_tasks
    return np.sqrt(var) / sow


def get_mg_weight(fname, run='run0', runs=[1,2,3], xsec=1, sow=1):
    hname = f'{fname}_{run}.pkl.gz'
    if 'run0' in run:
        hname = f'{fname}.pkl.gz'
    if not os.path.exists(hname):
        print(f'WARNING: {hname} does not exist! Skipping!')
        return None
    hists = get_hist_from_pkl(hname)
    run = run[3:]
    weights = []
    skip = False
    for irun in runs: #[0, 1, 2, 3]:
        int_run = (int(irun[3:]))
        if irun == f'run{int(run)}' and int_run == 0:
            skip = True
            weights.append(0)
            continue
        if f'EFT_weights{int_run-skip}' in hists:
            #weights.append(np.sum(hists[f'EFT_weights{int_run-skip}'].values()) / hists['originalXWGTUP'])
            #weights.append(np.sum(hists[f'EFT_weights{int_run-skip}'].values()) / np.sum(hists['SumOfWeights'].values()) / sow)
            weights.append(np.sum(hists[f'EFT_weights{int_run-skip}'].values()) / sow)
            #weights.append(hists[f'EFT_weights{irun-skip}'] / sow)
            key = f"EFT_weights{int_run-skip}"
            if args.debug:
                print(
                    f"Adding run{run} wgt{int_run-skip} "
                    f"{np.sum(hists[key].values())} -> "
                    f"{np.sum(hists[key].values()) / sow} "
                    f"{sow=} {xsec=}"
                )
            #weights.append(hists[f'EFT_weights{irun}'] / sow * xsec)
        else:
            print(irun, f'EFT_weights{int_run-skip} not found, adding 0')
            weights.append(0)
    if 'SM_weights' in hists:
        if args.debug:
            val = np.sum(hists["SM_weights"].values())
            print(
                f"Adding run{run} SM {val} -> {val / sow} {sow=} {xsec=}"
            )
        weights.append(np.sum(hists['SM_weights'].values()) / sow)
    else:
        weights.append(0)
    return weights


def make_plots(runs, wc_pts, jobs, h, orig_xsecs, mg_weights, mg_sm, sample, rel=True):
    user = os.getlogin()
    xsec_points = []
    wc_points = []
    poly = []

    mg_draw = True
    mgw_draw = True
    if args.debug: print(f'HERE {runs=} {jobs=}')
    for irun,run in enumerate(runs):
        wc_points.append([])
        xsec_points.append([])
        if run is None: continue
        for ipt, wc_pt in enumerate(wc_pts):
            h = get_sow(dname + '/' + fname, run)
            if h is None: continue
            if wc_pt is None: continue

            # Catch edge case where first file is missing _and_ the WC has a lepotn flavor
            # (e.g. `cte1` -> `ctei`)
            bad = False
            for wc in wc_pt:
                if wc not in h.wc_names: bad = True
            if bad: continue

            xsec = get_xsec(h, wc_pt, jobs[irun]) / (get_xsec(h, {}, jobs[irun]) if rel else 1.)
            # if irun == ipt: print(f'{rel=} {np.sum(h.as_hist({}).values())} {run=} {xsec=} ({get_xsec(h, wc_pt, jobs[irun])} / {get_xsec(h, {}, jobs[irun]) if rel else 1.} = {get_xsec(h, wc_pt, jobs[irun]) / get_xsec(h, {}, jobs[irun]) if rel else 1.})')
            if irun==ipt and args.debug: print(f'{xsec=} {jobs[irun]=} {mg_weights[irun]=}')
            xsec_err = get_xsec_error(h, wc_pts[irun], jobs[irun]) # Errors at starting points
            xsec_points[irun].append(xsec)
            wc_points[irun].append(list(wc_pt.values())[0])
            #if xsec > 100: # Don't plot extreme values
                #continue
            # xsec_err = False # FIXME
            if xsec_err > 10: xsec_err = False # FIXME
            # if irun==ipt and not rel: print(f'HERE {run} {xsec_err=} {xsec_err / xsec =}')
            if ipt == irun:
                label = f'st. pt. {list(wc_pts[irun].keys())[0]}={np.round(list(wc_pts[irun].values())[0], 3)}'
                plt.errorbar(list(wc_pt.values())[0], xsec, yerr=xsec_err, color=colors[irun], markerfacecolor='none', markeredgecolor=colors[irun], label=label, marker='o')
                #plt.errorbar(list(wc_pt.values())[0], xsec, yerr=xsec_err, color=colors[irun], markerfacecolor='none', markeredgecolor=colors[irun], label=f'st. pt. {wc_pts[irun]}', marker='o')
                if orig_xsecs[irun] is not None and orig_xsecs[irun] > 0:
                    if args.debug: print(f'{run} {wc_pt} {orig_xsecs[irun]=} / ' + str(get_xsec(h, {}, jobs[irun]) if rel else 1.) + ' -> ' + str(orig_xsecs[irun] / (get_xsec(h, {}, jobs[irun]) if rel else 1.)))
                    if mg_draw:
                        plt.plot(list(wc_pt.values())[0], orig_xsecs[irun] / (get_xsec(h, {}, jobs[irun]) if rel else 1.), marker='*', color='k', label='MG xsec', zorder=100, markersize=10, fillstyle='none')
                        mg_draw = False
                    else:
                        plt.plot(list(wc_pt.values())[0], orig_xsecs[irun] / (get_xsec(h, {}, jobs[irun]) if rel else 1.), marker='*', color='k', zorder=100, markersize=10, fillstyle='none')
            else:
                if len(xsecs) <= ipt or xsecs[ipt] is None:
                    plt.errorbar(list(wc_pt.values())[0], xsec, yerr=xsec_err, color=colors[irun], markerfacecolor='none', markeredgecolor=colors[irun])
                else:
                    plt.errorbar(list(wc_pt.values())[0], xsec, yerr=xsec_err, color=colors[irun], markerfacecolor='none', markeredgecolor=colors[irun], marker='o')
                if mg_weights[irun] is None: continue
                if len(mg_weights[irun]) <= ipt: continue
                if mg_weights[irun][ipt] is None: continue
                #if args.debug: print(f"normalizing {run}")
                #if args.debug: print(f"{wc_pt}")
                #if args.debug: print(f"{mg_weights[irun][ipt]}")
                #if args.debug: print(f"to {(get_xsec(h, {}, jobs[irun]) if rel else 1.)}")
                #if args.debug: print(f"-> {mg_weights[irun][ipt] / (get_xsec(h, {}, jobs[irun]) if rel else 1.)}")
                #if args.debug: print(f"normalizing {run} {wc_pt} {mg_weights[irun][ipt]} to {(get_xsec(h, {}, jobs[irun]) if rel else 1.)} -> {mg_weights[irun][ipt] / (get_xsec(h, {}, jobs[irun]) if rel else 1.)}")
                #if args.debug: print(f"For {irun} {wc_pt} hist predicts {get_xsec(h, wc_pt, jobs[irun])} and MG gave {mg_weights[irun][ipt]}.")
                if args.debug and np.abs(get_xsec(h, wc_pt, jobs[irun]) - mg_weights[irun][ipt]) / mg_weights[irun][ipt] > 10**(-1*TOLERANCE): print(f"For {irun} {wc_pt} hist predicts {get_xsec(h, wc_pt, jobs[irun])} and MG gave {mg_weights[irun][ipt]}.")
                if len(mg_weights) > irun and mg_weights[irun] is not None and len(mg_weights[irun]) > ipt and mg_weights[irun][ipt] is not None and mg_weights[irun][ipt] > 0: # and False: # FIXME
                    # TODO understand why MG SM is sometimes _very_ off, using hist predicted SM for now
                    #if args.debug: print(f'Drawing {mg_weights[irun][ipt]} -> {mg_weights[irun][ipt] * (mg_sm[irun] if not rel else 1.)}')
                    if args.debug: print(f'Drawing {mg_weights[irun][ipt]} -> {mg_weights[irun][ipt] * (get_xsec(h, {}, jobs[irun]) if not rel else 1.)}')
                    if mgw_draw:
                        plt.plot(list(wc_pt.values())[0], mg_weights[irun][ipt] * (get_xsec(h, {}, jobs[irun]) if not rel else 1.), marker='s', color=colors[irun], label='MG weight', zorder=100, markersize=10, fillstyle='none', linestyle='--')
                        #plt.plot(list(wc_pt.values())[0], mg_weights[irun][ipt] * (mg_sm[irun] if not rel else 1.), marker='s', color=colors[irun], label='MG weight', zorder=100, markersize=10, fillstyle='none', linestyle='--')
                        mgw_draw = False
                    else:
                        plt.plot(list(wc_pt.values())[0], mg_weights[irun][ipt] * (get_xsec(h, {}, jobs[irun]) if not rel else 1.), marker='s', color=colors[irun], zorder=100, markersize=10, fillstyle='none', linestyle='--')
                        #plt.plot(list(wc_pt.values())[0], mg_weights[irun][ipt] * (mg_sm[irun] if not rel else 1.), marker='s', color=colors[irun], zorder=100, markersize=10, fillstyle='none', linestyle='--')
        if len(wc_points[irun])==0:
            poly.append(None)
            continue
        #if any(xsec_pt > 100 for xsec_pt in xsec_points[irun]):
        #    continue
        poly.append(np.polyfit(wc_points[irun], xsec_points[irun], 2))

    #print(f'{poly=}')
    for irun,run in enumerate(runs):
        if len(wc_points[irun]) == 0: continue
        if len(poly) < irun+1: continue
        if len(wc_points) < irun+1: continue
        if poly[irun] is None: continue
        if wc_points[irun] is None: continue
        xaxis = np.linspace(min(wc_points[irun]),max(wc_points[irun]),100)
        plt.plot(xaxis, np.polyval(poly[irun], xaxis), color=colors[irun], ls=styles[irun % 4])

    if poly is None: return
    plt.legend(ncol=2)
    wc = args.wc
    plt.xlabel(wc)
    if rel:
        plt.ylabel(f'xsec({wc}) / xsec(SM)')
    else:
        plt.ylabel(f'xsec({wc})')
    postfix = '_rel' if rel else ''
    ax = plt.gca()
    # ax.set_yscale('log')
    xsec_points = [x if x is not None else 1. for x in xsec_points]
    max_xsec = np.max(np.array(xsec_points))
    if isinstance(max_xsec, list): # TODO undertand why this is different for tttt
        max_xsec = max(max_xsec)
    if max_xsec is None:
        max_xsec = 1.
    max_xsec *= 1.5
    plt.ylim(0, min(max_xsec, 20))
    if rel:
        ax.set_yticks(np.arange(0, min(max_xsec, 20), 1))
        plt.grid()
    if args.wc_tag is not None:
        sample = sample.replace(f'_{args.wc_tag}', '')
    if args.debug: print(f'/users/{user}/afs/www/EFT/1D/mg_xsec/{args.proc}/{sample}{postfix}.png')
    plt.savefig(f'/users/{user}/afs/www/EFT/1D/mg_xsec/{args.proc}/{sample}{postfix}.png')
    plt.close()


if __name__ == '__main__':
    dname  = f'/scratch365/{os.environ["USER"]}/wc_validation/1D/'
    # dname  = f'/afs/crc.nd.edu/user/y/ywan2/Public/forBrent/1D_gridpack_tllq/'  # FIXME this is for Wynona's pkl files
    fname  = 'sow_ttG_di_eft_histEFT'
    fname = f'2022_ttH_{args.wc}.pkl.gz'
    #fname  = 'UL16_TTGamma_Dilept_nanoGEN'
    proc = args.proc.replace('NuNuJetNoHiggs', '')#.replace('Jet', '')
    fname = f'2022_{proc}_{args.wc}'
    # fname = f'2022_nanoGEN_{proc}_{args.wc}'  # FIXME this is for Wynona's pkl files
    sample = f"{proc}_{args.wc}"
    if args.wc_tag is not None:
        sample += f'_{args.wc_tag}'
        fname += f'_{args.wc_tag}'
    if args.debug: print(sample)
    default_points = [-20, -6.67, 6.67, 20]
    ipt = 0
    #sample = 'TTGamma_Dilept_nanoGEN'
    #runs   = ['run0', 'full11', '9_run0']
    runs   = ['run0', 'run1']
    #runs   = ['run0', 'run1', 'run2', 'run3']
    #runs   = ['run1', 'run2']
    runs = [f'run{irun}' for irun in range(7)]
    lumi = 137 * 1000
    #runs = [run for run in runs if '3' not in run]
    #runs   = ['run0', 'run1']
    good_runs = []
    wc_pts = []
    xsecs  = []
    xsec_errs  = []
    orig_xsecs  = []
    mg_weights  = []
    mg_sm  = []
    sows   = []
    files  = []
    jobs   = []
    BLUE  = '\033[94m'
    GREEN = '\033[92m'
    RED   = '\033[91m'
    ENDC  = '\033[0m'
    match = GREEN+'GOOD'+ENDC
    bad   = RED+'BAD'+ENDC
    fail = [False] * len(runs)
    for irun,run in enumerate(runs):
        j = get_json(sample, run)
        h = get_sow(dname + '/' + fname, run)
        wc_pt = get_start('/cms/cephfs/data/store/user/' + os.environ['USER'] + '/Run3_gridpacks', f'{process_map[args.proc]}_{args.wc}{args.tag}', run)
        # wc_pt = get_start('/cms/cephfs/data/store/user/ywan2/Run3_gridpacks/tllq/', f'{process_map[args.proc]}_{args.wc}{args.tag}', run)  # FIXME Wynona's gridpacks
        wc_pts.append(wc_pt)
        if h is None or j is None or wc_pt is None: # or run == 'run3' or run == 'run4' or run == 'run2':
            jobs.append(-1)
            #wc_pts.append({args.wc: default_points[ipt]})
            ipt += 1
            xsecs.append(None)
            xsec_errs.append(None)
            orig_xsecs.append(None)
            mg_weights.append(None)
            mg_sm.append(None)
            good_runs.append(None)
            fail[irun] = True
            continue
        # ttHJet_cHQ1Run3With52WCsSMEFTsimTopMasslessTOP22006AxisScan_run1_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz
        #wc_pt = {"cpQM": 62.000000, "ctW": 1.580000, "ctq1": 1.190000, "cQq81": 2.430000, "ctZ": 2.560000, "cQq83": 2.780000, "ctG": 0.310000, "ctq8": 2.020000, "cQq13": 1.340000, "cQq11": 1.350000, "cpt": 32.930000}
        #xsec = get_xsec(h, wc_pt, j['nSumOfWeights'])# * jobs[irun])
        #jobs.append(j['nTasks'])
        wc_pt = get_start('/cms/cephfs/data/store/user/' + os.environ['USER'] + '/Run3_gridpacks', f'{process_map[args.proc]}_{args.wc}{args.tag}', run)
        # wc_pt = get_start('/cms/cephfs/data/store/user/ywan2/Run3_gridpacks/tllq/', f'{process_map[args.proc]}_{args.wc}{args.tag}', run)  # FIXME Wynona's gridpacks
        wc_pt = rename_wcs(wc_pt, h.wc_names)
        wc_pts[-1] = wc_pt
        #wc_pts.append(wc_pt)
        jobs.append(get_tasks(j))
        #jobs.append(100)
        xsec = get_xsec(h, wc_pt, jobs[irun])
        #xsec = lumi * get_xsec(h, wc_pt, jobs[irun]) / get_xsec(h, {}, jobs[irun])
        orig_xsecs.append(get_orig_xsec(dname + '/' + fname, run) / jobs[irun])
        mgw = get_mg_weight(dname + '/' + fname, run, runs, orig_xsecs[-1], jobs[irun])
        mg_sm.append(mgw.pop())
        mg_sm[-1] = 1 if mg_sm[-1] == 0 else mg_sm[-1]
        mg_weights.append([w / mg_sm[-1] for w in mgw])
        if '0.0' in wc_pt:
            mg_sm[-1] = mg_weights[-1]
        if args.debug: print(f'{run=} {wc_pt=} {xsec=} {orig_xsecs[-1]=} {mg_weights[-1]=} {mg_sm=}')
        #mg_weights.append(get_mg_weight(dname + '/' + fname, run, runs, orig_xsecs[-1], jobs[irun]) / get_xsec(h, {}, jobs[irun]))
        xsec_err = get_xsec_error(h, wc_pts[irun], jobs[irun]) # Errors at starting points
        print(f'Run {run} starts at:\n{wc_pt}\nwith {xsec} pb ({jobs[irun]} tasks)')
        sows.append(j['nSumOfWeights'])
        xsecs.append(np.round(xsec, TOLERANCE))
        xsec_errs.append(np.round(xsec_err, TOLERANCE))
        good_runs.append(run)
        if args.debug: print(f'Added {xsecs[-1]} {xsec_errs[-1]} {orig_xsecs[-1]}')
        files.append(len(j['files']))
        print(f'{BLUE}{sample} {run}: {xsec}{ENDC}')
    if all(fail):
        print(f'WARNING: No pkl files found for {sample}!')
        exit()
    colors = ['red', 'green', 'blue', 'orange', 'yellow', 'brown', 'magenta']
    styles = ['-', '--', '-.', ':']
    '''
    for irun,run in enumerate(runs):
        wc_points.append([])
        xsec_points.append([])
        for ipt, wc_pt in enumerate(wc_pts):
            h = get_sow(dname + '/' + fname, run)
            xsec = get_xsec(h, wc_pt, jobs[irun]) / get_xsec(h, {}, jobs[irun])
            xsec_err = get_xsec_error(h, wc_pts[irun], jobs[irun]) # Errors at starting points
            xsec_points[irun].append(xsec)
            wc_points[irun].append(list(wc_pt.values())[0])
            if ipt == irun:
                plt.errorbar(list(wc_pt.values())[0], xsec, yerr=xsec_err, color=colors[irun], markerfacecolor='none', markeredgecolor=colors[irun], label=f'st. pt. {wc_pts[irun]}', marker='o')
            else:
                plt.errorbar(list(wc_pt.values())[0], xsec, yerr=xsec_err, color=colors[irun], markerfacecolor='none', markeredgecolor=colors[irun], marker='o')
        poly.append(np.polyfit(wc_points[irun], xsec_points[irun], 2))
    for irun,run in enumerate(runs):
        xaxis = np.linspace(min(wc_points[irun]),max(wc_points[irun]),100)
        plt.plot(xaxis, np.polyval(poly[irun], xaxis), color=colors[irun], ls=styles[irun])
    '''

    user = os.getlogin()
    os.makedirs(f'/users/{user}/afs/www/EFT/1D/mg_xsec/{args.proc}/', exist_ok=True)
    make_plots(good_runs, wc_pts, jobs, h, orig_xsecs, mg_weights, mg_sm, sample, rel=True)
    make_plots(good_runs, wc_pts, jobs, h, orig_xsecs, mg_weights, mg_sm, sample, rel=False)
    make_html(f'/users/{user}/afs/www/EFT/1D/mg_xsec/{args.proc}')

    for irun,run in enumerate(runs):
        for ipt, wc_pt in enumerate(wc_pts):
            if irun == ipt:
                continue
            h = get_sow(dname + '/' + fname, run)
            if h is None: continue
            if wc_pt is None: continue
            if sows is None: continue
            if len(xsecs) < ipt+1: continue
            if xsecs[ipt] is None: continue
            if xsecs[irun] is None: continue
            if len(xsec_errs) < ipt+1: continue
            if xsec_errs[ipt] is None: continue
            if xsec_errs[irun] is None: continue
            if len(sows) < ipt+1: continue
            if len(runs) < ipt+1: continue
            if runs[ipt] is None: continue
            if xsecs[irun] == 0.: # Skim the SM since it can't reweight to some points reliably
                continue
            # sow = sows[irun]
            #xsec = get_xsec(h, wc_pt, sow)# * jobs[irun])
            xsec = get_xsec(h, wc_pt, jobs[irun])
            if xsec is None:
                continue
            #print(wc_pt, f'{xsec=}')
            #print(f'Reweighting {run} to {runs[ipt]} ({wc_pt})')
            pdiff = np.round(100 * (1 - xsecs[ipt] / xsec), TOLERANCE)
            if np.abs(xsec - xsecs[ipt]) < 10**(-1*TOLERANCE) or np.abs(xsecs[ipt] - xsec) < xsec_errs[ipt]:
            #if np.abs(xsec - xsecs[ipt]) < 10**(-1*TOLERANCE) or (np.abs(pdiff) < 4):
                print(f'{sample} {run}: {np.round(xsecs[irun], TOLERANCE)} -> {np.round(xsec, TOLERANCE)} {runs[ipt]}: {np.round(xsecs[ipt], TOLERANCE)} +/- {xsec_errs[ipt]} {match} ({pdiff}% diff) (MG xsec={orig_xsecs[irun]})')
            else:
                print(f'{sample} {run}: {np.round(xsecs[irun], TOLERANCE)} -> {np.round(xsec, TOLERANCE)} {runs[ipt]}: {np.round(xsecs[ipt], TOLERANCE)} +/- {xsec_errs[ipt]} {bad} {pdiff}% diff (expected {xsecs[ipt]}) (MG xsec={orig_xsecs[irun]})')
    '''
    plt.legend()
    plt.savefig(f'/users/byates2/afs/www/mg_xsec_{sample}.png')
    '''
    for irun,run in enumerate(runs):
        h = get_sow(dname + '/' + fname, run)
        if h is None: continue
        if len(xsecs) < irun+1: continue
        if xsecs[irun] is None: continue
        #sow = sows[irun]
        #xsec = get_xsec(h, wc_pt, sow)# * jobs[irun])
        xsec = get_xsec(h, {}, jobs[irun])
        print(f'{sample} {run}: {np.round(xsecs[irun], TOLERANCE)} @SM: {BLUE}{np.round(xsec, TOLERANCE)}{ENDC}')
