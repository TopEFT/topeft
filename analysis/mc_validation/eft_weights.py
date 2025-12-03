'''
This script produces the distributions of weights from nanoGEN files
(assuming the MG weights were also saved)
Example Run
python eft_weights.py \
--input /cms/cephfs/data/store/user/byates/tttt/nanoGEN_Run3/2022/tttt_LO_EFT/crab_tttt_nanoGEN_Run3/250715_223705/0000/ \
--output /users/byates2/afs/www/EFT/tttt_Run3/weights/weights.pdf
'''

#!/usr/bin/env python3

import uproot
import hist
import os
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import warnings
import topcoffea.modules.utils as utils
from topcoffea.modules.histEFT import HistEFT
plt.style.use(hep.style.CMS)

NanoAODSchema.warn_missing_crossrefs = False

if __name__ == '__main__':

    import argparse
    # 'EFTrwgt66_ctW_0.0_ctq1_0.0_cQq81_0.0_ctZ_0.0_cQq83_0.0_ctG_0.0_ctq8_0.0_cQq13_0.0_cQq11_0.0'

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='root://cmseos.fnal.gov//store/user/dspitzba/EFT/nanogen_small.root', help="Input file")
    argParser.add_argument('--output', action='store', default='./weights.pdf', help="Output file")
    args = argParser.parse_args()
 
    path = args.input
    #files = ['nanogen_12.root', 'nanogen_49.root', 'nanogen_51.root', 'nanogen_54.root', 'nanogen_5.root', 'nanogen_82.root', 'nanogen_13.root', 'nanogen_4.root', 'nanogen_52.root', 'nanogen_55.root', 'nanogen_63.root', 'nanogen_9.root', 'nanogen_2.root', 'nanogen_50.root', 'nanogen_53.root', 'nanogen_56.root', 'nanogen_7.root']
    files = [f for f in os.listdir(path) if '.root' in f]
    #files = [files[0]] # Only process a single file
    #files = [files[x] for x in range(10)] # Only process the first 10 files
    #files = ['nanogen_123_220.root']

    weight_axis = hist.axis.Regular(22, -20, 2, name="weight_ax", label="weight", underflow=True, overflow=True)
    #weight_axis = hist.axis.Regular(100, -1, 1e1, name="weight_ax", label="weight", underflow=True, overflow=True)
    h_SM = hist.Hist(weight_axis)
    events = NanoEventsFactory.from_root(
        #args.input,
        path+files[0],
        schemaclass=NanoAODSchema,
    ).events()

    w = events.LHEWeight
    eft_weight_names = [ x for x in w.fields if x.startswith('EFTrwgt') ]
    #h_ttG = hist.Hist(weight_axis)
    h_ttG = []
    eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
    #h_ttG_rwgt = HistEFT(weight_axis, wc_names=['ctZ', 'cpt', 'cpQM', 'cpQ3', 'ctW', 'ctp', 'ctG'], label=r"Events")
    for weight in eft_weight_names:
        h_ttG.append(hist.Hist(weight_axis))

    print('[', end='')
    for fin in files:
        events = NanoEventsFactory.from_root(
            #args.input,
            path+fin,
            schemaclass=NanoAODSchema,
        ).events()

        w = events.LHEWeight
        eft_weight_names = [ x for x in w.fields if x.startswith('EFTrwgt') ]
        if not eft_weight_names:
            eft_weight_names = utils.get_list_of_wc_names(path+fin)
            eft_weight_names = ['_0.0_'.join(eft_weight_names)]
            print(fin, eft_weight_names)

        sm_wgt = getattr(events.LHEWeight, eft_weight_names[-1])
        print('.', end='')
        #print(f'Smallest non-zero SM weight: {np.min(sm_wgt[sm_wgt != 0.0])}')
        h_SM.fill(weight_ax=np.log10(getattr(events.LHEWeight, eft_weight_names[-1])))
        #h_SM.fill(weight_ax=getattr(events.LHEWeight, eft_weight_names[-1]))

        #h_ttG = [(w,hist.Hist(weight_axis)) for w in ttG]
        #h_ctg1 = hist.Hist(weight_axis)
        #h_ctg1.fill(weight_ax=getattr(events.LHEWeight, 'EFTrwgt66_ctW_0.0_ctq1_0.0_cQq81_0.0_ctZ_0.0_cQq83_0.0_ctG_0.0_ctq8_0.0_cQq13_0.0_cQq11_0.0'))

        # EFTrwgt10_ctGRe_2.0_ctGIm_0.0_ctWRe_0.0_ctWIm_0.0_ctBRe_0.0_ctBIm_0.0_cHtbRe_0.0_cHtbIm_0.0_cHt_0.0
        #h_ctg2 = hist.Hist(weight_axis)
        #h_ctg2.fill(weight_ax=getattr(events.LHEWeight, 'EFTrwgt0_ctW_-1.722436_ctq1_1.171197_cQq81_1.34397_ctZ_-6.408086_cQq83_1.555205_ctG_0.2893_ctq8_-0.625025_cQq13_-1.305265_cQq11_1.762244'))
        #for (weight,h) in h_ttG:
        #    h.fill(weight_ax=getattr(events.LHEWeight, weight))


        #h_ctg1.plot1d(ax=ax, label=r'$C_{tG}=1$')
        #h_ctg2.plot1d(ax=ax, label=r'$C_{tG}=2$')
        #for i,(_,h) in enumerate(h_ttG):
        #    h.plot1d(ax=ax, label=f'wgt_{i}')

        for nw,weight in enumerate(eft_weight_names[:-1]):
            h_ttG[nw].fill(weight_ax=np.log10(getattr(events.LHEWeight, weight)))
            #h_ttG[nw].fill(weight_ax=getattr(events.LHEWeight, weight))
    #wgts = getattr(events.LHEWeight, 'EFTrwgt_ctZ_0.0_cpt_0.0_cpQM_0.0_cpQ3_0.0_ctW_0.0_ctp_0.0_ctG_0.0')
    #h_ttG_rwgt.fill(weight_ax=wgts, eft_coeff=eft_coeffs)
    print(']')

    for iweight,weight in enumerate(eft_weight_names):
        fig, ax = plt.subplots()

        hep.histplot(h_SM, ax=ax, label=r'$SM=0$', flow='show', histtype='errorbar', yerr=False)
        label = '$tt\gamma$'
        label = 'EFT'
        hep.histplot(h_ttG[iweight], ax=ax, label=label, flow='show', histtype='errorbar', yerr=False)
        # 'EFTrwgt66_ctW_0.0_ctq1_0.0_cQq81_0.0_ctZ_0.0_cQq83_0.0_ctG_0.0_ctq8_0.0_cQq13_0.0_cQq11_0.0'
        eft_weight = weight.split('_')[1:]
        wcs  = eft_weight[:-1:2]
        vals = eft_weight[1::2]
        vals = [float(v) for v in vals]
        eft_pt = dict(zip(wcs,vals))
        #hep.histplot(h_ttG_rwgt.as_hist(eft_pt), ax=ax, label=label+' reweight', flow='show', histtype='errorbar', yerr=False)
        ax.set_ylabel(r'# Events')
        ax.set_xlabel(r'log(weight)')

        #ax.set_xscale("log")
        ax.set_yscale("log")
        #ax.set_xlim([1e-2, 1e2])
        #ax.set_xlim([1e-6, 1e2])
        #ax.set_xlim([1e-2, 1e2])

        plt.legend()

        fig.savefig(args.output)
        fig.savefig(args.output.replace('.pdf', '_rwgt{}.pdf'.format(iweight)))
        fig.savefig(args.output.replace('.pdf', '_rwgt{}.png'.format(iweight)))
        print(f"Figure saved in {args.output.replace('.pdf', '_rwgt{}.pdf'.format(iweight))}")
        #fig.savefig(args.output.replace('.pdf', '_{}.pdf'.format(eft_weight_names[iweight])))
        #fig.savefig(args.output.replace('.pdf', '_{}.png'.format(eft_weight_names[iweight])))
        #print(f"Figure saved in {args.output.replace('.pdf', '_rwgt{}.pdf'.format(iweight))}")
        plt.close()
