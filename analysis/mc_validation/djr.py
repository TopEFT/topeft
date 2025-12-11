#!/usr/bin/env python3
'''
This script produces the DJR plots from nanoGEN files
(assuming the DJR were also saved)
Example Run
python djr.py \
--input /cms/cephfs/data/store/user/byates/tttt/nanoGEN_Run3/2022/tttt_LO_EFT/crab_tttt_nanoGEN_Run3/250715_223705/0000/ \
--output /users/byates2/afs/www/EFT/tttt_Run3/weights/weights.pdf
'''

import uproot
import os
import hist
import awkward as ak
import numpy as np
np.seterr(invalid='ignore')
import matplotlib.pyplot as plt
import mplhep as hep
import warnings
plt.style.use(hep.style.CMS)

if __name__ == '__main__':
    '''
    good example:
    root://cmseos.fnal.gov//store/user/dspitzba/EFT/qcut30.root

    bad example:
    root://cmseos.fnal.gov//store/user/cmsdas/2023/short_exercises/Generators/wjets_2j/w2jets_qcut10.root
    '''

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='root://cmseos.fnal.gov//store/user/dspitzba/EFT/qcut80.root', help="Input file")
    argParser.add_argument('--output', action='store', default='./djr.pdf', help="Output file")
    argParser.add_argument('--nevents', action='store', default=50e3, help="Number of events generated")
    args = argParser.parse_args()

    n_events_in = int(args.nevents)

    djr_axis = hist.axis.Regular(40, -0.5, 3.5, name="djr", label=r"$\Delta JR$")
    parton_axis = hist.axis.Integer(0, 4, name="n", label="Number of partons")
    #parton_axis = hist.axis.Regular(5, 0, 4, name="n", label="Number of partons")
    transition_axis = hist.axis.Integer(0, 6, name="t", label="DJR X->Y")
    djr = hist.Hist(djr_axis, parton_axis, transition_axis)

    files = [f for f in os.listdir(args.input) if '.root' in f and 'COPYING' not in f]
    #n_events_in *= len(files)
    tot = 0
    for fin in files:
        #print(f"Loading input file {args.input}/{fin}")
        fin = uproot.open(args.input+fin)
        events = fin["Events"]
        #ar = events["GenEventInfoProduct_generator__GEN./GenEventInfoProduct_generator__GEN.obj"].arrays()
        #djr_values = ar['GenEventInfoProduct_generator__GEN.obj']['DJRValues_']
        #nMEPartons = ar['GenEventInfoProduct_generator__GEN.obj']['nMEPartons_']
        djr_values10 = ak.Array(events['LHEWeight_DJR10'].array())
        djr_values21 = ak.Array(events['LHEWeight_DJR21'].array())
        djr_values32 = ak.Array(events['LHEWeight_DJR32'].array())
        #djr_values = ak.concatenate([events['LHEWeight_DJR10'].array(), events['LHEWeight_DJR21'].array(), events['LHEWeight_DJR32'].array()])
        #nMEPartons = events['LHEWeight_nMEPartons'].array()
        nMEPartons = events['Generator_nMEPartons'].array()

        djr.fill(
            djr = np.log10(ak.flatten(djr_values10, axis=0)),
            n = ak.values_astype(ak.flatten(ak.ones_like(djr_values10)*nMEPartons, axis=0), np.int32),
            #n = ak.flatten(ak.ones_like(djr_values10)*nMEPartons, axis=0),
            t = ak.flatten(ak.local_index(djr_values10), axis=0),
        )
        djr.fill(
            djr = np.log10(ak.flatten(djr_values21, axis=0)),
            n = ak.values_astype(ak.flatten(ak.ones_like(djr_values10)*nMEPartons, axis=0), np.int32),
            #n = ak.flatten(ak.ones_like(djr_values21)*nMEPartons, axis=0),
            t = ak.flatten(ak.local_index(djr_values21), axis=0),
        )
        djr.fill(
            djr = np.log10(ak.flatten(djr_values32, axis=0)),
            n = ak.values_astype(ak.flatten(ak.ones_like(djr_values10)*nMEPartons, axis=0), np.int32),
            #n = ak.flatten(ak.ones_like(djr_values32)*nMEPartons, axis=0),
            t = ak.flatten(ak.local_index(djr_values32), axis=0),
        )
        n_events = ak.num(djr_values10, axis=0)
    print(f"Efficiency is {n_events/n_events_in}, assuming {n_events_in} where simulated")

    print("Plotting...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, axs = plt.subplots(3,2, figsize=(15,21))

        for i in range(3):
            for j in range(2):
                transition = 2*i+j
                djr[:, :, transition].plot1d(
                    overlay='n',
                    ax=axs[i][j],
                    label= [f'{k} partons' for k in range(4)]
                )
                djr[:, :, transition][{'n':sum}].plot1d(
                    ax=axs[i][j],
                    label = ['total'],
                    color = 'gray',
                )

                axs[i][j].set_xlabel(r'$DJR\ %s \to %s$'%(transition, transition+1))
                axs[i][j].set_yscale('log')
                axs[i][j].set_ylim(0.3,n_events*1000)
                axs[i][j].legend(
                    loc='upper right',
                    bbox_to_anchor=(0.03, 0.88, 0.90, .11),
                    mode="expand",
                    ncol=2,
                )

        fig.savefig(args.output)
        print(f"Figure saved in {args.output}")
