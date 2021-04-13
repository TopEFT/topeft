from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import os
import uproot
import matplotlib.pyplot as plt
import numpy as np
from coffea import hist, processor 
from coffea.hist import plot
import os, sys

from topcoffea.plotter.plotter import plotter

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('--filepath1','-i1'   , default='histos/plotsTopEFT.pkl.gz', help = 'path of first file with histograms')
parser.add_argument('--filepath2','-i2'   , default='histos/second_plotsTopEFT.pkl.gz', help = 'path of second file with histograms')
parser.add_argument('--outpath','-p'   , default='../www/', help = 'Name of the output directory')
args = parser.parse_args()

path = args.filepath1
path2 = args.filepath2

with gzip.open(path) as fin:
  fin2 = gzip.open(path2)
  hin = pickle.load(fin)
  hin2 = pickle.load(fin2)
  hists = ['njets', 'nbtags', 'met', 'm3l', 'e0pt', 'm0pt', 'j0pt', 'e0eta', 'm0eta', 'j0eta', 'ht', 'j1pt', 'j1eta', 'j2pt', 'j2eta', 'j3pt', 'j3eta', 'e1pt', 'e1eta', 'e2pt',
           'e2eta', 'm1pt', 'm1eta', 'm2pt', 'm2eta']
  categories = {'channel': ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ'], 'cut': ['2jets', '4jets','4j1b', '4j2b']}
  prDic = {'NonPrompt': "TTTo2L2Nu"}
  for thing in hists:
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    h = hin[thing]
    h2 = hin2[thing]
    for cat in categories: 
      h = h.integrate(cat, categories[cat])
      h2 = h2.integrate(cat, categories[cat])
    hist.plot1d(h, overlay="sample", ax=ax, clear=False)
    hist.plot1d(h2, overlay="sample", ax=ax, clear=False)
    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(0, None)
    ax.set_xlabel(None)
    fig.savefig(os.path.join(args.outpath, thing))

