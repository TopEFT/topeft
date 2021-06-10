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

path = 'pods/tt/tt.pkl.gz'

processDic = {
  'VV'       : 'WZTo3LNU,WWTo2L2Nu,ZZTo2L2Nu,ZZTo4L',
  #'Nonprompt': 'TTsemilep, W0JetsToLNu,W1JetsToLNu,W2JetsToLNu,W3JetsToLNu',#'WJetsToLNu,TTsemilep',
  'Nonprompt': 'TTsemilep',#'WJetsToLNu,TTsemilep',
  'tW'       : 'tW_noFullHad,  tbarW_noFullHad',
  'DY'       : 'DYJetsToLL_M_10to50,DYJetsToLL_MLL50',
  'tt'       : 'TT',
  #'data': 'HighEGJet, SingleMuon'
}
bkgist = ['DY', 'VV', 'Nonprompt', 'tW', 'tt']

colors = ['#a6cee3', '#1f78b4','#b2df8a','#33a02c','#fb9a99']

from plotter.plotter import plotter

plt = plotter(path, prDic=processDic, bkgList=bkgist)
plt.SetColors(colors)
plt.Stack('invmass', xtit='', ytit='')
