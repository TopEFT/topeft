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
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)

path = 'histos/plotsTopEFT.pkl.gz'

processDic = {
  'Nonprompt' : 'TTTo2L2Nu,tW_noFullHad, tbarW_noFullHad, TTG,WJetsToLNu_MLM, WWTo2L2Nu',
  'Other': 'WWW,WZG,WWZ,WZZ,ZZG,ZZZ,tttt,ttWW,ttWZ,ttZH,ttZZ,ttHH,ttZZ,tZq',
  'DY' : 'DYJetsToLL_M_10to50_MLM, DYJetsToLL_M_50_a',
  'WZ' : 'WZTo2L2Q,WZTo3LNu',
  'ZZ' : 'ZZTo2L2Nu,ZZTo2L2Q,ZZTo4L',
  'ttW': 'TTWJetsToLNu',
  'ttZ': 'TTZToLL_M_1to10,TTZToLLNuNu_M_10_a',
  #'VVV' : 'WWW,WZG,WWZ,WZZ,ZZG,ZZZ',
  #'tttt' : 'tttt',
  #'ttVV' : 'ttWW,ttWZ,ttZH,ttZZ,ttHH,ttZZ',
  #'tZq' : 'tZq',
  #'tHq' : 'tHq',
  'ttH' : 'ttHnobb,tHq',
  'data' : 'EGamma_2018, SingleMuon_2018',
}
bkglist = ['Nonprompt', 'Other', 'DY',  'ttH', 'WZ', 'ZZ', 'ttZ', 'ttW'] #'VVV', 'ttVV', 'tttt', 'tZq', 'tHq',

colordic ={
  'Other' : '#808080',
  'DY' : '#fbff00',
  'WZ' : '#ffa200',
  'ZZ' : '#8fff00',
  'ttW': '#00a278',
  'ttZ': '#6603ab',
  'VVV' : '#c688b4',
  'tttt' : '#0b23f0',
  'Nonprompt' : '#0b23f0',
  'ttVV' : '#888db5',
  'tHq' : '#5b0003',
  'ttH' : '#f00b0b',
  'tZq' : '#00065b',
}

categories = {
 'channel' : ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS'],
 'Zcat' : ['onZ', 'offZ'],
 'lepCat' : ['2lSS'],
 'cut' : ['base', '2jets', '4jets', '4j1b', '4j2b'],
}

colors = [colordic[k] for k in bkglist]

from plotter.plotter import plotter

plt = plotter(path, prDic=processDic, bkgList=bkglist)
plt.plotData = True
plt.SetCategories(categories)
plt.SetColors(colors)
plt.Stack('invmass', xtit='', ytit='')


