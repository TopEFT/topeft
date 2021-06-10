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
parser.add_argument('--year', '-y', default=2018                            , help = 'Run year to access lumi')
parser.add_argument('--lumiJson', '-l', default='topcoffea/json/lumi.json'     , help = 'Lumi json file')
args = parser.parse_args()
year  = args.year
lumiJson  = args.lumiJson

with open(lumiJson) as jf:
  lumi = json.load(jf)
  lumi = lumi[year]

path = 'histos/plotsTopEFT.pkl.gz'

processDic = {
  'Nonprompt' : 'TTTo2L2Nu,tW_noFullHad, tbarW_noFullHad, WJetsToLNu_MLM, WWTo2L2Nu',
  'DY' : 'DYJetsToLL_M_10to50_MLM, DYJetsToLL_M_50_a',
  'Other': 'WWW,WZG,WWZ,WZZ,ZZZ,tttt,ttWW,ttWZ,ttZH,ttZZ,ttHH,tZq,TTG',
  'WZ' : 'WZTo2L2Q,WZTo3LNu',
  'ZZ' : 'ZZTo2L2Nu,ZZTo2L2Q,ZZTo4L',
  'ttW': 'TTWJetsToLNu',
  'ttZ': 'TTZToLL_M_1to10,TTZToLLNuNu_M_10_a',
  'ttH' : 'ttHnobb,tHq',
  'data' : 'EGamma, SingleMuon, DoubleMuon',
}
bkglist = ['Nonprompt', 'Other', 'DY',  'ttH', 'WZ', 'ZZ', 'ttZ', 'ttW']
allbkg  = ['tt', 'tW', 'WW', 'ttG', 'WW', 'WJets', 'Other', 'DY',  'ttH', 'WZ', 'ZZ', 'ttZ', 'ttW']

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
  'tt' : '#0b23f0',
  'tW' : '#888db5',
  'ttG' : '#5b0003',
  'WW' : '#f00b0b',
  'WJets' : '#00065b',
}

ch3l = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']
ch2lss = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
categories = {
 'channel' : ch3l,#['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ'],#'eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS'],
 'cut' : ['2jets', '4jets','4j1b', '4j2b'],#['base', '2jets', '4jets', '4j1b', '4j2b'],
 #'Zcat' : ['onZ', 'offZ'],
 #'lepCat' : ['3l'],
}

colors = [colordic[k] for k in bkglist]


def Draw(var, categories, label=''):
  plt = plotter(path, prDic=processDic, bkgList=bkglist, lumi=lumi)
  plt.plotData = True
  plt.SetColors(colors)
  plt.SetCategories(categories)
  plt.SetRegion(label)
  plt.Stack(var, xtit='', ytit='')
  plt.PrintYields('counts')

Draw('met', categories, '3 leptons')
