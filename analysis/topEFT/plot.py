#cat_dict = {
#  "2l" : {
#    "lep_chan_lst" : ["2lss_p" , "2lss_m" , "2lssCR"],
#    "lep_flav_lst" : ["ee" , "em" , "mm"],
#    "njets_lst"    : ["exactly_2j" , "exactly_4j" , "exactly_5j" , "exactly_6j" , "atleast_7j"],
#    "appl_lst"     : ['isSR_2l' , 'isAR_2l'],
#  },
#  "3l" : {
#    "lep_chan_lst" : ["3l_p_offZ_1b" , "3l_m_offZ_1b" , "3l_p_offZ_2b" , "3l_m_offZ_2b" , "3l_onZ_1b" , "3l_onZ_2b" , "3lCR"],
#    "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
#    "njets_lst"    : ["atleast_1j" , "exactly_2j" , "exactly_3j" , "exactly_4j" , "atleast_5j"],
#    "appl_lst"     : ['isSR_3l', 'isAR_3l'],
#  },
#  "4l" : {
#     "lep_chan_lst" : ["4l"],
#     "lep_flav_lst" : ["llll"], # Not keeping track of these separately
#     "njets_lst"    : ["exactly_2j" , "exactly_3j" , "atleast_4j"],
#     "appl_lst"     : ['isSR_4l'],
#  }
#}


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

def construct_cat_name(chan_str,njet_str=None,flav_str=None):

    # Get the component strings
    nlep_str = chan_str.split("_")[0] # Assumes n leps comes first in the str
    chan_str = "_".join(chan_str.split("_")[1:]) # The rest of the channel name is everything that comes after nlep
    if chan_str == "": chan_str = None # So that we properly skip this in the for loop below
    if flav_str is not None:
        flav_str = flav_str
    if njet_str is not None:
        njet_str = njet_str[-2:] # Assumes number of n jets comes at the end of the string
        if "j" not in njet_str:
            # The njet string should really have a "j" in it
            raise Exception(f"Something when wrong while trying to consturct channel name, is \"{njet_str}\" an njet string?")

    # Put the component strings into the channel name
    ret_str = nlep_str
    for component in [flav_str,chan_str,njet_str]:
        if component is None: continue
        ret_str = "_".join([ret_str,component])
    return ret_str

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('--year',     '-y', default = '2017'                           , help = 'Run year to access lumi')
parser.add_argument('--lumiJson', '-l', default = '../../topcoffea/json/lumi.json' , help = 'Lumi json file')
#parser.add_argument('--path',     '-p', default = 'histos/plotsTopEFT.pkl.gz'      , help = 'Path to pkl file')
parser.add_argument('--variable', '-v', default = 'counts'                         , help = 'Variable')
parser.add_argument('--channel',  '-c', default = '3l_CR'                          , help = 'Channels')
parser.add_argument('--njets',    '-j', default = 'atleast_1j'                     , help = 'Variable')
parser.add_argument('--flav',     '-f', default = 'all'                            , help = 'Variable')
parser.add_argument('--title',    '-t', default = '3l_CR'                          , help = 'Title of the plot')
parser.add_argument('--output',   '-o', default = None                             , help = 'Name of the output png file')
args = parser.parse_args()

year  = args.year
lumiJson  = args.lumiJson
path  = 'histos/plotsTopEFT.pkl.gz' #['histos/background.pkl.gz', 'histos/signal.pkl.gz', 'histos/data_Run2017B.pkl.gz']#args.path
var = args.variable
ch = args.channel
njets = args.njets
flav = args.flav
title = args.title
output = args.output

# determine SR or AR
appl = []
if   ch == '2lss_CR': appl = ['isSR_2lss', 'isAR_2lss']
elif "2lss" in ch   : appl = 'isSR_2lss'
elif "3l" in ch     : appl = 'isSR_3l' # 3 lepton control region requires 3 tight leptons
elif "4l" in ch     : appl = 'isSR_4l'
elif "CR" in ch     : appl = ['isSR_2lss', 'isAR_2lss']

if ch == '2lss'     : ch   = ['2lss_p', '2lss_m']
if ch == '3l'       : ch   = ["3l_p_offZ_1b", "3l_m_offZ_1b", "3l_p_onZ_1b", "3l_m_onZ_1b"]

# define "all" flavors
flav_lst = []
if flav == "all":
  if "2lss" in ch: flav_lst = ['ee', 'mm', 'em']
  if "3l"   in ch: flav_lst = ['eee', 'mmm', 'eem', 'mme']
  if "4l"   in ch: flav_lst = ['llll']
else             : flav_lst = [flav]

# Convert string to list
if isinstance(ch, str): ch = [ch]

# Construct the hist name
ch_list = []
for lep_chan in ch:
  for lep_flav in flav_lst:
    ch_name = construct_cat_name(lep_chan, njets, lep_flav)
    ch_list += [ch_name]

with open(lumiJson) as jf:
  lumi = json.load(jf)
  lumi = lumi[year]

processDic = {
  'Diboson': 'WZTo2L2Q_centralUL17, WZTo3LNu_centralUL17, ZZTo2L2Nu_centralUL17, ZZTo2L2Q_centralUL17, ZZTo4L_centralUL17, WWTo2L2Nu_centralUL17',
  'Triboson': 'WWW_centralUL17, WZG_centralUL17, WWZ_centralUL17, WZZ_centralUL17, ZZZ_centralUL17',
  'ttH': 'ttHnobb, ttHH',
  'ttll': 'TTZToLL_M_1to10, TTZToLLNuNu_M_10_a, TTG, ttZH, ttZZ',
  'ttlv': 'TTWJetsToLNu, ttWW, ttWZ',
  'tllq': 'tZq',
  'tHq': 'tHq',
  'TT': 'TTTo2L2Nu',
  'DY': 'DYJetsToLL_centralUL17',
  'data' : 'data',
}
bkglist = ['TT', 'DY', 'Diboson', 'Triboson', 'ttlv', 'ttll', 'ttH', 'tllq', 'tHq']
allbkg  = ['Diboson', 'Triboson', 'ttH', 'ttll', 'ttlv', 'tllq', 'tHq']

colordic ={
  'Other' : '#808080',
  'Diboson' : '#ff00ff',
  'Triboson': '#66ff66',
  'ttH' : '#CC0000',
  'ttll': '#00a278',
  'ttlv': '#009900',
  'tllq': '#ff66ff',
  'tHq' : '#00ffff',
  'TT': '#ffff33',
  'DY': '#33ff33',
}

colors = [colordic[k] for k in bkglist]


def Draw(var, categories, label=''):
  plt = plotter(path, prDic=processDic, bkgList=bkglist, lumi=lumi)
  plt.plotData = True
  plt.SetColors(colors)
  plt.SetCategories(categories)
  plt.SetRegion(label)
  plt.SetOutput(output)
  plt.Stack(var, xtit='', ytit='')
  plt.PrintYields('counts')

categories = {
 'channel'    : ch_list,
 'systematic' : 'nominal',
 'appl'       : appl,
}
Draw(var, categories, title)
