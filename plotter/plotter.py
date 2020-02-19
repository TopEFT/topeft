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
from cycler import cycler

class plotter:
  def __init__(self, path, prDic={}, colors={}, bkgList=[], dataName='data', outpath='./temp/', lumi=294.6):
    self.SetPath(path)
    self.SetProcessDic(prDic)
    self.SetBkgProcesses(bkgList)
    self.SetDataName(dataName)
    self.Load()
    self.SetOutpath(outpath)
    self.SetLumi(lumi)
    self.SetColors(colors)
    self.SetRegion()

  def SetPath(self, path):
    ''' Set path to sample '''
    self.path = path

  def Load(self, path=''):
    ''' Get a dictionary histoname : histogram '''
    if path != '': self.SetPath(path)
    self.hists = {}
    with gzip.open(self.path) as fin:
      hin = pickle.load(fin)
      for k in hin.keys():
        if k in self.hists: self.hists[k]+=hin[k]
        else:               self.hists[k]=hin[k]
    self.GroupProcesses()

  def SetProcessDic(self, prdic, sampleLabel='sample', processLabel='process'):
    ''' Set a dictionary process : samples '''
    self.prDic = OrderedDict()
    self.sampleLabel = sampleLabel
    self.processLabel = processLabel
    if len(prdic) == 0: return
    var = prdic[list(prdic.keys())[0]]
    if isinstance(var, str):
      for k in prdic:
        self.prDic[k] = (prdic[k].replace(' ', '').split(','))
    else:
      for k in groupDic:
        self.prDic[k] = (prdic[k])

  def GroupProcesses(self, prdic={}):
    ''' Move from grouping in samples to groping in processes '''
    if prdic != {}: self.SetProcessDic(prdic)
    for k in self.hists.keys(): 
      if len(self.hists[k].identifiers('sample')) == 0: continue
      self.hists[k] = self.hists[k].group(hist.Cat(self.sampleLabel, self.sampleLabel), hist.Cat(self.processLabel, self.processLabel), self.prDic)

  def SetBkgProcesses(self, bkglist=[]):
    ''' Set the list of background processes '''
    self.bkglist = bkglist
    if isinstance(self.bkglist, str): 
      self.bkglist = self.bkglist.replace(' ', '').split(',')
    self.bkgdic = OrderedDict()
    for b in self.bkglist: self.bkgdic[b] = b

  def SetDataName(self, dataName='data'):
    ''' Set the name of the data process '''
    self.dataName = dataName

  def SetHistoDic(self, histoDic={}):
    ''' Set dictionary with histoName : x-axis title '''
    self.histoDic = histoDic

  def SetOutpath(self, outpath='./temp/'):
    ''' Set output path '''
    self.outpath = outpath

  def SetColors(self, colors={}):
    ''' Set a dictionary with a color for each process '''
    if isinstance(colors, str):
      colors = colors.replace(' ', '').split(',')
      self.SetColors(colors)
      return
    elif isinstance(colors, list):
      self.colors = {}
      print('colors = ', colors)
      for i in range(len(self.prDic)):
        key = list(self.prDic.keys())[i]
        if i < len(colors): self.colors[key] = colors[i]
        else              : self.colors[key] = '#000000'
      return
    elif isinstance(colors, dict):
      self.colors = colors
      for key in list(self.prDic.keys()):
        if not key in self.colors: self.colors[key] = 1

  def GetColors(self, processes=[]):
    ''' Get a list of colors for each process '''
    col = []
    for k in processes: 
      c = self.colors[k] if k in self.colors else 1
      col.append(c)
    return col

  def SetLumi(self, lumi=296.4, lumiunit='pb$^{-1}$', sqrts='5.02 TeV'):
    self.lumi = lumi
    self.lumiunit = lumiunit
    self.sqrts = sqrts

  def SetRegion(self, region='$t\bar{t}$'):
    self.region = region

  def Stack(self, hname={}, xtit='', ytit=''):
    ''' prName can be a list of histograms or a dictionary 'histoName : xtit' '''
    if isinstance(hname, dict):
      for k in hname:
        self.Stack(k, hname[k], ytit)
      return
    if isinstance(hname, list):
      for k in hname:
        self.Stack(k, xtit, ytit)
      return
     
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # Get colors for the stack
    colors = self.GetColors(self.bkglist)
    #ax.set_prop_cycle(cycler(color=colors))

    # Data
    dataOpts = {'linestyle':'none', 'marker':'.', 'markersize':10., 'color':'k', 'elinewith':1, 'emarker':'_'}
    if self.dataName in [str(x) for x in list(self.hists[hname].identifiers(self.processLabel))]:
      plot.plot1d(self.hists[hname].sum('level')[self.dataName].sum('channel'), 
        overlay=self.processLabel, ax=ax, clear=False, error_opts=dataOpts)

    # Background
    fillOpt = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
    mcOpt   = {'label':'Stat. Unc.', 'hatch':'///', 'facecolor':'none', 'edgecolor':(0,0,0,.5), 'linewidth': 0}
    for bkg in self.bkgdic:
      fillOpti = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
      fillOpti['color'] = self.colors[bkg]
      #fillOpti['label'] = bkg
      print('fillopt', fillOpti)
      h = self.hists[hname].sum('level').sum('channel')[bkg]#.sum(self.processLabel)
      plot.plot1d(h, ax=ax, clear=False, stack=True, fill_opts=fillOpti, overlay=self.processLabel )#, error_opts=mcOpt)
    hbkg = self.hists[hname].sum('level').sum('channel').group(hist.Cat(self.processLabel,self.processLabel), hist.Cat(self.processLabel, self.processLabel), {'All bkg' : self.bkglist})
    #hbkg = hbkg.sum(self.processLabel)
    print('erropt = ', mcOpt)
    plot.plot1d(hbkg, ax=ax, clear=False, overlay=self.processLabel)#, error_opts={'hatch':'///', 'facecolor':'none', 'edgecolor':(0,0,0,.5), 'linewidth': 0}, overlay=self.processLabel)
    #hbkg = self.hists[hname].group(hist.Cat(self.processLabel,self.processLabel), hist.Cat(self.processLabel, self.processLabel), self.bkgdic)
    #plot.plot1d(hbkg.sum('level').sum('channel'),
    #  overlay=self.processLabel, ax=ax, clear=False, stack=True, fill_opts=fillOpt, error_opts=mcOpt)

    # Signal
    #ax._get_lines.prop_cycler = ax._get_patches_for_fill.prop_cycler
    #args = {'linestyle':'--', 'linewidth': 5}
    #plot.plot1d(signal_hists[key].project('jet_selection','baggy').project('region','iszeroL'),
    #            ax=ax, overlay="process", clear=False, stack=False, line_opts=args)

    # Options
    ax.autoscale(axis='x', tight=True)
    ax.set_yscale('log')
    ax.set_ylim(.1, None)
    leg = ax.legend()
    region = plt.text(0., 1., u"☕ %s"%self.region, fontsize=20, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    os.system('mkdir -p %s'%self.outpath)
    fig.savefig(os.path.join(self.outpath, hname+'.png'))
