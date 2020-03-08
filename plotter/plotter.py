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
  def __init__(self, path, prDic={}, colors={}, bkgList=[], dataName='data', outpath='./temp/', lumi=59.7):
    self.SetPath(path)
    self.SetProcessDic(prDic)
    self.SetBkgProcesses(bkgList)
    self.SetDataName(dataName)
    self.Load()
    self.SetOutpath(outpath)
    self.SetLumi(lumi)
    self.SetColors(colors)
    self.SetRegion()
    self.categories = {}
    self.doLegend = True
    self.doRatio = True
    self.doStack = True
    self.doLogY = False
    self.invertStack = False
    self.plotData = True
    self.fill_opts = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
    self.error_opts = {'label':'Stat. Unc.','hatch':'///','facecolor':'none','edgecolor':(0,0,0,.5),'linewidth': 0}
    self.textParams = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
    self.data_err_opts = {'linestyle':'none', 'marker': '.', 'markersize': 10., 'color':'k', 'elinewidth': 1,}#'emarker': '_'


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

  def SetLumi(self, lumi=59.7, lumiunit='fb$^{-1}$', sqrts='13 TeV'):
    self.lumi = lumi
    self.lumiunit = lumiunit
    self.sqrts = sqrts

  def SetCategories(self, dic):
    self.categories = dic

  def SetCategoty(self, catname, values):
    self.categories[catname] = values

  def SetRegion(self, region='$t\bar{t}$'):
    self.region = region

  def GetHistogram(self, hname, process, categories=None):
    ''' Returns a histogram with all categories contracted '''
    if categories == None: categories = self.categories
    h = self.hists[hname]
    for cat in categories: 
      print('cat : categories --> ', cat, ' : ', categories[cat])
      h = h.integrate(cat, categories[cat])
    if isinstance(process, str) and ',' in process: process = process.split(',')
    if isinstance(process, list): 
      prdic = {}
      for pr in process: prdic[pr] = pr
      h = h.group("process", hist.Cat("process", "process"), prdic)
    elif isinstance(process, str): 
      h = h[process].sum("process")
    return h

  def doData(self, hname):
    ''' Check if data histogram exists '''
    return self.dataName in [str(x) for x in list(self.hists[hname].identifiers(self.processLabel))] and self.plotData

  def SetLegend(self, do=True):
    self.doLegend = do

  def SetRatio(self, do=True):
    self.doRatio = do

  def SetStack(self, do=True):
    self.doStack = do

  def SetInvertStack(self, do=True):
    self.invertStack = do

  def SetLogY(self, do=True):
    self.doLogY = do

  def Stack(self, hname={}, xtit='', ytit=''):
    ''' prName can be a list of histograms or a dictionary 'histoName : xtit' '''
    if isinstance(hname, dict):
      for k in hname: self.Stack(k, hname[k], ytit)
      return
    if isinstance(hname, list):
      for k in hname: self.Stack(k, xtit, ytit)
      return
     
    density = False; binwnorm = None
    plt.rcParams.update(self.textParams)

    if self.doData(hname) and self.doRatio:
      fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
      fig.subplots_adjust(hspace=.07)
    else:
      fig, ax = plt.subplots(1, 1, figsize=(7,7))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    # Colors
    from cycler import cycler
    colors = self.GetColors(self.bkglist)
    if self.invertStack: 
      _n = len(h.identifiers(overlay))-1
      colors = colors[_n::-1]
    ax.set_prop_cycle(cycler(color=colors))

    fill_opts  = self.fill_opts
    error_opts = self.error_opts
    data_err_opts = self.data_err_opts
    if not self.doStack:
      error_opts = None
      fill_opts  = None

    if self.invertStack and type(h._axes[0])==hist.hist_tools.Cat:  h._axes[0]._sorted.reverse() 
    h = self.GetHistogram(hname, self.bkglist)
    hist.plot1d(h, overlay="process", ax=ax, clear=False, stack=self.doStack, density=density, line_opts=None, fill_opts=fill_opts, error_opts=error_opts, binwnorm=binwnorm)

    if self.doData(hname):
      hData = self.GetHistogram(hname, self.dataName)
      hist.plot1d(hData, ax=ax, clear=False, error_opts=data_err_opts, binwnorm=binwnorm)

    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(0, None)
    '''
    if not binwnorm is None:
      ax.set_ylabel(f"<Counts/{binwnorm}>")
      if '[' in ax.get_xlabel():
        units = ax.get_xlabel().split('[')[-1].split(']')[0]
        ax.set_ylabel(f"<Counts / {binwnorm} {units}>")
            
    ax.set_xlabel(None)
    '''

    if self.doLegend:
      leg_anchor=(1., 1.)
      leg_loc='upper left'
      handles, labels = ax.get_legend_handles_labels()
      if self.doData(hname):
        handles = handles[-1:]+handles[:-1]
        labels = ['Data']+labels[:-1]            
      ax.legend(handles, labels)#,bbox_to_anchor=leg_anchor,loc=leg_loc)
    
    if self.doData(hname) and self.doRatio:
      #hbkg = self.hists[hname].group(hist.Cat(self.processLabel,self.processLabel), hist.Cat(self.processLabel, self.processLabel), self.bkgdic)
      hist.plotratio(hData, h.sum("process"), ax=rax, error_opts=data_err_opts, denom_fill_opts={}, guide_opts={}, unc='num')
      rax.set_ylabel('Ratio')
      rax.set_ylim(0.5, 1.5)#ratioRange[0],ratioRange[1])

    if self.doLogY:
      ax.set_yscale("log")
      ax.set_ylim(1,ax.get_ylim()[1]*5)        

    #if not xRange is None: ax.set_xlim(xRange[0],xRange[1])
    #if not yRange is None: ax.set_ylim(yRange[0],yRange[1])

    CMS = plt.text(0., 1., r"$\bf{CMS}$ Preliminary", fontsize=16, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    #if not extraText is None:
    # extraLabel = plt.text(0.02, .99, extraText, fontsize=16, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # ax.set_ylim(0,ax.get_ylim()[1]*1.1)
    
    #region = plt.text(0., 1., u"☕ %s"%self.region, fontsize=20, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)


    os.system('mkdir -p %s'%self.outpath)
    fig.savefig(os.path.join(self.outpath, hname+'.png'))


  '''
    # Get colors for the stack
    colors = self.GetColors(self.bkglist)
    ax.set_prop_cycle(cycler(color=colors))

    # Data
    dataOpts = {'linestyle':'none', 'marker':'.', 'markersize':10., 'color':'k', 'elinewidth':1}
    if self.dataName in [str(x) for x in list(self.hists[hname].identifiers(self.processLabel))]:
      plot.plot1d(self.hists[hname].sum('cut').sum('channel').sum('Zcat').sum('lepCat')[self.dataName],
        overlay=self.processLabel, ax=ax, clear=False, error_opts=dataOpts)

    # Background
    fillOpt = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
    mcOpt   = {'label':'Stat. Unc.', 'hatch':'///', 'facecolor':'none', 'edgecolor':(0,0,0,.5), 'linewidth': 0}
    for bkg in self.bkgdic:
      fillOpti = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
      fillOpti['color'] = self.colors[bkg]
      #h = self.hists[hname].sum('cut').sum('channel').sum('Zcat').sum('lepCat')[bkg] #.sum(self.processLabel)
      h = self.hists[hname]
      for cat in self.categories: h = h.integrate(cat, self.categories[cat])
      h = h[bkg]
      h.scale(self.lumi*1000)
      y = h.values(overflow='all')
      print(bkg, ' : ', y[list(y.keys())[0]].sum())
    h = self.hists[hname]
    for cat in self.categories: h = h.integrate(cat, self.categories[cat])
    plot.plot1d(h, ax=ax, clear=False, stack=True, fill_opts=fillOpti, overlay=self.processLabel )#, error_opts=mcOpt)
    hbkg = self.hists[hname]
    for cat in self.categories: hbkg = hbkg.integrate(cat, self.categories[cat])
    hbkg = hbkg.group(hist.Cat(self.processLabel,self.processLabel), hist.Cat(self.processLabel, self.processLabel), {'All bkg' : self.bkglist})
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
    if self.doLogY: ax.set_yscale('log')
    ax.set_ylim(.1, None)
    leg = ax.legend()
    region = plt.text(0., 1., u"☕ %s"%self.region, fontsize=20, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    os.system('mkdir -p %s'%self.outpath)
    fig.savefig(os.path.join(self.outpath, hname+'.png'))
    '''
