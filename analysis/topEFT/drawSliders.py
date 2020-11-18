from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import os,sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import uproot
import matplotlib.pyplot as plt
import numpy as np
from coffea import hist, processor
from coffea.hist import plot
from cycler import cycler
from plotter.OutText import OutText
from modules.HistEFT import HistEFT
from modules.WCPoint import WCPoint
from modules.WCFit import WCFit
from matplotlib.widgets import Slider, Button, RadioButtons


path = 'histos/plotsTopEFT.pkl.gz'
hists = {}
with gzip.open(path) as fin:
  hin = pickle.load(fin)
  for k in hin.keys():
    if k in hists: hists[k]+=hin[k]
    else:          hists[k]=hin[k]

ch3l = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']

# Create figure
#fig, ax = plt.subplots(1, 1, figsize=(14,7))
fig, (ax, rax) = plt.subplots(2, 1, figsize=(14,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.9)

# Get histogram
h = hists['met']
h = h.integrate('channel', ch3l)
h = h.integrate('cut', 'base')
h = h.sum('sample')
      

# Get the histogram with the sum of weights, for normalization
hsow = hists['SumOfEFTweights']
hsow = hsow.sum('sample')
hsow.SetSMpoint()
smsow = hsow.values()[()][0]

# The SM point -- this won't be updated
hSM = h.copy()
hSM.SetSMpoint()
hSM.scale(1./smsow)

# Draw at SM point
h.SetSMpoint()
h.scale(1./smsow)
hist.plot1d(h, ax=ax, line_opts={'color':'orange'})

# Create sliders
sliders = {}; saxes = []
ypos_min = 0.1
ypos_max = 0.9
wcnames = h.GetWCnames()
ystep = (ypos_max-ypos_min)/len(wcnames)
for i in range(len(wcnames)): 
  ypos = ypos_min+ystep*i
  saxes.append(plt.axes([0.65, ypos, 0.3, 0.02]))
  sliders[wcnames[i]] = Slider(saxes[-1], wcnames[i], -3, 3, valinit=0)

# This function is called when moving a slider
def updatePlot(amount, name):
  norm = h.values()[()][0]
  h.SetStrength(name, amount)
  hsow.SetStrength(name, amount)
  norm = hsow.values()[()][0]
  h.scale(1./norm)
  hist.plot1d(h, ax=ax, line_opts={'color':'orange'})
  hist.plotratio(h, hSM, ax=rax, clear=True, denom_fill_opts={}, error_opts={'linestyle':'none', 'marker': '.', 'markersize': 10., 'color':'k', 'elinewidth': 1}, guide_opts={}, unc='num')
  rax.set_ylim(0, 3)
  rax.set_ylabel('Ratio to SM')
  fig.canvas.draw_idle()
  return norm
  
# Activate the sliders
for n in wcnames:
  sliders[n].on_changed(lambda x, n=n : updatePlot(x,n))

# Plot!
plt.show()
