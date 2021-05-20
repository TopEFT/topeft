"""
 WCPoint class
 This is basically a python copy of the c++ WCPoint class
   (see: https://github.com/TopEFT/EFTGenReader)
"""

from math import sqrt
import numpy as np

class WCPoint:

  def ParseRwgtId(self, _str):
    """ Parse weight string (from miniAOD format) to weight IDs and weight values """
    if _str.startswith('EFTrwgt'):
      self.idnum = int(_str[7:_str.index('_')])
      _str = _str[_str.index('_')+1:]
    inputs = _str.split('_');
    for i in range(0,len(inputs),2):
      label = inputs[i]
      self.wclist.append(label)
      val = float(inputs[i+1])
      self.inputs[label] = val
    #print('labels = [%i]: '%len(self.inputs.keys()), self.inputs.keys())

  def Scale(self, _val):
    """ Scale the weight """
    self.wgt *= _val

  def SetStrength(self, wcName, strength):
    """ Set value for a given wc name """
    self.inputs[wcName] = strength

  def SetSMPoint(self):
    """ Sets all WCs to SM value (i.e. 0.) """
    for i in range(len(self.inputs)): 
      self.inputs[i] = 0.0

  def GetStrength(self, wcName):
    """ Get strength for a particular WC """
    return self.inputs[wcName] if wcName in self.inputs else 0.0

  def GetEuclideanDistance(self, pt=None):
    """ Calculates the distance from the origin (SM point) using euclidean metric """
    if pt == None:
      return sqrt(sum([k*k for k in self.inputs.values()]))
    else:
      return sqrt(sum([(k-q)*(k-q) for k,q in zip(self.inputs.values(),pt.inputs.values())]))

  def GetDim(self):
    """ Returns the number of WC whose strength is non-zero """
    return len(list(filter(lambda x : x!=0, self.inputs.values())))

  def HasWC(self, wc_name):
    """ Returns if the point actually has an entry for a particular WC """
    return wc_name in self.inputs

  def IsEqualTo(self, pt):
    """ Compares if two WC points are equal """
    for k in self.inputs.keys():
      if k not in pt.inputs.keys(): return False
      if pt.inputs[k] != self.inputs[k]: return False
    return True
      
  def IsSMPoint(self):
    """ Checks if the point is equal SM (i.e. 0 for all WC) """
    return self.GetDim() != 0

  def Dump(self, _str='', append=False):
    """ Print out """
    ss1 = ('wgt: %g'%(self.wgt)).ljust(12)
    ss2 = ('').ljust(12)
    for k in self.inputs.keys():
      ss1 += (', '+k).ljust(12)
      ss2 += (', %g'%self.inputs[k]).ljust(12)
    print(ss1)
    print(ss2)

  def __init__(self,brname=None, wgt=0., names=None):
    """ Constructor """
    self.inputs = {}
    self.wgt = 0;
    self.tag = '';
    self.idnum = 0
    self.wclist = []
    # It can be a miniAOD branch name...
    if isinstance(brname, str) and brname.startswith('EFT'): 
      self.ParseRwgtId(brname)

    # Or a dictiionary
    elif isinstance(brname, dict):
      self.inputs = brname.copy()

    else:
      values = brname
      if isinstance(brname, str) and ',' in brname:
        values = brname.replace(' ', '').split(',')
      if isinstance(names, str) and ',' in names:
        names = names.replace(' ', '').split(',')
      if brname is None and isinstance(names, list):
        values = [0.]*len(names)

      if not (isinstance(values, list) and isinstance(names, list)):
        print("ERROR -- WCPoint: wrong inputs. Try setting WC and values with lists or a dictionary")
      for n,v in zip(names, values):
        self.inputs[n] = v
    self.wgt = float(wgt)

  def buildMatrix(self, wcs):
    #43.3 µs ± 658 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    tmp = np.zeros_like(wcs, dtype=float)
    tab = {k: v for k,v in zip(wcs,tmp)}
    wc = (self.inputs[k] if k in self.wclist else tab[k] for k in wcs)
    return np.fromiter(wc, dtype=float)

