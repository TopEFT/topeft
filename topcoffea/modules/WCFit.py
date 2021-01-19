"""
 WCFit class
 This is basically a python copy of the c++ WCFit class
   (see: https://github.com/TopEFT/EFTGenReader)
 Per-event fits of WC points are done with numpy.lstsq
"""

import numpy as np
from numpy import sqrt
from topcoffea.modules.WCPoint import WCPoint

kSMstr = 'sm' # For global use

class WCFit:

  def SetTag(self, tag):
    """ Set tag """
    self.tag = tag

  def Size(self):
    """ The number of pairs in the fit, should be equal to 1 + 2N + N(N-1)/2
        Note: len(pairs) and len(coeffs) should always be in 1-to-1 correspondance! """
    return len(self.pairs)

  def ErrSize(self):
    """ The number of pairs in the error fit
        Note: len(err_pairs) and len(err_coeffs) should always be in 1-to-1 correspondance! """
    return len(self.err_pairs)

  def GetTag(self):
    """ Returns self.tag """
    return self.tag

  def GetNames(self):
    """ A vector of all non-zero WCs in the fit (includes 'sm') """
    return this.names

  def GetPairs(self):
    """ A vector of (ordered) indicies, indicating the WC names of the pairs in the quadratic function """
    return self.pairs

  def GetCoefficients(self):
    """ A vector of the coefficients for each term in the quadratic function """
    return self.coeffs

  def GetErrorPairs(self):
    """ Returns err_pairs """
    return err_pairs

  def GetErrorCoefficients(self):
    """ Returns err_coeffs """
    return self.err_coeffs

  def GetIndexPair(self, n1, n2=''):
    """ Returns a (ordered) pair of indicies corresponding to a particular quadratic term
        Convention note: idx1 <= idx2 always! """
    if len(n1) == 2: n1,n2 = n1 
    idx1, idx2, which = -1, -1, -1
    for i in range(len(self.names)):
      if   which == -1 and n1 == self.names[i]:
        idx1 = i
        which = 1
      elif which == -1 and n2 == self.names[i]:
        idx1 = i
        which = 2
      if idx1 == -1: #We haven't found the first index yet!
        continue;
      if   which == 1 and n2 == self.names[i]:
        idx2 = i
        break
      elif which == 2 and n1 == self.names[i]:
        idx2 = i
        break
    return idx2, idx1

  def GetCoefficient(self, n1, n2=''):
    """ Returns a particular structure constant from the fit function
        Note: This is a very brute force method of finding the corresponding coefficient,
        the overloaded function method should be used whenever possible """
    if isinstance(n1,int): return self.coeffs[n1]
    id1, id2 = self.GetIndexPair(n1, n2)
    if id1 == -1 or id2 == -1: return 0. # We don't have the fit parameter pair, assume 0 (i.e. SM value)
    print('ncoeffs = ', len(self.coeffs))
    print('npairs = ', len(self.pairs))
    print('pairs = ', self.pairs)

    if len(self.pairs) != len(self.coeffs):
      print('[ERROR] WCFit pairs and coeffs vectors dont match! (getCoefficient)')
      return 0.0;

    for i in range(self.Size()):
      if self.pairs[i][0] == id1 and self.pairs[i][1] == id2:
        return self.coeffs[i]

    print("[ERROR] WCFit unable to find WC pair! (getCoefficient)")
    return 0.

  def GetErrorCoefficient(self, idx):
    """ Can only access the error coefficients directly via the err_coeffs vector """
    return self.err_coeffs[idx]

  def GetDim(self):
    """ Returns the dimensionality of the fit (i.e. the number of WCs) """
    return len(self.names) - 1 # Exclude 'sm' term

  def HasCoefficient(self, wc_name):
    """ Checks to see if the fit includes the specified WC """
    return wc_name in self.names

  def EvalPoint(self, pt, val=0.0):
    """ Evaluate the fit at a particular WC phase space point """
    if not isinstance(pt, WCPoint):
      wc_name = pt
      pt = WCPoint()
      pt.SetStrength(wc_name, val)
    v = 0
    for i in range(self.Size()):
      c = self.coeffs[i]
      n1 = self.names[(self.pairs[i])[0]]
      n2 = self.names[(self.pairs[i])[1]]
      x1 = 1 if n1 == kSMstr else pt.GetStrength(n1)
      x2 = 1 if n2 == kSMstr else pt.GetStrength(n2)
      v += x1*x2*c;
    return v;

  def EvalPointError(self, pt, val = 0.0):
    """ Evaluate the error fit at a particular WC phase space point """
    if not isinstance(pt, WCPoint):
      wc_name = pt
      pt = WCPoint()
      pt.SetStrength(wc_name, val)
    i = 0
    v,x1,x2,x3,x4,c = 0., 0., 0., 0., 0., 0.
    n1,n2,n3,n4 = '', '', '', ''
    err_pair,idx_pair = (0.,0.),(0.,0.)
    for i in range(self.ErrSize()):
      c = self.err_coeffs[i]
      err_pair = self.err_pairs[i]
      idx_pair = self.pairs[err_pair[0]]; n1 = self.names[idx_pair[0]]; n2 = self.names[idx_pair[1]]
      idx_pair = self.pairs[err_pair[1]]; n3 = self.names[idx_pair[0]]; n4 = self.names[idx_pair[1]]
      x1 = 1.0 if n1 == kSMstr else pt.GetStrength(n1)
      x2 = 1.0 if n2 == kSMstr else pt.GetStrength(n2)
      x3 = 1.0 if n3 == kSMstr else pt.GetStrength(n3)
      x4 = 1.0 if n4 == kSMstr else pt.GetStrength(n4)
      v += x1*x2*x3*x4*c;
    return sqrt(v);

  def AddFit(self, added_fit):
    """ Add fit """
    if added_fit.Size() == 0: return

    if self.Size() == 0: # We are an empty fit, set all values to those of the added fit
      self.names      = added_fit.GetNames()
      self.pairs      = added_fit.GetPairs()
      self.coeffs     = added_fit.GetCoefficients()
      self.err_pairs  = added_fit.GetErrorPairs()
      self.err_coeffs = added_fit.GetErrorCoefficients()
      if len(self.tag) == 0: self.tag = added_fit.GetTag()
      return;

    if self.Size() != added_fit.Size():
      print("[ERROR] WCFit mismatch in pairs! (addFit), self.Size(): ", self.Size(), ", added_fit.Size(): ", added_fit.Size())
      return
    elif self.ErrSize() != added_fit.ErrSize():
      print("[ERROR] WCFit mismatch in error pairs! (addFit)")
      return

    for i in range(self.ErrSize()):
      if i < self.Size(): self.coeffs[i] += added_fit.GetCoefficient(i)
      # It is *very* important that we keep track of the err fit coeffs separately, since Sum(f^2) != (Sum(f))^2
      self.err_coeffs[i] += added_fit.GetErrorCoefficient(i)

  def Scale(self, val):
    """ Scaling fit """
    for i in range(self.Size()):    self.coeffs[i]     *= val
    for i in range(self.ErrSize()): self.err_coeffs[i] += val*val

  def Clear(self):
    self.names.clear()
    self.pairs.clear()
    self.coeffs.clear()
    self.err_pairs.clear()
    self.err_coeffs.clear()

  def Save(self, fpath, append=False):
    """ Save the fit to a text file """
    if not append: print('Producing fitparams table...')
    ss1 = ''.ljust(10)
    ss2 = self.tag.ljust(10)
    for i in range(self.Size()):
      idx_pair = self.pairs[i]
      n1 = self.names[idx_pair[0]]
      n2 = self.names[idx_pair[1]]
      ss1 += (n1+'*'+n2).ljust(15)
      ss2 += ('%g'%self.coeffs[i]).ljust(15)

    outf = open(fpath, 'a+' if append else 'w')
    outf.write(ss1+'\n'+ss2)
    outf.close()
    self.Dump(append)


  def Dump(self, append=False, max_cols=10, wc_name=''):
    ss1 = ''.ljust(10)
    ss2 = self.tag.ljust(10)
    for i in range(self.Size()):
      if i >= max_cols:
        ss1 += ' ...'; ss2 += ' ...'
        break
      idx_pair = self.pairs[i]
      n1 = self.names[idx_pair[0]]
      n2 = self.names[idx_pair[1]]
      if wc_name == '':
        ss1 += (n1+'*'+n2).ljust(15)
        ss2 += ('%g'%self.coeffs[i]).ljust(15)
      else:
        if (n1 in [wc_name, 'sm']) and (n2 in [wc_name, 'sm']):
          ss1 += (n1+'*'+n2).ljust(15)
          ss2 += ('%g'%self.coeffs[i]).ljust(15)
    if not append: print(ss1)
    print(ss2)

  def Extend(self, newName = ''):
    """ This is how we build up all the vectors which store the fit and err_fit info
        Quadratic Form Convention:
          Dim=0: (0,0)
          Dim=1: (0,0) (1,0) (1,1)
          Dim=2: (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
          Dim=3: (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
          etc.
        Note: For ALL pairs --> p[0] >= p[1]
    """
    if self.HasCoefficient(newName):
      print('[ERROR] Tried to extend WCFit with a name already present! (extend)')
      return;

    self.names.append(newName)
    new_idx1 = len(self.names)-1
    # Extend the pairs and coeffs vectors
    for i in range(new_idx1+1):
      idx_pair1 = (new_idx1, i)
      self.pairs.append(idx_pair1)
      self.coeffs.append(0.0) # Extending makes no assumptions about the fit coefficients
      # Extend the err_pairs and err_coeffs vectors
      new_idx2 = len(self.pairs)-1
      for j in range(new_idx2+1):
        idx_pair2 = (new_idx2,j)
        self.err_pairs.append(idx_pair2)
        self.err_coeffs.append(0.0)

  def FitPoints(self,pts):
    """ Extract a n-Dim quadratic fit from a collection of WC phase space points """
    self.Clear()
    if len(pts) == 0: return

    self.Extend(kSMstr) # The SM term is always first

    # This assumes that all WCPoints have exact same list of WC names
    for kv in pts[0].inputs.keys(): 
      self.Extend(kv)

    nCols = self.Size() # Should be equal to 1 + 2*N + N*(N - 1)/2
    nRows = len(pts)

    A = []; b = []
    for rid in range(nRows):
      A.append([])
      for cid in range(nCols):
        idx_pair = self.pairs[cid]
        n1 = self.names[idx_pair[0]]
        n2 = self.names[idx_pair[1]]
        x1 = 1. if n1 == kSMstr else pts[rid].inputs[n1]
        x2 = 1. if n2 == kSMstr else pts[rid].inputs[n1]
        A[rid].append(x1*x2)
      b.append(pts[rid].wgt)
    A = np.array(A)
    b = np.array(b)

    c_x, _, _, _ = np.linalg.lstsq(A,b, rcond=None) # Solve for the fit parameters
    for i in range(self.ErrSize()):
      if i < self.Size(): self.coeffs[i] = c_x[i]
      idx_pair = self.err_pairs[i]
      self.err_coeffs[i] = c_x[idx_pair[0]]*c_x[idx_pair[1]] if idx_pair[0] == idx_pair[1] else 2*c_x[idx_pair[0]]*c_x[idx_pair[1]]

  def SetNamesAndCoefficients(self, names, coefficients, errors=[]):
    self.Clear()
    if len(names) == 0: return
    self.Extend(kSMstr)

    for name in names: self.Extend(name)
    if self.Size() != len(coefficients):
      print('ERROR : %i coefficients are needed but %i are given'%(self.Size(), len(coefficients)))
      return
 
    for i in range(self.Size()):
      self.coeffs[i] = coefficients[i]

    if len(errors) == self.ErrSize():
      for i in range(self.ErrSize()):
        self.err_coeffs[i] = errors[i]
    else:
      for i in range(self.ErrSize()):
        idx_pair = self.err_pairs[i]
        self.err_coeffs[i] = coefficients[idx_pair[0]]*coefficients[idx_pair[1]] if idx_pair[0] == idx_pair[1] else 2*coefficients[idx_pair[0]]*coefficients[idx_pair[1]]

  def __init__(self, wcpoints=None, tag='', names=None, coeffs=None, errors=None):
    """ Constructor """
    self.names = [] # Includes 'sm'
    self.pairs = [] # pair doublets of the 'names' list
    self.coeffs = [] # fit structure constants
    self.err_pairs = [] # pair doubles, indices of the 'pairs' list
    self.err_coeffs = [] # The error fit structure constants
    self.tag = ''
    self.tag = tag
    if wcpoints != None:
      self.FitPoints(wcpoints)
    if not coeffs is None:
      n = len(names)
      if len(coeffs) != 1+2*n+n*(n-1)/2:
        print('ERROR : found %i names but %i coefficients (and should be 1+2*n+n(n-1) = %i)'%(n, len(coeffs), 1+2*n+n*(n-1)/2))
        return
      self.SetNamesAndCoefficients(names, coeffs, errors)
