"""
 HistEFT class - built on top of coffea.hist.Hist
 Deals with EFT coefficients. Most common methods of the parent class are redefined in order to account for the EFT coefficients
 The plotting settings are inherited.
 Uses WCPoint and WCFit to evaluate the weight fuctions (at plotter level).

 Example of initizalization: 
  HistEFT("Events", ['c1', 'c2', 'c3'], hist.Cat("sample", "sample"), hist.Cat("cut", "cut"), hist.Bin("met", "MET (GeV)", 40, 0, 400))

 TODO: check the group and rebin functions... in particular, the part of grouping/rebinning the coefficients
 TODO: add sum of weights for normalization? 
"""


import coffea
import numpy as np
import copy
from topcoffea.modules.WCFit import WCFit
from topcoffea.modules.WCPoint import WCPoint

class HistEFT(coffea.hist.Hist):

  def __init__(self, label, wcnames, *axes, **kwargs):
    """ Initialize """
    if isinstance(wcnames, str) and ',' in wcnames: wcnames = wcnames.replace(' ', '').split(',')
    n = len(wcnames) if isinstance(wcnames, list) else wcnames
    self._wcnames = wcnames
    self._nwc = n
    self._ncoeffs = int(1+2*n+n*(n-1)/2)
    self.CreatePairs()
    self._WCPoint = None

    super().__init__(label, *axes, **kwargs)

    self.EFTcoeffs = {}
    self.EFTerrs   = {}
    self.WCFit     = {}

  def CreatePairs(self):
    """ Create pairs... same as for WCFit class """
    self.idpairs  = []
    self.errpairs = []
    n = self._nwc
    for f in range(n+1):
      for i in range(f+1):
        self.idpairs.append((f,i))
        for j in range(len(self.idpairs)-1):
          self.errpairs.append([i,j])
    self.errpairs = np.array(self.errpairs)

  def GetErrCoeffs(self, coeffs):
    """ Get all the w*w coefficients """
    #return [coeffs[p[0]]*coeffs[p[1]] if (p[1] == p[0]) else 2*(coeffs[p[0]]*coeffs[p[1]]) for p in self.errpairs]
    return np.where(self.errpairs[:,0]==self.errpairs[:,1], coeffs[self.errpairs[:,0]]*coeffs[self.errpairs[:,1]], 2*coeffs[self.errpairs[:,0]]*coeffs[self.errpairs[:,1]])

  def copy(self, content=True):
    """ Copy """
    out = HistEFT(self._label, self._wcnames, *self._axes, dtype=self._dtype)
    if self._sumw2 is not None: out._sumw2 = {}
    if content:
        out._sumw = copy.deepcopy(self._sumw)
        out._sumw2 = copy.deepcopy(self._sumw2)
    out.EFTcoeffs = copy.deepcopy(self.EFTcoeffs)
    out.EFTerrs =  copy.deepcopy(self.EFTerrs)
    return out

  def identity(self):
    return self.copy(content=False)

  def clear(self):
    self._sumw = {}
    self._sumw2 = None
    self.EFTcoeffs = {}
    self.EFTerrs   = {}
    self.WCFit     = {}

  def GetNcoeffs(self):
    """ Number of coefficients """
    return self._ncoeffs

  def GetNcoeffsErr(self):
    """ Number of w*w coefficients """
    return int((self._ncoeffs+1)*(self._ncoeffs)/2)

  def GetSparseKeys(self, **values):
    """ Get tuple from values """
    return tuple(d.index(values[d.name]) for d in self.sparse_axes())

  def fill(self, EFTcoefficients, **values):
    """ Fill histogram, incuding EFT fit coefficients """
    if EFTcoefficients is None or len(EFTcoefficients) == 0:
      super().fill(**values)
      return
    values_orig = values.copy()
    weight = values.pop("weight", None)

    sparse_key = tuple(d.index(values[d.name]) for d in self.sparse_axes())
    if sparse_key not in self.EFTcoeffs:
      self.EFTcoeffs[sparse_key] = []
      self.EFTerrs  [sparse_key] = []
      for i in range(self.GetNcoeffs()   ): self.EFTcoeffs[sparse_key].append(np.zeros(shape=self._dense_shape, dtype=self._dtype))
      for i in range(self.GetNcoeffsErr()): self.EFTerrs  [sparse_key].append(np.zeros(shape=self._dense_shape, dtype=self._dtype))

    errs = []
    iCoeff, iErr = 0,0
    if self.dense_dim() > 0:
      dense_indices = tuple(d.index(values[d.name]) for d in self._axes if isinstance(d, coffea.hist.hist_tools.DenseAxis))
      xy = np.atleast_1d(np.ravel_multi_index(dense_indices, self._dense_shape))
      if len(EFTcoefficients) > 0: 
        #EFTcoefficients = EFTcoefficients.regular()
        errs = [self.GetErrCoeffs(x) for x in EFTcoefficients]
      for coef in np.transpose(EFTcoefficients):
        #coef = coffea.util._ensure_flat(coef)
        self.EFTcoeffs[sparse_key][iCoeff][:] += np.bincount(xy, weights=coef, minlength=np.array(self._dense_shape).prod() ).reshape(self._dense_shape)
        iCoeff += 1
      
      # Calculate errs...
      for err in np.transpose(errs):
        self.EFTerrs[sparse_key][iErr][:] += np.bincount(xy, weights=err, minlength=np.array(self._dense_shape).prod() ).reshape(self._dense_shape)
        iErr+=1
    else:
      for coef in np.transpose(EFTcoefficients):
        self.EFTcoeffs[sparse_key][iCoeff] += np.sum(coef)
      # Calculate errs...
      for err in np.transpose(errs):
        self.EFTerrs[sparse_key][iErr][:] += np.sum(err)
    super().fill(**values_orig)

  #######################################################################################
  def SetWCFit(self, key=None):
    if key==None: 
      for key in list(self._sumw.keys())[-1:]: self.SetWCFit(key)
      return
    self.WCFit[key] = []
    bins = np.transpose(self.EFTcoeffs[key]) #np.array((self.EFTcoeffs[key])[:]).transpose()
    errs = np.array((self.EFTerrs  [key])[:]).transpose()
    ibin = 0
    for fitcoeff, fiterrs in zip(bins, errs):
      self.WCFit[key].append(WCFit(tag='%i'%ibin, names=self._wcnames, coeffs=fitcoeff, errors=fiterrs))
      #self.WCFit[key][-1].Dump()
      ibin+=1

  def add(self, other):
    """ Add another histogram into this one, in-place """
    #super().add(other)
    if not self.compatible(other):
      raise ValueError("Cannot add this histogram with histogram %r of dissimilar dimensions" % other)
    raxes = other.sparse_axes()

    def add_dict(left, right):
      for rkey in right.keys():
        lkey = tuple(self.axis(rax).index(rax[ridx]) for rax, ridx in zip(raxes, rkey))
        if lkey in left:
          left[lkey] += right[rkey]
        else:
          left[lkey] = copy.deepcopy(right[rkey])

    if self._sumw2 is None and other._sumw2 is None: pass
    elif self._sumw2 is None:
      self._init_sumw2()
      add_dict(self._sumw2, other._sumw2)
    elif other._sumw2 is None:
      add_dict(self._sumw2, other._sumw)
    else:
      add_dict(self._sumw2, other._sumw2)
    add_dict(self._sumw, other._sumw)
    add_dict(self.EFTcoeffs, other.EFTcoeffs)
    add_dict(self.EFTerrs, other.EFTerrs)
    return self

  def DumpFits(self, key=''):
   """ Display all the fit parameters for all bins """
   if key == '': 
     for k in self.EFTcoeffs.keys(): self.DumpFits(k)
     return
   for fit in (len(self.WCFit[key])):
     fit.Dump()

  def ScaleFits(self, SF, key=''):
   """ Scale all the fits by some amount """
   if key == '': 
     for k in self.EFTcoeffs.keys(): self.ScaleFits(SF, k)
     return
   for fit in self.WCFit[key]:
     fit.Scale(SF)  
 

  def __getitem__(self, keys):
    """ Extended from parent class """
    if not isinstance(keys, tuple): keys = (keys,)
    if len(keys) > self.dim():  raise IndexError("Too many indices for this histogram")
    elif len(keys) < self.dim():
      if Ellipsis in keys:
        idx = keys.index(Ellipsis)
        slices = (slice(None),) * (self.dim() - len(keys) + 1)
        keys = keys[:idx] + slices + keys[idx + 1:]
      else:
        slices = (slice(None),) * (self.dim() - len(keys))
        keys += slices
    sparse_idx, dense_idx, new_dims = [], [], []

    for s, ax in zip(keys, self._axes):
      if isinstance(ax, coffea.hist.hist_tools.SparseAxis):
        sparse_idx.append(ax._ireduce(s))
        new_dims.append(ax)
      else:
        islice = ax._ireduce(s)
        dense_idx.append(islice)
        new_dims.append(ax.reduced(islice))
    dense_idx = tuple(dense_idx)

    def dense_op(array):
      return np.block(coffea.hist.hist_tools.assemble_blocks(array, dense_idx))

    out = HistEFT(self._label, self._wcnames, *new_dims, dtype=self._dtype)
    if self._sumw2 is not None: out._init_sumw2()
    for sparse_key in self._sumw:
      if not all(k in idx for k, idx in zip(sparse_key, sparse_idx)): continue
      if sparse_key in out._sumw:
        out._sumw[sparse_key] += dense_op(self._sumw[sparse_key])
        if self._sumw2 is not None:
          out._sumw2[sparse_key] += dense_op(self._sumw2[sparse_key])
      else:
        out._sumw[sparse_key] = dense_op(self._sumw[sparse_key]).copy()
        if self._sumw2 is not None:
          out._sumw2[sparse_key] = dense_op(self._sumw2[sparse_key]).copy()
    for sparse_key in self.EFTcoeffs:
      if not all(k in idx for k, idx in zip(sparse_key, sparse_idx)): continue
      if sparse_key in out.EFTcoeffs:
        for i in range(len(out.EFTcoeffs[sparse_key])):
          out.EFTcoeffs[sparse_key][i] += dense_op(self.EFTcoeffs[sparse_key][i])
          out.EFTerrs  [sparse_key][i] += dense_op(self.EFTerrs  [sparse_key][i])
      else: 
        out.EFTcoeffs[sparse_key]=[]; out.EFTerrs[sparse_key]=[]; 
        for i in range(self.GetNcoeffs()   ): out.EFTcoeffs[sparse_key].append(np.zeros(shape=self._dense_shape, dtype=self._dtype))
        for i in range(self.GetNcoeffsErr()): out.EFTerrs  [sparse_key].append(np.zeros(shape=self._dense_shape, dtype=self._dtype))
        for i in range(len(self.EFTcoeffs[sparse_key])):
          out.EFTcoeffs[sparse_key][i] += dense_op(self.EFTcoeffs[sparse_key][i]).copy()
          out.EFTerrs  [sparse_key][i] += dense_op(self.EFTerrs  [sparse_key][i]).copy()
    return out

  def sum(self, *axes, **kwargs):
    """ Integrates out a set of axes, producing a new histogram 
        Project() and integrate() depends on sum() and are heritated """
    overflow = kwargs.pop('overflow', 'none')
    axes = [self.axis(ax) for ax in axes]
    reduced_dims = [ax for ax in self._axes if ax not in axes]
    out = HistEFT(self._label, self._wcnames, *reduced_dims, dtype=self._dtype)
    if self._sumw2 is not None: out._init_sumw2()

    sparse_drop = []
    dense_slice = [slice(None)] * self.dense_dim()
    dense_sum_dim = []
    for axis in axes:
      if isinstance(axis, coffea.hist.hist_tools.DenseAxis):
        idense = self._idense(axis)
        dense_sum_dim.append(idense)
        dense_slice[idense] = overflow_behavior(overflow)
      elif isinstance(axis, coffea.hist.hist_tools.SparseAxis):
        isparse = self._isparse(axis)
        sparse_drop.append(isparse)
    dense_slice = tuple(dense_slice)
    dense_sum_dim = tuple(dense_sum_dim)

    def dense_op(array):
      if len(dense_sum_dim) > 0:
        return np.sum(array[dense_slice], axis=dense_sum_dim)
      return array

    for key in self._sumw.keys():
      new_key = tuple(k for i, k in enumerate(key) if i not in sparse_drop)
      if new_key in out._sumw:
        out._sumw[new_key] += dense_op(self._sumw[key])
        if self._sumw2 is not None:
          out._sumw2[new_key] += dense_op(self._sumw2[key])
      else:
        out._sumw[new_key] = dense_op(self._sumw[key]).copy()
        if self._sumw2 is not None:
          out._sumw2[new_key] = dense_op(self._sumw2[key]).copy()

    for key in self.EFTcoeffs.keys():
      new_key = tuple(k for i, k in enumerate(key) if i not in sparse_drop)
      if new_key in out.EFTcoeffs:
        #out.EFTcoeffs[new_key] += dense_op(self.EFTcoeffs[key])
        #out.EFTerrs  [new_key] += dense_op(self.EFTerrs  [key])
        for i in range(len(self.EFTcoeffs[key])):
          out.EFTcoeffs[new_key][i] += dense_op(self.EFTcoeffs[key][i])
        for i in range(len(self.EFTerrs[key])):
          out.EFTerrs  [new_key][i] += dense_op(self.EFTerrs[key][i])
      else:
        out.EFTcoeffs[new_key] = []
        out.EFTerrs[new_key] = []
        for i in range(len(self.EFTcoeffs[key])):
          out.EFTcoeffs[new_key].append( dense_op(self.EFTcoeffs[key][i]).copy() )
        for i in range(len(self.EFTerrs[key])):
          out.EFTerrs  [new_key].append( dense_op(self.EFTerrs  [key][i]).copy() )
    return out

  def project(self, *axes, **kwargs):
    """ Project histogram onto a subset of its axes
         Same as in parent class """
    overflow = kwargs.pop('overflow', 'none')
    axes = [self.axis(ax) for ax in axes]
    toremove = [ax for ax in self.axes() if ax not in axes]
    return self.sum(*toremove, overflow=overflow)

  def integrate(self, axis_name, int_range=slice(None), overflow='none'):
    """ Integrates current histogram along one dimension
          Same as in parent class """
    axis = self.axis(axis_name)
    full_slice = tuple(slice(None) if ax != axis else int_range for ax in self._axes)
    if isinstance(int_range, coffea.hist.hist_tools.Interval):
      # Handle overflow intervals nicely
      if   int_range.nan()        : overflow = 'justnan'
      elif int_range.lo == -np.inf: overflow = 'under'
      elif int_range.hi ==  np.inf: overflow = 'over'
    return self[full_slice].sum(axis.name, overflow=overflow)  # slice may make new axis, use name

  def remove(self, bins, axis):
    """ Remove bins from a sparse axis
        Same as in parent class """
    axis = self.axis(axis)
    if not isinstance(axis, coffea.hist.hist_tools.SparseAxis):
      raise NotImplementedError("Hist.remove() only supports removing items from a sparse axis.")
    bins = [axis.index(binid) for binid in bins]
    keep = [binid.name for binid in self.identifiers(axis) if binid not in bins]
    full_slice = tuple(slice(None) if ax != axis else keep for ax in self._axes)
    return self[full_slice]

  def group(self, old_axes, new_axis, mapping, overflow='none'): 
    """ Group a set of slices on old axes into a single new axis """
    ### WARNING: check that this function works properly... (TODO) --> Are the EFT coefficients properly grouped?
    if not isinstance(new_axis, coffea.hist.hist_tools.SparseAxis):
      raise TypeError("New axis must be a sparse axis.  Note: Hist.group() signature has changed to group(old_axes, new_axis, ...)!")
    if new_axis in self.axes() and self.axis(new_axis) is new_axis:
      raise RuntimeError("new_axis is already in the list of axes.  Note: Hist.group() signature has changed to group(old_axes, new_axis, ...)!")
    if not isinstance(old_axes, tuple): old_axes = (old_axes,)
    old_axes = [self.axis(ax) for ax in old_axes]
    old_indices = [i for i, ax in enumerate(self._axes) if ax in old_axes]
    new_dims = [new_axis] + [ax for ax in self._axes if ax not in old_axes]
    out = HistEFT(self._label, self._wcnames, *new_dims, dtype=self._dtype)
    if self._sumw2 is not None: out._init_sumw2()
    for new_cat in mapping.keys():
      the_slice = mapping[new_cat]
      if not isinstance(the_slice, tuple): the_slice = (the_slice,)
      if len(the_slice) != len(old_axes):
        raise Exception("Slicing does not match number of axes being rebinned")
      full_slice = [slice(None)] * self.dim()
      for idx, s in zip(old_indices, the_slice): full_slice[idx] = s
      full_slice = tuple(full_slice)
      reduced_hist = self[full_slice].sum(*tuple(ax.name for ax in old_axes), overflow=overflow)  # slice may change old axis binning
      new_idx = new_axis.index(new_cat)
      for key in reduced_hist._sumw:
        new_key = (new_idx,) + key
        out._sumw[new_key] = reduced_hist._sumw[key]
        if self._sumw2 is not None:
          out._sumw2[new_key] = reduced_hist._sumw2[key]

      # Will this piece work??
      out.EFTcoeffs = copy.deepcopy(reduced_hist.EFTcoeffs)
      out.EFTerrs   = copy.deepcopy(reduced_hist.EFTerrs)

    return out

  def rebin(self, old_axis, new_axis):
    """ Rebin a dense axis """
    old_axis = self.axis(old_axis)
    if isinstance(new_axis, numbers.Integral):
        new_axis = Bin(old_axis.name, old_axis.label, old_axis.edges()[::new_axis])
    new_dims = [ax if ax != old_axis else new_axis for ax in self._axes]
    out = HistEFT(self._label, self._wcnames, *new_dims, dtype=self._dtype)
    if self._sumw2 is not None: out._init_sumw2()
    idense = self._idense(old_axis)

    def view_ax(idx):
      fullindex = [slice(None)] * self.dense_dim()
      fullindex[idense] = idx
      return tuple(fullindex)
    binmap = [new_axis.index(i) for i in old_axis.identifiers(overflow='allnan')]

    def dense_op(array):
      anew = np.zeros(out._dense_shape, dtype=out._dtype)
      for iold, inew in enumerate(binmap):
        anew[view_ax(inew)] += array[view_ax(iold)]
      return anew

    for key in self._sumw:
      out._sumw[key] = dense_op(self._sumw[key])
      if self._sumw2 is not None:
        out._sumw2[key] = dense_op(self._sumw2[key])

    ### TODO: check that this is working!
    for key in self.EFTcoeffs.keys():
      if key in out.EFTcoeffs:
        for i in range(len(self.EFTcoeffs[key])):
          out.EFTcoeffs[key][i] += dense_op(self.EFTcoeffs[key][i])
        for i in range(len(self.EFTerrs[key])):
          out.EFTerrs  [key][i] += dense_op(self.EFTerrs[key][i])
      else:
        out.EFTcoeffs[key] = []
        out.EFTerrs[key] = []
        for i in range(len(self.EFTcoeffs[key])):
          out.EFTcoeffs[key].append( dense_op(self.EFTcoeffs[key][i]).copy() )
        for i in range(len(self.EFTerrs[key])):
          out.EFTerrs  [key].append( dense_op(self.EFTerrs  [key][i]).copy() )
    return out

  ###################################################################
  ### Evaluation
  def Eval(self, wcp=None):
    """ Set a WC point and evaluate """
    if isinstance(WCPoint, dict):
      wcp = WCPoint(wcp)
    elif isinstance(wcp, str):
      values = wcp.replace(" ", "").split(',')
      wcp = WCPoint(values, names=self.GetWCnames())
    elif isinstance(wcp, list):
      wcp = WCPoint(wcp, names=self.GetWCnames())

    if wcp is None:
      if self._WCPoint is None: self._WCPoint = WCPoint(names=self._wcnames)
    else: 
      self._WCPoint = wcp
    self.EvalInSelfPoint()

  def EvalInSelfPoint(self):
    """ Evaluate to self._WCPoint """
    if len(self.WCFit.keys()) == 0: self.SetWCFit()
    if not hasattr(self,'_sumw_orig'): 
      self._sumw_orig  = self._sumw.copy()
      self._sumw2_orig = self._sumw2.copy()
    for key in self.WCFit.keys():
      weights = np.array([wc.EvalPoint(     self._WCPoint) for wc in self.WCFit[key]])
      errors  = np.array([wc.EvalPointError(self._WCPoint) for wc in self.WCFit[key]])
      self._sumw [key] = self._sumw_orig [key]*weights
      self._sumw2[key] = self._sumw2_orig[key]*errors

  def SetSMpoint(self):
    """ Set SM WC point and evaluate """
    wc = WCPoint(names=self._wcnames)
    wc.SetSMPoint()
    self.Eval(wc)

  def SetStrength(self, wc, val):
    """ Set a WC strength and evaluate """
    self._WCPoint.SetStrength(wc, val)
    self.EvalInSelfPoint()

  def GetWCnames(self):
    return self._wcnames
