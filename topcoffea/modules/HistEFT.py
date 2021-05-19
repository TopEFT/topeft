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
import awkward as ak
import numbers

from coffea.hist.hist_tools import DenseAxis

from topcoffea.modules.EFTHelper import EFTHelper

class HistEFT(coffea.hist.Hist):

  def __init__(self, label, wcnames, *axes, **kwargs):
    """ Initialize """
    if isinstance(wcnames, str) and ',' in wcnames: wcnames = wcnames.replace(' ', '').split(',')
    n = len(wcnames) if isinstance(wcnames, list) else wcnames
    self._wcnames = wcnames
    self._eft_helper = EFTHelper(wcnames)
    self._nwc = n
    self._ncoeffs = self._eft_helper.get_w_coeffs()
    self._nerrcoeffs = self._eft_helper.get_w2_coeffs()
    self._wcs = np.zeros(n)
    
    super().__init__(label, *axes, **kwargs)


  def _init_sumw2(self):
    self._sumw2 = {}
    for key in self._sumw.keys():
      # Check if this is an EFT bin or a regular bin
      if self.dense_dim() > 0:
        is_eft_bin = (self._sumw[key].shape != self._dense_shape)
      else:
        is_eft_bin = isinstance(self._sumw[key],np.ndarray)
      
      if is_eft_bin:
        # EFT bins that already existed prior to calling sumw2 can't
        # be converted into bins with errors
        self._sumw2[key] = None
      else:
        self._sumw2[key] = self._sumw[key].copy()
            
  def set_wilson_coefficients(self,values):
    """Set the WC values used to evaluate the bin contents of this histogram"""
    self._wcs = np.asarray(values).copy()

  def copy(self, content=True):
    """ Copy """
    out = HistEFT(self._label, self._wcnames, *self._axes, dtype=self._dtype)
    if self._sumw2 is not None: out._sumw2 = {}
    out._wcs = copy.deepcopy(self._wcs)
    if content:
        out._sumw = copy.deepcopy(self._sumw)
        out._sumw2 = copy.deepcopy(self._sumw2)
    return out

  def identity(self):
    return self.copy(content=False)

  def clear(self):
    self._sumw = {}
    self._sumw2 = None

  def fill(self, **values):
    """ Fill histogram, incuding EFT fit coefficients """

    # If we're not filling with EFT coefficients, just do the normal coffea.hist.Hist.fill().
    eft_coeff = values.pop("eft_coeff",None)

    # First, let's pull out the weight and handle that.
    weight = values.pop("weight", None)
    if isinstance(weight, (ak.Array, np.ndarray)):
      weight = np.asarray(weight)
    if isinstance(weight, numbers.Number):
      weight = np.atleast_1d(weight)

    # Next up, we need to pull out the error coefficients, if those exist
    eft_err_coeff = values.pop("eft_err_coeff",None)

    # Now all that's left should be axes, so let's double check that
    # there aren't any extra or missing.  Note: We do this after
    # popping the weight and EFT stuff so that we can feel confident
    # that everything left is an axis.
    if not all(d.name in values for d in self._axes):
        missing = ", ".join(d.name for d in self._axes if d.name not in values)
        raise ValueError(
            "Not all axes specified for %r.  Missing: %s" % (self, missing)
        )
    if not all(name in self._axes for name in values):
        extra = ", ".join(name for name in values if name not in self._axes)
        raise ValueError(
            "Unrecognized axes specified for %r.  Extraneous: %s" % (self, extra)
        )

    # Lookup which sparse bin (i.e. the indices of this bin along the non-dense axes.
    sparse_key = tuple(d.index(values[d.name]) for d in self.sparse_axes())

    # Check if we're actually doing all this EFT stuff or not
    if eft_coeff is None:
      # But wait.  Does this sparse bin look like it's got EFT coefficients stored in it?
      if sparse_key in self._sumw:
        if len(np.atleast_1d(self._sumw[sparse_key]).shape) != len(self._dense_shape):
          raise ValueError("Attempt to fill an EFT bin with non-EFT events.")
      # Put the weights back in!  We're just going to rely on the
      # regular coffea.hist.Hist fill method to handle this.
      values['weight']=weight
      super().fill(**values)
      return

    # OK then, we're doing this with EFT coefficients.    

    # We want this as a numpy array
    eft_coeff = np.asarray(eft_coeff)

    # Check that we have the right number of coefficients
    if self._ncoeffs != eft_coeff.shape[1]:
      raise ValueError(
        "Wrong number of EFT coefficients.  "+
        "Expecting {}, received {}".format(self._ncoeffs, eft_coeff.shape[1])
      )
    if eft_err_coeff is not None:
      if self._nerrcoeffs != eft_err_coeff.shape[1]:
        raise ValueError(
          "Wrong number of EFT w*w coefficients.  "+
          "Expecting {}, received {}".format(self._nerrcoeffs, eft_err_coeff.shape[1])
        )
      
    # Next, if there are weights, we should multiply the EFT coefficients by those weights
    if weight is not None:
      eft_coeff *= weight[:,np.newaxis]
      # Also, if there are EFT error coefficients, those need to be scaled by weight**2
      if eft_err_coeff is not None:
        eft_err_coeff *= (weight[:,np.newaxis]**2)

    # At this point, we're ready to accumulate these coefficients with
    # any of our previous ones.  Note: we're going to use the same
    # "dense bins" structure as a regular histogram, but just extend
    # it by one more index to track the necessary coefficients.

    # If this bin has never been filled before, initialize the numpy
    # arrays to hold the coefficient sums Note, if there are no dense
    # axes, then this just becomes 1D numpy array to store the
    # coefficients for this sparse bin (see below when we go to fill).
    if sparse_key not in self._sumw:
      self._sumw[sparse_key] = np.zeros(
        shape=(*self._dense_shape,self._ncoeffs), dtype=self._dtype
      )
      if eft_err_coeff is not None:
        if self._sumw2 is None:
          self._init_sumw2()
        self._sumw2[sparse_key] = np.zeros(
          shape=(*self._dense_shape,self._nerrcoeffs), dtype=self._dtype
        )
      else:
        if self._sumw2 is not None:
          self._sumw2[sparse_key] = None

    # This get a little weird now.  We want to use np.bincount to sum
    # up the coefficients in our bins, but this only works for a 1D
    # array.  So, we're going to construct flat arrays of which
    # coefficients need to be incremented and then use np.bincount to
    # do the actual summing.
    if self.dense_dim() > 0: 
      # repeat the dense indices ncoeffs times to account for the fact
      # that we're filling that many numbers
      dense_indices = tuple(
        np.repeat(d.index(values[d.name]),self._ncoeffs) for d in self._axes if isinstance(d, DenseAxis)
      )
      # This will make sure that all the coefficients in the flattened
      # array corresponding to the appropriate dense bin get filled
      xy = np.atleast_1d(
        np.ravel_multi_index(
          (*dense_indices,np.tile(np.arange(self._ncoeffs),eft_coeff.shape[0])),
          (*self._dense_shape,self._ncoeffs))
      )
      # This little bit of magic sums the coefficients into the right
      # bins.  It's just like filling a regular histogram bin, except
      # that we're doing it _ncoeffs times.
      self._sumw[sparse_key] += np.bincount(
        xy, weights=eft_coeff.flatten(),
        minlength=np.array((*self._dense_shape,self._ncoeffs)).prod()
      ).reshape((*self._dense_shape,self._ncoeffs))
      # Ah, but what about those darned w**2 coefficients?
      if eft_err_coeff is not None:
        # We're doing the same thing as for the EFT weight coefficients, but with the err array
        dense_indices = tuple(
          np.repeat(d.index(values[d.name]),self._nerrcoeffs) for d in self._axes if isinstance(d, DenseAxis)
        )
        xy = np.atleast_1d(
          np.ravel_multi_index(
            (*dense_indices,np.tile(np.arange(self._nerrcoeffs),eft_err_coeff.shape[0])),
          (*self._dense_shape,self._nerrcoeffs))
        )
        self._sumw2[sparse_key] += np.bincount(
          xy, weights=eft_err_coeff.flatten(),
          minlength=np.array((*self._dense_shape,self._nerrcoeffs)).prod()
        ).reshape((*self._dense_shape,self._nerrcoeffs))
    else:
      # If we end up here, then the sparse bin has no dense bins.
      # That means all our eft_coeff and eft_err_coeffs just need to
      # be summed and stored in this bin directly
      self._sumw[sparse_key] += np.sum(eft_coeff,axis=0)
      if eft_err_coeff is not None:
        self._sumw2[sparse_key] += np.sum(eft_err_coeff,axis=0)

  def add(self, other):
    """ Add another histogram into this one, in-place """

    if not self.compatible(other):
      raise ValueError("Cannot add this histogram with histogram %r of dissimilar dimensions" % other)
    raxes = other.sparse_axes()

    def add_dict(left, right):
      for rkey in right.keys():
        lkey = tuple(self.axis(rax).index(rax[ridx]) for rax, ridx in zip(raxes, rkey))
        if lkey in left and left[lkey] is not None:
          # Checking to make sure we don't accidentally try to sum a regular and EFT bin
          if self.dense_dim() > 0:
            if left[lkey].shape != right[rkey].shape:
              raise ValueError("Attempt to add histogram bins with EFT weights to ones without.")
          else:
            if isinstance(left[lkey],np.ndarray) != isinstance(right[rkey],np.ndarray):
              raise ValueError("Attempt to add histogram bins with EFT weights to ones without.")
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
    if len(self.EFTcoeffs) > 0: self.SetWCFit()
    return self

  def DumpFits(self, key=''):
   """ Display all the fit parameters for all bins """
   if key == '': 
     for k in self.EFTcoeffs.keys(): self.DumpFits(k)
     return
   for fit in self.WCFit[key]:
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
    out._wcs = copy.deepcopy(self._wcs)
    if self._sumw2 is not None: out._init_sumw2()
    for sparse_key in self._sumw:
      if not all(k in idx for k, idx in zip(sparse_key, sparse_idx)):
        continue
      if sparse_key in out._sumw:
        out._sumw[sparse_key] += dense_op(self._sumw[sparse_key])
        if self._sumw2 is not None:
          if self._sumw2[sparse_key] is not None:
            if out._sumw2[sparse_key] is not None:
              out._sumw2[sparse_key] += dense_op(self._sumw2[sparse_key])
            else:
              raise ValueError('Cannot combine bins where only some have EFT error weights.')
          else:
            if out_sumw2[sparse_key] is not None:
              raise ValueError('Cannot combine bins where only some have EFT error weights.')
      else:
        out._sumw[sparse_key] = dense_op(self._sumw[sparse_key]).copy()
        if self._sumw2 is not None:
          if self._sumw2[sparse_key] is not None:
            out._sumw2[sparse_key] = dense_op(self._sumw2[sparse_key]).copy()
          else:
            out._sumw2[sparse_key] = None
    return out

  def sum(self, *axes, **kwargs):
    """ Integrates out a set of axes, producing a new histogram 
        Project() and integrate() depends on sum() and are heritated """
    overflow = kwargs.pop('overflow', 'none')
    axes = [self.axis(ax) for ax in axes]
    reduced_dims = [ax for ax in self._axes if ax not in axes]
    out = HistEFT(self._label, self._wcnames, *reduced_dims, dtype=self._dtype)
    out._wcs = copy.deepcopy(self._wcs)
    if self._sumw2 is not None: out._init_sumw2()

    sparse_drop = []
    dense_slice = [slice(None)] * self.dense_dim()
    dense_sum_dim = []
    for axis in axes:
      if isinstance(axis, coffea.hist.hist_tools.DenseAxis):
        idense = self._idense(axis)
        dense_sum_dim.append(idense)
        dense_slice[idense] = coffea.hist.hist_tools.overflow_behavior(overflow)
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
        # Check that we're not trying to combine EFT and non-EFT bins
        if self.dense_dim() > 0:
          if out._sumw[new_key].shape != self._sumw[key].shape:
            raise ValueError("Attempt to sum bins with EFT weights to ones without.")
        else:
          if isinstance(out._sumw[new_key],np.ndarray) != isinstance(self._sumw[key],np.ndarray):
            raise ValueError("Attempt to sum bins with EFT weights to ones without.")
        out._sumw[new_key] += dense_op(self._sumw[key])
        if self._sumw2 is not None:
          if self._sumw2[key] is not None:
            if out._sumw2[new_key] is not None:
              out._sumw2[new_key] += dense_op(self._sumw2[key])
            else:
              raise ValueError('Cannot combine bins where only some have EFT error weights')
          else:
            if out._sumw2[new_key] is not None:
              raise ValueError('Tried to combine bins with and without EFT error weights')
      else:
        out._sumw[new_key] = dense_op(self._sumw[key]).copy()
        if self._sumw2 is not None:
          if self._sumw2[key] is not None:
            out._sumw2[new_key] = dense_op(self._sumw2[key]).copy()
          else:
            out._sumw2[new_key] = None
              
    return out


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
    out._wcs = copy.deepcopy(self._wcs)
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
          if reduced_hist._sumw2[key] is not None:
            out._sumw2[new_key] = reduced_hist._sumw2[key]
          else:
            out._sumw2[new_key] = None

    return out

  def rebin(self, old_axis, new_axis):
    """ Rebin a dense axis """
    old_axis = self.axis(old_axis)
    if isinstance(new_axis, numbers.Integral):
        new_axis = Bin(old_axis.name, old_axis.label, old_axis.edges()[::new_axis])
    new_dims = [ax if ax != old_axis else new_axis for ax in self._axes]
    out = HistEFT(self._label, self._wcnames, *new_dims, dtype=self._dtype)
    out._wcs = copy.deepcopy(self._wcs)
    if self._sumw2 is not None: out._init_sumw2()
    idense = self._idense(old_axis)

    def view_ax(idx):
      fullindex = [slice(None)] * self.dense_dim()
      fullindex[idense] = idx
      return tuple(fullindex)
    binmap = [new_axis.index(i) for i in old_axis.identifiers(overflow='allnan')]

    def dense_op(array):
      anew = np.zeros(shape=(*out._dense_shape,out._ncoeffs), dtype=out._dtype)
      for iold, inew in enumerate(binmap):
        anew[view_ax(inew)] += array[view_ax(iold)]
      return anew

    for key in self._sumw:
      out._sumw[key] = dense_op(self._sumw[key])
      if self._sumw2 is not None:
        if self._sumw2[key] is not None:
          out._sumw2[key] = dense_op(self._sumw2[key])
        else:
          out._sumw2[key] = None

    return out

  def values(self, sumw2=False, overflow="none"):
    """Extract the sum of weights arrays from this histogram
    Parameters
    ----------
        sumw2 : bool
            If True, frequencies is a tuple of arrays (sum weights, sum squared weights)
        overflow
           See `sum` description for meaning of allowed values
    
    Returns a mapping ``{(sparse identifier, ...): numpy.array(...), ...}``
    where each array has dimension `dense_dim` and shape matching
    the number of bins per axis, plus 0-3 overflow bins depending
    on the ``overflow`` argument.
    """

    def view_dim(arr):
      if self.dense_dim() == 0:
        return arr
      else:
        return arr[
          tuple(coffea.hist.hist_tools.overflow_behavior(overflow) for _ in range(self.dense_dim()))
        ]

    out = {}
    for sparse_key in self._sumw.keys():
      id_key = tuple(ax[k] for ax, k in zip(self.sparse_axes(), sparse_key))

      # Now we have to "pay the piper" an actually calculate bin contents from the bin coefficients
      # Start by figuring out if this is an EFT bin or not
      if self.dense_dim() > 0:
        is_eft_bin = (self._sumw[sparse_key].shape != self._dense_shape)
      else:
        is_eft_bin = isinstance(self._sumw[sparse_key],np.ndarray)

      if is_eft_bin:
        _sumw = self._eft_helper.calc_eft_weights(self._sumw[sparse_key],self._wcs)
      else:
        _sumw = self._sumw[sparse_key]

      if sumw2:
        if self._sumw2 is not None:
            if is_eft_bin:
              if self._sumw2[sparse_key] is not None:
                _sumw2 = self._eft_helper.calc_eft_w2(self._sumw2[sparse_key],self._wcs)  
              else:
                # Set really tiny error bars (e.g. one one-millionth the size of the average bin)
                _sumw2 = np.full_like(_sumw,1e-30*np.mean(_sumw))
            else:
              _sumw2 = self._sumw2[sparse_key]
        else:
          if is_eft_bin:
            # Set really tiny error bars (e.g. one one-millionth the size of the average bin)
            _sumw2 = np.full_like(_sumw,1e-30*np.mean(_sumw))
          else:
            _sumw2 = _sumw
        w2 = view_dim(_sumw2)
        out[id_key] = (view_dim(_sumw), w2)
      else:
        out[id_key] = view_dim(_sumw)

    return out

  def scale(self, factor, axis=None):
    """Scale histogram in-place by factor
    Parameters
    ----------
      factor : float or dict
              A number or mapping of identifier to number
      axis : optional
             Which (sparse) axis the dict applies to, may be a tuples of axes.
             The dict keys must follow the same structure.
    Examples
    --------
    This function is useful to quickly reweight according to some
    weight mapping along a sparse axis, such as the ``species`` axis
    in the `Hist` example:
    >>> h.scale({'ducks': 0.3, 'geese': 1.2}, axis='species')
    >>> h.scale({('ducks',): 0.5}, axis=('species',))
    >>> h.scale({('geese', 'honk'): 5.0}, axis=('species', 'vocalization'))
    """
    if self._sumw2 is None:
      self._init_sumw2()
    if isinstance(factor, numbers.Number) and axis is None:
      for key in self._sumw.keys():
        self._sumw[key] *= factor
        if self._sumw2[key] is not None:
          self._sumw2[key] *= factor ** 2
    elif isinstance(factor, dict):
      if not isinstance(axis, tuple):
        axis = (axis,)
        factor = {(k,): v for k, v in factor.items()}
      axis = tuple(map(self.axis, axis))
      isparse = list(map(self._isparse, axis))
      factor = {
        tuple(a.index(e) for a, e in zip(axis, k)): v for k, v in factor.items()
      }
      for key in self._sumw.keys():
        factor_key = tuple(key[i] for i in isparse)
        if factor_key in factor:
          self._sumw[key] *= factor[factor_key]
          if self._sumw2[key] is not None:
            self._sumw2[key] *= factor[factor_key] ** 2
    elif isinstance(factor, numpy.ndarray):
      axis = self.axis(axis)
      raise NotImplementedError("Scale dense dimension by a factor")
    else:
      raise TypeError("Could not interpret scale factor")
