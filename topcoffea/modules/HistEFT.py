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
from topcoffea.modules.utils import regex_match

import topcoffea.modules.eft_helper as efth

class HistEFT(coffea.hist.Hist):

  def __init__(self, label, wcnames, *axes, **kwargs):
    """ Initialize """
    if isinstance(wcnames, str) and ',' in wcnames: wcnames = wcnames.replace(' ', '').split(',')
    n = len(wcnames) if isinstance(wcnames, list) else wcnames
    self._wcnames = wcnames
    self._nwc = n
    self._ncoeffs = efth.n_quad_terms(n)
    self._nerrcoeffs = efth.n_quartic_terms(n)
    self._wcs = np.zeros(n)
    self.forceSMsumW2=False    
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
            
  def set_wilson_coeff_from_array(self, values):
    """Set the WC values used to evaluate the bin contents of this histogram to the contents of 
       an array where the elements are ordered according to the order defined by wcnames.
    """
    self._wcs = np.asarray(values).copy()

  def set_sm(self):
    """Conveniece method to set the WC values to SM (all zero)"""
    self._wcs = np.zeros(self._nwc)
    
  def set_wilson_coefficients(self, **values):
    """Set the WC values used to evaluate the bin contents of this histogram
       where the WCs are specified as keyword arguments.  Any WCs not listed are set to zero.
    """
    self.set_sm()
    for key in values:
      try:
        index = self._wcnames.index(key)
        self._wcs[index] = values[key]
      except ValueError:
        msg = 'This HistEFT does not know about the "{}" Wilson coefficient.  '.format(key)
        if self._nwc == 0:
          msg += 'There are no Wilson coefficients defined for this HistEFT.'
        else:
          wc_string = ', '.join(self._wcnames)
          part1, _, part2 = wc_string.rpartition(', ')
          msg += ('Defined Wilson coefficients: '+part1 + ', and ' + part2+'.')
        raise LookupError(msg)

  def split_by_terms(self,axis_bins,axis_name='sample'):
    """ Split the EFT contributions by unique term from the quadratic parameterization
    Parameters
    ----------
        axis_bins : list
            A list of axis bins that should correspond to bins with an EFT parmeterization. These
            bins will be summed together and the resulting parameterization split term-by-term to
            fill a set of new bins in the same axis. Typically, this should correspond to one or
            more of the private EFT MC samples. The axis bin strings can be regular expressions
            which will be used to find any matches in the axis.
        axis_name : str
            The name of the sparse axis that is to be regrouped. This should almsot always
            correspond to whichever sparse axis defines the different MC process samples.

    Returns:
        A new HistEFT with the matched axis bins summed over and the resulting EFT parameterization
        split up term-by-term, with each term appearing as a new bin in the specified axis. The
        original axis bins are removed from the axis before returning the histogram

    TODO: We could probably preserve the EFT quadratic information by filling the new histogram with
          the quadratic coefficient that corresponds to the specific bin being filled.
    """
    if self.dense_dim() > 1:
      raise RuntimeError("Splitting by terms not implemented for histograms with more than 1 dense axis")
    if not axis_name in self._axes:
      raise KeyError(f"No axis {axis_name} found in {self}")
    dense_ax = self.dense_axes()[0]

    # Combine together bins that we want to have included in the EFT contributions
    old_ax = self.axis(axis_name)
    new_ax = coffea.hist.Cat(old_ax.name,old_ax.label)

    GROUP_NAME = 'signals'

    ident_names = [x.name for x in self.identifiers(axis_name)]
    to_group = {GROUP_NAME: regex_match(ident_names,regex_lst=axis_bins)}
    for ident in old_ax.identifiers():  # Should this be 'old_ax' or self.identifiers()?
      n = ident.name
      if not n in to_group[GROUP_NAME]:
        to_group[n] = [n]
    new_h = self.group(old_ax,new_ax,to_group)

    # Now begin the actual splitting
    wcs = np.hstack(("sm",new_h._wcnames))
    wc_vals = np.hstack((np.ones(1),new_h._wcs))

    # First we evaluate the wc0*wc1 part of the quadratic, since this will be the same for every
    #   bin in the histogram. Each element of 'wc_terms' will correspond to a different term of
    #   the quadratic, so all that is left is to get the structure constants and multiply
    #   element-by-element to get the expected yield
    n_wcs = len(wcs)
    n_terms = int(n_wcs*(n_wcs+1)/2)
    iarr = np.zeros(2)
    wc_terms = np.zeros(n_terms)
    for idx in range(n_terms):
      efth.quadratic_term_to_factors(idx,iarr)
      i,j = [int(x) for x in iarr]
      val = wc_vals[i]*wc_vals[j]
      wc_terms[idx] = val

    # This relies heavily on the fact that the ordering of the sparse axes matches the ordering
    #   in the sparse_key tuple that you get from self._sumw
    sparse_axes = [x.name for x in new_h.sparse_axes()]
    sparse_keys = [x for x in new_h._sumw.keys()]
    for sparse_key in sparse_keys:
      v = new_h._sumw[sparse_key]
      if new_h.dense_dim() > 0:
        is_eft_bin = (v.shape != new_h._dense_shape)
      else:
        is_eft_bin = isinstance(v,np.ndarray)

      if is_eft_bin:
        bins = v[1:-2,...]  # Chop off the '*-flow' bins
        term_vals = bins*wc_terms
        for idx,wgts in enumerate(term_vals.T):
          efth.quadratic_term_to_factors(idx,iarr)    # Get the 'name' of this term
          i,j = [int(x) for x in iarr]
          n1 = wcs[i]
          n2 = wcs[j]
          fill_info = {}
          # Just to reiterate, this for loop relies on the fact that the ordering of the sparse axes
          #   matches the ordering of the sparse key tuple that you get from self._sumw
          for sp_axis_name,sp_axis_bin in zip(sparse_axes,sparse_key):
            fill_info[sp_axis_name] = sp_axis_bin
          fill_info[dense_ax.name] = dense_ax.centers()
          fill_info['weight'] = wgts
          fill_info[axis_name] = f"{n1}.{n2}"  # This overwrites the category (which should've been GROUP_NAME)
          new_h.fill(**fill_info)
    new_h = new_h.remove([GROUP_NAME],axis_name)
    return new_h

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
      # Wait! If weight is not None, but some axes have EFT weights,
      # the base class _init_sumw2() will do the wrong thing, so call
      # ours here!
      if weight is not None and self._sumw2 is None:
        self._init_sumw2()
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
      eft_coeff = eft_coeff*weight[:,np.newaxis]
      # Also, if there are EFT error coefficients, those need to be scaled by weight**2
      if eft_err_coeff is not None:
        eft_err_coeff = eft_err_coeff*(weight[:,np.newaxis]**2)

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

    # Adds right to left
    def add_dict(left, right):
      for rkey in right.keys():
        lkey = tuple(self.axis(rax).index(rax[ridx]) for rax, ridx in zip(raxes, rkey))
        # If the lkey is not already in left, just take the value from right
        # Note: We do not have to check if rkey is in right, since we're looping over right.keys()
        if lkey not in left:
          left[lkey] = copy.deepcopy(right[rkey])
        # Check that neither value is none
        elif (left[lkey] is not None) and (right[rkey] is not None):
          # Check if we're trying to sum a regular and EFT bin
          if ((self.dense_dim() > 0) and (left[lkey].shape != right[rkey].shape)):
            if left[lkey].shape[0] == right[rkey].shape[0]:
              # Add the non-EFT bin contents to the 0th element (SM element) of the EFT bin
              # But first we have to know which hist is the EFT one
              if len(left[lkey].shape) == 2:
                # The left hist is the one with eft weights
                left[lkey][:,0] = left[lkey][:,0] + right[rkey] 
              elif len(right[rkey].shape) == 2:
                # The right hist is the one with eft weights
                # So we want left to be equal to left plus right (where left is just added to the SM part of right), without modifying right
                tmp = left[lkey]
                left[lkey] = copy.deepcopy(right[rkey])
                left[lkey][:,0] = left[lkey][:,0] + tmp
              else:
                raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
            else:
              raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
          elif ((self.dense_dim() < 1) and (isinstance(left[lkey],np.ndarray) != isinstance(right[rkey],np.ndarray))):
            raise ValueError("Attempt to add histogram bins with EFT weights to ones without.")
          else:
            left[lkey] += right[rkey]
        # If either is None, we want the value of the one that is not none (at least we take into account the uncertainties we can).
        # if both are None, its None
        if left[lkey] is None:
          left[lkey] = right[rkey]
        if right[rkey] is None:
          pass
        

    # Add the sumw2 values
    if self._sumw2 is None and other._sumw2 is None: 
      pass
    elif self._sumw2 is None:
      self._init_sumw2()
      add_dict(self._sumw2, other._sumw2)
    elif other._sumw2 is None:
      
      # This is a tricky case because we want to put in "sumw" for
      # "sumw2" for any non-EFT sparse bins, but None for any EFT
      # sparse bins
      temp = {}
      for key in other._sumw:
        # If it's an EFT bin, there will be a bigger shape
        if other._sumw[key].shape != self._dense_shape:
          temp[key] = None
        else:
          # Regular bins can just copy over sumw for sumw2
          temp[key] = other._sumw[key]
      add_dict(self._sumw2, temp)
    else:
      add_dict(self._sumw2, other._sumw2)

    # Add the sumw values
    add_dict(self._sumw, other._sumw)

    return self 

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

        # Handle the sumw2 values
        # Note: This is copied directly from the implementation in sum(), with key->sparse_key, new_key->sparse_key
        if self._sumw2 is not None:
          # If neither sumw2 value is None, add them
          # First check if we're trying to add regular errors to eft errors
          # Note: This is really the same as in sumw, so would be better to have a function instead of copy paste
          if (self._sumw2[sparse_key] is not None) and (out._sumw2[sparse_key] is not None):
            if (out._sumw2[sparse_key].shape != self._sumw2[sparse_key].shape):
              if out._sumw2[sparse_key].shape[0] == self._sumw2[sparse_key].shape[0]:
                # Add the non-EFT bin contents to the 0th element (SM element) of the EFT bin
                # But first we have to know which hist is the EFT one
                if len(out._sumw2[sparse_key].shape) == 2:
                  # The out hist is the one with eft weights
                  out._sumw2[sparse_key][:,0] = out._sumw2[sparse_key][:,0] + self._sumw2[sparse_key]
                elif len(self._sumw2[sparse_key].shape) == 2:
                  # The original hist self is the one with eft weights
                  # So we want out to be equal to self plus out (where out is just added to the SM part of self), without modifying self
                  tmp2 = out._sumw2[sparse_key]
                  out._sumw2[sparse_key] = copy.deepcopy(self._sumw2[sparse_key])
                  out._sumw2[sparse_key][:,0] = out._sumw2[sparse_key][:,0] + tmp2
                else:
                  raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
              else:
                raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
            else:
              out._sumw2[sparse_key] += dense_op(self._sumw2[sparse_key])
          # If either sumw2 value is None, we will set the out sumw2 to None
          else:
            out._sumw2[sparse_key] = None

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

        # Handle the sumw values
        # Check if we're trying to combine EFT and non-EFT bins
        if ((self.dense_dim() > 0) and (out._sumw[new_key].shape != self._sumw[key].shape)):
          if out._sumw[new_key].shape[0] == self._sumw[key].shape[0]:
            # Add the non-EFT bin contents to the 0th element (SM element) of the EFT bin
            # But first we have to know which hist is the EFT one
            if len(out._sumw[new_key].shape) == 2:
              # The out hist is the one with eft weights
              out._sumw[new_key][:,0] = out._sumw[new_key][:,0] + self._sumw[key]
            elif len(self._sumw[key].shape) == 2:
              # The original hist self is the one with eft weights
              # So we want out to be equal to self plus out (where out is just added to the SM part of self), without modifying self
              tmp = out._sumw[new_key]
              out._sumw[new_key] = copy.deepcopy(self._sumw[key])
              out._sumw[new_key][:,0] = out._sumw[new_key][:,0] + tmp
            else:
              raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
          else:
            raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
        elif ((self.dense_dim() == 0) and (isinstance(out._sumw[new_key],np.ndarray) != isinstance(self._sumw[key],np.ndarray))):
          raise ValueError("Attempt to sum bins with EFT weights to ones without.")
        else:
          out._sumw[new_key] += dense_op(self._sumw[key])

        # Handle the sumw2 values
        if self._sumw2 is not None:
          # If neither sumw2 value is None, add them
          # First check if we're trying to add regular errors to eft errors
          # Note: This is really the same as in sumw, so would be better to have a function instead of copy paste
          if (self._sumw2[key] is not None) and (out._sumw2[new_key] is not None):
            if (out._sumw2[new_key].shape != self._sumw2[key].shape):
              if out._sumw2[new_key].shape[0] == self._sumw2[key].shape[0]:
                # Add the non-EFT bin contents to the 0th element (SM element) of the EFT bin
                # But first we have to know which hist is the EFT one
                if len(out._sumw2[new_key].shape) == 2:
                  # The out hist is the one with eft weights
                  out._sumw2[new_key][:,0] = out._sumw2[new_key][:,0] + self._sumw2[key]
                elif len(self._sumw2[key].shape) == 2:
                  # The original hist self is the one with eft weights
                  # So we want out to be equal to self plus out (where out is just added to the SM part of self), without modifying self
                  tmp2 = out._sumw2[new_key]
                  out._sumw2[new_key] = copy.deepcopy(self._sumw2[key])
                  out._sumw2[new_key][:,0] = out._sumw2[new_key][:,0] + tmp2
                else:
                  raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
              else:
                raise ValueError("Cannot sum these histograms, the values are not an expected shape.")
            else:
              out._sumw2[new_key] += dense_op(self._sumw2[key])
          # If either sumw2 value is None, we will set the out sumw2 to None
          else:
            out._sumw2[new_key] = None

      # The new_key in not in out._sumw yet, so just take the val from self
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
        new_axis = coffea.hist.Bin(old_axis.name, old_axis.label, old_axis.edges()[::new_axis])
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
      if array.shape != self._dense_shape:
        newshape = (*out._dense_shape,out._ncoeffs)
      else:
        newshape = out._dense_shape

      anew = np.zeros(shape=newshape, dtype=out._dtype)
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

  def values(self, sumw2=False, overflow="none", debug=False):
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
        _sumw = efth.calc_eft_weights(self._sumw[sparse_key],self._wcs)
      else:
        _sumw = self._sumw[sparse_key]

      if sumw2:
        if self._sumw2 is not None:
            if is_eft_bin:
              if self._sumw2[sparse_key] is not None:
                if self.forceSMsumW2: _sumw2 = self._sumw2[sparse_key]
                else:                 _sumw2 = efth.calc_eft_w2(self._sumw2[sparse_key],self._wcs)  
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
