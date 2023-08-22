#! /usr/bin/env python

import hist
import boost_histogram as bh

import awkward as ak
import numpy as np

from itertools import chain, product, repeat

from typing import Mapping, Union, Sequence


class SparseHist(hist.Hist, family=hist):
    """ Histogram specialized for sparse categorical data."""

    def __init__(self, *axes, **kwargs):
        """ Arguments:
            axes: List of categorical and regular/variable axes. Categorical access should come first. At least one regular or variable axis should be specified.
            kwargs: Same as for hist.Hist
        """

        self._init_args = dict(kwargs)

        categorical_axes, dense_axes = self._check_args(axes)

        self._dense_hists: dict[tuple, hist.Hist] = {}

       # we use self to keep track of the bins in the categorical axes.
        super().__init__(*categorical_axes, storage="Double")

        self._categorical_axes = super().axes
        self._dense_axes = hist.axis.NamedAxesTuple(dense_axes)

        self.axes = hist.axis.NamedAxesTuple(chain(super().axes, dense_axes))


    def _check_args(self, axes):
        on_cats = True
        categorical_axes = []
        dense_axes = []

        for axis in axes:
            if isinstance(axis, (hist.axis.StrCategory, hist.axis.IntCategory)):
                if not on_cats:
                    ValueError("All categorical axes should be specified first.")
                categorical_axes.append(axis)
            else:
                on_cats = False
                dense_axes.append(axis)

        if len(dense_axes) < 1:
            raise ValueError("At least one dense axis should be specified.")

        return categorical_axes, dense_axes

    def from_categorical_axes_like(self, *categorical_axes):
        """ Create an empty histogram with the same dense axes as self, but with the categorical axes provided.
        """
        return type(self)(*categorical_axes, *self.dense_axes, **self._init_args)

    def __copy__(self):
        """ Empty histograms with the same bins. """
        return self.from_categorical_axes_like(*self.categorical_axes)

    def __deepcopy__(self, memo):
        if len(self._dense_hists) < 1:
            return self.from_categorical_axes_like(*self.categorical_axes)
        else:
            return self[{}]

    def __str__(self):
        return repr(self)

    def _split_axes(self, axes: dict):
        """ Split axes dictionaries in categorical or dense.
        Axes returned in the order they were created. All axes of the histogram should be specified.
        """
        cats = {axis.name: axes[axis.name] for axis in self.categorical_axes}
        nocats = {axis.name: axes[axis.name] for axis in self._dense_axes}
        return (cats, nocats)

    def _cats_as_dict(self, values):
        return dict(zip(self.categorical_axes.name, values))

    def categories_to_index(self, bins: Union[Sequence, Mapping], collapsed=None, as_dict=False):
        if collapsed is None:
            collapsed = repeat(False)
        t = tuple(axis.index(bin) for axis, bin, mask in zip(self.categorical_axes, bins, collapsed) if not mask)
        if as_dict:
            return self._cats_as_dict(t)
        else:
            return t

    def index_to_categories(self, indices: Sequence, collapsed=None, as_dict=None):
        if collapsed is None:
            collapsed = repeat(False)
        t = tuple(axis[index] for index, axis, mask in zip(indices, self.categorical_axes, collapsed) if not mask)
        if as_dict:
            return self._cats_as_dict(t)
        else:
            return t

    @property
    def categorical_axes(self):
        return self._categorical_axes

    @property
    def dense_axes(self):
        return self._dense_axes

    def categorical_keys(self, as_dict=False):
        for indices in self._dense_hists:
            key = self.index_to_categories(indices)
            if as_dict:
                key = self._cats_as_dict(key)
            yield key

    def _fill_bookkeep(self, *args):
        super().fill(*args)
        sparse_key = self.categories_to_index(args)
        if sparse_key not in self._dense_hists:
            h = hist.Hist(*self._dense_axes, **self._init_args)
            self._dense_hists[sparse_key] = h
        return self._dense_hists[sparse_key]

    def fill(self, weight=None, sample=None, threads=None, **kwargs):
        cats, nocats = self._split_axes(kwargs)

        # fill the bookkeeping first, so that the index of the key exists.
        h = self._fill_bookkeep(*list(cats.values()))
        return h.fill(weight=weight, sample=sample, threads=threads, **nocats)

    def _to_bin(self, cat_name, value, offset=0):
        """ Converts category value into its index slice in a StrCategory or IntCategory axis. """
        if isinstance(value, int):
            # already an index
            if value > -1:
                return value + offset
            else:
                return len(self._bookkeep_hist.axes[cat_name]) + value + offset
        elif isinstance(value, str):
            return self.categorical_axes[cat_name].index(value) + offset
        elif isinstance(value, complex):
            return self.categorical_axes[cat_name].index(int(value.imag)) + offset
        elif isinstance(value, bh.tag.loc):
            return self._to_bin(cat_name, value.value, value.offset)
        elif isinstance(value, slice):
            start = value.start if value.start else 0
            stop = value.stop if value.stop else len(self.axes[cat_name])
            step = value.step if value.step else 1
            return slice(
                self._to_bin(cat_name, start, offset),
                self._to_bin(
                    cat_name, stop, offset + (stop < 0)  # add 1 if stop negative, e.g. [-1] index
                ),
                step,
            )
        elif value == sum:
            return sum
        elif isinstance(value, Sequence):
            return tuple(self._to_bin(cat_name, v, offset) for v in value)
        raise ValueError(f"Invalid index specification: {cat_name}: {value}")

    def _make_index_key(self, key):
        if isinstance(key, Mapping):
            index_key = {
                axis.name: slice(None) for axis in self.axes
            }
            index_key.update(key)
        elif isinstance(key, Sequence):
            if len(key) == len(self.categorical_axes):
                # assume just the name of the categories
                index_key = dict(zip(self.categorical_axes.name, key))
                index_key.update({axis.name: slice(None) for axis in self._dense_axes})
            elif len(key) == len(self.axes):
                # assume all axes specified, including dense axes
                index_key = dict(zip((a.name for a in self.axes), key))
            else:
                raise ValueError("Incorrect dimensions were specified.")
        else:
            raise ValueError("Index should be a mapping or a tuple.")

        for a in self.categorical_axes:
            index_key[a.name] = self._to_bin(a.name, index_key[a.name])
        return index_key

    def _from_hists(self, hists: dict, categorical_axes: list, collapsed: Union[None, Sequence] = None):
        """Construct a sparse hist from a dictionary of dense histograms.
        hists: a dictionary of dense histograms.
        categorical_axes: axes to use for the new histogram.
        collapsed: mask that indicates which sparse keys disappear from the indices of hists.
                  (I.e., the new categorical_axes correspond to False values in collapsed.
        """
        dense_axes = list(hists.values())[0].axes

        new_hist = self.from_categorical_axes_like(*categorical_axes)
        for index_key, dense_hist in hists.items():
            named_key = self.index_to_categories(index_key, collapsed)
            new_hist._fill_bookkeep(*named_key)
            index_key = new_hist.categories_to_index(named_key)
            if index_key not in new_hist._dense_hists:
                new_hist._dense_hists[index_key] = hist.Hist(*dense_axes, **self._init_args)
            new_hist._dense_hists[index_key] += dense_hist
        return new_hist

    def _from_hists_no_dense(self, hists: dict, categorical_axes: list, collapsed: Union[None, Sequence] = None):
        """Construct a hist.Hist from a dictionary of histograms where all the dense axes have collapsed. """
        new_hist = hist.Hist(*categorical_axes, **self._init_args)
        for index_key, weight in hists.items():
            named_key = self.index_to_categories(index_key, collapsed)
            new_hist.fill(*named_key, weight=weight)
        return new_hist

    def _filter_dense(self, index_key):
        def asseq(cat_name, x):
            if isinstance(x, int):
                return range(x, x + 1)
            elif isinstance(x, slice):
                step = x.step if isinstance(x.step, int) else 1
                return range(x.start, x.stop, step)
            elif x == sum:
                return range(len(self.axes[cat_name]))
            return x

        cats, nocats = self._split_axes(index_key)
        filtered = {}
        for sparse_key in product(*(asseq(name, v) for name, v in cats.items())):
            if sparse_key in self._dense_hists:
                filtered[sparse_key] = self._dense_hists[sparse_key][tuple(nocats.values())]
        return filtered

    def __getitem__(self, key):
        index_key = self._make_index_key(key)
        filtered = self._filter_dense(index_key)

        if len(filtered) == 0:
            raise KeyError("No bins found")

        collapsed = [index_key[name] is sum or isinstance(index_key[name], int) for name in self.categorical_axes.name]
        new_cats = [
            type(axis)([], growth=True, name=axis.name, label=axis.label)
            for axis, mask in zip(self.categorical_axes, collapsed) if not mask
        ]

        first = list(filtered.values())[0]
        if not isinstance(first, hist.Hist):
            if len(new_cats) == 0:
                # whole histogram collapsed to singe value
                return first
            else:
                # dense axes have collapsed to a single value
                return self._from_hists_no_dense(filtered, new_cats, collapsed)
        else:
            return self._from_hists(filtered, new_cats, collapsed)

    def _ak_rec_op(self, op_on_dense):
        builder = ak.ArrayBuilder()

        def rec(key, depth):
            axis = list(self.categorical_axes)[-1 * depth]
            for i in range(len(axis)):
                next_key = (*key, i) if key else (i, )
                if depth > 1:
                    with builder.list():
                        rec(next_key, depth - 1)
                else:
                    if next_key in self._dense_hists:
                        builder.append(op_on_dense(self._dense_hists[next_key]))
                    else:
                        builder.append(None)
        rec(None, len(self.categorical_axes.name))
        return builder.snapshot()

    def values(self, flow=False):
        return self._ak_rec_op(lambda h: h.values(flow=flow))

    def counts(self, flow=False):
        return self._ak_rec_op(lambda h: h.counts(flow=flow))

    def _do_op(self, op_on_dense):
        for h in self._dense_hists.values():
            op_on_dense(h)

    def reset(self):
        self._do_op(lambda h: h.reset())

    def view(self, flow=False, as_dict=True):
        if not as_dict:
            key = ", ".join([f"'{name}': ..." for name in self.categorical_axes.name])
            raise ValueError(f"If not a dict, only view of particular dense histograms is currently supported. Use h[{{{key}}}].view(flow=...) instead.")
        return {self.index_to_categories(k): h.view(flow=flow) for k, h in self._dense_hists.items()}

    def integrate(self, name: str, value=None):
        if value is None:
            value = sum
        return self[{name: value}]

    def group(self, axis_name: str, groups: dict[str, list[str]]):
        """ Generate a new SparseHist where bins of axis are merged
        according to the groups mapping.
        """
        old_axis = self.axes[axis_name]
        new_axis = hist.axis.StrCategory(
            groups.keys(), name=axis_name, label=old_axis.label, growth=True
        )

        cat_axes = []
        for axis in self.categorical_axes:
            if axis.name == axis_name:
                cat_axes.append(new_axis)
            else:
                cat_axes.append(axis)

        hnew = self.from_categorical_axes_like(*cat_axes)
        for target, sources in groups.items():
            old_key = self._make_index_key({axis_name: sources})
            filtered = self._filter_dense(old_key)

            for old_index, dense in filtered.items():
                new_key = self.index_to_categories(old_index, as_dict=True)
                new_key[axis_name] = target
                new_index = hnew.categories_to_index(new_key.values())

                hnew._fill_bookkeep(*new_key.values())
                hnew._dense_hists[new_index] += dense
        return hnew

    def remove(self, axis_name, bins):
        """Remove bins from a categorical axis

        Parameters
        ----------
            bins : iterable
                A list of bin identifiers to remove
            axis : str
                Sparse axis name

        Returns a copy of the histogram with specified bins removed.
        """
        if axis_name not in self.categorical_axes.name:
            raise ValueError(f"{axis_name} is not a categorical axis of the histogram.")

        axis = self.axes[axis_name]
        keep = [bin for bin in axis if bin not in bins]
        index = [axis.index(bin) for bin in keep]

        full_slice = tuple(slice(None) if ax != axis else index for ax in self.axes)
        return self[full_slice]

    def prune(self, axis, to_keep):
        """Convenience method to remove all categories except for a selected subset.
        """
        to_remove = [x for x in self.axes[axis] if x not in to_keep]
        return self.remove(axis, to_remove)

    def scale(self, factor: float):
        self *= factor
        return self

    def empty(self):
        for h in self._dense_hists.values():
            if np.any(h.view(flow=True) != 0):
                return False
        return True

    def _ibinary_op(self, other, op: str):
        if not isinstance(other, SparseHist):
            for h in self._dense_hists.values():
                getattr(h, op)(other)
        else:
            if self.categorical_axes.name != other.categorical_axes.name:
                raise ValueError("Category names are different, or in different order, and therefore cannot be merged.")
            for index_oh, oh in other._dense_hists.items():
                cats = other.index_to_categories(index_oh)
                self._fill_bookkeep(*cats)
                index = self.categories_to_index(cats)
                getattr(self._dense_hists[index], op)(oh)
        return self

    def _binary_op(self, other, op: str):
        h = self.copy()
        op = op.replace("__", "__i", 1)
        return h._ibinary_op(other, op)

    def __reduce__(self):
        return (type(self)._read_from_reduce, (list(self.categorical_axes), list(self.dense_axes), self._init_args, self._dense_hists))

    @classmethod
    def _read_from_reduce(cls, cat_axes, dense_axes, init_args, dense_hists):
        hnew = cls(*cat_axes, *dense_axes, **init_args)
        for k, h in dense_hists.items():
            hnew._fill_bookkeep(*hnew.index_to_categories(k))
            hnew._dense_hists[k] = h
        return hnew

    def __iadd__(self, other):
        return self._ibinary_op(other, "__iadd__")

    def __add__(self, other):
        return self._binary_op(other, "__add__")

    def __radd__(self, other):
        return self._binary_op(other, "__add__")

    def __isub(self, other):
        return self._ibinary_op(other, "__isub__")

    def __sub__(self, other):
        return self._binary_op(other, "__sub__")

    def __rsub__(self, other):
        return self._binary_op(other, "__sub__")

    def __imul__(self, other):
        return self._ibinary_op(other, "__imul__")

    def __mul__(self, other):
        return self._binary_op(other, "__mul__")

    def __rmul__(self, other):
        return self._binary_op(other, "__mul__")

    def __idiv__(self, other):
        return self._ibinary_op(other, "__idiv__")

    def __div__(self, other):
        return self._binary_op(other, "__div__")

    def __itruediv__(self, other):
        return self._ibinary_op(other, "__itruediv__")

    def __truediv__(self, other):
        return self._binary_op(other, "__truediv__")

    # compatibility methods for old coffea
    # all of these are deprecated
    def identity(self):
        h = self.copy(deep=False)
        h.reset()
        return h
