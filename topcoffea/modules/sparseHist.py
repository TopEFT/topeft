#! /usr/bin/env python

import hist
import boost_histogram as bh

import awkward as ak

from itertools import chain, product, repeat

from typing import Mapping, Union, Sequence


class SparseHist(hist.Hist, family=hist):
    def __init__(self, *axes, **kwargs):
        self._init_args = dict(kwargs)

        categorical_axes, dense_axes = self._check_args(axes)

        # actual "histogram". One histogram with one dense axis per combination of categorical bins.
        self._dense_hists: dict[tuple, hist.Hist] = {}

        self._cat_names = list(axis.name for axis in categorical_axes)
        self._dense_axes = list(dense_axes)

        # we use self to keep track of the bins in the categorical axes.
        super().__init__(*categorical_axes, storage="Double")
        self.axes = hist.axis.NamedAxesTuple(chain(super().axes, self._dense_axes))

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

        if len(categorical_axes) < 1 or len(dense_axes) < 1:
            raise ValueError("At least one categorical axis and at least one dense axis should be specified.")

        return categorical_axes, dense_axes

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
        return dict(zip(self._cat_names, values))

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
        return hist.axis.NamedAxesTuple(self.axes[name] for name in self._cat_names)

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
        raise ValueError("Invalid index specification.")

    def _make_full_key(self, key):
        if isinstance(key, Mapping):
            full_key = {
                axis.name: slice(None) for axis in self.axes
            }
            full_key.update(key)
        elif isinstance(key, Sequence):
            if len(key) == len(self._cat_names):
                # assume just the name of the categories
                full_key = dict(zip(self._cat_names, key))
                full_key.update({axis.name: slice(None) for axis in self._dense_axes})
            elif len(key) == len(self.axes):
                # assume all axes specified, including dense axes
                full_key = dict(zip((a.name for a in self.axes), key))
            else:
                raise ValueError("Incorrect dimensions were specified.")
        else:
            raise ValueError("Index should be a mapping or a tuple.")

        for a in self.categorical_axes:
            full_key[a.name] = self._to_bin(a.name, full_key[a.name])
        return full_key

    def _from_hists(self, hists: dict, categorical_axes: list, collapsed: Union[None, Sequence] = None):
        dense_axes = list(hists.values())[0].axes

        new_hist = type(self)(*categorical_axes, *dense_axes, **self._init_args)

        for index_key, dense_hist in hists.items():
            named_key = self.index_to_categories(index_key, collapsed)
            new_hist._fill_bookkeep(*named_key)
            index_key = new_hist.categories_to_index(named_key)
            if index_key not in new_hist._dense_hists:
                new_hist._dense_hists[index_key] = hist.Hist(*dense_axes, **self._init_args)
            new_hist._dense_hists[index_key] += dense_hist
        return new_hist

    def _from_sum_all(self, hists: dict):
        dense_axes = list(hists.values())[0].axes
        new_hist = hist.Hist(*dense_axes, **self._init_args)
        for dense_hist in hists.values():
            new_hist += dense_hist
        return new_hist

    def _from_hists_no_dense(self, hists: dict, categorical_axes: list, collapsed: Union[None, Sequence] = None):
        new_hist = hist.Hist(*categorical_axes, **self._init_args)
        for index_key, weight in hists.items():
            named_key = self.index_to_categories(index_key, collapsed)
            new_hist.fill(*named_key, weight=weight)
        return new_hist

    def _filter_dense(self, full_key):
        def asseq(cat_name, x):
            if isinstance(x, int):
                return range(x, x + 1)
            elif isinstance(x, slice):
                return range(x.start, x.stop, x.step)
            elif x == sum:
                return range(len(self.axes[cat_name]))
            return x

        cats, nocats = self._split_axes(full_key)
        filtered = {}
        for sparse_key in product(*(asseq(name, v) for name, v in cats.items())):
            if sparse_key in self._dense_hists:
                filtered[sparse_key] = self._dense_hists[sparse_key][tuple(nocats.values())]

        if len(filtered) == 0:
            raise KeyError("No bins found")
        return filtered

    def __getitem__(self, key):
        full_key = self._make_full_key(key)
        filtered = self._filter_dense(full_key)

        collapsed = [full_key[name] is sum or isinstance(full_key[name], int) for name in self._cat_names]
        new_cats = [
            type(axis)([], growth=True, name=axis.name, label=axis.label)
            for axis, mask in zip(self.categorical_axes, collapsed) if not mask
        ]

        first = list(filtered.values())[0]
        if len(new_cats) == 0:
            if isinstance(first, hist.Hist):
                return self._from_sum_all(filtered)
            else:
                # collapsed to singe value
                return first
        if not isinstance(first, hist.Hist):
            # there was only one dense axis, and it collapse into one value
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
        rec(None, len(self._cat_names))
        return builder.snapshot()

    def values(self, flow=False):
        return self._ak_rec_op(lambda h: h.values(flow=flow))

    def counts(self, flow=False):
        return self._ak_rec_op(lambda h: h.counts(flow=flow))

    def view(self, flow=False, as_dict=True):
        if not as_dict:
            key = ", ".join([f"'{name}': ..." for name in self._cat_names])
            raise ValueError(f"If not a dict, only view of particular dense histograms is currently supported. Use h[{{{key}}}].view(flow=...) instead.")
        return {k: h.values(flow=flow) for k, h in self._dense_hists}

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
        if axis_name not in self._cat_names:
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

    def _ibinary_op(self, other, op: str):
        if not isinstance(other, SparseHist):
            for h in self._dense_hists.values():
                getattr(h, op)(other)
        else:
            if self._cat_names != other._cat_names:
                raise ValueError("Category names are different, or in different order, and therefore cannot be merged.")
            for index, oh in other._dense_hists.items():
                if index not in self._dense_hists:
                    self._fill_bookkeep(*other.index_to_categories(index))
                getattr(self._dense_hists[index], op)(oh)
        return self

    def _binary_op(self, other, op: str):
        h = self.copy()
        op = op.replace("__", "__i", 1)
        return h._ibinary_op(other, op)

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

    def __deepcopy__(self, deep=False):
        return self[{}]
