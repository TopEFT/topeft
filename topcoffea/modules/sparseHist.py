#! /usr/bin/env python

import hist
import boost_histogram as bh

from itertools import chain, product, repeat

from typing import Any, List, Mapping, Union, Sequence


class SparseHist(hist.Hist, family=hist):
    def __init__(self, *, categorical_axes, dense_axes, **kwargs):
        self._init_args = dict(kwargs)

        # actual "histogram". One histogram with one dense axis per combination of categorical bins.
        self._dense_hists: dict[tuple, hist.Hist] = {}

        self._cat_names = list(axis.name for axis in categorical_axes)
        self._dense_axes = list(dense_axes)

        # we use self to keep track of the bins in the categorical axes.
        super().__init__(*categorical_axes, storage="Double")
        self.axes = hist.axis.NamedAxesTuple(chain(super().axes, self._dense_axes))

        if len(self._cat_names) < 1 or len(self._dense_axes) < 1:
            raise ValueError("At least one categorical axis and one dense axis should be specified.")

        if not all(map(lambda a: isinstance(a, (hist.axis.StrCategory, hist.axis.IntCategory)), self.categorical_axes)):
            raise ValueError("Category axes should be of type hist.axis.StrCategory or hist.axis.IntCategory")

    def __str__(self):
        return repr(self)

    def _split_axes(self, axes: dict):
        """ Split axes dictionaries in categorical or dense.
        Axes returned in the order they were created. All axes of the histogram should be specified.
        """
        cats = {axis.name: axes[axis.name] for axis in self.categorical_axes}
        nocats = {axis.name: axes[axis.name] for axis in self._dense_axes}
        return (cats, nocats)

    def categories_to_index(self, bins: Union[Sequence, Mapping], collapsed=None):
        if collapsed is None:
            collapsed = repeat(False)
        return tuple(axis.index(bin) for axis, bin, mask in zip(self.categorical_axes, bins, collapsed) if not mask)

    def index_to_categories(self, indices: Sequence, collapsed=None):
        if collapsed is None:
            collapsed = repeat(True)
        return tuple(axis[index] for index, axis, mask in zip(indices, self.categorical_axes, collapsed) if not mask)

    @property
    def categorical_axes(self):
        return hist.axis.NamedAxesTuple(self.axes[name] for name in self._cat_names)

    @property
    def categorical_keys(self):
        return chain(*self.categorical_axes)

    def _fill_bookkeep(self, *args, **kwargs):
        super().fill(*args, **kwargs)

    def fill(self, weight=None, sample=None, threads=None, **kwargs):
        cats, nocats = self._split_axes(kwargs)

        # fill the bookkeeping first, so that the index of the key exists.
        self._fill_bookkeep(**cats)

        sparse_key = self.categories_to_index(tuple(kwargs[name] for name in self._cat_names))
        try:
            h = self._dense_hists[sparse_key]
        except KeyError:
            h = hist.Hist(*self._dense_axes, **self._init_args)
            self._dense_hists[sparse_key] = h

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
            # boost_histogram.tag.loc
            if len(key) != len(self.axes):
                raise ValueError("Incorrect dimensions were specified.")
            full_key = dict(zip((a.name for a in self.axes), key))
        else:
            raise ValueError("Index should be a mapping or a tuple.")

        for a in self.categorical_axes:
            full_key[a.name] = self._to_bin(a.name, full_key[a.name])

        return full_key

    def _from_hists(self, hists: dict, categorical_axes: list, collapsed: Union[None, Sequence] = None):
        dense_axes = list(hists.values())[0].axes

        new_hist = type(self)(categorical_axes=categorical_axes, dense_axes=dense_axes, **self._init_args)
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
