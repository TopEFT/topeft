#! /usr/bin/env python

import hist
import boost_histogram as bh
import numpy as np

from itertools import product, chain

from typing import Any, Dict, List, Mapping, Union

from topcoffea.modules.sparseHist import SparseHist
import topcoffea.modules.eft_helper as efth

try:
    from numpy.typing import ArrayLike, Self
except ImportError:
    ArrayLike = Any
    Number = Any
    Self = Any


_family = hist


class HistEFT(SparseHist, family=_family):
    """ Histogram specialized to hold Wilson Coefficients.
    Example:
    ```
    h = HistEFT(
        hist.axis.StrCategory(["ttH"], name="process", growth=True),
        hist.axis.Regular(
            name="ht",
            label="ht [GeV]",
            bins=3,
            start=0,
            stop=30,
            flow=True,
        ),
        wc_names=["ctG"],
        label="Events",
    )

    h.fill(
        process="ttH",
        ht=np.array([1, 1, 2, 15, 25, 100, -100]),
        # per row, quadratic coefficient values associated with one event.
        eft_coeff=[
            [1.1, 2.1, 3.1],     # to (ttH, 1j) bins (one bin per coefficient)
            [1.2, 2.2, 3.2],     # to (ttH, 1j) bins
            [1.3, 2.3, 3.3],     # to (ttH, 2j) bins
            [1.4, 2.4, 3.4],     # to (ttH, 15j) bins
            [1.5, 2.5, 3.5],     # to (ttH, 25j) bins
            [100, 200, 300],     # to (ttH, overflow given 100 >= stop) bins
            [-100, -200, -300],  # to (ttH, underflow given -100 < start) bins
        ],
    )

    # eval at 0, returns a dictionary from categorical axes bins to array, same as justsm,
    # {('ttH',): array([-100. ,   3.6,    1.4,    1.5,  600. ])}
    h.eval({})
    h.eval({"ctG": 0})     # same thing
    h.eval(np.zeros((1,))  # same thing

    # eval at 1, same as adding all bins together per bins of dense axis.
    # {('ttH',): array([-600. ,   19.8,    7.2,    7.5,  600. ])}
    h.eval({"ctG": 1})     # same thing
    h.eval(np.ones((1,))  # same thing

    # instead of h.eval(...), h.as_hist(...) may be used to create a standard hist.Hist with the
    # result of the evaluation:
    hn = h.as_hist({"ctG": 0.02})
    hn.plot1d()
    ```
    """

    def __init__(
        self,
        *args,
        wc_names: Union[List[str], None] = None,
        **kwargs,
    ) -> None:
        """ HistEFT initialization is similar to hist.Hist, with the following restrictions:
        - All axes should have a name.
        - Exactly one axis can be dense (i.e. hist.axis.Regular, hist.axis.Variable, or his.axis.Integer)
        - The dense axis should be the last specified in the list of arguments.
        - Categorical axes should be specified with growth=True.
        """

        if not wc_names:
            wc_names = []

        n = len(wc_names)
        self._wc_names = {n: i for i, n in enumerate(wc_names)}
        self._wc_count = n
        self._quad_count = efth.n_quad_terms(n)

        # a little ugly, but we need to keep these arguments in case we need to create a new histogram with group(...)
        self._init_args_base = dict(kwargs)
        self._init_args_eft = {"wc_names": wc_names}

        self._needs_rebinning = kwargs.pop("rebin", False)
        if self._needs_rebinning:
            raise ValueError("Do not know how to rebin yet...")

        kwargs.setdefault("storage", "Double")
        if kwargs["storage"] != "Double":
            raise ValueError("only 'Double' storage is supported")

        if args[-1].name == "quadratic_term":
            self._coeff_axis = args[-1]
            args = args[:-1]
        else:
            # no axis for quadratic_term found, creating our own.
            self._coeff_axis = hist.axis.Integer(
                start=0, stop=self._quad_count, name="quadratic_term"
            )

        self._dense_axis = args[-1]
        if not isinstance(
            self._dense_axis, (bh.axis.Regular, bh.axis.Variable, bh.axis.Integer)
        ):
            raise ValueError("dense axis should be the last specified")

        reserved_names = ["quadratic_term", "sample", "weight", "thread"]
        if any([axis.name in reserved_names for axis in args]):
            raise ValueError(
                f"No axis may have one of the following names: {','.join(reserved_names)}"
            )

        super().__init__(*args, self._coeff_axis, **kwargs)

    def from_categorical_axes_like(self, *axes):
        return type(self)(*axes, self._dense_axis, **self._init_args_base, **self._init_args_eft)

    def wc_names(self):
        return list(self._wc_names)

    def index_of_wc(self, wc: str):
        return self._wc_names[wc]

    def quadratic_term_index(self, *wcs: List[str]):
        """Given the name of two coefficients, it returns the index
        of the corresponding quadratic coefficient. E.g., if the
        histogram was defined with wc_names=["ctG"]:

        h.quadratic_term_index("sm", "sm")   -> 0
        h.quadratic_term_index("sm", "ctG")  -> 1
        h.quadratic_term_index("ctG", "ctG") -> 2
        """

        def str_to_index(s):
            if s == "sm":
                return 0
            else:
                return self.index_of_wc(s) + 1

        if len(wcs) != 2:
            raise ValueError("List of coefficient names should have length 2")

        wc1, wc2 = map(str_to_index, wcs)
        if wc1 < wc2:
            wc1, wc2 = wc2, wc1

        return int((((wc1 + 1) * wc1) / 2) + wc2)

    def should_rebin(self):
        return self._needs_rebinning

    @property
    def dense_axis(self):
        return self._dense_axis

    def _fill_flatten(self, a, n_events):
        # manipulate input arrays into flat arrays. broadcast_to and ravel used so that arrays are not duplicated in memory
        a = np.asarray(a)
        if a.ndim > 2 or (a.ndim == 2 and (a.shape != (n_events, 1))):
            raise ValueError(
                "Incompatible dimensions between data and Wilson coefficients."
            )

        if a.ndim > 1:
            a = a.ravel()

        # turns [e0, e1, ...] into [[e0, e0, ...],
        #                           [e1, e1, ...],
        #                            [...       ]]
        # and then into       [e0, e0, ..., e1, e1, ..., e2, e2, ...]
        # each value repeated the number of quadratic coefficients.
        return np.broadcast_to(a, (self._quad_count, n_events)).T.ravel()

    def _fill_indices(self, n_events):
        # turns [0, 1, 2, ..., num of quadratic coeffs - 1]
        # into:
        # [0, 1, 2, ..., 0, 1, 2 ...,]
        # repeated n_events times.
        return np.broadcast_to(
            np.ogrid[0 : self._quad_count], (n_events, self._quad_count)
        ).ravel()

    def fill(
        self,
        eft_coeff: ArrayLike = None,  # [num of events x (num of wc coeffs + 1)]
        **values,
    ) -> Self:
        """
        Insert data into the histogram using names and indices, return
        a HistEFT object.

        cat axes:  "s1"                    each categorical axis with one value to fill
        dense axis:[ e0, e1, ... ]         each entry is the value for one event.
        weight:    [ w0, w1, ... ]         weight per event
        eft_coeff: [[c00, c01, c02, ...]   each row is the coefficient values for one event,
                    [c10, c11, c12, ...]
                    ...                 ]  cij is the value of jth coefficient for the ith event.
                                           ei, wi, and ci* go together.

        If eft_coeff is not given, then it is assumed to be [[1, 0, 0, ...], [1, 0, 0, ...], ...]
        """

        n_events = len(values[self.dense_axis.name])

        if eft_coeff is None:
            # if eft_coeff not given, assume values only for sm
            eft_coeff = np.broadcast_to(
                np.concatenate((np.ones((1,)), np.zeros((self._quad_count - 1,)))),
                (n_events, self._quad_count),
            )

        eft_coeff = np.asarray(eft_coeff)

        # turn into [e0, e0, ..., e1, e1, ..., e2, e2, ...]
        values[self._dense_axis.name] = self._fill_flatten(values[self._dense_axis.name], n_events)

        # turn into: [c00, c01, c02, ..., c10, c11, c12, ...]
        eft_coeff = eft_coeff.ravel()

        # index for coefficient axes.
        # [ 0, 1, 2, ..., 0, 1, 2, ...]
        indices = self._fill_indices(n_events)

        weight = values.pop("weight", None)
        if weight is not None:
            weight = self._fill_flatten(weight, n_events)
            eft_coeff = eft_coeff * weight

        # fills:
        # [e0,      e0,      e0    ..., e1,     e1,     e1,     ...]
        # [ 0,      1,       2,    ..., 0,      1,      2,      ...]
        # [c00*w0, c01*w0, c02*w0, ..., c10*w1, c11*w1, c12*w1, ...]
        super().fill(quadratic_term=indices, **values, weight=eft_coeff)

    def _wc_for_eval(self, values):
        """Set the WC values used to evaluate the bin contents of this histogram
        where the WCs are specified as keyword arguments.  Any WCs not listed are set to zero.
        """
        if not values:
            return np.zeros(self._wc_count)

        result = values
        if isinstance(values, Mapping):
            result = np.zeros(self._wc_count)
            for wc, val in values.items():
                try:
                    index = self._wc_names[wc]
                    result[index] = val
                except KeyError:
                    msg = f'This HistEFT does not know about the "{wc}" Wilson coefficient. Known coefficients: {list(self._wc_names.keys())}'
                    raise LookupError(msg)

        return np.asarray(result)

    def eval(self, values):
        """Extract the sum of weights arrays from this histogram
        Parameters
        ----------
        values: ArrayLike or Mapping or None
            The WC values used to evaluate the bin contents of this histogram. Either an array with the values, or a dictionary. If None, use an array of zeros.
        """

        values = self._wc_for_eval(values)

        out = {}
        for sparse_key, hvs in self.view(flow=True, as_dict=True).items():
            out[sparse_key] = efth.calc_eft_weights(hvs, values)
        return out

    def as_hist(self, values):
        """Construct a regular histogram evaluated at values.
        (Like self.eval(...) but result is a histogram.)
        Parameters
        ----------
        values: ArrayLike or Mapping or None
            The WC values used to evaluate the bin contents of this histogram. Either an array with the values, or a dictionary. If None, use an array of zeros.
        overflow: bool
            Whether to include under and overflow bins.
        """
        evals = self.eval(values=values)
        nhist = hist.Hist(
            *[axis for axis in self.axes if axis != self._coeff_axis],
            **self._init_args_base,
        )

        sparse_names = list(axis.name for axis in self.categorical_axes())
        for sp_vals, arrs in evals.items():
            sp_key = dict(zip(sparse_names, sp_vals))
            nhist[sp_key] = arrs
        return nhist

    def group(self, old_name: str, new_name: str, groups: Dict[str, List[str]]):
        """ Generate a new HistEFT where bins of axis named old_name are merged
        according to the groups mapping. The axis of the resuling histogram is
        named new_name.
        """
        if old_name != new_name:
            for ax in self.axes:
                if new_name == ax.name:
                    raise ValueError("Name of new axis is already in use")

        old_axis = self.axes[old_name]
        new_axis = hist.axis.StrCategory(
            groups.keys(), name=new_name, label=old_axis.label, growth=True
        )

        new_axes = []
        for axis in self.axes:
            if axis.name == old_name:
                new_axes.append(new_axis)
            elif axis != self._coeff_axis:
                new_axes.append(axis)

        hnew = type(self)(
            *new_axes,
            **self._init_args_base,
            **self._init_args_eft,
        )

        hnew._coeff_axis.label = self._coeff_axis.label

        for join, splits in groups.items():
            if isinstance(splits, str):
                splits = [splits]
            joined = sum(self[{old_name: split}] for split in splits)
            hnew[{new_name: join}] = joined.view(flow=True)
        return hnew

    def scale(self, factor: float):
        self *= factor
        return self

    def empty(self):
        return np.all(self.values(flow=True) == 0)

    # compatibility methods for old coffea
    # all of these are deprecated
    def identity(self):
        h = self.copy(deep=False)
        h.reset()
        return h
