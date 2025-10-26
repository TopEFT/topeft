import collections
import importlib
import json
import sys
import types

try:  # pragma: no cover - prefer real awkward when available
    import awkward as ak
except ModuleNotFoundError:  # pragma: no cover - shim for test isolation
    class _SimpleArray:
        def __init__(self, data):
            if isinstance(data, _SimpleArray):
                self._data = data._data
            else:
                self._data = np.array(data, dtype=object)

        def __array__(self, dtype=None):
            return np.array(self._data, dtype=dtype) if dtype else np.array(self._data)

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def __repr__(self):
            return f"SimpleArray({self._data!r})"

        def __getitem__(self, item):
            return _SimpleArray(self._data[item])

        def __setitem__(self, key, value):
            self._data[key] = value

        def _extract_field(self, item, name):
            if isinstance(item, dict):
                return item.get(name)
            if isinstance(item, (list, tuple, np.ndarray)):
                return [self._extract_field(sub, name) for sub in item]
            if hasattr(item, name):
                return getattr(item, name)
            return None

        def __getattr__(self, name):
            try:
                extractor = np.vectorize(lambda item: self._extract_field(item, name), otypes=[object])
                return _SimpleArray(extractor(self._data))
            except Exception as exc:  # pragma: no cover - defensive
                raise AttributeError(name) from exc

        @property
        def layout(self):  # pragma: no cover - compatibility shim
            return self._data

    def _ensure_array(value):
        return value._data if isinstance(value, _SimpleArray) else np.array(value, dtype=object)

    def _wrap_array(value):
        return value if isinstance(value, _SimpleArray) else _SimpleArray(value)

    class _AwkwardModule(types.SimpleNamespace):
        def Array(self, data):
            return _SimpleArray(data)

        def ones_like(self, array, dtype=float):
            data = _ensure_array(array)
            return _SimpleArray(np.ones_like(data, dtype=dtype))

        def fill_none(self, array, value):
            data = _ensure_array(array)

            def _replace(item):
                if isinstance(item, (list, tuple, np.ndarray)):
                    return [_replace(sub) for sub in item]
                return value if item is None else item

            replacer = np.vectorize(_replace, otypes=[object])
            return _SimpleArray(replacer(data))

        def flatten(self, array):
            data = _ensure_array(array)
            flat = []
            for item in data:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat.extend(item)
                else:
                    flat.append(item)
            return _SimpleArray(np.array(flat, dtype=object))

        def num(self, array, axis=0):
            data = _ensure_array(array)
            if axis != 0:
                raise NotImplementedError("Only axis=0 is supported in the test shim")
            counts = []
            for item in data:
                if isinstance(item, (list, tuple, np.ndarray)):
                    counts.append(len(item))
                else:
                    counts.append(1 if item is not None else 0)
            return np.array(counts, dtype=int)

        def unflatten(self, flat_array, counts):
            values = list(_ensure_array(flat_array).ravel())
            counts_arr = np.array(_ensure_array(counts), dtype=int).ravel()
            result = []
            index = 0
            for count in counts_arr:
                segment = values[index : index + count]
                index += count
                result.append(segment)
            return _SimpleArray(np.array(result, dtype=object))

    ak = _AwkwardModule()
    ak.combinations = lambda array, *args, **kwargs: _SimpleArray([])
    ak.with_name = lambda array, name: array
    sys.modules.setdefault("awkward", ak)
import numpy as np
import pytest


class _EventNamespace(dict):
    """Simple container supporting attribute and key access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive fallback
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _install_stubs(monkeypatch):
    """Install lightweight module stubs required by the analysis processor."""

    def _module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    topcoffea_pkg = _module("topcoffea")
    topcoffea_pkg.__path__ = []  # type: ignore[attr-defined]
    topcoffea_modules_pkg = _module("topcoffea.modules")
    topcoffea_modules_pkg.__path__ = []  # type: ignore[attr-defined]
    topcoffea_pkg.modules = topcoffea_modules_pkg  # type: ignore[attr-defined]

    # HistEFT stub
    hist_module = _module("topcoffea.modules.histEFT")

    class _DummyHistEFT:
        def __init__(self, *axes, wc_names=None, label=None, **kwargs):
            self.axes = tuple(axes)
            self.wc_names = list(wc_names or [])
            self.label = label
            self._fills = []
            self._sumw = 0.0

        def fill(self, weight=None, **kwargs):
            if weight is None:
                weight = 1.0
            array = np.asarray(weight, dtype=float)
            self._fills.append({"weight": array, "fields": kwargs})
            self._sumw += float(np.sum(array))

        def __repr__(self):  # pragma: no cover - debug helper
            return f"DummyHistEFT(sumw={self._sumw}, fills={len(self._fills)})"

    hist_module.HistEFT = _DummyHistEFT

    # ``hist`` axis helpers
    hist_pkg = _module("hist")

    class _BaseAxis:
        def __init__(self, kind, *args, **kwargs):
            self.kind = kind
            self.args = args
            self.kwargs = kwargs
            self.name = kwargs.get("name")
            self.label = kwargs.get("label")
            self.metadata = {"kind": kind}

        def __repr__(self):  # pragma: no cover - debug helper
            return f"Axis(kind={self.kind!r}, name={self.name!r})"

    class _DenseAxis(_BaseAxis):
        def __init__(self, kind, *args, **kwargs):
            super().__init__(kind, *args, **kwargs)
            if kind == "Variable":
                self.edges = np.asarray(args[0], dtype=float)
            elif kind == "Regular":
                bins, start, stop = args[:3]
                self.edges = np.linspace(start, stop, int(bins) + 1, dtype=float)
            elif kind == "Integer":
                start, stop = args[:2]
                self.edges = np.arange(start, stop + 1, dtype=int)
            else:
                self.edges = np.asarray(args, dtype=float)

    class _CategoryAxis(_BaseAxis):
        def __init__(self, kind, *args, **kwargs):
            super().__init__(kind, *args, **kwargs)
            categories = []
            if args:
                categories = list(args[0])
            elif kind == "Boolean":
                categories = [False, True]
            self._categories = list(categories)

        def __iter__(self):
            return iter(self._categories)

        def __contains__(self, value):
            return value in self._categories

        def __len__(self):
            return len(self._categories)

        def __repr__(self):  # pragma: no cover - debug helper
            return f"CategoryAxis(kind={self.kind!r}, name={self.name!r}, categories={self._categories!r})"

    def _axis_factory(kind, axis_type):
        def _factory(*args, **kwargs):
            return axis_type(kind, *args, **kwargs)

        return _factory

    hist_pkg.axis = types.SimpleNamespace(
        Regular=_axis_factory("Regular", _DenseAxis),
        Variable=_axis_factory("Variable", _DenseAxis),
        Integer=_axis_factory("Integer", _DenseAxis),
        IntCategory=_axis_factory("IntCategory", _CategoryAxis),
        StrCategory=_axis_factory("StrCategory", _CategoryAxis),
        Boolean=_axis_factory("Boolean", _CategoryAxis),
    )

    # Coffea interfaces
    coffea_pkg = _module("coffea")
    processor_module = _module("coffea.processor")

    class _ProcessorABC:  # pragma: no cover - interface shim
        """Minimal stand-in for coffea.processor.ProcessorABC."""

        def __init_subclass__(cls, **kwargs):  # pragma: no cover - compatibility
            return super().__init_subclass__(**kwargs)

    class _ExecutorBase:  # pragma: no cover - simple executor stub
        def __init__(self, **config):
            self.config = config

        def __call__(self, *args, **kwargs):
            return {}

    class _IterativeExecutor(_ExecutorBase):
        pass

    class _FuturesExecutor(_ExecutorBase):
        pass

    class _DaskExecutor(_ExecutorBase):
        pass

    class _ParslExecutor(_ExecutorBase):
        pass

    class _TaskVineExecutor(_ExecutorBase):
        pass

    class _Runner:  # pragma: no cover - invoked indirectly in tests
        def __init__(self, executor=None, schema=None, chunksize=None, maxchunks=None, **kwargs):
            self.executor = executor
            self.schema = schema
            self.chunksize = chunksize
            self.maxchunks = maxchunks
            self.runner_config = kwargs

        def __call__(self, items, processor_instance, *args, **kwargs):
            if processor_instance is None:
                return {}
            process = getattr(processor_instance, "process", None)
            if callable(process):
                return process(items)
            return {}

    def _iterative_executor(*args, **kwargs):  # pragma: no cover - legacy shim
        return _IterativeExecutor(*args, **kwargs)

    def _futures_executor(*args, **kwargs):  # pragma: no cover - legacy shim
        return _FuturesExecutor(*args, **kwargs)

    def _dask_executor(*args, **kwargs):  # pragma: no cover - legacy shim
        return _DaskExecutor(*args, **kwargs)

    def _parsl_executor(*args, **kwargs):  # pragma: no cover - legacy shim
        return _ParslExecutor(*args, **kwargs)

    def _taskvine_executor(*args, **kwargs):  # pragma: no cover - legacy shim
        return _TaskVineExecutor(*args, **kwargs)

    # Accumulator helpers mimicking coffea.processor.accumulator
    class _AccumulatorABC:  # pragma: no cover - interface shim
        def identity(self):
            return type(self)()

        def add(self, other):
            return other

    class _Accumulatable(_AccumulatorABC):
        pass

    class _ValueAccumulator(_AccumulatorABC):
        def __init__(self, value=None):
            self.value = value

        def add(self, other):
            self.value = other
            return self

    class _ListAccumulator(list):
        def add(self, other):
            self.extend(other)
            return self

    class _SetAccumulator(set):
        def add(self, other):
            super().update(other)
            return self

    class _DictAccumulator(dict):
        def add(self, other):
            self.update(other)
            return self

    def _defaultdict_accumulator(factory):
        return collections.defaultdict(factory)

    def _column_accumulator(array):
        return np.asarray(array)

    def _accumulate(sequence):
        result = {}
        for item in sequence:
            if isinstance(item, dict):
                result.update(item)
        return result

    accumulator_module = _module("coffea.processor.accumulator")
    accumulator_module.AccumulatorABC = _AccumulatorABC
    accumulator_module.Accumulatable = _Accumulatable
    accumulator_module.accumulate = _accumulate
    accumulator_module.value_accumulator = _ValueAccumulator
    accumulator_module.list_accumulator = _ListAccumulator
    accumulator_module.set_accumulator = _SetAccumulator
    accumulator_module.dict_accumulator = _DictAccumulator
    accumulator_module.defaultdict_accumulator = _defaultdict_accumulator
    accumulator_module.column_accumulator = _column_accumulator

    processor_module.ProcessorABC = _ProcessorABC
    processor_module.ExecutorBase = _ExecutorBase
    processor_module.IterativeExecutor = _IterativeExecutor
    processor_module.FuturesExecutor = _FuturesExecutor
    processor_module.DaskExecutor = _DaskExecutor
    processor_module.ParslExecutor = _ParslExecutor
    processor_module.TaskVineExecutor = _TaskVineExecutor
    processor_module.Runner = _Runner
    processor_module.iterative_executor = _iterative_executor
    processor_module.futures_executor = _futures_executor
    processor_module.dask_executor = _dask_executor
    processor_module.parsl_executor = _parsl_executor
    processor_module.taskvine_executor = _taskvine_executor
    processor_module.AccumulatorABC = _AccumulatorABC
    processor_module.Accumulatable = _Accumulatable
    processor_module.accumulate = _accumulate
    processor_module.value_accumulator = _ValueAccumulator
    processor_module.list_accumulator = _ListAccumulator
    processor_module.set_accumulator = _SetAccumulator
    processor_module.dict_accumulator = _DictAccumulator
    processor_module.defaultdict_accumulator = _defaultdict_accumulator
    processor_module.column_accumulator = _column_accumulator
    processor_module.accumulator = accumulator_module

    executor_submodule = _module("coffea.processor.executor")
    executor_submodule.ExecutorBase = _ExecutorBase
    executor_submodule.IterativeExecutor = _IterativeExecutor
    executor_submodule.FuturesExecutor = _FuturesExecutor
    executor_submodule.DaskExecutor = _DaskExecutor
    executor_submodule.ParslExecutor = _ParslExecutor
    executor_submodule.TaskVineExecutor = _TaskVineExecutor
    executor_submodule.Runner = _Runner
    executor_submodule.accumulate = _accumulate
    executor_submodule.AccumulatorABC = _AccumulatorABC
    executor_submodule.Accumulatable = _Accumulatable
    executor_submodule.list_accumulator = _ListAccumulator
    executor_submodule.set_accumulator = _SetAccumulator
    executor_submodule.dict_accumulator = _DictAccumulator
    executor_submodule.defaultdict_accumulator = _defaultdict_accumulator
    executor_submodule.column_accumulator = _column_accumulator

    taskvine_submodule = _module("coffea.processor.taskvine_executor")
    taskvine_submodule.TaskVineExecutor = _TaskVineExecutor

    processor_submodule = _module("coffea.processor.processor")
    processor_submodule.ProcessorABC = _ProcessorABC

    processor_module.executor = executor_submodule  # type: ignore[attr-defined]
    processor_module.taskvine_executor = taskvine_submodule  # type: ignore[attr-defined]
    processor_module.processor = processor_submodule  # type: ignore[attr-defined]

    coffea_pkg.processor = processor_module  # type: ignore[attr-defined]

    analysis_tools_module = _module("coffea.analysis_tools")

    def _boolean_masks_to_categorical_integers(
        masks,
        insert_unmasked_as_zeros=False,
        insert_commonmask_as_zeros=None,
        return_mask=False,
    ):
        mask_arrays = [np.asarray(mask, dtype=bool) for mask in masks]
        if mask_arrays:
            mask_stack = np.stack(mask_arrays, axis=1)
        else:
            mask_stack = np.zeros((0, 0), dtype=bool)

        if insert_unmasked_as_zeros:
            unmasked = np.all(mask_stack, axis=1, keepdims=True)
            mask_stack = np.concatenate([unmasked, mask_stack], axis=1)

        if insert_commonmask_as_zeros is not None:
            common = np.asarray(insert_commonmask_as_zeros, dtype=bool).reshape(-1, 1)
            mask_stack = np.concatenate([common, mask_stack], axis=1)

        if return_mask:
            return mask_stack

        categories = [np.nonzero(row)[0].tolist() for row in mask_stack]
        return ak.Array(categories)

    class _WeightStatistics:
        def __init__(self, sumw=0.0, sumw2=0.0, minw=np.inf, maxw=-np.inf, n=0):
            self.sumw = float(sumw)
            self.sumw2 = float(sumw2)
            self.minw = float(minw)
            self.maxw = float(maxw)
            self.n = int(n)

        def identity(self):
            return type(self)()

        def add(self, other):
            self.sumw += other.sumw
            self.sumw2 += other.sumw2
            self.minw = min(self.minw, other.minw)
            self.maxw = max(self.maxw, other.maxw)
            self.n += other.n
            return self

        def __iadd__(self, other):
            return self.add(other)

    class _PackedSelection:
        _max_items = {
            np.dtype("uint16"): 16,
            np.dtype("uint32"): 32,
            np.dtype("uint64"): 64,
        }

        def __init__(self, dtype="uint32"):
            self._dtype = np.dtype(dtype)
            if self._dtype not in self._max_items:
                raise ValueError(f"dtype {dtype} is not supported")
            self._names = []
            self._data = None

        @property
        def names(self):
            return list(self._names)

        @property
        def maxitems(self):
            return self._max_items[self._dtype]

        def _ensure_mask(self, selection, fill_value=False):
            try:
                import awkward as _ak  # pragma: no cover - use real awkward when available

                if isinstance(selection, _ak.Array):
                    selection = _ak.to_numpy(selection, allow_missing=True)
            except ModuleNotFoundError:  # pragma: no cover - rely on numpy fallback
                pass

            if isinstance(selection, np.ma.MaskedArray):
                selection = selection.filled(bool(fill_value))
            arr = np.asarray(selection, dtype=bool)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr

        def add(self, name, selection, fill_value=False):
            mask = self._ensure_mask(selection, fill_value=fill_value)
            if self._data is None:
                self._data = np.zeros(mask.shape, dtype=self._dtype)
            elif mask.shape != self._data.shape:
                raise ValueError(
                    f"Selection '{name}' has shape {mask.shape}, expected {self._data.shape}"
                )
            if len(self._names) >= self.maxitems:
                raise RuntimeError("PackedSelection has reached maximum capacity")

            bit = self._dtype.type(1 << len(self._names))
            np.bitwise_or(self._data, bit, out=self._data, where=mask)
            self._names.append(name)

        def _data_view(self):
            if self._data is None:
                return np.zeros(1, dtype=self._dtype)
            return self._data

        def require(self, **names):
            data = self._data_view()
            consider = self._dtype.type(0)
            required = self._dtype.type(0)
            for name, value in names.items():
                idx = self._names.index(name)
                consider |= self._dtype.type(1 << idx)
                if value:
                    required |= self._dtype.type(1 << idx)
            return (data & consider) == required

        def all(self, *names):
            data = self._data_view()
            if not names:
                return np.ones_like(data, dtype=bool)
            requirements = {name: True for name in names}
            return self.require(**requirements)

        def any(self, *names):
            data = self._data_view()
            if not names:
                return np.zeros_like(data, dtype=bool)
            consider = self._dtype.type(0)
            for name in names:
                idx = self._names.index(name)
                consider |= self._dtype.type(1 << idx)
            return (data & consider) != 0

    class _Weights:
        def __init__(self, size, storeIndividual=False):
            self._store_individual = bool(storeIndividual)
            self._central = np.ones(int(size), dtype=float) if size is not None else None
            self._weights = {}
            self._modifiers = {}
            self._statistics = {}

        @property
        def weightStatistics(self):
            return self._statistics

        def _ensure_array(self, value):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if self._central is None:
                self._central = np.ones_like(arr, dtype=float)
            return arr

        def _register_stat(self, name, array):
            if array.size:
                minw = float(np.min(array))
                maxw = float(np.max(array))
            else:  # pragma: no cover - degenerate
                minw = float("inf")
                maxw = float("-inf")
            self._statistics[name] = _WeightStatistics(
                sumw=np.sum(array),
                sumw2=np.sum(array ** 2),
                minw=minw,
                maxw=maxw,
                n=array.size,
            )

        def _add_variation(self, label, base, up=None, down=None, shift=False):
            if up is not None:
                up_arr = self._ensure_array(up)
                ratio_up = np.ones_like(base, dtype=float)
                if shift:
                    up_arr = up_arr + base
                np.divide(up_arr, base, out=ratio_up, where=base != 0)
                self._modifiers[label + "Up"] = ratio_up
            else:
                ratio_up = None

            if down is not None:
                down_arr = self._ensure_array(down)
                ratio_down = np.ones_like(base, dtype=float)
                if shift:
                    down_arr = base - down_arr
                np.divide(down_arr, base, out=ratio_down, where=base != 0)
                self._modifiers[label + "Down"] = ratio_down
            elif ratio_up is not None:
                inv = np.ones_like(ratio_up, dtype=float)
                np.divide(1.0, ratio_up, out=inv, where=ratio_up != 0)
                self._modifiers[label + "Down"] = inv

        def add(self, name, weight, weightUp=None, weightDown=None, shift=False):
            base = self._ensure_array(weight)
            self._central = self._central * base
            if self._store_individual:
                self._weights[name] = base
            self._register_stat(name, base)
            self._add_variation(name, base, up=weightUp, down=weightDown, shift=shift)

        def add_multivariation(
            self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
        ):
            base = self._ensure_array(weight)
            self._central = self._central * base
            if self._store_individual:
                self._weights[name] = base
            self._register_stat(name, base)

            for modifier, up, down in zip(modifierNames, weightsUp, weightsDown):
                label = f"{name}_{modifier}"
                self._add_variation(label, base, up=up, down=down, shift=shift)

        @property
        def variations(self):
            keys = set(self._modifiers.keys())
            for key in list(keys):
                if key.endswith("Up"):
                    keys.add(key[:-2] + "Down")
            return keys

        def weight(self, modifier=None):
            if modifier in (None, "nominal"):
                return self._central

            if modifier not in self._modifiers and isinstance(modifier, str) and modifier.endswith("Down"):
                up_key = modifier[:-4] + "Up"
                if up_key in self._modifiers:
                    inv = np.ones_like(self._modifiers[up_key], dtype=float)
                    np.divide(1.0, self._modifiers[up_key], out=inv, where=self._modifiers[up_key] != 0)
                    self._modifiers[modifier] = inv

            if modifier in self._modifiers:
                return self._central * self._modifiers[modifier]

            raise KeyError(f"Unknown weight modifier '{modifier}'")

        def partial_weight(self, include=None, exclude=None):
            if not self._store_individual:
                raise ValueError("storeIndividual must be True to compute partial weights")
            include = include or []
            exclude = exclude or []
            if include and exclude:
                raise ValueError("Specify only include or exclude, not both")
            names = set(self._weights)
            if include:
                names &= set(include)
            if exclude:
                names -= set(exclude)
            result = np.ones_like(self._central, dtype=float)
            for name in names:
                result *= self._weights[name]
            return result

    class _Cutflow:
        def __init__(self):
            self._counts = collections.OrderedDict()

        def add(self, name, count):
            value = float(np.sum(count)) if np.size(count) else float(count)
            self._counts[name] = self._counts.get(name, 0.0) + value

        def __getitem__(self, key):
            return self._counts[key]

        def items(self):
            return self._counts.items()

        def to_dict(self):  # pragma: no cover - convenience helper
            return dict(self._counts)

    class _NminusOne(_Cutflow):
        pass

    def _cutflow_to_npz(cutflow, filename, **metadata):  # pragma: no cover - shim
        return {
            "filename": filename,
            "cutflow": dict(getattr(cutflow, "items", lambda: [])()),
            "metadata": metadata,
        }

    def _nminus_one_to_npz(nminus_one, filename, **metadata):  # pragma: no cover
        return {
            "filename": filename,
            "cutflow": dict(getattr(nminus_one, "items", lambda: [])()),
            "metadata": metadata,
        }

    analysis_tools_module.boolean_masks_to_categorical_integers = (
        _boolean_masks_to_categorical_integers
    )
    analysis_tools_module.WeightStatistics = _WeightStatistics
    analysis_tools_module.PackedSelection = _PackedSelection
    analysis_tools_module.Weights = _Weights
    analysis_tools_module.Cutflow = _Cutflow
    analysis_tools_module.NminusOne = _NminusOne
    analysis_tools_module.CutflowToNpz = _cutflow_to_npz
    analysis_tools_module.NminusOneToNpz = _nminus_one_to_npz

    class _IncompatiblePartitions(Exception):
        """Lightweight proxy for dask_awkward.utils.IncompatiblePartitions."""

    def _compatible_partitions(*_args, **_kwargs):
        return True

    analysis_tools_module.IncompatiblePartitions = _IncompatiblePartitions
    analysis_tools_module.compatible_partitions = _compatible_partitions
    analysis_tools_module.numpy = np
    analysis_tools_module.awkward = ak
    analysis_tools_module.hist = hist_pkg
    analysis_tools_module.coffea = coffea_pkg
    analysis_tools_module.warnings = __import__("warnings")
    analysis_tools_module.lru_cache = __import__("functools").lru_cache
    analysis_tools_module.namedtuple = __import__("collections").namedtuple

    dask_module = _module("dask")
    dask_awkward_module = _module("dask_awkward")
    analysis_tools_module.dask = dask_module  # type: ignore[attr-defined]
    analysis_tools_module.dask_awkward = dask_awkward_module  # type: ignore[attr-defined]

    nanoevents_module = _module("coffea.nanoevents")
    coffea_pkg.nanoevents = nanoevents_module  # type: ignore[attr-defined]

    class _NanoEventsFactory:
        warn_missing_crossrefs = False

        def __init__(self, events=None):
            self._events = events if events is not None else ak.Array([])

        @classmethod
        def from_root(cls, *_args, **_kwargs):
            return cls()

        @classmethod
        def from_parquet(cls, *_args, **_kwargs):  # pragma: no cover - compatibility shim
            return cls()

        @classmethod
        def from_awkd(cls, events, *_args, **_kwargs):  # pragma: no cover - compatibility shim
            return cls(events)

        @classmethod
        def from_preloaded(cls, events, *_args, **_kwargs):  # pragma: no cover
            return cls(events)

        def events(self):
            return self._events

    class _NanoAODSchema:
        @classmethod
        def schema(cls, *_args, **_kwargs):  # pragma: no cover - simple placeholder
            return {}

    nanoevents_module.NanoEventsFactory = _NanoEventsFactory
    nanoevents_module.NanoAODSchema = _NanoAODSchema

    lumi_tools_module = _module("coffea.lumi_tools")

    class _DummyLumiMask:
        def __init__(self, *_, **__):
            pass

        def __call__(self, run, lumi):
            return ak.ones_like(run, dtype=bool)

    class _LumiList(list):  # pragma: no cover - lightweight shim
        def __init__(self, *lumis, **kwargs):
            super().__init__(lumis or kwargs.get("lumis", []))

        def __contains__(self, item):
            return list.__contains__(self, item)

    class _LumiData(dict):  # pragma: no cover - lightweight shim
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    lumi_tools_module.LumiMask = _DummyLumiMask
    lumi_tools_module.LumiList = _LumiList
    lumi_tools_module.LumiData = _LumiData

    lumi_submodule = _module("coffea.lumi_tools.lumi_tools")
    lumi_submodule.LumiMask = _DummyLumiMask
    lumi_submodule.LumiList = _LumiList
    lumi_submodule.LumiData = _LumiData
    lumi_tools_module.lumi_tools = lumi_submodule  # type: ignore[attr-defined]

    # Parameter helpers
    paths_module = _module("topcoffea.modules.paths")
    paths_module.topcoffea_path = lambda path: str(path)

    topeft_pkg = _module("topeft")
    topeft_pkg.__path__ = []  # type: ignore[attr-defined]
    topeft_modules_pkg = _module("topeft.modules")
    topeft_modules_pkg.__path__ = []  # type: ignore[attr-defined]
    topeft_pkg.modules = topeft_modules_pkg  # type: ignore[attr-defined]

    te_paths_module = _module("topeft.modules.paths")
    te_paths_module.topeft_path = lambda path: str(path)

    eft_helper_module = _module("topcoffea.modules.eft_helper")
    eft_helper_module.remap_coeffs = lambda *args, **kwargs: args[-1]
    eft_helper_module.calc_w2_coeffs = lambda coeffs, dtype=None: coeffs

    def _dummy_get_param(mapping):
        defaults = {
            "lo_xsec_samples": [],
            "conv_samples": [],
            "prompt_and_conv_samples": [],
            "eta_j_cut": 2.5,
            "jet_id_cut": 2,
            "lumi_2018": 1.0,
            "btag_wp_loose_UL18": 0.0,
            "btag_wp_medium_UL18": 0.0,
        }
        return defaults.get(mapping, 0.0)

    get_param_module = _module("topcoffea.modules.get_param_from_jsons")
    get_param_module.GetParam = lambda *args, **kwargs: _dummy_get_param

    # Corrections stubs
    corrections_module = _module("topcoffea.modules.corrections")

    class _JetCorrections:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, collection, *_, **__):
            return collection

    corrections_module.ApplyJetCorrections = lambda *args, **kwargs: _JetCorrections()
    corrections_module.AttachScaleWeights = lambda *args, **kwargs: None
    corrections_module.AttachPSWeights = lambda *args, **kwargs: None
    corrections_module.GetPUSF = lambda *args, **kwargs: np.ones(1)
    corrections_module.btag_sf_eval = lambda *args, **kwargs: np.ones(1)
    corrections_module.get_method1a_wgt_doublewp = (
        lambda *args, **kwargs: (np.ones(1), np.ones(1))
    )

    topeft_corr_module = _module("topeft.modules.corrections")
    topeft_corr_module.ApplyJetCorrections = corrections_module.ApplyJetCorrections
    topeft_corr_module.ApplyJetSystematics = lambda *args, **kwargs: args[1]
    topeft_corr_module.GetBtagEff = lambda *args, **kwargs: np.ones(1)
    topeft_corr_module.AttachMuonSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachMuonTrigSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachElectronSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachTauSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachElectronTrigSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachPdfWeights = lambda *args, **kwargs: None
    topeft_corr_module.AttachScaleWeights = corrections_module.AttachScaleWeights
    topeft_corr_module.AttachPSWeights = corrections_module.AttachPSWeights
    topeft_corr_module.ApplyMuonPtCorr = lambda *args, **kwargs: args[1]
    topeft_corr_module.ApplyTES = lambda *args, **kwargs: (args[1], args[1])
    topeft_corr_module.ApplyTESSystematic = lambda *args, **kwargs: (args[1], args[1])
    topeft_corr_module.ApplyFESSystematic = lambda *args, **kwargs: (args[1], args[1])
    topeft_corr_module.AttachPerLeptonFR = lambda *args, **kwargs: None
    topeft_corr_module.ApplyRochesterCorrections = (
        lambda mu, year, isData: mu
    )
    topeft_corr_module.GetTriggerSF = lambda *args, **kwargs: np.ones(1)
    topeft_corr_module.ApplyJetVetoMaps = lambda *args, **kwargs: args[1]

    channel_metadata_module = _module("topeft.modules.channel_metadata")

    class _ChannelMetadataHelper:
        def __init__(self, *args, **kwargs):
            pass

        def build_channel_mapping(self, *args, **kwargs):
            return {}

        def get_channel(self, *args, **kwargs):
            return {}

    channel_metadata_module.ChannelMetadataHelper = _ChannelMetadataHelper

    btag_module = _module("topeft.modules.btag_weights")
    btag_module.register_btag_sf_weights = lambda *args, **kwargs: None

    # Selection helpers
    te_obj_module = _module("topeft.modules.object_selection")

    class _DummyLeptonSelection:
        def coneptElec(self, ele):
            return ak.fill_none(ele.pt, 0)

        def coneptMuon(self, mu):
            return ak.fill_none(mu.pt, 0)

        def isPresElec(self, ele):
            return True

        def isLooseElec(self, ele):
            return True

        def isFOElec(self, ele, year):
            return True

        def tightSelElec(self, ele):
            return True

        def isPresMuon(self, mu):
            return True

        def isLooseMuon(self, mu):
            return True

        def isFOMuon(self, mu, year):
            return True

        def tightSelMuon(self, mu):
            return True

    te_obj_module.run2leptonselection = lambda: _DummyLeptonSelection()
    te_obj_module.run3leptonselection = lambda: _DummyLeptonSelection()
    te_obj_module.ttH_idEmu_cuts_E3 = lambda *args, **kwargs: ak.Array([0])
    te_obj_module.isPresTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isClean = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isGood = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isVLooseTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isLooseTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.iseTightTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.ismTightTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isFwdJet = lambda *args, **kwargs: ak.Array([False])

    tc_obj_module = _module("topcoffea.modules.object_selection")
    tc_obj_module.is_tight_jet = (
        lambda pt, eta, jet_id, pt_cut=30.0, eta_cut=2.5, id_cut=2: ak.ones_like(pt, dtype=bool)
    )

    te_evt_module = _module("topeft.modules.event_selection")
    te_evt_module.add1lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is1l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.add2lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is2l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.add3lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is3l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.add4lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is4l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.addLepCatMasks = lambda events: events.__setitem__("is_e", ak.ones_like(events.nom, dtype=bool))

    tc_evt_module = _module("topcoffea.modules.event_selection")
    tc_evt_module.get_Z_peak_mask = lambda *args, **kwargs: ak.Array([True])
    tc_evt_module.get_off_Z_mask_low = lambda *args, **kwargs: ak.Array([True])
    tc_evt_module.get_any_sfos_pair = lambda *args, **kwargs: ak.Array([True])
    tc_evt_module.trg_pass_no_overlap = lambda *args, **kwargs: ak.Array([True])

    systematics_module = _module("topeft.modules.systematics")
    systematics_module.add_fake_factor_weights = lambda *args, **kwargs: None
    systematics_module.apply_theory_weight_variations = lambda **kwargs: {}
    systematics_module.register_lepton_sf_weight = lambda *args, **kwargs: None
    systematics_module.register_trigger_sf_weight = lambda *args, **kwargs: None
    systematics_module.register_weight_variation = lambda *args, **kwargs: None
    systematics_module.validate_data_weight_variations = lambda *args, **kwargs: None


@pytest.fixture
def processor(monkeypatch, tmp_path):
    # Ensure we import the processor with the test stubs.
    for module_name in list(sys.modules):
        if module_name.startswith("analysis.topeft_run2.analysis_processor"):
            sys.modules.pop(module_name)

    _install_stubs(monkeypatch)

    # Create minimal golden JSON
    golden_json = tmp_path / "golden.json"
    golden_json.write_text(json.dumps({}), encoding="utf-8")

    analysis_processor = importlib.import_module("analysis.topeft_run2.analysis_processor")

    def _fake_combinations(array, *args, **kwargs):
        return ak.Array([[{"l0": 0.0, "l1": 0.0}]])[:, :0]

    monkeypatch.setattr(analysis_processor.ak, "combinations", _fake_combinations)
    monkeypatch.setattr(analysis_processor.ak, "with_name", lambda array, name: array)

    sample = {
        "isData": True,
        "histAxisName": "DummyData",
        "year": "2018",
        "xsec": 1.0,
        "nSumOfWeights": 1.0,
        "WCnames": [],
        "path": "store/data/Run2018A-UL/RAW",
    }

    hist_keys = {
        "nominal": (
            (
                "observable",
                "3l_p_offZ_1b_2j",
                "isSR_3l",
                "DummyDataset",
                "nominal",
            ),
        )
    }

    var_info = {
        "label": "Dummy",
        "regular": (1, 0.0, 1.0),
        "definition": 'np.ones_like(events["event"])',
    }

    channel_dict = {
        "jet_selection": "exactly_2j",
        "chan_def_lst": ["3l_p_offZ_1b"],
        "lep_flav_lst": ["eee"],
        "appl_region": "isSR_3l",
        "features": (),
    }

    processor = analysis_processor.AnalysisProcessor(
        sample=sample,
        wc_names_lst=[],
        hist_keys=hist_keys,
        var_info=var_info,
        ecut_threshold=None,
        do_errors=False,
        split_by_lepton_flavor=False,
        channel_dict=channel_dict,
        golden_json_path=str(golden_json),
        available_systematics={
            "object": (),
            "weight": (),
            "theory": (),
            "data_weight": (),
        },
    )

    return processor


def _build_minimal_events(metadata=None):
    events = _EventNamespace()
    events.metadata = metadata if metadata is not None else {"dataset": "DummyDataset"}
    events.caches = [dict()]
    events.run = ak.Array([1])
    events.luminosityBlock = ak.Array([1])
    events.event = ak.Array([1])

    events.MET = ak.Array([{"pt": 40.0, "phi": 0.0}])
    electron_template = ak.Array(
        [
            [
                {
                    "pt": 30.0,
                    "eta": 0.1,
                    "phi": 0.0,
                    "mass": 0.0005,
                    "hoe": 0.1,
                    "deltaEtaSC": 0.0,
                    "eInvMinusPInv": 0.0,
                    "sieie": 0.0,
                    "matched_jet": {"btagDeepFlavB": 0.1},
                    "matched_gen": {"pdgId": 11},
                    "jetIdx": 0,
                    "charge": 1,
                }
            ]
        ]
    )
    muon_template = ak.Array(
        [
            [
                {
                    "pt": 25.0,
                    "eta": 0.2,
                    "phi": 0.1,
                    "mass": 0.105,
                    "matched_jet": {"btagDeepFlavB": 0.05},
                    "matched_gen": {"pdgId": 13},
                    "jetIdx": 1,
                    "charge": -1,
                }
            ]
        ]
    )
    events.Electron = electron_template[:, :0]
    events.Muon = muon_template[:, :0]
    events.Tau = ak.Array(
        [
            [
                {
                    "pt": 20.0,
                    "eta": 0.3,
                    "phi": 0.2,
                    "mass": 1.77,
                    "dxy": 0.0,
                    "dz": 0.0,
                    "idDeepTau2017v2p1VSjet": 1,
                    "idDeepTau2017v2p1VSe": 1,
                    "idDeepTau2017v2p1VSmu": 1,
                    "decayMode": 0,
                }
            ]
        ]
    )
    events.Jet = ak.Array(
        [
            [
                {
                    "pt": 50.0,
                    "eta": 0.5,
                    "phi": 0.3,
                    "mass": 5.0,
                    "rawFactor": 0.0,
                    "jetId": 6,
                    "btagDeepFlavB": 0.2,
                    "hadronFlavour": 0,
                },
                {
                    "pt": 45.0,
                    "eta": -0.4,
                    "phi": -0.2,
                    "mass": 4.0,
                    "rawFactor": 0.0,
                    "jetId": 6,
                    "btagDeepFlavB": 0.05,
                    "hadronFlavour": 1,
                },
            ]
        ]
    )
    events.fixedGridRhoFastjetAll = ak.Array([10.0])
    return events


def test_process_nominal_run_is_quiet(processor, capsys, monkeypatch):
    events = _build_minimal_events(metadata=types.MappingProxyType({"dataset": "DummyDataset"}))

    def _fake_process(self, events):
        dataset = events.metadata["dataset"]
        self._debug(
            "Processing variation '%s' (type: %s, base: %s)",
            "nominal",
            None,
            None,
        )
        self._debug(
            "Variation group mapping for '%s': mapping=%s key=%s info=%s",
            "nominal",
            {},
            (),
            {},
        )
        self._debug(
            "Filling histograms for channel '%s' (base '%s') with cuts %s",
            self.channel,
            self.channel,
            {},
        )
        self._debug(
            "Filled histkey %s with %d selected events",
            (self.var, self.channel, self.appregion, dataset, self.syst),
            0,
        )
        return {dataset: 0}

    monkeypatch.setattr(processor, "process", _fake_process.__get__(processor, type(processor)))

    result = processor.process(events)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert result == {"DummyDataset": 0}


def test_metadata_to_mapping_handles_various_inputs():
    analysis_processor = importlib.import_module("analysis.topeft_run2.analysis_processor")

    mapping_proxy = types.MappingProxyType({"dataset": "proxy", "value": 1})
    sequence_items = [("dataset", "sequence"), ("value", 2)]
    namespace = types.SimpleNamespace(dataset="namespace", value=3, helper=lambda: None)

    metadata_from_proxy = analysis_processor.AnalysisProcessor._metadata_to_mapping(mapping_proxy)
    metadata_from_sequence = analysis_processor.AnalysisProcessor._metadata_to_mapping(sequence_items)
    metadata_from_namespace = analysis_processor.AnalysisProcessor._metadata_to_mapping(namespace)

    assert metadata_from_proxy["dataset"] == "proxy"
    assert metadata_from_sequence["dataset"] == "sequence"
    assert metadata_from_namespace["dataset"] == "namespace"
    assert "helper" not in metadata_from_namespace
