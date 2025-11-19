"""Fallback histogram utilities when topcoffea lacks ``hist_utils``."""
from __future__ import annotations

import gzip
import pickle
import queue
import threading
from typing import Dict, Iterator, Tuple, Union

from pickle import UnpicklingError

try:  # pragma: no cover - exercised in environments without cloudpickle
    import cloudpickle
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    cloudpickle = pickle  # type: ignore[assignment]

try:  # pragma: no cover - exercised via versioned environments
    from pickle import DICT, EMPTY_DICT, _Stop, _Unframer, _Unpickler
except ImportError:  # Python < 3.11 lacks the streaming helpers
    _STREAMING_SUPPORT = False
else:
    _STREAMING_SUPPORT = True

HAS_STREAMING_SUPPORT = _STREAMING_SUPPORT

__all__ = [
    "HAS_STREAMING_SUPPORT",
    "LazyHist",
    "get_hist_dict_non_empty",
    "iterate_hist_from_pkl",
    "iterate_histograms_from_pkl",
]


def get_hist_dict_non_empty(h: Dict[str, object]) -> Dict[str, object]:
    """Return a shallow copy of *h* that omits entries with empty histograms."""

    return {k: v for k, v in h.items() if not _is_hist_empty(v)}


class _StreamingHistDict(dict):
    """Dictionary subclass that forwards assignments to a sink function."""

    __slots__ = ("_sink",)

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def __setitem__(self, key, value):  # type: ignore[override]
        self._sink(key, value)


_QUEUE_END = object()


if HAS_STREAMING_SUPPORT:

    class _StreamingHistUnpickler(_Unpickler):
        """Unpickler that emits items as the top-level histogram dict is filled."""

        dispatch = _Unpickler.dispatch.copy()

        def __init__(self, file, *, allow_empty=True, **kwargs):
            super().__init__(file, **kwargs)
            self._allow_empty = allow_empty
            self._root_dict = None
            self._queue: "queue.Queue[Tuple[str, object] | object]" = queue.Queue(
                maxsize=1
            )
            self._stop_event = threading.Event()
            self._worker_exc: Exception | None = None

        def _should_emit(self, hist) -> bool:
            if self._allow_empty:
                return True
            return not _is_hist_empty(hist)

        def _emit(self, key, hist):
            if self._should_emit(hist):
                self._push_queue((key, hist))

        def iterate(self) -> Iterator[Tuple[str, object]]:
            worker = threading.Thread(target=self._consume_pickle, daemon=True)
            worker.start()
            try:
                while True:
                    item = self._queue.get()
                    if item is _QUEUE_END:
                        if self._worker_exc is not None:
                            raise self._worker_exc
                        return
                    yield item  # type: ignore[misc]
            finally:
                self._stop_event.set()
                worker.join()

        def _push_queue(self, value):
            while not self._stop_event.is_set():
                try:
                    self._queue.put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def _consume_pickle(self):
            try:
                self._run()
                if self._root_dict is None:
                    raise UnpicklingError(
                        "Histogram pickle did not contain a dictionary"
                    )
            except Exception as exc:  # pragma: no cover - propagated to caller
                self._worker_exc = exc
            finally:
                self._push_queue(_QUEUE_END)

        def _run(self):
            if not hasattr(self, "_file_read"):
                raise UnpicklingError(
                    "Unpickler.__init__() was not called by %s.__init__()"
                    % (self.__class__.__name__,)
                )
            self._unframer = _Unframer(self._file_read, self._file_readline)
            self.read = self._unframer.read
            self.readinto = self._unframer.readinto
            self.readline = self._unframer.readline
            self.metastack = []
            self.stack = []
            mark = self.mark
            read = self.read
            dispatch = self.dispatch
            push_mark = self.push_mark
            try:
                while True:
                    key = read(1)
                    if not key:
                        raise EOFError
                    if key[0] == b"."[0]:
                        return
                    if key[0] == b"("[0]:
                        push_mark()
                        continue
                    if key[0] == b"g"[0]:
                        markobject = mark()
                        if isinstance(markobject, tuple):
                            key = markobject[0]
                            self._emit(key, markobject[1])
                    dispatch[key[0]](self)
            except _Stop:
                return

        def _is_root_context(self) -> bool:
            return self._root_dict is None and not self.stack and not self.metastack

        def load_empty_dictionary(self):  # type: ignore[override]
            if self._is_root_context():
                root = _StreamingHistDict(self._emit)
                self._root_dict = root
                self.append(root)
            else:
                super().load_empty_dictionary()

        dispatch[EMPTY_DICT[0]] = load_empty_dictionary

        def load_dict(self):  # type: ignore[override]
            if self._is_root_context():
                items = self.pop_mark()
                root = _StreamingHistDict(self._emit)
                self._root_dict = root
                self.append(root)
                for i in range(0, len(items), 2):
                    root[items[i]] = items[i + 1]
            else:
                super().load_dict()

        dispatch[DICT[0]] = load_dict


def _is_hist_empty(hist: object) -> bool:
    empty_method = getattr(hist, "empty", None)
    if callable(empty_method):
        try:
            return bool(empty_method())
        except TypeError:
            return bool(empty_method)
    return False


class LazyHist:
    """Wrapper that defers histogram materialization until explicitly requested."""

    __slots__ = ("_payload", "_value")

    def __init__(self, payload: bytes):
        self._payload = payload
        self._value = _QUEUE_END  # sentinel reused privately

    @classmethod
    def from_hist(cls, hist: object) -> "LazyHist":
        payload = cloudpickle.dumps(hist, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(payload)

    def materialize(self) -> object:
        if self._value is _QUEUE_END:
            self._value = cloudpickle.loads(self._payload)
        return self._value

    def release(self) -> None:
        if self._value is not _QUEUE_END:
            self._value = _QUEUE_END

    def empty(self) -> bool:
        hist = self.materialize()
        return _is_hist_empty(hist)

    def unwrap(self) -> object:
        return self.materialize()


def _iterate_hist_entries(
    path_to_pkl: str, allow_empty: bool
) -> Iterator[Tuple[str, object]]:
    with gzip.open(path_to_pkl, "rb") as fin:
        if HAS_STREAMING_SUPPORT:
            streamer = _StreamingHistUnpickler(fin, allow_empty=allow_empty)
            yield from streamer.iterate()
        else:
            mapping = pickle.load(fin)
            if not isinstance(mapping, dict):
                raise UnpicklingError("Histogram pickle did not contain a dictionary")
            for key, hist in mapping.items():
                if allow_empty or not _is_hist_empty(hist):
                    yield key, hist


def iterate_histograms_from_pkl(
    path_to_pkl: str, *, allow_empty: bool = True
) -> Iterator[Tuple[str, LazyHist]]:
    """Yield ``(key, LazyHist)`` pairs for the histograms stored in *path_to_pkl*."""

    for key, hist in _iterate_hist_entries(
        path_to_pkl, allow_empty=allow_empty
    ):
        lazy = LazyHist.from_hist(hist)
        del hist
        yield key, lazy


def iterate_hist_from_pkl(
    path_to_pkl: str,
    *,
    allow_empty: bool = True,
    materialize: Union[bool, str] = False,
) -> Union[Iterator[Tuple[str, object]], Dict[str, object]]:
    """Iterate over histogram pickle entries, materializing as requested."""

    if isinstance(materialize, str):
        normalized = materialize.lower()
        if normalized not in {"lazy", "eager"}:
            raise ValueError(
                "materialize must be a boolean or one of 'lazy'/'eager'"
            )
        materialize = normalized == "eager"

    iterator = _iterate_hist_entries(path_to_pkl, allow_empty=allow_empty)
    if materialize:
        return {key: hist for key, hist in iterator}
    return iterator
