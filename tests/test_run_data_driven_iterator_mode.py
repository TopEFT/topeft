import gzip
import importlib.util
from pathlib import Path

import cloudpickle
import numpy as np
import pytest


def _load_run_data_driven_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "analysis" / "topeft_run2" / "run_data_driven.py"
    spec = importlib.util.spec_from_file_location("run_data_driven", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


run_data_driven = _load_run_data_driven_module()

REFERENCE_GOLDEN = Path("/scratch365/apiccine/2023CRs_flips_ftau_SFs_merged_2023_np.pkl.gz")
REFERENCE_CANDIDATE = Path("/scratch365/apiccine/2023CRs_flips_ftau_SFs_merged_2023_np_codex.pkl.gz")


class FakeHist:
    def __init__(self, processes):
        self._processes = list(processes)
        self.axes = {"process": tuple(self._processes)}

    def remove(self, axis_name, labels):
        assert axis_name == "process"
        keep = [p for p in self._processes if p not in set(labels)]
        return FakeHist(keep)


class IteratorDummyProducer:
    output_hist = {}
    calls = []
    get_calls = 0
    iter_calls = 0

    def __init__(self, inputHist, outputName, iterator_mode=False):
        self.inputHist = inputHist
        self.outputName = outputName
        self.iterator_mode = iterator_mode
        IteratorDummyProducer.calls.append((inputHist, outputName, iterator_mode))

    def getDataDrivenHistogram(self):
        IteratorDummyProducer.get_calls += 1
        return IteratorDummyProducer.output_hist

    def iter_data_driven_histograms(self):
        IteratorDummyProducer.iter_calls += 1
        yield from IteratorDummyProducer.output_hist.items()


@pytest.fixture(autouse=True)
def clear_dummy_state():
    IteratorDummyProducer.output_hist = {}
    IteratorDummyProducer.calls.clear()
    IteratorDummyProducer.get_calls = 0
    IteratorDummyProducer.iter_calls = 0
    yield
    IteratorDummyProducer.output_hist = {}
    IteratorDummyProducer.calls.clear()
    IteratorDummyProducer.get_calls = 0
    IteratorDummyProducer.iter_calls = 0


def _load_pkl(pkl_path: Path):
    with gzip.open(pkl_path, "rb") as stream:
        return cloudpickle.load(stream)


def _write_with_cloudpickle(path: str, payload):
    with gzip.open(path, "wb") as stream:
        cloudpickle.dump(payload, stream)


def test_default_mode_uses_streaming_writer(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"

    IteratorDummyProducer.output_hist = {"njets": FakeHist(["flipsUL18", "ttbarUL18"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", IteratorDummyProducer)

    captured = {}

    def fake_dump_streaming(out_name, items, **_kwargs):
        payload = dict(items)
        captured["payload"] = payload
        captured["kwargs"] = _kwargs
        _write_with_cloudpickle(out_name, payload)

    monkeypatch.setattr(run_data_driven.utils, "dump_dict_streaming", fake_dump_streaming)
    monkeypatch.setattr(
        run_data_driven.utils,
        "dump_to_pkl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dict writer should not be used")),
    )

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--only-flips",
        ]
    )

    assert IteratorDummyProducer.calls == [(str(input_path), str(output_path), True)]
    assert IteratorDummyProducer.get_calls == 0
    assert IteratorDummyProducer.iter_calls == 1
    assert "payload" in captured
    assert captured["kwargs"].get("protocol") == 3
    assert captured["kwargs"].get("clear_memo_interval") == 1
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL18"]


def test_default_mode_matches_legacy_dict_mode(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_default_mode = tmp_path / "default_mode.pkl.gz"
    output_legacy_mode = tmp_path / "legacy_mode.pkl.gz"

    IteratorDummyProducer.output_hist = {
        "njets": FakeHist(["flipsUL18", "nonpromptUL18"]),
        "ht": FakeHist(["flipsUL18", "ttbarUL18"]),
    }
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", IteratorDummyProducer)
    monkeypatch.setattr(
        run_data_driven.utils,
        "dump_dict_streaming",
        lambda out_name, items, **_kwargs: _write_with_cloudpickle(out_name, dict(items)),
    )

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_default_mode),
        ]
    )
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_legacy_mode),
            "--legacy-dict-mode",
        ]
    )

    assert IteratorDummyProducer.calls == [
        (str(input_path), str(output_default_mode), True),
        (str(input_path), str(output_legacy_mode), False),
    ]

    default_mode = _load_pkl(output_default_mode)
    legacy_mode = _load_pkl(output_legacy_mode)

    assert set(default_mode.keys()) == set(legacy_mode.keys())
    for key in default_mode:
        assert list(default_mode[key].axes["process"]) == list(legacy_mode[key].axes["process"])


def compare_histogram_pickles(reference_path: Path, candidate_path: Path, *, rtol=1e-12, atol=1e-12):
    reference = _load_pkl(reference_path)
    candidate = _load_pkl(candidate_path)

    assert set(reference.keys()) == set(candidate.keys()), "Histogram-key mismatch"

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    bitwise_equal = True

    for key in reference:
        href = reference[key]
        hcan = candidate[key]

        ref_axes = [axis.name for axis in href.axes]
        can_axes = [axis.name for axis in hcan.axes]
        assert ref_axes == can_axes, f"Axis-name mismatch for {key}"
        for axis_name in ref_axes:
            assert list(href.axes[axis_name]) == list(hcan.axes[axis_name]), (
                f"Axis-category mismatch for {key}:{axis_name}"
            )

        ref_view = href.view(as_dict=True, flow=True)
        can_view = hcan.view(as_dict=True, flow=True)
        assert set(ref_view.keys()) == set(can_view.keys()), f"Bin-key mismatch for {key}"

        for bin_key in ref_view:
            ref_vals = np.asarray(ref_view[bin_key])
            can_vals = np.asarray(can_view[bin_key])
            abs_diff = np.abs(ref_vals - can_vals)
            local_max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
            denom = np.maximum(np.abs(ref_vals), atol)
            rel_diff = abs_diff / denom
            local_max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0
            max_abs_diff = max(max_abs_diff, local_max_abs)
            max_rel_diff = max(max_rel_diff, local_max_rel)
            if not np.allclose(ref_vals, can_vals, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"Value mismatch for {key}:{bin_key}; "
                    f"max_abs={local_max_abs:.3e}, max_rel={local_max_rel:.3e}"
                )
            bitwise_equal = bitwise_equal and np.array_equal(ref_vals, can_vals)

        ref_sumw2 = getattr(href, "_sumw2", None)
        can_sumw2 = getattr(hcan, "_sumw2", None)
        if ref_sumw2 is None or can_sumw2 is None:
            assert ref_sumw2 is can_sumw2, f"sumw2 presence mismatch for {key}"
            continue

        assert set(ref_sumw2.keys()) == set(can_sumw2.keys()), f"sumw2-key mismatch for {key}"
        for sw2_key in ref_sumw2:
            ref_arr = ref_sumw2[sw2_key]
            can_arr = can_sumw2[sw2_key]
            if ref_arr is None or can_arr is None:
                assert ref_arr is can_arr, f"sumw2 None mismatch for {key}:{sw2_key}"
                continue
            ref_vals = np.asarray(ref_arr)
            can_vals = np.asarray(can_arr)
            abs_diff = np.abs(ref_vals - can_vals)
            local_max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
            denom = np.maximum(np.abs(ref_vals), atol)
            rel_diff = abs_diff / denom
            local_max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0
            max_abs_diff = max(max_abs_diff, local_max_abs)
            max_rel_diff = max(max_rel_diff, local_max_rel)
            if not np.allclose(ref_vals, can_vals, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"sumw2 mismatch for {key}:{sw2_key}; "
                    f"max_abs={local_max_abs:.3e}, max_rel={local_max_rel:.3e}"
                )
            bitwise_equal = bitwise_equal and np.array_equal(ref_vals, can_vals)

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "bitwise_equal": bitwise_equal,
    }


@pytest.mark.skipif(
    not (REFERENCE_GOLDEN.exists() and REFERENCE_CANDIDATE.exists()),
    reason="Reference/candidate pickle pair is not available on this machine",
)
def test_iterator_output_matches_golden_reference():
    metrics = compare_histogram_pickles(
        REFERENCE_GOLDEN,
        REFERENCE_CANDIDATE,
        rtol=1e-12,
        atol=1e-12,
    )
    assert metrics["max_abs_diff"] <= 1e-12
    assert metrics["max_rel_diff"] <= 1e-12
