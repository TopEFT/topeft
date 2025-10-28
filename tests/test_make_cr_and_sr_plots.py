import warnings

import numpy as np
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


class _DummyHist:
    def __init__(self):
        self.scale_factors = []

    def eval(self, _):
        return {"sample": np.zeros(4)}

    def scale(self, factor):
        self.scale_factors.append(factor)


def test_unit_normalization_skips_empty_histograms(monkeypatch):
    dummy_mc = _DummyHist()
    dummy_data = _DummyHist()

    def _stop_after_normalization(*args, **kwargs):
        raise RuntimeError("stop-after-normalization")

    monkeypatch.setattr(make_cr_and_sr_plots.plt, "subplots", _stop_after_normalization)

    logged_messages = []

    def _capture_warning(msg, *args, **kwargs):
        if args:
            msg = msg % args
        logged_messages.append(msg)

    monkeypatch.setattr(make_cr_and_sr_plots.logger, "warning", _capture_warning)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeError, match="stop-after-normalization"):
            make_cr_and_sr_plots.make_region_stacked_ratio_fig(
                h_mc=dummy_mc,
                h_data=dummy_data,
                unit_norm_bool=True,
                bins=np.array([0.0, 1.0]),
            )

    assert not dummy_mc.scale_factors
    assert not dummy_data.scale_factors

    assert any("Skipping MC unit normalization" in msg for msg in logged_messages)
    assert any("Skipping data unit normalization" in msg for msg in logged_messages)
