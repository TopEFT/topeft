import pathlib
import sys

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import analysis.topeft_run2.make_cr_and_sr_plots as mcp
import analysis.topeft_run2.tauFitter as tau


def _make_hist():
    histo = HistEFT(
        hist.axis.StrCategory(["a", "b", "c"], name="process", growth=True),
        hist.axis.Regular(3, 0, 3, name="score"),
        wc_names=[],
    )
    histo.fill(process="a", score=[0.5], weight=[1.0])
    histo.fill(process="b", score=[1.5], weight=[2.0])
    histo.fill(process="c", score=[2.5], weight=[4.0])
    return histo


@pytest.mark.parametrize("group_fn", [mcp.group_bins, tau.group_bins])
def test_group_bins_combines_contributions(group_fn):
    original = _make_hist()
    grouped = group_fn(
        original,
        {"combo": ["a", "b"]},
        axis_name="process",
        drop_unspecified=True,
    )

    grouped_sm = grouped.as_hist({})
    combo_values = grouped_sm[{"process": "combo"}].values()

    original_sm = original.as_hist({})
    expected = (
        original_sm[{"process": "a"}].values()
        + original_sm[{"process": "b"}].values()
    )

    np.testing.assert_allclose(combo_values, expected)
    assert list(grouped_sm.axes["process"]) == ["combo"]


@pytest.mark.parametrize("group_fn", [mcp.group_bins, tau.group_bins])
def test_group_bins_preserves_unspecified_when_requested(group_fn):
    original = _make_hist()
    grouped = group_fn(
        original,
        {"combo": ("a", "b")},
        axis_name="process",
        drop_unspecified=False,
    )

    grouped_sm = grouped.as_hist({})
    assert list(grouped_sm.axes["process"]) == ["combo", "c"]

    combo_values = grouped_sm[{"process": "combo"}].values()
    c_values = grouped_sm[{"process": "c"}].values()

    original_sm = original.as_hist({})
    np.testing.assert_allclose(
        combo_values,
        original_sm[{"process": "a"}].values()
        + original_sm[{"process": "b"}].values(),
    )
    np.testing.assert_allclose(
        c_values,
        original_sm[{"process": "c"}].values(),
    )


@pytest.mark.parametrize("group_fn", [mcp.group_bins, tau.group_bins])
def test_group_bins_raises_for_unknown_sources(group_fn):
    histo = _make_hist()
    with pytest.raises(ValueError, match="missing"):
        group_fn(
            histo,
            {"combo": ["missing"]},
            axis_name="process",
            drop_unspecified=True,
        )
