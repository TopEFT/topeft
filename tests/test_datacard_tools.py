import numpy as np
import pytest

from topeft.modules.datacard_tools import to_hist


def test_to_hist_handles_missing_sumw2_entries():
    arr = [np.array([0.0, 3.0, 4.0]), None]

    hist_obj = to_hist(arr, name="test_hist", zero_wgts=True)

    np.testing.assert_allclose(hist_obj.values(), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(hist_obj.variances(), np.zeros(2))


def test_to_hist_raises_without_zero_wgts_when_sumw2_missing():
    arr = [np.array([0.0, 3.0, 4.0]), None]

    with pytest.raises(ValueError, match="sumw2"):
        to_hist(arr, name="test_hist", zero_wgts=False)
