import hist
from topcoffea.modules.sparseHist import SparseHist

import numpy as np
import awkward as ak

nbins = 12
data_ptz = np.arange(0, 600, 600 / nbins)


def make_hist():
    h = SparseHist(
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.Regular(nbins, 0, 600, name="ptz"),
    )
    h.fill(process="ttH", channel="ch0", ptz=data_ptz)

    return h


def test_simple_fill():
    h = make_hist()

    # expect one count per bin
    ones = np.ones((1, 1, nbins))
    values = h.values(flow=False)
    assert ak.all(values == ak.Array(ones))

    # expect one count per bin, plus 0s for overflow
    ones_with_flow = np.zeros((1, 1, nbins + 2))
    ones_with_flow[0, 0, 1:-1] += ones[0, 0, :]
    values = h.values(flow=True)
    assert ak.all(values == ones_with_flow)


def test_integrate():
    h = make_hist()
    values = h.values()

    h.fill(process="ttH", channel="ch1", ptz=data_ptz)

    h2 = h[{"channel": sum}]

    assert ak.all(h2.values() == values[0, :] * 2)


def test_slice():
    h = make_hist()
    h.fill(process="ttH", channel="ch1", ptz=data_ptz * 0.5)

    h0 = h[{"channel": "ch0"}]
    h1 = h[{"channel": "ch1"}]

    assert ak.all(h.values()[:, 0, :] == h0.values())
    assert ak.all(h.values()[:, 1, :] == h1.values())
    assert ak.sum(h.values()) == ak.sum(h0.values()) + ak.sum(h1.values())


def test_remove():
    h = make_hist()
    h.fill(process="ttH", channel="ch1", ptz=data_ptz * 0.5)

    ha = h[{"channel": ["ch1"]}]

    hr = h.remove("channel", ["ch0"])

    assert ak.all(hr.values() == ha.values())


def test_flow():
    h = make_hist()

    flowed = np.array([-10000, -10000, -10000, 10000, 10000, 10000, 10000])
    h.fill(process="ttH", channel="ch0", ptz=flowed)

    # expect one count per bin, plus the overflow
    ones_with_flow = np.ones((1, 1, nbins + 2))
    ones_with_flow[0, 0, 0] = np.count_nonzero(flowed < 0)
    ones_with_flow[0, 0, -1] = np.count_nonzero(flowed > 1000)

    values = h.values(flow=True)
    assert(ak.all(values == ones_with_flow))


def test_addition():
    h = make_hist()

    flowed = np.array([-10000, -10000, -10000, 10000, 10000, 10000, 10000])
    h.fill(process="ttH", channel="ch0", ptz=flowed)

    values = h.values(flow=True)
    values2 = values * 2

    h2 = h + h
    assert(ak.all(h2.values(flow=True) == values2))


def test_scale():
    h = make_hist()
    values = h.values(flow=True)

    h *= 3
    h12 = 4 * h
    values12 = values * 12

    assert(ak.all(h12.values(flow=True) == values12))
