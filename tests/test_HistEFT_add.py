import numpy as np
import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth

import awkward as ak

# Let's generate some fake data to use for testing
wc_names_lst = [
    "cpt",
    "ctp",
    "cptb",
    "cQlMi",
    "cQq81",
    "cQq11",
    "cQl3i",
    "ctq8",
    "ctlTi",
    "ctq1",
    "ctli",
    "cQq13",
    "cbW",
    "cpQM",
    "cpQ3",
    "ctei",
    "cQei",
    "ctW",
    "ctlSi",
    "cQq83",
    "ctZ",
    "ctG"
]

nevts = 1000
wc_count = len(wc_names_lst)
ncoeffs = efth.n_quad_terms(wc_count)
rng = np.random.default_rng()
eft_fit_coeffs = rng.normal(0.3, 0.5, (nevts, ncoeffs))
eft_one_coeffs = np.ones((nevts, ncoeffs))
sums = np.sum(eft_fit_coeffs, axis=0)

a = HistEFT(
    hist.axis.StrCategory([], name="type", label="type", growth=True),
    hist.axis.Regular(1, 0, 1, name="x", label="x"),
    label="Events",
    wc_names=wc_names_lst,
)

# Just need another one where I won't fill the EFT coefficients
b = a.copy()

# Fill the EFT histogram
a.fill(type="eft", x=np.full(nevts, 0.5), eft_coeff=eft_fit_coeffs)
b.fill(type="non-eft", x=np.full(nevts, 0.5))


a_w = HistEFT(
    hist.axis.StrCategory([], name="type", label="type", growth=True),
    hist.axis.Regular(1, 0, 1, name="x", label="x"),
    label="Events",
    wc_names=wc_names_lst,
)

# Just need another one where I won't fill the EFT coefficients
b_w = a_w.copy()

# Fill the EFT histogram
weight_val = 0.9
a_w.fill(
    type="eft",
    x=np.full(nevts, 0.5),
    eft_coeff=eft_fit_coeffs,
    weight=np.full(nevts, weight_val),
)
b_w.fill(type="non-eft", x=np.full(nevts, 0.5), weight=np.full(nevts, weight_val))


def test_scale_a_weights():
    assert np.all(
        np.abs(a_w.integrate("type", "eft").view(as_dict=True)[()][0] - weight_val * sums) < 1e-10
    )

    integral = a_w.integrate("type", "eft").view(as_dict=True)[()].sum()

    assert a_w.integrate("type", "eft").eval({})[()].sum() != integral

    ones = np.ones(wc_count)
    assert np.abs(a_w.integrate("type", "eft").eval(ones)[()].sum() - integral) < 1e-10

    ones = dict(zip(wc_names_lst, np.ones(wc_count)))
    assert np.abs(a_w.integrate("type", "eft").eval(ones)[()].sum() - integral) < 1e-10



def test_ac_deepcopy():
    c_w = a_w.copy()

    assert np.all(
        a_w.integrate("type", "eft").view(as_dict=True)[()] == c_w.integrate("type", "eft").view(as_dict=True)[()]
    )
    c_w.scale(1)
    c_w.reset()

    assert ak.sum(a_w.values()) != 0
    assert ak.sum(c_w.values()) == 0


def test_group():
    c_w = a_w.group("type", "all", {"all": ["eft", "non-eft"]})
    assert (
        c_w.integrate("all").view(as_dict=True)[()].sum()
        == a_w.integrate("type").view(as_dict=True)[()].sum()
    )


def test_add_ab():
    ab = a + b

    assert np.all(ab.integrate("type", "eft").view(as_dict=True)[()][0] == sums)
    assert ab.integrate("type", "non-eft").view(as_dict=True)[()][0][0] == nevts


def test_add_ba():
    ba = b + a
    assert np.all(ba.integrate("type", "eft").view(as_dict=True)[()][0] == sums)
    assert ba.integrate("type", "non-eft").view(as_dict=True)[()][0][0] == nevts


def test_add_aba():
    ab = a + b
    aba = ab + a
    assert np.all(aba.integrate("type", "eft").view(as_dict=True)[()][0] == 2 * sums)
    assert aba.integrate("type", "non-eft").view(as_dict=True)[()][0][0] == nevts


def test_add_baa():
    ba = b + a
    baa = ba + a
    assert np.all(baa.integrate("type", "eft").view(as_dict=True)[()][0] == 2 * sums)
    assert baa.integrate("type", "non-eft").view(as_dict=True)[()][0][0] == nevts


def test_add_abb():
    ab = a + b
    abb = ab + b
    assert np.all(abb.integrate("type", "eft").view(as_dict=True)[()][0] == sums)
    assert abb.integrate("type", "non-eft").view(as_dict=True)[()][0][0] == 2 * nevts


def test_add_bab():
    ba = b + a
    bab = ba + b
    assert np.all(bab.integrate("type", "eft").view(as_dict=True)[()][0] == sums)
    assert bab.integrate("type", "non-eft").view(as_dict=True)[()][0][0] == 2 * nevts


def test_add_ab_weights():
    ab = a_w + b_w
    assert np.all(
        np.abs(ab.integrate("type", "eft").view(flow=False)[()][0] - weight_val * sums) < 1e-10
    )

    assert np.all(
        np.abs(ab.integrate("type", "non-eft").view(flow=False)[()][0][0] - weight_val * nevts) < 1e-10
    )


def test_add_ba_weights():
    ba = b_w + a_w
    assert np.all(
        np.abs(ba.integrate("type", "eft").view(flow=False)[()][0] - weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(ba.integrate("type", "non-eft").view(flow=False)[()][0][0] - weight_val * nevts) < 1e-10
    )


def test_add_aba_weights():
    ab = a_w + b_w
    aba = ab + a_w
    assert np.all(
        np.abs(aba.integrate("type", "eft").view(flow=False)[()][0] - 2 * weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(aba.integrate("type", "non-eft").view(flow=False)[()][0][0] - weight_val * nevts) < 1e-10
    )


def test_add_baa_weights():
    ba = b_w + a_w
    baa = ba + a_w
    assert np.all(
        np.abs(baa.integrate("type", "eft").view(flow=False)[()][0] - 2 * weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(baa.integrate("type", "non-eft").view(flow=False)[()][0][0] - weight_val * nevts) < 1e-10
    )


def test_add_abb_weights():
    ab = a_w + b_w
    abb = ab + b_w
    assert np.all(
        np.abs(abb.integrate("type", "eft").view(flow=False)[()][0] - weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(abb.integrate("type", "non-eft").view(flow=False)[()][0][0] - 2 * weight_val * nevts) < 1e-10
    )


def test_add_bab_weights():
    ba = b_w + a_w
    bab = ba + b_w
    assert np.all(
        np.abs(bab.integrate("type", "eft").view(flow=False)[()][0] - weight_val * sums) < 1e-10
    )

    assert np.all(
        np.abs(bab.integrate("type", "non-eft").view(flow=False)[()][0][0] - 2 * weight_val * nevts) < 1e-10
    )


def test_split_by_terms():
    integral = a.sum("type").view(as_dict=True)[()].sum()
    c = a.split_by_terms(["x"], "type")
    assert (
        integral
        == c.integrate("type", [k[0] for k in c.view(as_dict=True) if "eft" not in k[0]])
        .view(as_dict=True)[()]
        .sum()
    )
