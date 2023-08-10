import numpy as np
import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth

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

# Just need another one like the one above
b = HistEFT(
    hist.axis.StrCategory([], name="type", label="type", growth=True),
    hist.axis.Regular(1, 0, 1, name="x", label="x"),
    wc_names=wc_names_lst,
    label="Events",
)

# Fill the EFT histogram
a.fill(type="eft_a", x=np.full(nevts, 0.5), eft_coeff=eft_fit_coeffs)
b.fill(type="eft_b", x=np.full(nevts, 0.5), eft_coeff=eft_one_coeffs)


def test_add_ab_noerrors():
    ab = a + b
    assert np.all(ab.integrate("type", "eft_a").view(flow=False)[0] == sums)
    assert np.all(ab.integrate("type", "eft_b").view(flow=False)[0] == nevts)


def test_add_ba_noerrors():
    ba = a + b
    assert np.all(ba.integrate("type", "eft_a").view(flow=False)[0] == sums)
    assert np.all(ba.integrate("type", "eft_b").view(flow=False)[0] == nevts)


def test_add_aba_noerrors():
    ab = a + b
    aba = ab + a
    assert np.all(aba.integrate("type", "eft_a").view(flow=False)[0] == 2 * sums)
    assert np.all(aba.integrate("type", "eft_b").view(flow=False)[0] == nevts)


def test_add_baa_noerrors():
    ba = b + a
    baa = ba + a
    assert np.all(baa.integrate("type", "eft_a").view(flow=False)[0] == 2 * sums)
    assert np.all(baa.integrate("type", "eft_b").view(flow=False)[0] == nevts)


def test_add_abb_noerrors():
    ab = a + b
    abb = ab + b
    assert np.all(abb.integrate("type", "eft_a").view(flow=False)[0] == sums)
    assert np.all(abb.integrate("type", "eft_b").view(flow=False)[0] == 2 * nevts)


def test_add_bab_noerrors():
    ba = b + a
    bab = ba + b
    assert np.all(bab.integrate("type", "eft_a").view(flow=False)[0] == sums)
    assert np.all(bab.integrate("type", "eft_b").view(flow=False)[0] == 2 * nevts)


a_w = HistEFT(
    hist.axis.StrCategory([], name="type", label="type", growth=True),
    hist.axis.Regular(1, 0, 1, name="x", label="x"),
    label="Events",
    wc_names=wc_names_lst,
)

# Just need another one like the one above
b_w = HistEFT(
    hist.axis.StrCategory([], name="type", label="type", growth=True),
    hist.axis.Regular(1, 0, 1, name="x", label="x"),
    label="Events",
    wc_names=wc_names_lst,
)

# Fill the EFT histogram
weight_val = 0.9
a_w.fill(
    type="eft_a",
    x=np.full(nevts, 0.5),
    eft_coeff=eft_fit_coeffs,
    weight=np.full(nevts, weight_val),
)
b_w.fill(
    type="eft_b",
    x=np.full(nevts, 0.5),
    eft_coeff=eft_one_coeffs,
    weight=np.full(nevts, weight_val),
)


def test_scale_a_weights():
    ones = dict(zip(wc_names_lst, np.ones(wc_count)))
    zeros = dict(zip(wc_names_lst, np.zeros(wc_count)))

    intf = a_w.integrate("type", "eft_a").view(flow=True)[1]
    assert intf[0] == 0
    assert intf[-1] == 0
    assert np.all(np.abs(intf[1:-1] - weight_val * sums) < 1e-10)  # drop under and overflow bins

    integral = a_w.integrate("type", "eft_a").eval({})[()]
    assert np.all(
        np.abs(
            a_w.integrate("type", "eft_a").eval(zeros)[()] == integral
        )
    )

    assert np.any(
        np.abs(
            a_w.integrate("type", "eft_a").eval(ones)[()] != integral
        )
    )

    a_w.integrate("type", "eft_a").eval(ones)[()].sum() - weight_val * sums.sum() < 1e-10


def test_ac_deepcopy():
    c_w = a_w.copy(deep=True)
    assert np.all(
        a_w.integrate("type", "eft_a").view(flow=False)[0] == c_w.integrate("type", "eft_a").view(flow=False)[0]
    )
    c_w.scale(1)
    c_w.reset()
    assert np.all(c_w.view(flow=False)[0] == 0)


def test_group():
    ab = a + b
    c = ab.group("type", "all", {"efts": ["eft_a", "eft_b"]})
    assert (
        c.integrate("all", "efts").eval({})[()].sum()
        == a.integrate("type", "eft_a").eval({})[()].sum()
        + b.integrate("type", "eft_b").eval({})[()].sum()
    )


def test_add_ab_weights():
    ab = a_w + b_w
    assert np.all(
        np.abs(ab.integrate("type", "eft_a").view(flow=False)[0] - weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(ab.integrate("type", "eft_b").view(flow=False)[0] - nevts * weight_val) < 1e-10
    )


def test_add_ba_weights():
    ba = b_w + a_w
    assert np.all(
        np.abs(ba.integrate("type", "eft_a").view(flow=False)[0] - weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(ba.integrate("type", "eft_b").view(flow=False)[0] - nevts * weight_val) < 1e-10
    )


def test_add_aba_weights():
    ab = a_w + b_w
    aba = ab + a_w
    assert np.all(
        np.abs(aba.integrate("type", "eft_a").view(flow=False)[0] - 2 * weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(aba.integrate("type", "eft_b").view(flow=False)[0] - nevts * weight_val) < 1e-10
    )


def test_add_baa_weights():
    ba = b_w + a_w
    baa = ba + a_w
    assert np.all(
        np.abs(baa.integrate("type", "eft_a").view(flow=False)[0] - 2 * weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(baa.integrate("type", "eft_b").view(flow=False)[0] - nevts * weight_val) < 1e-10
    )


def test_add_abb_weights():
    ab = a_w + b_w
    abb = ab + b_w
    assert np.all(
        np.abs(abb.integrate("type", "eft_a").view(flow=False)[0] - weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(abb.integrate("type", "eft_b").view(flow=False)[0] - 2 * nevts * weight_val) < 1e-10
    )


def test_add_bab_weights():
    ba = b_w + a_w
    bab = ba + b_w
    assert np.all(
        np.abs(bab.integrate("type", "eft_a").view(flow=False)[0] - weight_val * sums) < 1e-10
    )
    assert np.all(
        np.abs(bab.integrate("type", "eft_b").view(flow=False)[0] - 2 * nevts * weight_val) < 1e-10
    )


def test_split_by_terms():
    integral = a[{'type': sum}]
    c = a.split_by_terms(['x'], 'type')
    assert integral == c.integrate('type', [k[0] for k in c.view(flow=True) if 'eft' not in k[0]]).view(flow=True)[()].sum()
