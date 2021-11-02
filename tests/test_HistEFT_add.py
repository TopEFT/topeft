import numpy as np
from coffea import hist
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth

# Let's generate some fake data to use for testing
nevts = 1000
rng = np.random.default_rng()
eft_fit_coeffs = rng.normal(0.3, 0.5, (nevts,276))
sums = np.sum(eft_fit_coeffs, axis=0)

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

a = HistEFT("Events", wc_names_lst,
                   hist.Cat("type", "type"),
                   hist.Bin("x",  "x", 1, 0, 1))
# Just need another one where I won't fill the EFT coefficients
b = a.copy(content=False)

# Fill the EFT histogram
a.fill(type='eft', x=np.full(nevts,0.5), eft_coeff=eft_fit_coeffs)
b.fill(type='non-eft', x=np.full(nevts,0.5))

def test_scale_a_weights():
    assert np.all(np.abs(a_w.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    integral = a_w.integrate('type','eft').values()[()].sum()
    a_w.set_wilson_coeff_from_array(np.ones(a_w._nwc))
    assert a_w.integrate('type','eft').values()[()].sum() != integral
    assert np.all(np.abs(a_w.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    ones = dict(zip(wc_names_lst, np.ones(a_w._nwc)))
    a_w.set_wilson_coefficients(**ones)
    assert a_w.integrate('type','eft').values()[()].sum() != integral #FIXME we should compute the actual values
    a_w.set_sm()
    assert a_w.integrate('type','eft').values()[()].sum() == integral

def test_ac_deepcopy():
    c_w = a_w.copy(content=True)
    assert np.all(a_w.integrate('type','eft')._sumw[()] == c_w.integrate('type','eft')._sumw[()])
    c_w.scale(1)
    c_w.clear()
    assert c_w._sumw == {}
    assert c_w._sumw2 ==  None

def test_group():
    c_w = a_w.group('type', hist.Cat('all', 'all'), {'all': ['eft', 'non-eft']})
    assert c_w.integrate('all').values()[()].sum() == a_w.integrate('type').values()[()].sum()

def test_add_ab_noerrors():
    ab = a + b
    assert np.all(ab.integrate('type','eft')._sumw[()][1] == sums) 
    assert ab.integrate('type','eft')._sumw2 is None
    assert ab.integrate('type','non-eft')._sumw[()][1] == nevts 
    assert ab.integrate('type','non-eft')._sumw2 is None

def test_add_ba_noerrors():
    ba = b + a
    assert np.all(ba.integrate('type','eft')._sumw[()][1] == sums) 
    assert ba.integrate('type','eft')._sumw2 is None
    assert ba.integrate('type','non-eft')._sumw[()][1] == nevts 
    assert ba.integrate('type','non-eft')._sumw2 is None

def test_add_aba_noerrors():
    ab = a + b
    aba = ab + a
    assert np.all(aba.integrate('type','eft')._sumw[()][1] == 2*sums) 
    assert aba.integrate('type','eft')._sumw2 is None
    assert aba.integrate('type','non-eft')._sumw[()][1] == nevts 
    assert aba.integrate('type','non-eft')._sumw2 is None

def test_add_baa_noerrors():
    ba = b + a
    baa = ba + a
    assert np.all(baa.integrate('type','eft')._sumw[()][1] == 2*sums) 
    assert baa.integrate('type','eft')._sumw2 is None
    assert baa.integrate('type','non-eft')._sumw[()][1] == nevts 
    assert baa.integrate('type','non-eft')._sumw2 is None

def test_add_abb_noerrors():
    ab = a + b
    abb = ab + b
    assert np.all(abb.integrate('type','eft')._sumw[()][1] == sums) 
    assert abb.integrate('type','eft')._sumw2 is None
    assert abb.integrate('type','non-eft')._sumw[()][1] == 2*nevts 
    assert abb.integrate('type','non-eft')._sumw2 is None

def test_add_bab_noerrors():
    ba = b + a
    bab = ba + b
    assert np.all(bab.integrate('type','eft')._sumw[()][1] == sums) 
    assert bab.integrate('type','eft')._sumw2 is None
    assert bab.integrate('type','non-eft')._sumw[()][1] == 2*nevts 
    assert bab.integrate('type','non-eft')._sumw2 is None
    

a_w = HistEFT("Events", wc_names_lst,
              hist.Cat("type", "type"),
              hist.Bin("x",  "x", 1, 0, 1))
# Just need another one where I won't fill the EFT coefficients
b_w = a_w.copy(content=False)

# Fill the EFT histogram
weight_val = 0.9
a_w.fill(type='eft', x=np.full(nevts,0.5), eft_coeff=eft_fit_coeffs, weight=np.full(nevts,weight_val))
b_w.fill(type='non-eft', x=np.full(nevts,0.5), weight=np.full(nevts,weight_val))

def test_add_ab_weights():
    ab = a_w + b_w
    assert np.all(np.abs(ab.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert ab.integrate('type','eft')._sumw2[()] is None
    assert abs(ab.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(ab.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_ba_weights():
    ba = b_w + a_w
    assert np.all(np.abs(ba.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert ba.integrate('type','eft')._sumw2[()] is None
    assert abs(ba.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(ba.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_aba_weights():
    ab = a_w + b_w
    aba = ab + a_w
    assert np.all(np.abs(aba.integrate('type','eft')._sumw[()][1] - 2*weight_val*sums) < 1e-10) 
    assert aba.integrate('type','eft')._sumw2[()] is None
    assert abs(aba.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(aba.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_baa_weights():
    ba = b_w + a_w
    baa = ba + a_w
    assert np.all(np.abs(baa.integrate('type','eft')._sumw[()][1] - 2*weight_val*sums) < 1e-10) 
    assert baa.integrate('type','eft')._sumw2[()] is None
    assert abs(baa.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(baa.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_abb_weights():
    ab = a_w + b_w
    abb = ab + b_w
    assert np.all(np.abs(abb.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert abb.integrate('type','eft')._sumw2[()] is None
    assert abs(abb.integrate('type','non-eft')._sumw[()][1] - 2*nevts*weight_val) < 1e-10
    assert abs(abb.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*2*nevts) < 1e-10

def test_add_bab_weights():
    ba = b_w + a_w
    bab = ba + b_w
    assert np.all(np.abs(bab.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert bab.integrate('type','eft')._sumw2[()] is None
    assert abs(bab.integrate('type','non-eft')._sumw[()][1] - 2*nevts*weight_val) < 1e-10
    assert abs(bab.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*2*nevts) < 1e-10

a_e = HistEFT("Events", wc_names_lst,
              hist.Cat("type", "type"),
              hist.Bin("x",  "x", 1, 0, 1))
# Just need another one where I won't fill the EFT coefficients
b_e = a_e.copy(content=False)

#Now let's do the sum of the weights squared quartic too
eft_w2_coeffs = efth.calc_w2_coeffs(eft_fit_coeffs)
sums_w2 = np.sum(eft_w2_coeffs, axis=0)

# Fill the EFT histogram
a_e.fill(type='eft', x=np.full(nevts,0.5), eft_coeff=eft_fit_coeffs, eft_err_coeff=eft_w2_coeffs,
         weight=np.full(nevts,weight_val))
b_e.fill(type='non-eft', x=np.full(nevts,0.5), weight=np.full(nevts,weight_val))


def test_add_ab_errors():
    ab = a_e + b_e
    assert np.all(np.abs(ab.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert np.all(np.abs(ab.integrate('type','eft')._sumw2[()][1] - (weight_val**2)*sums_w2) < 1e-10) 
    assert abs(ab.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(ab.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_ba_errors():
    ba = b_e + a_e
    assert np.all(np.abs(ba.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert np.all(np.abs(ba.integrate('type','eft')._sumw2[()][1] - (weight_val**2)*sums_w2) < 1e-10) 
    assert abs(ba.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(ba.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_aba_errors():
    ab = a_e + b_e
    aba = ab + a_e
    assert np.all(np.abs(aba.integrate('type','eft')._sumw[()][1] - 2*weight_val*sums) < 1e-10) 
    assert np.all(np.abs(aba.integrate('type','eft')._sumw2[()][1] - 2*(weight_val**2)*sums_w2) < 1e-10) 
    assert abs(aba.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(aba.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_baa_errors():
    ba = b_e + a_e
    baa = ba + a_e
    assert np.all(np.abs(baa.integrate('type','eft')._sumw[()][1] - 2*weight_val*sums) < 1e-10) 
    assert np.all(np.abs(baa.integrate('type','eft')._sumw2[()][1] - 2*(weight_val**2)*sums_w2) < 1e-10) 
    assert abs(baa.integrate('type','non-eft')._sumw[()][1] - nevts*weight_val) < 1e-10
    assert abs(baa.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*nevts) < 1e-10

def test_add_abb_errors():
    ab = a_e + b_e
    abb = ab + b_e
    assert np.all(np.abs(abb.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert np.all(np.abs(abb.integrate('type','eft')._sumw2[()][1] - (weight_val**2)*sums_w2) < 1e-10) 
    assert abs(abb.integrate('type','non-eft')._sumw[()][1] - 2*nevts*weight_val) < 1e-10
    assert abs(abb.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*2*nevts) < 1e-10

def test_add_bab_errors():
    ba = b_e + a_e
    bab = ba + b_e
    assert np.all(np.abs(bab.integrate('type','eft')._sumw[()][1] - weight_val*sums) < 1e-10) 
    assert np.all(np.abs(bab.integrate('type','eft')._sumw2[()][1] - (weight_val**2)*sums_w2) < 1e-10) 
    assert abs(bab.integrate('type','non-eft')._sumw[()][1] - 2*nevts*weight_val) < 1e-10
    assert abs(bab.integrate('type','non-eft')._sumw2[()][1] - (weight_val**2)*2*nevts) < 1e-10
    
def test_split_by_terms():
    integral = a.sum('type').values()[()].sum()
    c = a.split_by_terms(['x'], 'type')
    assert integral == c.integrate('type', [k[0] for k in c.values() if 'eft' not in k[0]]).values()[()].sum()
