import numpy as np
import awkward as ak
from coffea import hist
from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.WCPoint import WCPoint
from topcoffea.modules.WCFit import WCFit

def fval(xvals = [], svals = []):
    # Ordering convention for the structure constants:
    # Dim=0 (0,0)
    # Dim=1 (0,0) (1,0) (1,1)
    # Dim=2 (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
    y = 0.0
    idx = 0
    for i in range(len(xvals)):
        for j in range(i+1):
            c1 = xvals[i]
            c2 = xvals[j]
            s  = svals[idx]
            y += s*c1*c2
            #print(f'{i},{j} ')
            idx += 1
    #print()
    return y

########################### WCFit unit tests ###########################

def test_wcfit():
    chk_str = ''

    unit_chk = True
    all_chks,units = [0]*2
    tolerance = 1e-4

    # The structure constants
    s00 = 1.0
    s10 = 1.5
    s11 = 1.25

    pts = []
    vals = [-1.0,1.25,0.5,2.5,4]
    for x in vals:
        y = s00*1.0 + s10*x + s11*x*x
        pts.append(WCPoint(f'EFTrwgt0_ctG_{x}',y))

    chk_x = 1.5
    chk_y = s00*1.0 + s10*chk_x + s11*chk_x*chk_x
    chk_pt = WCPoint(f'EFTrwgt0_ctG_{chk_x}',0.0)

    print('Running unit tests for WCFit class')
    all_chks = 0
    units = 0

    fit_base = WCFit(pts,'base')
    unit_chk = (abs(fit_base.EvalPoint(chk_pt) - chk_y) < tolerance)
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 1 ---')
    print('chk_x    : ', chk_x)
    print('chk_y    : ', chk_y)
    print('EvalPoint: ', fit_base.EvalPoint(chk_pt))
    print('test: ', chk_str)
    print('--------------\n')

    fit_new = WCFit()
    fit_new.SetTag('new')
    fit_new.AddFit(fit_base)
    unit_chk = (abs(fit_base.EvalPoint(chk_pt) - chk_y) < tolerance)
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 2 ---')
    print('chk_x    : ', chk_x)
    print('chk_y    : ', chk_y)
    print('EvalPoint: ', fit_new.EvalPoint(chk_pt))
    print('test: ', chk_str)
    print('--------------\n')

    fit_new.AddFit(fit_base) #CAREFUL b/c WCFit is mutable
    unit_chk = (abs(fit_new.EvalPoint(chk_pt) - 2*chk_y) < tolerance)
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 3 ---')
    print('chk_x    : ', chk_x)
    print('chk_y    : ', 2*chk_y)
    print('EvalPoint: ', fit_new.EvalPoint(chk_pt))
    print('test: ', chk_str)
    print('--------------\n')

    #fit_base = WCFit(pts,'base') #redefine b/c WCFit is mutable
    unit_chk = (abs(fit_base.EvalPoint(chk_pt) - chk_y) < tolerance)
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 4 ---')
    print('chk_x    : ', chk_x)
    print('chk_y    : ', chk_y)
    print('EvalPoint: ', fit_base.EvalPoint(chk_pt))
    print('test: ', chk_str)
    print('--------------\n')

    print(f'Passed Checks: {all_chks}/{units}')
    assert (all_chks == units)

########################### Stats unit tests ###########################

def test_stats():
    chk_str = ''
    unit_chk = True
    all_chks,units = [0]*2
    result,expected,diff,tolerance = [0]*4
    tolerance = 0.001

    # Basically the SM 'strength'
    x0 = 1.0

    # Dummy WC names to use (needs to match dimension of pt
    wc_names = ['sm','ctG','ctZ']

    # The structure constants, need to match dimension of pt
    svals = [
        1.15, # (00)
        1.35,1.25, # (10) (11)
        0.25,0.75,1.00, # (20) (21) (22)
    ]
    # Make sure there are enough pts to fully determine the fit!
    pts = [
        [x0,-1.00, 0.00],
        [x0,-0.50, 0.25],
        [x0, 0.00, 0.35],
        [x0, 0.25, 0.05],
        [x0, 0.50,-0.05],
        [x0, 0.75, 0.25],
        [x0, 1.00,-0.35],
    ]

    wc_pts = []
    idx=0
    for pt in pts:
        y = fval(pt,svals)
        s = f'EFTrwgt{idx}'
        for i in range(1, len(pt)): # NOTE: pt better not be size 0!!
            wc_str = wc_names[i]
            s += f'_{wc_str}_{pt[i]}'
        #print(s,y)
        wc_pts.append(WCPoint(s,y))
        idx += 1

    fit_1 = WCFit(wc_pts,'f1')
    fit_2 = WCFit()
    fit_2.SetTag('f2')

    nevents = 5000
    for i in range(nevents):
        fit_2.AddFit(fit_1)

    ###########################

    print('Running unit tests for stats unc.')
    all_chks = 0
    units = 0

    # Needs to be the same size as wc_names
    chk_x = [x0,1.2,0.4]
    chk_y = 0.0
    chk_e = 0.0
    for i in range(nevents):
        v = fval(chk_x,svals)
        chk_y += v
        chk_e += v*v
    chk_e = chk_e**.5

    chk_wcstr = 'EFTrwgt0'
    sidx = 0
    for i in range(len(wc_names)):
        if i: # Need to skip first entry since that's the SM 'strength'
            chk_wcstr += f'_{wc_names[i]}_{chk_x[i]}'
        for j in range(i+1):
            v = svals[sidx]
            #print(f'{i}{j}: {v}')
            sidx = sidx + 1
    print()
    chk_pt = WCPoint(chk_wcstr,0.0)

    ###########################

    # Basic check for proper adding of quadratic structure constants
    # Note: We expect the diff to grow with increased number of events due to the numeric precison
    expected = chk_y
    result = fit_2.EvalPoint(chk_pt)
    diff = abs(expected - result)
    tolerance = 1e-4

    unit_chk = (diff < tolerance)
    all_chks += unit_chk
    units += 1
    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 1 ---')
    print('evts     : ', nevents)
    print('chk_wcstr: ', chk_wcstr)
    print('expected : ', expected)
    print('result   : ', result)
    print('diff     : ', diff)
    print('tolerance: ', tolerance)
    print('test: ', chk_str)
    print('--------------\n')


    # Check the error calculation
    # Note: We expect the diff to grow with increased number of events due to the numeric precison
    expected = chk_e
    result = fit_2.EvalPointError(chk_pt)
    diff = abs(expected - result)
    tolerance = 1e-05*(10*nevents)**.5

    unit_chk = (diff < tolerance)
    all_chks += unit_chk
    units += 1
    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 2 ---')
    print('evts     : ', nevents)
    print('chk_wcstr: ', chk_wcstr)
    print('expected : ', expected)
    print('result   : ', result)
    print('diff     : ', diff)
    print('tolerance: ', tolerance)
    print('test: ', chk_str)
    print('--------------\n')

    # Now do the percent error
    # Note: The diff here also appears to grow apparently due to numeric precison, but much more slowly (it is still kind of concerning)
    expected = chk_e / chk_y
    result = fit_2.EvalPointError(chk_pt) / fit_2.EvalPoint(chk_pt)
    diff = abs(expected - result)
    tolerance = 1e-04

    unit_chk = (diff < tolerance)
    all_chks += unit_chk
    units += 1
    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 3 ---')
    print('evts     : ', nevents)
    print('chk_wcstr: ', chk_wcstr)
    print('expected : ', expected)
    print('result   : ', result)
    print('diff     : ', diff)
    print('tolerance: ', tolerance)
    print('test: ', chk_str)
    print('--------------\n')

    ###########################

    print(f'Passed Checks: {all_chks}/{units}')
    assert (all_chks == units)

########################### HistEFT unit tests ###########################

def test_histeft():
    chk_str = ''
    unit_chk = True
    all_chks,units = [0]*2
    result,expected,diff,tolerance = [0]*4
    tolerance = 1e-4

    wc_names = ['sm','ctG','ctZ']

    # The structure constants
    s00 = 1.0
    s10 = 1.5
    s11 = 1.25
    sconst = [s00, s10, s11]

    # A dummy WC name to use
    wc_name = 'ctG'

    pts = []
    vals = [-1.0,1.25,0.5,2.5,4]
    for x in vals:
        y = s00*1.0 + s10*x + s11*x*x
        pts.append(WCPoint(f'EFTrwgt0_{wc_name}_{x}',y))

    fit_1 = WCFit(pts,'f1')
    fit_2 = WCFit()
    fit_2.SetTag('f2')

    fit_2.AddFit(fit_1)
    fit_2.AddFit(fit_1)

    chk_x = 1.5
    chk_y = s00*1.0 + s10*chk_x + s11*chk_x*chk_x
    chk_vals = {wc_name:chk_x, 'ctZ':0.0}
    chk_pt = WCPoint(f'EFTrwgt0_{wc_name}_{chk_x}',0.0)

    print('Running unit tests for HistEFT class')
    all_chks = 0
    units = 0

    h_base = HistEFT("h_base", wc_names[1::], hist.Cat("sample", "sample"), hist.Bin("n",  "", 1, 0, 1))

    val=ak.Array([0.5])
    eftval = ak.Array([0.0002579])
    sconst = sconst + sconst
    h_base.fill(n=val, sample='test', weight=np.ones_like(val), eft_coeff=[ak.Array(sconst)])

    expected = 1.0
    result = list(h_base.values().values())[0][0]

    unit_chk = (abs(result - expected) < tolerance)
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 1 ---')
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    ###########################

    h_base.set_wilson_coefficients(**chk_vals)

    expected = fit_1.EvalPoint(chk_pt)
    result = list(h_base.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 2 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    ###########################

    chk_x = 0.75
    chk_y = s00*1.0 + s10*chk_x + s11*chk_x*chk_x
    chk_vals = {wc_name:chk_x, 'ctZ':0.0}
    chk_pt.SetStrength(wc_name,chk_x)
    h_base.set_wilson_coefficients(**chk_vals)

    expected = fit_1.EvalPoint(chk_pt)
    result = list(h_base.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 3 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    ###########################

    h_base.fill(n=val, sample='test', weight=np.ones_like(val), eft_coeff=[ak.Array(sconst)*2])
    h_base.set_wilson_coefficients(**chk_vals)

    # First make sure the original WCFits weren't messed with
    expected = chk_y + 2*chk_y
    result = fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt)
    diff = abs(expected - result)
    tolerance = 1e-10

    unit_chk = (diff < tolerance)
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 4 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('fit_1 + fit_2: ', result)
    print('difference   : ', diff)
    print('tolerance    : ', tolerance)
    print('test: ', chk_str)
    print('--------------\n')

    # Now check that the TH1EFT actually worked
    expected = fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt)
    result = list(h_base.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 5 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    ###########################

    h_new = h_base.copy()

    chk_x = 0.975
    chk_y = s00*1.0 + s10*chk_x + s11*chk_x*chk_x
    chk_vals = {wc_name:chk_x, 'ctZ':0.0}
    chk_pt.SetStrength(wc_name,chk_x)

    h_new.set_wilson_coefficients(**chk_vals)

    # First check that h_new has the right value
    expected = fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt)
    result = list(h_new.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 6 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    chk_x = 0.75    # Needs to be w/e chk_x was before UNIT 6
    chk_y = s00*1.0 + s10*chk_x + s11*chk_x*chk_x
    chk_vals = {wc_name:chk_x, 'ctZ':0.0}
    chk_pt.SetStrength(wc_name,chk_x)

    # Next check that the h_base was unaffected when we scaled h_new
    expected = fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt)
    result = list(h_base.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 7 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    # Check HistEFT.add()
    expected = fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt) #fits for h_base
    expected += fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt) #fits for h_new
    h_base.add(h_new)
    h_base.set_wilson_coefficients(**chk_vals) #evaluate h_base at chk_pt
    result = list(h_base.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 8 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    # Check HistEFT.add() reweight
    chk_x = 0.75    # Needs to be w/e chk_x was before UNIT 6
    chk_y = s00*1.0 + s10*chk_x + s11*chk_x*chk_x
    chk_vals = {wc_name:chk_x, 'ctZ':0.0}
    chk_pt.SetStrength(wc_name,chk_x)
    expected = fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt)
    expected += fit_1.EvalPoint(chk_pt) + fit_2.EvalPoint(chk_pt)
    h_base.set_wilson_coefficients(**chk_vals)
    result = list(h_base.values().values())[0][0]

    unit_chk = abs(result - expected) < tolerance
    all_chks += unit_chk
    units += 1

    chk_str = 'Passed' if unit_chk else 'Failed'
    print('--- UNIT 9 ---')
    print('chk_x        : ', chk_pt.GetStrength(wc_name))
    print('expected     : ', expected)
    print('GetBinContent: ', result)
    print('test: ', chk_str)
    print('--------------\n')

    ###########################

    print(f'Passed Checks: {all_chks}/{units}')
    assert (all_chks == units)
