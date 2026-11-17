"""Relative Tau pruning thresholds and fail-closed pair protections."""
from collections import namedtuple
import inspect
import ast
import textwrap

import numpy as np
import pytest
from topeft.modules import datacard_tools as dt

key = namedtuple('key', 'systematic')
base = 'CMS_eff_t_DeepTau_test'


def classify(nominal=(0., 2000.), up=(0., 2001.), down=(0., 1999.),
             threshold=None, nuisance=base, run='_run3', missing=None, raw=None):
    templates = {key('nominal'): np.array([nominal, np.zeros(len(nominal))])}
    for direction, values in [('Up', up), ('Down', down)]:
        if direction != missing:
            templates[key(nuisance + direction)] = np.array([values, np.zeros(len(values))])
    kwargs = {} if threshold is None else {'nominal_like_tau_shape_threshold': threshold}
    return dt._nominal_like_tau_shape_bases(templates, templates if raw is None else raw,
                                          len(nominal), run, [], 'per-run', **kwargs)


def test_default_and_historical_discrimination():
    assert dt.DEFAULT_NOMINAL_LIKE_TAU_SHAPE_THRESHOLD == 1e-3
    assert classify() == {base}
    assert classify(threshold=1e-3) == {base}
    assert classify(threshold=1e-5) == set()
    assert classify(up=(0., 2000.01), down=(0., 1999.99), threshold=1e-5) == {base}


@pytest.mark.parametrize('nominal,up,down,threshold', [
    ((0., 1000.), (0., 1001.), (0., 999.), 1e-3),
    ((0., 100000.), (0., 100001.), (0., 99999.), 1e-5),
    ((0., 2000.), (0., 2001.), (0., 1996.), 1e-3),
])
def test_strict_equal_or_one_side_above_retains(nominal, up, down, threshold):
    assert classify(nominal, up, down, threshold) == set()


@pytest.mark.parametrize('missing', ['Up', 'Down'])
def test_missing_direction_retains(missing):
    assert classify(missing=missing) == set()


def test_zero_bin_and_underflow():
    assert classify((0., 2000., 0.), (0., 2001., 0.), (0., 1999., 0.)) == {base}
    assert classify((0., 2000., 0.), (0., 2001., 0.01), (0., 1999., 0.)) == set()
    assert classify(up=(1., 2001.)) == set()


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, -1.])
def test_unsafe_values_retain(value):
    assert classify(up=(0., value)) == set()
    assert classify(nominal=(0., value)) == set()


def test_raw_negative_total_and_ratio_safety():
    raw = {key('nominal'): [[0., 2000.]], key(base+'Up'): [[0., -1.]],
           key(base+'Down'): [[0., 1999.]]}
    assert classify(raw=raw) == set()
    assert classify((0., 0.), (0., 0.), (0., 0.)) == set()
    assert classify((0., 1e-320), (0., 1.), (0., 1e-320)) == set()


@pytest.mark.parametrize('nuisance,run', [
    ('CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2018', '_run2'),
    ('CMS_scale_t_DeepTau2018v2p5_DM0_genTau_2018', '_run3'),
    ('unrelated', '_run3'),
])
def test_genuine_tes_quarantine_and_eligibility(nuisance, run):
    assert classify(nuisance=nuisance, run=run) == set()


@pytest.mark.parametrize('value', [True, False, np.bool_(True), 0, -1, np.nan, np.inf, -np.inf])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError, match='finite positive'):
        classify(threshold=value)
    with pytest.raises(ValueError, match='finite positive'):
        dt.DatacardMaker(hists={}, nominal_like_tau_shape_threshold=value)


def test_aac_incompatibility():
    with pytest.raises(ValueError, match='unsupported with use_AAC'):
        dt.DatacardMaker(hists={}, suppress_nominal_like_tau_shapes=True, use_AAC=True)


def test_toggle_guards_classifier_and_forwards_parameter():
    # Execute the actual producer's bounded guard, without entering its card writer.
    tree = ast.parse(textwrap.dedent(inspect.getsource(dt.DatacardMaker)))
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and isinstance(n.test, ast.BoolOp)
              and 'self.suppress_nominal_like_tau_shapes' in ast.unparse(n.test)
              and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                      and c.func.id == '_nominal_like_tau_shape_bases' for c in ast.walk(n))]
    guard, = guards
    from types import SimpleNamespace
    calls = []
    env = dict(self=SimpleNamespace(suppress_nominal_like_tau_shapes=False,
               nominal_like_tau_shape_threshold=1e-5, suffix='_run3', run_decorrelate=[],
               lepton_mva_correlation='per-run'), base='sm', p='ttll',
               process_retains_stat_uncertainty=lambda p: False, v={}, raw_templates={},
               ch_hist=SimpleNamespace(dense_axis=SimpleNamespace(extent=2)),
               all_shapes=set(), syst_width=0,
               _nominal_like_tau_shape_bases=lambda *a, **k: calls.append(k) or set())
    code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), '<producer_guard>', 'exec')
    exec(code, env)
    assert calls == []
    env['self'].suppress_nominal_like_tau_shapes = True
    exec(code, env)
    assert calls == [{'nominal_like_tau_shape_threshold': 1e-5}]
