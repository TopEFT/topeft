"""Bounded real correction and processor-boundary OFF/ON checks; no WQ."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
from coffea.analysis_tools import Weights
import numpy as np
import pytest

from topcoffea.modules import corrections as tc_cor
from topeft.modules import corrections as te_cor


SOURCE = Path(__file__).resolve().parents[1] / 'analysis' / 'topeft_run2'


def emitter_namespace():
    import os
    import resource
    import sys
    import time
    namespace = dict(json=json, os=os, resource=resource, sys=sys, time=time)
    tree = ast.parse((SOURCE / 'analysis_processor.py').read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', '') in
             {'_phase_memory_mib', '_PhaseTelemetry', '_IterationTelemetry'}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    namespace['analysis_processor'] = SimpleNamespace(**namespace)
    tree = ast.parse((SOURCE / 'run_analysis.py').read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', '') == '_persist_task_phase_telemetry']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('year', ['2018', '2022'])
@pytest.mark.parametrize('variation', ['nominal', 'up', 'down'])
def test_native_pileup_off_on_and_persistence(year, variation, capsys, tmp_path):
    ns = emitter_namespace()
    inputs = ak.Array([8., 22., 39.])
    off = tc_cor.GetPUSF(inputs, year, variation)
    assert capsys.readouterr().out == ''
    reporter = ns['_PhaseTelemetry'](event_count=3)
    with tc_cor.telemetry_context(reporter, variation, 2):
        on = tc_cor.GetPUSF(inputs, year, variation)
    np.testing.assert_array_equal(on, off)
    raw = capsys.readouterr().out
    stdout = tmp_path / 'worker.stdout'
    stdout.write_text(raw)
    sink = tmp_path / 'records.jsonl'
    ns['_persist_task_phase_telemetry'](SimpleNamespace(outfile_stdout=stdout, id=83), sink)
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert rows and all(r['task_id'] == 83 and r['syst_var'] == variation
                        and r['iteration_index'] == 2 for r in rows)
    assert any('payload_read_decompress_and_correctionset' in r['operation'] for r in rows)
    assert any(r['operation'] == 'topcoffea.GetPUSF.total' for r in rows)
    assert all(r['inclusive_wall_ns'] >= r['exclusive_wall_ns'] >= 0
               and r['inclusive_cpu_ns'] >= r['exclusive_cpu_ns'] >= 0 for r in rows)
    assert all(r['calls'] > 0 and r['failures'] == 0 for r in rows)
    assert 'private' not in sink.read_text()


@pytest.mark.parametrize('label', ['nominal', 'JES_AbsoluteUp'])
def test_real_processor_weight_copy_boundary_off_on(label, capsys):
    tree = ast.parse((SOURCE / 'analysis_processor.py').read_text())
    scope = next(n for n in ast.walk(tree) if isinstance(n, ast.With)
                 and 'processor.copy_weights' in ast.unparse(n.items))
    weights = Weights(3, storeIndividual=True)
    weights.add('norm', np.array([1., -2., 3.]))
    ns = emitter_namespace()
    expected = copy.deepcopy(weights)
    for reporter in [None, ns['_PhaseTelemetry'](event_count=3)]:
        env = dict(tc_cor=tc_cor, copy=copy, weights_obj_base=weights,
                   telemetry=reporter, syst_var=label, iteration_index=4)
        exec(compile(ast.Module(body=[scope], type_ignores=[]), 'actual_processor_copy', 'exec'), env)
        np.testing.assert_array_equal(env['weights_obj_base_for_kinematic_syst'].weight(),
                                      expected.weight())
        if reporter is None:
            assert capsys.readouterr().out == ''
    rows = [json.loads(line.split('] ', 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [(r['operation'], r['syst_var'], r['iteration_index'], r['calls']) for r in rows] == [
        ('processor.copy_weights', label, 4, 1)]


@pytest.mark.parametrize('is_data,enable_tau', [(True, False), (False, False), (False, True)])
def test_reached_guards_do_not_report_skipped_work(is_data, enable_tau, capsys):
    # Exercise the maintained guards with inert bodies: this qualifies reachability
    # and correlation only, never the substituted tau or MC physics calculation.
    tree = ast.parse((SOURCE / 'analysis_processor.py').read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and ast.unparse(n.target) == '(iteration_index, syst_var)')
    selected = []
    for node in loop.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) in ('self.enable_tau_blocks', 'not isData'):
            with_nodes = [n for n in node.body if isinstance(n, ast.With)]
            if with_nodes:
                guard = copy.deepcopy(node)
                guard.body = []
                for scope in with_nodes:
                    scope = copy.deepcopy(scope)
                    scope.body = [ast.copy_location(ast.Pass(), scope.body[0])]
                    guard.body.append(scope)
                selected.append(guard)
    assert len(selected) == 2
    reporter = emitter_namespace()['_PhaseTelemetry']()
    env = dict(tc_cor=tc_cor, telemetry=reporter, self=SimpleNamespace(enable_tau_blocks=enable_tau),
               isData=is_data, syst_var='nominal', iteration_index=0)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, 'actual_guards', 'exec'), env)
    operations = [json.loads(line.split('] ', 1)[1])['operation']
                  for line in capsys.readouterr().out.splitlines()]
    assert ('processor.taus' in operations) == enable_tau
    assert ('processor.btag_and_trigger_sf' in operations) == (not is_data)


def test_cross_repository_correction_interface_and_native_conversion():
    assert te_cor.telemetry_operation is tc_cor.telemetry_operation
    import correctionlib
    correction = correctionlib.CorrectionSet.from_string(json.dumps({
        'schema_version': 2, 'corrections': [{
            'name': 'identity', 'version': 1, 'inputs': [{'name': 'x', 'type': 'real'}],
            'output': {'name': 'w', 'type': 'real'},
            'data': {'nodetype': 'formula', 'expression': 'x*x', 'parser': 'TFormula',
                     'variables': ['x']}}]}))['identity']
    values = ak.Array([1., 2., 3.])
    np.testing.assert_array_equal(te_cor._evaluate_correctionlib(correction, values), [1., 4., 9.])
    records = []
    with tc_cor.telemetry_context(SimpleNamespace(mark=lambda phase, **row: records.append(row)), 'nominal', 0):
        actual = te_cor._evaluate_correctionlib(correction, values)
    np.testing.assert_array_equal(actual, [1., 4., 9.])
    assert {r['operation'] for r in records} >= {
        'topeft._evaluate_correctionlib.total',
        'topeft._evaluate_correctionlib.conversion_and_flatten',
        'topeft._evaluate_correctionlib.evaluate'}
