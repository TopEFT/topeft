"""Focused, scheduler-free coverage for the read-only operator dashboard."""
import contextlib
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / 'analysis' / 'topeft_run2' / 'wq_dashboard.py'
spec = importlib.util.spec_from_file_location('wq_dashboard', SOURCE)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def worker(origin='/scratch/a', selector='^project-.*', status=2, cores=2):
    return {'Owner': 'operator', 'Cmd': origin + '/with_oasis_certs',
            'Args': './work_queue_worker -M ' + selector, 'JobStatus': status,
            'RequestCpus': cores, 'RequestMemory': 8000, 'RequestDisk': 8192000}


class DashboardTests(unittest.TestCase):
    def test_catalog_owner_normalization_and_freshness(self):
        now = time.time()
        records = [{'owner': 'operator', 'project': 'a', 'name': 'h', 'port': 1,
                    'lastheardfrom': now - 2, 'tasks_complete': 3, 'workers': 4},
                   {'owner': 'operator', 'project': 'b', 'name': 'h', 'port': 2,
                    'lastheardfrom': now - 300},
                   {'project': 'unknown', 'name': 'h', 'port': 3},
                   {'owner': 'someone_else', 'project': 'foreign', 'name': 'h', 'port': 4}]
        rows = dashboard.owner_catalog(records, 'operator', now, 120)
        self.assertEqual([r['project'] for r in rows], ['a', 'b'])
        self.assertEqual(rows[0]['tasks_done'], 3)
        self.assertEqual(rows[0]['workers_connected'], 4)
        self.assertEqual([r['_fresh'] for r in rows], [True, False])
        self.assertEqual(dashboard.matching_catalog(records, 'h', 3, 'operator'), [])
        near_expiry = [dict(records[0], lastheardfrom=now - 119)]
        self.assertTrue(dashboard.owner_catalog(near_expiry, 'operator', now, 120)[0]['_fresh'])
        self.assertFalse(dashboard.owner_catalog(near_expiry, 'operator', now + 10, 120)[0]['_fresh'])

    def test_project_selection_exact_unique_zero_ambiguous(self):
        now = time.time()
        records = [{'owner': 'operator', 'project': 'exact', 'name': 'h1', 'port': 9,
                    'lastheardfrom': now},
                   {'owner': 'operator', 'project': 'exact-ish', 'name': 'h2', 'port': 10,
                    'lastheardfrom': now}]
        rows = dashboard.owner_catalog(records, 'operator', now, 120)
        self.assertEqual(dashboard.select_project(rows, 'exact'), (('h1', '9'), 'OK'))
        self.assertEqual(dashboard.select_project(rows, 'ex'), (None, 'NO_FRESH_MATCH'))
        rows.extend(dashboard.owner_catalog([dict(records[0], name='h3', port=11)], 'operator', now, 120))
        self.assertEqual(dashboard.select_project(rows, 'exact')[1], 'AMBIGUOUS_FRESH_MATCH')
        rows[0]['_fresh'] = False
        self.assertEqual(dashboard.select_project(rows, 'exact'), (('h3', '11'), 'OK'))

    def test_task_enumeration_disabled_makes_no_task_call(self):
        calls = []
        def query(args, timeout):
            calls.append(args)
            return [{'tasks_waiting': 2, 'tasks_running': 0}], 'OK'
        summary, status, tasks, task_status = dashboard.manager_queries('host', '9164', 5, False, 5, query)
        self.assertEqual((status, task_status, tasks), ('OK', 'DISABLED', None))
        self.assertEqual(len(calls), 1)
        self.assertNotIn('-T', calls[0])
        dashboard.manager_queries('host', '9164', 5, True, 7, query)
        self.assertEqual(len(calls), 3)
        self.assertIn('-T', calls[2])
        self.assertIn('7', calls[2])

    def test_project_identity_is_rechecked_before_detail_or_tasks(self):
        calls = []
        def wrong_project(args, timeout):
            calls.append(args)
            return [{'project': 'other', 'tasks_waiting': 1, 'tasks_running': 0}], 'OK'
        summary, status, tasks, task_status = dashboard.manager_queries(
            'host', '9164', 5, True, 7, wrong_project, expected_project='wanted')
        self.assertEqual((summary, status, tasks, task_status),
                         (None, 'DIRECT_PROJECT_MISMATCH', None, 'SKIPPED_DIRECT_PROJECT_MISMATCH'))
        self.assertEqual(len(calls), 1)
        self.assertNotIn('-T', calls[0])
        def timed_out(args, timeout):
            return None, 'TIMEOUT'
        self.assertEqual(dashboard.manager_queries('host', '9164', 5, True, 7, timed_out,
                                                   expected_project='wanted')[3],
                         'SKIPPED_UNVERIFIED_DIRECT_PROJECT')
        def correct_project(args, timeout):
            calls.append(args)
            return ([{'project': 'wanted', 'tasks_waiting': 1, 'tasks_running': 0}], 'OK')
        dashboard.manager_queries('host', '9164', 5, True, 7, correct_project,
                                  expected_project='wanted')
        self.assertIn('-T', calls[-1])

    def test_worker_origin_selector_and_pool_identity(self):
        a = worker()
        b = worker(selector='other')
        c = worker(cores=4)
        d = worker(origin='/scratch/b', status=5)
        self.assertEqual(dashboard.worker_origin(a), '/scratch/a')
        self.assertEqual(dashboard.worker_selector(a), '^project-.*')
        self.assertEqual(dashboard.worker_selector({'Args': './work_queue_worker'}), '?')
        self.assertEqual(len(dashboard.pool_counts([a, b, c, d])), 4)
        self.assertEqual(dashboard.pool_counts([d])[dashboard.pool_key(d)][5], 1)
        self.assertIsNone(dashboard.worker_origin({'Cmd': '/usr/bin/python', 'Args': 'analysis.py'}))
        self.assertIsNone(dashboard.worker_origin({'Cmd': '/bin/bash',
                                                   'Args': "-c 'echo work_queue_worker'"}))

    def test_factory_argv_config_and_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'factory.json'
            path.write_text(json.dumps({'manager-name': 'from-json', 'max-workers': 1,
                                        'min-workers': 0, 'scratch-dir': '/scratch/a'}))
            parsed = dashboard.factory_from_process('123 /bin/work_queue_factory -C %s -S /scratch/a' % path)
            self.assertEqual(parsed['selector'], 'from-json')
            self.assertEqual(parsed['config_status'], 'OK')
            self.assertEqual(dashboard.factory_warnings(parsed, [worker(), worker(status=1)])[1],
                             'OBSERVED_ACTIVE_JOBS_ABOVE_DECLARED_MAX')
            self.assertIsNone(dashboard.factory_warnings(parsed, [worker(origin='/scratch/b')])[1])
            cli = dashboard.factory_from_process('124 /bin/work_queue_factory -M cli -C %s' % path)
            self.assertEqual(cli['selector'], 'cli')
            bad = Path(directory) / 'bad.json'
            bad.write_text('{bad')
            self.assertTrue(dashboard.factory_from_process('125 /bin/work_queue_factory -C %s' % bad)
                            ['config_status'].startswith('UNAVAILABLE'))
            self.assertTrue(dashboard.factory_from_process('126 /bin/work_queue_factory -C %s' %
                             (Path(directory) / 'missing'))['config_status'].startswith('UNAVAILABLE'))

    def test_recent_exits_and_missing_fields(self):
        rows = [dict(worker(), ExitCode=0, RemoteWallClockTime=10),
                dict(worker(), ExitCode=2, RemoteWallClockTime=20),
                dict(worker(), ExitBySignal=True, ExitSignal=9, RemoteWallClockTime=30),
                worker(), {'Cmd': '/bin/other', 'ExitCode': 1},
                {'Cmd': '/bin/bash', 'Args': "-c 'echo work_queue_worker'", 'ExitCode': 2}]
        exits = dashboard.recent_exits(rows)['/scratch/a']
        self.assertEqual(exits[:4], (4, 1, 1, 1))
        self.assertEqual(exits[4], 20)
        self.assertEqual(exits[5][2], 1)
        self.assertEqual(dashboard.recent_exits(None), {})

    def test_history_failure_is_not_zero(self):
        with mock.patch.object(dashboard, 'run', return_value=(124, '', 'query deadline exceeded')):
            records, status = dashboard.query(['condor_history'], 3)
        self.assertIsNone(records)
        self.assertEqual(status, 'TIMEOUT')

    def test_rendered_history_unavailable_and_default_task_opt_out(self):
        calls = []
        history_timeouts = []
        def query(args, timeout):
            calls.append(args)
            if args[0] == 'condor_history':
                history_timeouts.append(timeout)
                return None, 'TIMEOUT'
            if args[0] == 'condor_q':
                return [], 'OK'
            if args[-2:] == ['host', '9164']:
                return [{'tasks_waiting': 0, 'tasks_running': 0,
                         'tasks_with_results': 0}], 'OK'
            return [], 'OK'
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=query), \
             mock.patch.object(dashboard, 'run', return_value=(1, '', '')) as run_mock, \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, {'OWNER': 'operator', 'MANAGER_HOST': 'host',
                                          'MANAGER_PORT': '9164'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        self.assertFalse(any('-T' in args for args in calls))
        history_args = [args for args in calls if args[0] == 'condor_history']
        self.assertEqual(len(history_args), 1)
        self.assertIn('-scanlimit', history_args[0])
        self.assertEqual(history_timeouts, [20.0])
        if getpass.getuser() != 'operator':
            run_mock.assert_not_called()
            self.assertIn('NOT_OBSERVABLE_FOR_SELECTED_OWNER', output.getvalue())
        self.assertIn('=== recent_worker_exits ===\nquery_status: TIMEOUT', output.getvalue())
        self.assertIn('recent_history: UNAVAILABLE', output.getvalue())
        self.assertIn('=== active_work_queue_tasks ===\nquery_status: DISABLED', output.getvalue())

    def test_project_mismatch_keeps_owner_sections_and_skips_tasks(self):
        calls = []
        def query(args, timeout):
            calls.append(args)
            if args[0] == 'condor_history':
                return [], 'OK'
            if args[0] == 'condor_q':
                return [], 'OK'
            if args[-2:] == ['host', '9164']:
                return [{'project': 'other', 'tasks_waiting': 1, 'tasks_running': 0}], 'OK'
            return [{'owner': 'operator', 'project': 'wanted', 'name': 'host',
                     'port': 9164, 'lastheardfrom': time.time()}], 'OK'
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=query), \
             mock.patch.object(dashboard, 'run', return_value=(1, '', '')), \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, {'OWNER': 'operator', 'MANAGER_PROJECT': 'wanted',
                                          'TASK_DETAILS': '1'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        self.assertFalse(any('-T' in args for args in calls))
        self.assertIn('owner_records=1', output.getvalue())
        self.assertIn('selection: DIRECT_PROJECT_MISMATCH', output.getvalue())
        self.assertIn('query_status: SKIPPED_DIRECT_PROJECT_MISMATCH', output.getvalue())

    def test_held_and_health_alerts(self):
        held = worker(status=5)
        factory = {'scratch': '/scratch/a', 'selector': 'selector', 'pid': '2',
                   'settings': {'max-workers': 0}}
        alerts = dashboard.supply_alerts([held], [dict(worker(), ExitCode=2)], [factory])
        self.assertEqual({a[0] for a in alerts},
                         {'HELD_WORKER_JOBS', 'RECENT_NONZERO_OR_SIGNALED',
                          'OBSERVED_ACTIVE_JOBS_ABOVE_DECLARED_MAX'})
        self.assertEqual(dashboard.held_job_attribution(held, {'/scratch/a'}),
                         ('/scratch/a', 'LIVE_FACTORY_BOUND', '^project-.*'))
        self.assertEqual(dashboard.held_job_attribution(held, set())[1], 'NO_LIVE_FACTORY_MATCH')
        rows = dashboard.owner_catalog([{'owner': 'operator', 'project': 'p',
                                         'tasks_waiting': 5, 'workers': 0,
                                         'lastheardfrom': time.time()}],
                                       'operator', time.time(), 120)
        self.assertEqual(len(dashboard.manager_alerts(rows)), 1)
        rows[0]['_fresh'] = False
        self.assertEqual(dashboard.manager_alerts(rows), [])

    def test_sanitization_and_summary_semantics(self):
        self.assertEqual(dashboard.clean('a\x1bb\n'), 'a b ')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dashboard.table(['value'], [['bad\nrow']])
        self.assertNotIn('bad\nrow', output.getvalue())
        self.assertEqual(dashboard.normalized_summary({'tasks_complete': 7})['tasks_done'], 7)


if __name__ == '__main__':
    unittest.main()
