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
    return {'Owner': 'operator', 'Cmd': origin + '/condor.sh',
            'Args': './with_oasis_certs ./work_queue_worker -M ' + selector, 'JobStatus': status,
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

    def test_worker_invocation_is_independent_of_scratch_name(self):
        generic = worker(origin='/tmp/wq-apiccine-tau')
        self.assertEqual(dashboard.worker_origin(generic), '/tmp/wq-apiccine-tau')
        misleading = {'Cmd': '/tmp/wq-apiccine-factory-misc/condor.sh',
                      'Args': '/bin/echo hello'}
        self.assertIsNone(dashboard.worker_origin(misleading))
        self.assertEqual(dashboard.worker_origin({'Cmd': '/some/path/work_queue_worker'}),
                         '/some/path')
        self.assertEqual(dashboard.worker_origin({'Cmd': '/some/path/work_queue_worker',
                                                  'Args': "'malformed"}), '/some/path')
        self.assertEqual(dashboard.worker_origin({'Cmd': generic['Cmd'],
                                                  'Arguments': './work_queue_worker -M manager'}),
                         '/tmp/wq-apiccine-tau')
        accepted_wrapper = {'Cmd': '/tmp/wq-apiccine-manual/with_oasis_certs',
                            'Args': './work_queue_worker -M manager'}
        self.assertEqual(dashboard.worker_origin(accepted_wrapper), '/tmp/wq-apiccine-manual')
        self.assertEqual(dashboard.worker_selector(accepted_wrapper), 'manager')
        self.assertIsNone(dashboard.worker_origin({'Cmd': accepted_wrapper['Cmd'],
                                                   'Args': '/bin/echo ./work_queue_worker'}))
        self.assertIsNone(dashboard.worker_origin({'Cmd': accepted_wrapper['Cmd'],
                                                   'Args': "'./work_queue_worker"}))
        fallback = {'Cmd': generic['Cmd'], 'Args': '', 'Arguments': './work_queue_worker -M fallback'}
        self.assertEqual(dashboard.worker_origin(fallback), '/tmp/wq-apiccine-tau')
        self.assertEqual(dashboard.worker_selector(fallback), 'fallback')
        self.assertIsNone(dashboard.worker_origin({'Cmd': generic['Cmd'],
                                                   'Args': "'./with_oasis_certs ./work_queue_worker"}))
        self.assertIsNone(dashboard.worker_origin({'Cmd': generic['Cmd'],
                                                   'Args': '/bin/echo ./work_queue_worker'}))

    def test_factory_argv_config_and_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'factory.json'
            path.write_text(json.dumps({'manager-name': 'from-json', 'max-workers': 1,
                                        'min-workers': 0, 'scratch-dir': '/scratch/a',
                                        'timeout': 900}))
            parsed = dashboard.factory_from_process('123 /bin/work_queue_factory -C %s -S /scratch/a' % path)
            self.assertEqual(parsed['selector'], 'from-json')
            self.assertEqual(parsed['config_status'], 'OK')
            self.assertEqual(parsed['settings']['worker_timeout'], 900)
            self.assertIsNone(parsed['settings']['factory_timeout'])
            distinct = dashboard.factory_from_process(
                '124 /bin/work_queue_factory -C %s --factory-timeout 0' % path)
            self.assertEqual(distinct['settings']['worker_timeout'], 900)
            self.assertEqual(distinct['settings']['factory_timeout'], '0')
            cli_only = dashboard.factory_from_process('125 /bin/work_queue_factory --factory-timeout 0')
            self.assertIsNone(cli_only['settings']['worker_timeout'])
            self.assertEqual(cli_only['settings']['factory_timeout'], '0')
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

    def test_cgroup_probe_requires_explicit_running_owner_snapshot_id(self):
        jobs = [dict(worker(), ClusterId=12, ProcId=0),
                dict(worker(status=1), ClusterId=13, ProcId=0),
                dict(worker(), Owner='other', ClusterId=14, ProcId=0),
                dict(worker(), ClusterId=15, ProcId=0)]
        calls = []
        def fake_run(args, timeout, input_text=None):
            calls.append((args, timeout, input_text))
            return 0, 'memory.current: 123\n', ''
        self.assertEqual(dashboard.cgroup_probes('', jobs, 'operator', 'schedd', 4, 8, fake_run), [])
        self.assertEqual(calls, [])
        rows = dashboard.cgroup_probes('bad 99.0,13.0 14.0 12.0 12.0', jobs,
                                      'operator', 'schedd', 4, 8, fake_run)
        self.assertEqual([row[0] for row in rows], ['12.0'])
        self.assertEqual(calls[0][:2],
                         (['condor_ssh_to_job', '-name', 'schedd', '12.0', 'bash', '-s'], 8))
        script = calls[0][2]
        for name in ('_CONDOR_JOB_PIDS', 'memory.current', 'memory.peak',
                     'memory.max', 'memory.events', '/proc/$pid/cgroup'):
            self.assertIn(name, script)
        self.assertEqual(dashboard.cgroup_probes('12.0', jobs, 'operator', 'schedd', 0, 8, fake_run), [])
        self.assertEqual(len(calls), 1)
        capped = dashboard.cgroup_probes('12.0 15.0', jobs, 'operator', 'schedd', 1, 8, fake_run)
        self.assertEqual([row[0] for row in capped], ['12.0'])
        self.assertEqual(len(calls), 2)

    def test_cgroup_script_is_piped_through_bounded_subprocess(self):
        process = mock.Mock(returncode=0)
        process.communicate.return_value = ('memory.current: 123\n', '')
        with mock.patch.object(dashboard.subprocess, 'Popen', return_value=process) as popen:
            self.assertEqual(dashboard.run(['condor_ssh_to_job', 'bash', '-s'], 8,
                                           input_text=dashboard.CGROUP_READ_SCRIPT),
                             (0, 'memory.current: 123\n', ''))
        self.assertEqual(popen.call_args.kwargs['stdin'], dashboard.subprocess.PIPE)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        process.communicate.assert_called_once_with(input=dashboard.CGROUP_READ_SCRIPT, timeout=8)

    def test_cgroup_probe_opt_out_and_rendered_timeout_labels(self):
        def query(args, timeout):
            if args[0] == 'condor_q':
                return [dict(worker(), ClusterId=12, ProcId=0)], 'OK'
            if args[0] == 'condor_history':
                return [], 'OK'
            if args[-2:] == ['host', '9164']:
                return [{'tasks_waiting': 0, 'tasks_running': 0}], 'OK'
            return [], 'OK'
        factory = {'pid': '1', 'selector': '?', 'config_path': None, 'config_status': 'OK',
                   'scratch': '/scratch/a', 'settings': {'worker_timeout': 900,
                                                       'factory_timeout': '0'}}
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=query), \
             mock.patch.object(dashboard, 'run', return_value=(0, '1 work_queue_factory\n', '')) as run_mock, \
             mock.patch.object(dashboard.pwd, 'getpwuid', return_value=mock.Mock(pw_name='operator')), \
             mock.patch.object(dashboard, 'factory_from_process', return_value=factory), \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, {'OWNER': 'operator', 'MANAGER_HOST': 'host',
                                          'MANAGER_PORT': '9164'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        self.assertIn('DISABLED: CGROUP_JOB_IDS unset', output.getvalue())
        self.assertIn('worker_timeout', output.getvalue())
        self.assertIn('factory_timeout', output.getvalue())
        self.assertRegex(output.getvalue(), r'900\s+0')
        self.assertFalse(any(call.args[0][0] == 'condor_ssh_to_job' for call in run_mock.call_args_list))

    def test_help_exits_before_any_queries_or_probes(self):
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query') as query_mock, \
             mock.patch.object(dashboard, 'run') as run_mock, \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py', '--help']), \
             mock.patch.dict(os.environ, {'CGROUP_JOB_IDS': '12.0'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        query_mock.assert_not_called()
        run_mock.assert_not_called()
        self.assertIn('CGROUP_JOB_IDS', output.getvalue())
        self.assertIn('Host/task matching alone does not authorize', output.getvalue())

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
