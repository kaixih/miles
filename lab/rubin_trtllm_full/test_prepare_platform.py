"""CPU-only contracts for preparation; no live SSH, allocation or GPU work."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('trtllm_prepare', Path(__file__).with_name('prepare_platform.py'))
P = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(P)


def options(platform='gb300'):
    return P.parser().parse_args([
        '--platform', platform, '--source', '/home/scratch.kaixih_ent/repro/frozen/source',
        '--source-manifest', '/home/scratch.kaixih_ent/repro/frozen/source-manifest.json',
        '--source-commit', 'a' * 40, '--wait-until', '2026-09-17T08:00:00Z',
        *(['--image', 'registry/image@sha256:' + 'b' * 64] if platform == 'gb300' else []),
    ])


def allocation():
    return {'JobId': '2212644', 'UserId': 'kaixih(28644)', 'GroupId': 'dip(30)',
            'JobState': 'RUNNING', 'NumNodes': '1', 'AllocTRES': 'cpu=144,node=1,gres/gpu=4',
            'NodeList': 'gb300-nvl-012-compute04', 'TimeLimit': '08:00:00',
            'StartTime': '2026-09-17T01:30:21', 'EndTime': '2026-09-17T09:30:21'}


class PreparationTests(unittest.TestCase):
    def test_exact_campaign_and_image_selection(self):
        for platform, job in [('rubin', '2212643'), ('gb300', '2212644')]:
            c = P.validate_options(options(platform))
            self.assertEqual(c['run_id'], f'20260917-{platform}-j{job}-trtllm')
            self.assertEqual(c['run_dir'], '/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/' + c['run_id'])
        for key, value in [('image', 'mutable:tag'), ('source_commit', 'main'),
                           ('source', Path('/tmp/source')), ('wait_until', '2026-09-17T01:00:00')]:
            a = options(); setattr(a, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): P.validate_options(a)
        a = options('gb300'); a.image = None
        with self.assertRaises(ValueError): P.validate_options(a)
        a = options('rubin'); a.image = 'registry/other@sha256:' + 'b' * 64
        with self.assertRaises(ValueError): P.validate_options(a)

    def test_allocation_guard_is_bound_to_real_eight_hour_lease(self):
        c = P.validate_options(options()); good = allocation()
        now = P.timestamp('2026-09-17T02:00:00Z')
        self.assertEqual(P.allocation_guard(c, good, now), 'ready')
        for key, value in [('JobId', '2212643'), ('UserId', 'other(123)'), ('GroupId', 'other(123)'),
                           ('JobState', 'COMPLETED'), ('NumNodes', '2'), ('NodeList', '(null)'),
                           ('AllocTRES', 'cpu=144,node=1,gres/gpu=8'), ('TimeLimit', '09:00:00'),
                           ('EndTime', '2026-09-17T10:30:21')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                P.allocation_guard(c, {**good, key: value}, now)
        with self.assertRaises(ValueError): P.allocation_guard(c, good, P.timestamp('2026-09-17T09:29:00Z'))
        with self.assertRaisesRegex(ValueError, 'lease changed'):
            P.allocation_guard(c, {**good, 'NodeList': 'another-node'}, now, good)
        self.assertEqual(P.allocation_guard(c, {**good, 'JobState': 'PENDING', 'NodeList': '(null)'}, now), 'wait')
        with self.assertRaises(ValueError):
            P.allocation_guard(c, {**good, 'JobState': 'PENDING'}, now, good)

    def test_source_views_are_explicit_and_cannot_escape_scratch(self):
        shared = '/home/scratch.kaixih_ent/repro/frozen/source'
        self.assertEqual(P.node_view('rubin', shared), shared)
        self.assertEqual(P.node_view('gb300', shared), '/mnt/cifs' + shared)
        with self.assertRaises(ValueError): P.node_view('gb300', '/tmp/source')

    def test_driver_configuration_keeps_full_recipe_and_model_only_save(self):
        for platform in ('gb300', 'rubin'):
            a = options(platform); c = P.validate_options(a); worker = P.Worker(c, a)
            worker.original = allocation()
            storage = {'node_ip': '10.0.0.1', 'local_root': '/raid/tmp/local', 'nic': 'verified-nic'}
            driver = worker.driver_config(storage, {'root': '/raid/tmp/verified-models'})
            self.assertEqual(driver['sglang_moe_runner_backend'], 'flashinfer_trtllm')
            self.assertFalse(driver['save_optimizer'])
            self.assertEqual(driver['uid'], 28644)
            self.assertEqual(driver['gid'], 30)
            self.assertEqual(driver['lease_deadline'], '2026-09-17T09:30:21Z')
            self.assertEqual(driver['node_repo'], ('/mnt/cifs' if platform == 'gb300' else '') + c['source'])
            self.assertEqual(driver['megatron_path'], '/root/Megatron-LM' if platform == 'gb300' else '/opt/Megatron-LM')

    def test_remote_programs_compile_without_execution(self):
        c = P.validate_options(options()); c['node'] = 'gb300-nvl-012-compute04'
        for function, args in [(P.host_storage, (c, None)), (P.host_model_inventory, ('/raid/tmp/models',))]:
            code = P.python_program(function, *args)
            compile(code, 'remote-preparation', 'exec')
        source = Path(P.__file__).read_text()
        self.assertNotIn("'--standalone'", source)
        self.assertIn("'--master-addr=127.0.0.1'", source)
        self.assertNotIn("'docker', 'build'", source)
        self.assertNotIn("'docker', 'rm'", source)

    def test_resume_stage_reuses_only_completed_receipt_and_runs_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            a = options(); c = {**P.validate_options(a), 'run_dir': d}
            worker = P.Worker(c, a); worker.attempt = Path(d) / 'attempt'
            result = worker.stage('sample', lambda: {'evidence': 'fixed'})
            self.assertEqual(result, {'evidence': 'fixed'})
            checks = []
            result = worker.stage('sample', lambda: self.fail('Must not rerun completed stage'), checks.append)
            self.assertEqual(checks, [result])
            with self.assertRaisesRegex(ValueError, 'changed'):
                worker.stage('sample', lambda: None, lambda r: P.require(False, 'changed'))

    def test_plan_path_never_accesses_source_or_remote(self):
        a = options()
        with patch.object(P, 'parser') as parser, patch.object(P.Worker, 'execute', side_effect=AssertionError), \
                patch.object(P.subprocess, 'check_output', side_effect=AssertionError), patch('builtins.print') as printed:
            parser.return_value.parse_args.return_value = a
            self.assertEqual(P.main(), 0)
        plan = json.loads(printed.call_args.args[0])
        self.assertEqual(plan['mode'], 'PLAN_ONLY_NO_REMOTE_CALLS')
        self.assertEqual(plan['recipe']['rollouts'], 50)
        self.assertEqual(plan['recipe']['updates'], 200)
        self.assertEqual(plan['capacity_gib'], 400)


if __name__ == '__main__':
    unittest.main()
