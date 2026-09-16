"""CPU-only snapshot transaction and lossless transfer contracts. No SSH calls."""
import base64
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import zlib

SPEC = importlib.util.spec_from_file_location('collector_under_test', Path(__file__).with_name('collect_qwen3_snapshot.py'))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def packet(raw):
    return {'bytes': len(raw), 'sha256': C.sha(raw), 'data': base64.b64encode(zlib.compress(raw)).decode()}


def payload(config, previous, tail):
    whole = previous + tail
    return {'schema': 'verified-log-append-v1', 'root': config['root'],
            'log': {'mode': 'append', 'offset': len(previous), 'prefix_sha256': C.sha(previous),
                    'bytes': len(whole), 'sha256': C.sha(whole), 'tail': packet(tail)},
            'files': {'planned-run-config.json': packet(b'{}'), 'train-launch.json': packet(b'{}')},
            'watchdog': {}, 'remote_seconds': 1, 'watchdog_read_seconds': .1}


class TransferTests(unittest.TestCase):
    def test_append_preserves_exact_bytes_including_partial_unicode_line(self):
        config = {'root': '/test'}
        previous = b'physical line\nutf8 ' + '\u2028'.encode() + b' unfin'
        tail = b'ished\n\x1b[36mnew line\x1b[0m\n'
        decoded = C.decode_payload(payload(config, previous, tail), config, previous)
        self.assertEqual(decoded[C.LOG], previous + tail)
        receipt = json.loads(decoded['collection-receipt.json'])
        self.assertEqual(receipt['tail_bytes'], len(tail))
        self.assertEqual(receipt['sha256'], C.sha(previous + tail))

    def test_corrupt_prefix_tail_full_hash_and_paths_rejected(self):
        config, old, tail = {'root': '/test'}, b'old\n', b'new\n'
        original = payload(config, old, tail)
        cases = []
        item = deepcopy(original); item['log']['prefix_sha256'] = '0' * 64; cases.append(item)
        item = deepcopy(original); item['log']['sha256'] = '0' * 64; cases.append(item)
        item = deepcopy(original); item['log']['tail']['bytes'] += 1; cases.append(item)
        item = deepcopy(original); item['files']['../unsafe'] = packet(b'x'); cases.append(item)
        item = deepcopy(original); item['root'] = '/other'; cases.append(item)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                C.decode_payload(item, config, old)
        with self.assertRaises(ValueError):
            C.decode_payload(original, config, old + b'changed')

    def test_remote_append_and_same_size_rewrite_full_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'logs').mkdir()
            config = {'root': directory, 'node': 'no-network', 'container': 'no-docker'}
            old = b'original\n'; current = old + b'append\n'
            path = root / C.LOG; path.write_bytes(current)
            with patch.object(signal, 'alarm'), patch.object(C.subprocess, 'check_output', return_value='{}'):
                first = C._remote_payload(config, {'bytes': len(old), 'sha256': C.sha(old)}, [])
                self.assertEqual(first['log']['mode'], 'append')
                self.assertEqual(first['log']['tail']['bytes'], len(current) - len(old))
                path.write_bytes(b'REWRITTEN' + current[9:])
                second = C._remote_payload(config, {'bytes': len(old), 'sha256': C.sha(old)}, [])
            self.assertEqual(second['log']['mode'], 'full')
            self.assertEqual(second['log']['offset'], 0)
            self.assertEqual(C.decode_packet(second['log']['tail'], C.MAX_LOG_BYTES), path.read_bytes())

    def test_decompression_trailing_data_or_oversize_rejected(self):
        record = packet(b'abc')
        record['data'] = base64.b64encode(base64.b64decode(record['data']) + b'extra').decode()
        with self.assertRaises(ValueError): C.decode_packet(record, 10)
        with self.assertRaises(ValueError): C.decode_packet(packet(b'abc'), 2)

    def test_physical_newline_evidence_number(self):
        argv = ['--max-tokens-per-gpu', '4096', '--save', '/ckpt']
        _recipe, evidence, _excluded = C.actual_recipe_and_save_events(argv,
            'sample text\u2028still first physical line\n'
            'saving checkpoint at iteration 9 to /ckpt\n')
        self.assertEqual(evidence['observed_save_events'][0]['line'], 2)


class TransactionTests(unittest.TestCase):
    def setup_files(self, directory):
        root = Path(directory)
        for label, config in C.RUNS.items():
            path = root / config['destination'] / C.LOG
            path.parent.mkdir(parents=True); path.write_bytes((label + ' old\n').encode())
        (root / 'comparison.json').write_bytes(b'old comparison')
        (root / 'runs.json').write_bytes(b'old metadata')
        (root / 'io-overlap-windows.json').write_text(json.dumps({'schema_version':1, 'snapshot_utc':'2026-09-16T02:40:00Z', 'windows':[]}))
        return root

    def original_bytes(self, root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()
                and p.name != '.collect_snapshot.lock'}

    def result(self, item):
        label, _config = item
        return label, {}, {C.LOG: (label + ' new\n').encode()}

    def fake_summary(self, stage, results, io_evidence):
        (stage / "io-overlap-windows.snapshot.json").write_bytes(io_evidence["raw"])
        report = {'runs': [{'label': label, 'source_log_sha256': C.sha(files[C.LOG])}
                           for label, _metadata, files in results]}
        (stage / 'comparison.json').write_text(json.dumps(report))
        (stage / 'runs.json').write_text('{}')
        return report

    def test_second_platform_failure_does_not_touch_first_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.setup_files(directory); before = self.original_bytes(root)
            def fetch(item):
                if item[0] == 'gb300': raise TimeoutError('GB SSH timed out')
                return self.result(item)
            with patch.object(C, 'ROOT', root), patch.object(C, 'collect', side_effect=fetch):
                with self.assertRaises(TimeoutError): C.collect_snapshot()
            self.assertEqual(self.original_bytes(root), before)

    def test_parser_failure_preserves_every_current_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.setup_files(directory); before = self.original_bytes(root)
            with patch.object(C, 'ROOT', root), patch.object(C, 'collect', side_effect=self.result), \
                    patch.object(C, 'build_snapshot', side_effect=ValueError('invalid metadata')):
                with self.assertRaises(ValueError): C.collect_snapshot()
            self.assertEqual(self.original_bytes(root), before)

    def test_success_commits_both_and_drops_stale_optional_not_other_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.setup_files(directory)
            obsolete = root / 'gb300/train_exit.json'; obsolete.write_text('{"old":true}')
            retained = root / 'gb300/unrelated-proof.txt'; retained.write_text('retain')
            with patch.object(C, 'ROOT', root), patch.object(C, 'collect', side_effect=self.result), \
                    patch.object(C, 'build_snapshot', side_effect=self.fake_summary):
                C.collect_snapshot()
            self.assertFalse(obsolete.exists()); self.assertEqual(retained.read_text(), 'retain')
            for label, config in C.RUNS.items():
                self.assertEqual((root / config['destination'] / C.LOG).read_bytes(), (label + ' new\n').encode())
            self.assertEqual(len(json.loads((root / 'comparison.json').read_text())['runs']), 2)

    def test_commit_io_failure_rolls_back_both_logs_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.setup_files(directory); before = self.original_bytes(root)
            original = os.replace; calls = [0]
            def fail_once(source, destination):
                calls[0] += 1
                if calls[0] == 5: raise OSError('simulated commit failure')
                return original(source, destination)
            with patch.object(C, 'ROOT', root), patch.object(C, 'collect', side_effect=self.result), \
                    patch.object(C, 'build_snapshot', side_effect=self.fake_summary), \
                    patch.object(C.os, 'replace', side_effect=fail_once):
                with self.assertRaises(OSError): C.collect_snapshot()
            self.assertEqual(self.original_bytes(root), before)

    def test_concurrent_collector_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with C.collection_lock(Path(directory)):
                with self.assertRaises(BlockingIOError):
                    with C.collection_lock(Path(directory)): pass


class IoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.helper = C.load_source_module('io_test_helper', C.WORKSPACE / 'lab/rubin_two_node/io_overlap.py')
        self.parser = C.load_source_module('summary_test_helper', C.WORKSPACE / 'lab/rubin_two_node/summarize_qwen3_runs.py')
        self.document = {'schema_version': 1, 'snapshot_utc': '2026-09-16T02:10:00Z',
                         'windows': [{'id': 'copy', 'platform': 'rubin', 'operation': 'model staging',
                             'status': 'complete', 'start_utc': '2026-09-16T02:01:10Z',
                             'end_utc': '2026-09-16T02:01:30Z'}]}

    def evidence(self, document=None):
        document = document or self.document
        raw = json.dumps(document).encode()
        return {'document': document, 'raw': raw, 'sha256': C.sha(raw), 'path': '/test/windows.json'}

    def synthetic_log(self):
        text = ''
        for i in range(6):
            stamp = f'[2026-09-16 02:0{i + 1}:00.000]'
            text += f"{stamp} step {i}: {{'train/grad_norm': 0.1}}\n"
            text += f"{stamp} rollout {i}: {{'rollout/raw_reward': {i / 10}}}\n"
            text += f"{stamp} perf {i}: {{'perf/train_time': 20, 'perf/step_time': 60}}\n"
        return text

    def test_union_keeps_warmup_save_profile_cohort_and_all_learning_values(self):
        meta = {'optimizer_steps_per_rollout': 1, 'expected_rollouts': 6, 'status': 'RUNNING',
                'profiling_coverage_known': True, 'profiled_rollouts': [5], 'exclude_timing_rollouts': [0, 4]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'train.log'; path.write_text(self.synthetic_log())
            before = self.parser.summarize_run('rubin', path, meta)
            original = deepcopy(before)
            after_meta = C.io_exclusions('rubin', before, meta, self.evidence(), self.helper)
            after = self.parser.summarize_run('rubin', path, after_meta)
        self.assertEqual(before, original)
        self.assertEqual(meta['exclude_timing_rollouts'], [0, 4])
        self.assertEqual(after_meta['exclude_timing_rollouts'], [0, 1, 2, 4])
        self.assertEqual([r['rollout_id'] for r in after['rows'] if r['unprofiled_timing_eligible']], [3])
        self.assertEqual(after['unprofiled_stage_statistics']['step']['count'], 1)
        self.assertEqual(after['unprofiled_stage_statistics']['step']['mean_seconds'], 60)
        for old, new in zip(before['rows'], after['rows']):
            for field in ('metrics', 'train_steps', 'eval', 'common', 'events', 'optimizer_steps_observed'):
                self.assertEqual(old[field], new[field])
        self.assertEqual(before['source_log_sha256'], after['source_log_sha256'])
        self.assertEqual(after_meta['io_overlap']['window_input']['sha256'], self.evidence()['sha256'])
        self.assertEqual(after_meta['io_overlap']['window_input']['evidence'], self.document)

    def test_bad_timestamp_and_stale_active_window_remain_explicit_unknown(self):
        meta = {'optimizer_steps_per_rollout': 1, 'profiling_coverage_known': True, 'exclude_timing_rollouts': [0]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'train.log'; path.write_text(self.synthetic_log())
            run = self.parser.summarize_run('rubin', path, meta)
        bad = deepcopy(self.document); bad['windows'][0]['start_utc'] = 'bad'
        result = C.io_exclusions('rubin', run, meta, self.evidence(bad), self.helper)
        self.assertEqual(result['io_overlap']['assessment']['unknown_rollout_ids'], list(range(6)))
        self.assertIn('invalid_window:copy', result['io_overlap']['assessment']['rows'][0]['unknown_reasons'])
        active = deepcopy(self.document); active['snapshot_utc'] = '2026-09-16T02:02:30Z'
        active['windows'][0].update(status='running', end_utc=None)
        result = C.io_exclusions('rubin', run, meta, self.evidence(active), self.helper)
        self.assertEqual(result['io_overlap']['assessment']['unknown_rollout_ids'], [2, 3, 4, 5])
        self.assertIn('active_window_not_observed_through_row:copy',
                      result['io_overlap']['assessment']['rows'][-1]['unknown_reasons'])

    def test_malformed_window_structure_rejected_before_remote_fetch(self):
        cases = [[], {'schema_version': 1, 'snapshot_utc': 'bad', 'windows': []},
                 {**self.document, 'windows': ['not a record']},
                 {**self.document, 'windows': [{**self.document['windows'][0], 'platform': 'typo'}]},
                 {**self.document, 'windows': self.document['windows'] * 2}]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for document in cases:
                (root / 'io-overlap-windows.json').write_text(json.dumps(document))
                (root / 'comparison.json').write_bytes(b'unchanged')
                with patch.object(C, 'ROOT', root), patch.object(C, 'collect') as fetch:
                    with self.subTest(document=document), self.assertRaises(ValueError):
                        C.collect_snapshot()
                fetch.assert_not_called()
                self.assertEqual((root / 'comparison.json').read_bytes(), b'unchanged')


if __name__ == '__main__': unittest.main()
