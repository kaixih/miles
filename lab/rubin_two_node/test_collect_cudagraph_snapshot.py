"""CPU-only tests: no SSH, no live jobs, no GPU operations."""
import base64
from copy import deepcopy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import zlib

SPEC = importlib.util.spec_from_file_location('cg_collector', Path(__file__).with_name('collect_cudagraph_snapshot.py'))
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)


def config(platform='gb300'):
    job = '2203647' if platform == 'gb300' else '2203648'
    run = '20260916-' + platform + '-j' + job + '-cg'
    return {'platform': platform, 'run_id': run, 'job_id': job, 'node': platform + '-test', 'node_ip': '10.0.0.1',
            'run_dir': '/home/scratch.kaixih_ent/repro/miles-qwen3-cudagraph/' + run,
            'node_run_dir': '/raid/tmp/miles-kaixih-j' + job + '-cudagraph/run',
            'dashboard_port': 28265, 'lease_deadline': '2026-09-16T14:50:55Z',
            'image': 'image@sha256:' + 'a' * 64, 'source_commit': 'source'}


def trtllm_config(platform='gb300'):
    c = config(platform)
    job = '2212644' if platform == 'gb300' else '2212643'
    run = '20260917-' + platform + '-j' + job + '-trtllm'
    c.update(job_id=job, run_id=run,
             run_dir='/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/' + run,
             node_run_dir='/raid/tmp/miles-kaixih-j' + job + '-trtllm/run',
             sglang_moe_runner_backend='flashinfer_trtllm', save_optimizer=False)
    return c


def packet(raw):
    return {'bytes': len(raw), 'sha256': C.sha(raw), 'data': base64.b64encode(zlib.compress(raw)).decode()}


def payload(c, prior=b'', tail=b''):
    return {'run_id': c['run_id'], 'root': c['run_dir'], 'observed_at': '2026-09-16T08:00:00Z',
            'files': {}, 'log': {'offset': len(prior), 'prefix_sha256': C.sha(prior), 'bytes': len(prior+tail),
                                'sha256': C.sha(prior+tail), 'tail': packet(tail)},
            'allocation': {'allowed': False, 'reason': 'fixture'}, 'live': None}


def evidence(c):
    entry = ('python3 train.py --num-rollout 50 --num-steps-per-rollout 4 --max-tokens-per-gpu 4096 '
             '--save /run-output/checkpoints --save-interval 50 --rollout-batch-size 256 '
             '--n-samples-per-prompt 8 --global-batch-size 512 --rollout-max-response-len 1024 '
             '--rollout-max-prompt-len 512 --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 '
             '--sglang-disable-piecewise-cuda-graph --sglang-bf16-gemm-backend torch')
    if 'sglang_moe_runner_backend' in c:
        entry += ' --sglang-moe-runner-backend ' + c['sglang_moe_runner_backend']
    if not c.get('save_optimizer', True):
        entry += ' --no-save-optim'
    source = b'{"source":"hash"}\n'
    files = {'driver-config.json': C.encoded(c), 'source-manifest.json': source,
             'train-launch.json': C.encoded({'git_commit': 'source', 'source_manifest_sha256': C.sha(source)}),
             C.LOG: ('Running entrypoint for job job123: ' + entry + '\n'
                     'saving checkpoint at iteration 9 to /run-output/checkpoints\n').encode()}
    p = payload(c)
    p['live'] = {'ray': {'run_id': c['run_id'], 'submission_id': 'job123', 'entrypoint': entry, 'status': 'RUNNING'},
                 'watchdog': {'run_id': c['run_id'], 'submission_id': 'job123', 'ray_status': 'RUNNING'}}
    return files, p


class CollectorTests(unittest.TestCase):
    def test_configuration_refuses_old_run_wrong_platform_job_and_path(self):
        good = config(); C.validate_config('gb300', good)
        for key, bad in [('run_id', 'old-run'), ('platform', 'rubin'), ('job_id', '100'),
                         ('run_dir', good['run_dir'] + '/other'), ('node_run_dir', '/raid/tmp/j2203647/../run'),
                         ('node_run_dir', '/raid/tmp/j2203648/run'), ('node', 'host;evil')]:
            c = dict(good, **{key: bad})
            with self.subTest(key=key), self.assertRaises(ValueError): C.validate_config('gb300', c)

    def test_exact_trtllm_campaign_configuration(self):
        for platform in ('gb300', 'rubin'):
            c = trtllm_config(platform); C.validate_config(platform, c)
            self.assertEqual(C.campaign(platform, c['run_id'])['name'], 'qwen3-trtllm-full-v1')
            changes = [
                {'run_id': c['run_id'].replace('20260917', '20260918')},
                {'job_id': '999999'},
                {'run_dir': c['run_dir'].replace('miles-qwen3-trtllm-full', 'miles-qwen3-cudagraph')},
                {'run_id': c['run_id'].replace('-trtllm', '-cg')},
                *({'sglang_moe_runner_backend': value} for value in ('triton', 'flashinfer_cutlass', '', None, [])),
                {'save_optimizer': 'false'},
            ]
            for change in changes:
                with self.subTest(platform=platform, change=change), self.assertRaises(ValueError):
                    C.validate_config(platform, {**c, **change})
            del c['sglang_moe_runner_backend']
            with self.assertRaisesRegex(ValueError, 'requires flashinfer_trtllm'):
                C.validate_config(platform, c)

    def test_allocation_requires_owner_node_running_and_exact_unexpired_lease(self):
        c = config(); now = dt.datetime(2026,9,16,8,tzinfo=dt.timezone.utc)
        text = 'JobId=2203647 JobState=RUNNING NodeList=gb300-test NumNodes=1 UserId=kaixih(28644) EndTime=2026-09-16T14:50:55'
        self.assertTrue(C.allocation_guard(c, text, now)[0])
        for old,new in [('RUNNING','COMPLETED'), ('28644','123'), ('gb300-test','other'), ('14:50:55','15:50:55'), ('2203647','2203648')]:
            self.assertFalse(C.allocation_guard(c, text.replace(old,new), now)[0])
        self.assertFalse(C.allocation_guard(c,text,now.replace(hour=15))[0])

    def test_expired_or_other_owner_never_connects_to_node(self):
        with tempfile.TemporaryDirectory() as d:
            c = config(); c['run_dir'] = str(Path(d).resolve())
            with patch.object(signal, 'alarm'), patch.object(C.subprocess, 'check_output', return_value=b'JobId=2203647 JobState=COMPLETED') as call:
                result = C.remote_read(c, {'bytes':0,'sha256':C.sha(b'')}, [], 'SHOULD NEVER EXECUTE')
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[0][:4], ['env','TZ=UTC','scontrol','show'])
            self.assertIsNone(result['live']); self.assertFalse(result['allocation']['allowed'])

    def test_live_api_uses_verified_host_ip_and_retains_only_exact_job_whitelist(self):
        import io, socket, urllib.request
        c=config(); c['lease_deadline']='2099-09-16T14:50:55Z'
        jobs=[{'submission_id':'job123','entrypoint':'python train.py','status':'RUNNING',
               'runtime_env':{'env_vars':{'RUBIN_RUN_ID':c['run_id'],'SECRET':'not retained'}}},
              {'submission_id':'unrelated','runtime_env':{'env_vars':{'RUBIN_RUN_ID':'other','SECRET':'other secret'}}}]
        with patch.object(socket,'gethostname',return_value=c['node']), patch.object(C.subprocess,'check_output',return_value=b'[{"addr_info":[{"local":"10.0.0.1"}]}]'), patch.object(urllib.request,'build_opener') as opener:
            opener.return_value.open.return_value=io.BytesIO(json.dumps(jobs).encode())
            result=C.node_read(c)
            self.assertEqual(opener.return_value.open.call_args.args[0],'http://10.0.0.1:28265/api/jobs/')
        self.assertEqual(result['ray']['submission_id'],'job123')
        self.assertNotIn('SECRET',json.dumps(result)); self.assertNotIn('unrelated',json.dumps(result))

    def test_incremental_transfer_preserves_unicode_and_partial_lines(self):
        c = config(); old = 'text\u2028not a newline\npar'.encode(); tail = b'tial\n'
        p = payload(c,old,tail)
        self.assertEqual(C.decode(c,p,old)[C.LOG],old+tail)
        for key,value in [('prefix_sha256','0'*64),('sha256','0'*64),('offset',1)]:
            bad = deepcopy(p); bad['log'][key] = value
            with self.assertRaises(ValueError): C.decode(c,bad,old)
        p['files']['../unrelated'] = packet(b'bad')
        with self.assertRaises(ValueError): C.decode(c,p,old)

    def test_actual_argv_graph_is_request_only_and_save_exclusions(self):
        c = config(); files,p = evidence(c); m = C.metadata('gb300',c,p,files)
        self.assertEqual(m['recipe']['max_logprob_tokens_per_gpu'],4096)
        self.assertEqual(m['expected_rollouts'],50); self.assertEqual(m['optimizer_steps_per_rollout'],4)
        self.assertEqual(m['exclude_timing_rollouts'],[0,9,10])
        self.assertTrue(m['graph']['decode_requested']); self.assertFalse(m['graph']['prefill_requested'])
        self.assertFalse(m['graph']['replay_verified']); self.assertTrue(m['profiling_coverage_known'])
        self.assertEqual(m['kernels']['sglang_bf16_gemm'],'torch')
        self.assertEqual(m['recipe']['sglang_moe_runner_backend'], 'triton')

    def test_trtllm_actual_argv_backend_graph_and_checkpoint_guards(self):
        c = trtllm_config(); files, p = evidence(c)
        m = C.metadata('gb300', c, p, files)
        self.assertEqual(m['kernels']['sglang_moe'], 'flashinfer_trtllm')
        self.assertEqual(m['recipe']['sglang_moe_runner_backend'], 'flashinfer_trtllm')
        self.assertFalse(m['recipe']['save_optimizer'])
        self.assertTrue(m['graph']['decode_requested'])
        self.assertFalse(m['graph']['prefill_requested'])
        self.assertEqual(m['expected_rollouts'], 50)
        self.assertEqual(m['optimizer_steps_per_rollout'], 4)
        replacements = [
            ('flashinfer_trtllm', 'triton', 'backend'),
            (' --sglang-moe-runner-backend flashinfer_trtllm', '', 'backend'),
            ('--sglang-bf16-gemm-backend torch', '--sglang-bf16-gemm-backend torch --sglang-disable-cuda-graph', 'graph'),
            ('--sglang-bf16-gemm-backend torch', '--sglang-bf16-gemm-backend torch --sglang-disable-decode-cuda-graph', 'graph'),
            ('--sglang-bf16-gemm-backend torch', '--sglang-bf16-gemm-backend torch --sglang-cuda-graph-backend-decode disabled', 'graph'),
            ('--sglang-disable-piecewise-cuda-graph', '', 'graph'),
            (' --no-save-optim', '', 'optimizer checkpoint'),
        ]
        for old, new, error in replacements:
            files, p = evidence(c)
            before = p['live']['ray']['entrypoint']; after = before.replace(old, new)
            p['live']['ray']['entrypoint'] = after
            files[C.LOG] = files[C.LOG].replace(before.encode(), after.encode())
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, error):
                C.metadata('gb300', c, p, files)

    def test_configured_legacy_backend_must_also_match_actual(self):
        c = config(); files, p = evidence(c)
        c['sglang_moe_runner_backend'] = 'flashinfer_cutlass'
        files['driver-config.json'] = C.encoded(c)
        with self.assertRaisesRegex(ValueError, 'backend'):
            C.metadata('gb300', c, p, files)

    def test_source_job_and_entrypoint_mismatch_rejected(self):
        c = config()
        for field in ('source','identity','entry','profiler'):
            files,p = evidence(c)
            if field=='source': files['source-manifest.json'] = b'changed'
            if field=='identity': p['live']['ray']['run_id']='other'
            if field=='entry': p['live']['ray']['entrypoint'] += ' --different'
            if field=='profiler':
                p['live']['ray']['entrypoint'] += ' --use-pytorch-profiler'
                files[C.LOG] = files[C.LOG].replace(b'--num-rollout',b'--use-pytorch-profiler --num-rollout')
                p['live']['ray']['entrypoint'] = files[C.LOG].decode().split('\n')[0].split(': ',1)[1]
            with self.subTest(field=field), self.assertRaises(ValueError): C.metadata('gb300',c,p,files)

    def test_retained_running_is_unknown_but_terminal_watch_wins(self):
        c = config(); files,p = evidence(c)
        files['ray-job.json'] = C.encoded(p['live']['ray']); p['live']=None
        self.assertEqual(C.metadata('gb300',c,p,files)['status'],'UNKNOWN')
        files['watchdog.json'] = C.encoded({'run_id':c['run_id'],'submission_id':'job123','terminal_status':'SUCCEEDED'})
        self.assertEqual(C.metadata('gb300',c,p,files)['status'],'SUCCEEDED')

    def test_pending_platform_commit_and_failed_fetch_no_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out=root/'new-cudagraph'; cfg=root/'config.json'
            for label in ('rubin','gb300'): (root/(label+'.json')).write_bytes(C.encoded(config(label)))
            cfg.write_text(json.dumps({'gb300':'gb300.json','rubin':None}))
            def fetch(label,c,output):
                files,p=evidence(c); return label,C.metadata(label,c,p,files),files
            got=C.collect(cfg,out,fetch)
            self.assertEqual(got['pending_platforms'],['rubin'])
            comparison=(out/'comparison.json').read_bytes(); raw=(out/'gb300'/C.LOG).read_bytes()
            self.assertEqual(json.loads((out/'health.json').read_bytes())['comparison_sha256'],C.sha(comparison))
            cfg.write_text(json.dumps({'gb300':'gb300.json','rubin':'rubin.json'}))
            def failure(label,c,output):
                if label=='rubin': raise RuntimeError('bounded read failed')
                return fetch(label,c,output)
            with self.assertRaises(RuntimeError): C.collect(cfg,out,failure)
            self.assertEqual((out/'comparison.json').read_bytes(),comparison)
            self.assertEqual((out/'gb300'/C.LOG).read_bytes(),raw)

    def test_final_completion_requires_200_positive_gradients_and_800_rank_steps(self):
        events=[{'logged_id':i,'metrics':{'train/grad_norm':0.2,'train/loss':0.0}} for i in range(200)]
        run={'metadata':{'status':'SUCCEEDED','expected_rollouts':50,'optimizer_steps_per_rollout':4,
                         'driver_exits':{n:{'exit_code':0} for n in ('train_exit.json','train-driver-exit.json')}},
             'rows':[{'train_steps':events}], 'completed_training_rollouts':list(range(50)), 'parse_errors':[], 'conflicts':[]}
        raw='\n'.join(f'[UTC actor_cell0_rank{rank}] model.py:643 - train op=train_step rollout={r} step={step} attempt=0 outcome=NORMAL valid_step=true'
                      for r in range(50) for step in range(4) for rank in range(4)).encode()
        result=C.completion_validation(run,raw)
        self.assertEqual(result['status'],'PASS'); self.assertEqual(result['normal_rank_steps_observed'],800)
        bad=deepcopy(run); bad['rows'][0]['train_steps'][99]['metrics']['train/grad_norm']=0.0
        self.assertEqual(C.completion_validation(bad,raw)['status'],'NOT_PROVEN')
        self.assertFalse(C.completion_validation(run,raw.replace(b'outcome=NORMAL',b'outcome=SKIPPED',1))['checks']['all_rank_steps_normal'])
        bad=deepcopy(run); bad['metadata']['driver_exits'].pop('train_exit.json')
        self.assertFalse(C.completion_validation(bad,raw)['checks']['driver_exits_zero'])

    def test_campaigns_cannot_be_mixed_and_new_comparison_is_labeled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root / 'comparison'; cfg = root / 'config.json'
            for platform in ('gb300', 'rubin'):
                (root / (platform + '.json')).write_bytes(C.encoded(trtllm_config(platform)))
            cfg.write_text(json.dumps({'gb300': 'gb300.json', 'rubin': 'rubin.json'}))
            def fetch(platform, c, output):
                files, p = evidence(c)
                return platform, C.metadata(platform, c, p, files), files
            C.collect(cfg, out, fetch)
            self.assertEqual(json.loads((out / 'comparison.json').read_bytes())['experiment_id'],
                             'qwen3-trtllm-full-v1')
            (root / 'rubin.json').write_bytes(C.encoded(config('rubin')))
            with self.assertRaisesRegex(ValueError, 'mix'), patch.object(C, 'collect_one', side_effect=AssertionError):
                C.collect(cfg, root / 'mixed-comparison', fetch)

    def test_old_artifact_roots_refused_before_read(self):
        for path in [C.WORKSPACE/'outputs/rubin-gb300-qwen3', C.WORKSPACE/'reports/rubin-gb300-qwen3/site', C.WORKSPACE]:
            with self.assertRaises(ValueError): C.collect('/does/not/exist',path)


if __name__ == '__main__': unittest.main()
