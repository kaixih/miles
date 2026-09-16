import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import unittest
import tempfile
from unittest.mock import patch
from lab.rubin_two_node import actor_update_profile as hook
from lab.rubin_two_node import plan_actor_update_replay as planner


def recipe(tmp_path):
    flags = {
        '--actor-num-nodes':'1', '--actor-num-gpus-per-node':'4', '--num-gpus-per-node':'4',
        '--tensor-model-parallel-size':'1', '--pipeline-model-parallel-size':'1',
        '--context-parallel-size':'1', '--expert-model-parallel-size':'4', '--expert-tensor-parallel-size':'1',
        '--global-batch-size':'512', '--rollout-batch-size':'256', '--n-samples-per-prompt':'8',
        '--num-steps-per-rollout':'4', '--num-rollout':'50', '--megatron-to-hf-mode':'raw',
        '--rollout-max-context-len':'4096', '--max-tokens-per-gpu':'8192',
        '--hf-checkpoint':'/models/hf', '--ref-load':'/models/release', '--prompt-data':'/main/inputs/train.jsonl',
        '--save':'/main/checkpoints', '--save-interval':'10', '--eval-interval':'5',
    }
    source = tmp_path/'source.json'
    source.write_text(json.dumps({'argv':[token for kv in flags.items() for token in kv],
                                  'extra_runtime_env':{'PYTHONPATH':'/opt/miles:/opt/Megatron-LM'}}))
    rollout = tmp_path/'rollout.pt'
    rollout.write_bytes(b'trusted rollout file fixture')
    return SimpleNamespace(source_plan=str(source), run_id='actor-profile-test', rollout_path=str(rollout),
        rollout_sha256=hook.digest_file(rollout), initial_model_id='a'*64, train_step_source_sha256='b'*64,
        schedule_source_sha256='d'*64,
        output_root=str(tmp_path/'new-output'), deadline_epoch=time.time()+1000, max_tokens_per_gpu=4096)


def iterator():
    return SimpleNamespace(offset=0, micro_batch_size=None, micro_batch_indices=[[i] for i in range(12)],
                           rollout_data={'tokens':[[i,i+1] for i in range(12)], 'total_lengths':[2]*12,
                                         'response_lengths':[1]*12})


class MonkeyPatch:
    def __init__(self, stack): self.stack = stack
    def setattr(self, obj, name, value): self.stack.enter_context(patch.object(obj, name, value))
    def setitem(self, mapping, key, value): self.stack.enter_context(patch.dict(mapping, {key:value}))
    def setenv(self, key, value): self.setitem(hook.os.environ, key, value)


class ActorProfileTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.monkeypatch = MonkeyPatch(self.stack)

    def test_rank0_captures_one_update(self):
        self.check_hook(0)

    def test_other_rank_never_starts_profiler(self):
        self.check_hook(1)

    def test_plan_preserves_learning_recipe_without_starting_or_creating_output(self):
        tmp_path = self.tmp_path
        opts = recipe(tmp_path)
        plan = planner.build_plan(opts)
        args = plan['argv']
        assert args[args.index('--num-rollout')+1] == '50'
        assert args[args.index('--debug-exit-after-rollout')+1] == '1'
        assert args[args.index('--custom-megatron-init-path')+1] == 'actor_update_profile.install'
        assert '--save' not in args and '--eval-interval' not in args and '--use-pytorch-profiler' not in args
        assert plan['runtime_env']['env_vars']['PYTHONPATH'].startswith('/opt/actor-profile:')
        assert plan['scope']['optimizer_updates'] == 4
        assert not Path(opts.output_root).exists()


    def test_refuses_modified_input_resume_wrong_deadline_and_budget(self):
        tmp_path = self.tmp_path
        opts = recipe(tmp_path)
        opts.rollout_sha256 = 'c'*64
        with self.assertRaisesRegex(ValueError, 'Rollout SHA256'):
            planner.build_plan(opts)
        opts.rollout_sha256 = hook.digest_file(opts.rollout_path)
        opts.max_tokens_per_gpu = 2048
        with self.assertRaisesRegex(ValueError, 'token budget'):
            planner.build_plan(opts)
        opts.max_tokens_per_gpu = 4096
        opts.deadline_epoch = time.time()-1
        with self.assertRaisesRegex(ValueError, 'deadline'):
            planner.build_plan(opts)
        opts.deadline_epoch = time.time()+1000
        value = json.loads(Path(opts.source_plan).read_text())
        value['argv'] += ['--load', '/main/checkpoints']
        Path(opts.source_plan).write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fresh main'):
            planner.build_plan(opts)




    def test_packing_detects_same_length_different_tokens_and_partition(self):
        item = iterator()
        first = hook.packing_record([item], 2)
        item.rollout_data['tokens'][0] = [9,9]
        assert hook.packing_record([item], 2)['fingerprint'] != first['fingerprint']
        item.rollout_data['tokens'][0] = [1,2]
        item.micro_batch_indices[:2] = [[1], [0]]
        assert hook.packing_record([item], 2)['fingerprint'] != first['fingerprint']
        assert item.offset == 0


    def test_budget_bounds_phase_rss_and_bytes(self):
        tmp_path, monkeypatch = self.tmp_path, self.monkeypatch
        config = planner.build_plan(recipe(tmp_path))['config']
        monkeypatch.setattr(hook, 'rss_bytes', lambda: 100)
        budget = hook.Budget(config, tmp_path)
        assert budget.reason(budget.until-1, 100, 0) is None
        assert budget.reason(budget.until, 100, 0) == 'capture_deadline'
        assert budget.reason(0, 101+config['max_rss_growth_bytes'], 0) == 'rss_growth'
        assert budget.reason(0, 100, config['max_trace_bytes']+1) == 'output_size'
        budget.exporting()
        assert budget.until <= config['deadline_epoch']
        assert budget.reason(budget.until, 100, 0) == 'export_deadline'


    def test_identical_native_duplicate_is_accepted_but_conflicting_is_rejected(self):
        opts = recipe(self.tmp_path)
        source = json.loads(Path(opts.source_plan).read_text())
        source['argv'] += ['--moe-token-dispatcher-type', 'alltoall', '--moe-token-dispatcher-type', 'alltoall']
        Path(opts.source_plan).write_text(json.dumps(source))
        plan = planner.build_plan(opts)
        assert plan['argv'].count('--moe-token-dispatcher-type') == 1
        assert plan['argv'][plan['argv'].index('--prompt-data')+1] == '/inputs/train.jsonl'
        source['argv'][-1] = 'allgather'
        Path(opts.source_plan).write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            planner.build_plan(opts)

    def test_plan_requires_explicit_reviewed_per_platform_schedule_hash(self):
        opts=recipe(self.tmp_path)
        opts.schedule_source_sha256='e'*64
        assert planner.build_plan(opts)['config']['schedule_source_sha256']=='e'*64
        opts.schedule_source_sha256='unreviewed'
        with self.assertRaisesRegex(ValueError,'SHA256'): planner.build_plan(opts)

    def test_budget_file_snapshot_tolerates_atomic_receipt_rename(self):
        disappearing = SimpleNamespace(is_file=lambda:True,
            stat=lambda: (_ for _ in ()).throw(FileNotFoundError()))
        stable = SimpleNamespace(is_file=lambda:True, stat=lambda:SimpleNamespace(st_size=13))
        folder = SimpleNamespace(iterdir=lambda:iter([disappearing, stable]))
        assert hook.output_bytes(folder) == 13

    def test_annotations_restore_descriptors_even_on_failure(self):
        observed = []
        @contextlib.contextmanager
        def record(name):
            observed.append(name)
            yield
        class Optimizer:
            def step(self): return 7
        class Checkpoint:
            @staticmethod
            def backward(value): return value + 1
        class MoELayer:
            def forward(self, value): return value * 2
        class Model:
            def modules(self): return [MoELayer()]
        schedule = SimpleNamespace(forward_step=lambda value:value, backward_step=lambda value:value)
        collective = lambda value:value
        torch = SimpleNamespace(profiler=SimpleNamespace(record_function=record),
                                distributed=SimpleNamespace(all_reduce=collective))
        original_step = Optimizer.step
        original_backward = Checkpoint.__dict__['backward']
        modules = {'megatron.core.pipeline_parallel.schedules':schedule,
                   'megatron.core.tensor_parallel.random':SimpleNamespace(CheckpointFunction=Checkpoint)}
        self.monkeypatch.setattr(hook.importlib, 'import_module', modules.__getitem__)
        with self.assertRaisesRegex(RuntimeError, 'original failure'):
            with hook.annotations(torch, None, Optimizer(), [Model()]) as names:
                assert Optimizer().step() == 7
                assert Checkpoint.backward(2) == 3
                assert MoELayer().forward(3) == 6
                assert torch.distributed.all_reduce(9) == 9
                assert 'actor.comm_enqueue.all_reduce' in names
                raise RuntimeError('original failure')
        assert Optimizer.step is original_step
        assert Checkpoint.__dict__['backward'] is original_backward
        assert torch.distributed.all_reduce is collective
        assert 'actor.moe_forward' in observed

    def capture_fixture(self, source):
        history, captured, state = [], [], {'active':False,'events':[]}
        class Profiler:
            def __init__(self, **kwargs): captured.append(kwargs)
            def __enter__(self):
                state['active']=True
                history.append('profile_start')
                return self
            def __exit__(self, *args):
                history.append('profile_stop')
                state['active']=False
            def export_chrome_trace(self, path):
                history.append('export')
                Path(path).write_text(json.dumps({'traceEvents':state['events']}))
        @contextlib.contextmanager
        def record(name):
            if state['active']:
                history.append('enter '+name)
                state['events'].append({'cat':'user_annotation','name':name})
            try:
                yield
            finally:
                if state['active']: history.append('exit '+name)
        def forward_step(data_iterator, num_microbatches, config, current_microbatch=None):
            history.append('fwd'+str(current_microbatch))
            data_iterator.offset+=1
            if state['active']: state['events'].append({'cat':'kernel','name':'gemm'})
            return object(), 1
        def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
            history.append('bwd')
        schedule=SimpleNamespace(__file__=str(source),forward_step=forward_step,backward_step=backward_step,
                                 forward_backward_no_pipelining=lambda:None)
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:history.append('sync')),
            profiler=SimpleNamespace(profile=Profiler,record_function=record,ProfilerActivity=SimpleNamespace(CPU=1,CUDA=2)))
        return torch,schedule,history,captured

    def test_microbatch_order_complete_pair_cleanup_and_no_early_export(self):
        torch,schedule,history,captured=self.capture_fixture(self.tmp_path/'schedule.py')
        data=iterator(); original=schedule.forward_step
        cfg=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
        with hook.MicrobatchCapture(torch,schedule,data,3) as capture:
            for i in range(3):
                output,_=schedule.forward_step(data,3,cfg,current_microbatch=i)
                schedule.backward_step(None,output,None,cfg)
        assert history.index('profile_start') > history.index('fwd0')
        assert history.index('profile_stop') < history.index('fwd2')
        assert history.count('profile_start')==history.count('profile_stop')==1
        assert 'export' not in history
        assert capture.observation()['forward_calls']==capture.observation()['backward_calls']==3
        assert capture.selected_indices==[1]
        assert schedule.forward_step is original

    def test_microbatch_refuses_wrong_order_tensor_and_cleans_active_profiler(self):
        cfg=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
        for mode in ('wrong_order','wrong_tensor','original_exception'):
            with self.subTest(mode=mode):
                torch,schedule,history,captured=self.capture_fixture(self.tmp_path/'schedule.py')
                data=iterator();original=schedule.forward_step
                with self.assertRaises(RuntimeError):
                    with hook.MicrobatchCapture(torch,schedule,data,3):
                        out,_=schedule.forward_step(data,3,cfg,current_microbatch=0)
                        if mode=='wrong_order':
                            schedule.forward_step(data,3,cfg,current_microbatch=1)
                        schedule.backward_step(None,out,None,cfg)
                        out,_=schedule.forward_step(data,3,cfg,current_microbatch=1)
                        if mode=='original_exception': raise RuntimeError('original body exception')
                        schedule.backward_step(None,object(),None,cfg)
                assert schedule.forward_step is original
                assert history.count('profile_start')==history.count('profile_stop')
                assert 'export' not in history

    def test_schedule_guard_refuses_pipeline_overlap_and_unknown_source(self):
        source=self.tmp_path/'schedule.py';source.write_text('# exact schedule')
        torch,schedule,_,_=self.capture_fixture(source)
        module=SimpleNamespace(get_forward_backward_func=lambda:schedule.forward_backward_no_pipelining)
        args=SimpleNamespace(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,
                             context_parallel_size=1,expert_model_parallel_size=4)
        hook.validate_native_schedule(args,module,schedule,hook.digest_file(source))
        args.overlap_moe_expert_parallel_comm=True
        with self.assertRaisesRegex(ValueError,'Overlapping'):hook.validate_native_schedule(args,module,schedule,hook.digest_file(source))
        args.overlap_moe_expert_parallel_comm=False
        args.pipeline_model_parallel_size=2
        with self.assertRaisesRegex(ValueError,'TP1'):hook.validate_native_schedule(args,module,schedule,hook.digest_file(source))
        args.pipeline_model_parallel_size=1
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):hook.validate_native_schedule(args,module,schedule,'c'*64)

    def check_hook(self, rank):
        tmp_path, monkeypatch = self.tmp_path, self.monkeypatch
        opts = recipe(tmp_path)
        config = planner.build_plan(opts)['config']
        source = tmp_path/'model.py'
        source.write_text('# frozen native model source\n')
        config['train_step_source_sha256'] = hook.digest_file(source)
        native_schedule=tmp_path/'schedules.py';native_schedule.write_text('# pinned no-pipeline scheduler')
        config['schedule_source_sha256']=hook.digest_file(native_schedule)
        torch,schedule,history,captured=self.capture_fixture(native_schedule)
        called=[]
        def original(args, rollout_id, step_id, data_iterator, model, optimizer,
                     opt_param_scheduler, num_microbatches, num_rollouts, witness_info, attempt):
            called.append(step_id)
            cfg=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
            for i in range(num_microbatches):
                out,_=schedule.forward_step(data_iterator[0],num_microbatches,cfg,current_microbatch=i)
                schedule.backward_step(None,out,None,cfg)
            history.append('optimizer'+str(step_id))
            history.append('update_return'+str(step_id))
            return ({'loss':0.1}, 2.0, SimpleNamespace(name='NORMAL'))
        model_module = SimpleNamespace(__file__=str(source), train_one_step=original,
                                       get_forward_backward_func=lambda:schedule.forward_backward_no_pipelining)
        monkeypatch.setitem(sys.modules, 'miles', SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'miles.backends', SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'miles.backends.megatron_utils', SimpleNamespace(model=model_module))
        torch.distributed=SimpleNamespace(get_world_size=lambda:4,get_rank=lambda:rank,barrier=lambda:None)
        torch.__version__='test'
        monkeypatch.setitem(sys.modules, 'torch', torch)
        modules={'megatron.core.pipeline_parallel.schedules':schedule,
                 'megatron.core.tensor_parallel.random':SimpleNamespace()}
        monkeypatch.setattr(hook.importlib, 'import_module', modules.__getitem__)
        monkeypatch.setattr(hook.os, 'getuid', lambda:28644)
        monkeypatch.setattr(hook.os, 'getgid', lambda:30)
        monkeypatch.setattr(hook, 'rss_bytes', lambda:100)
        monkeypatch.setenv(hook.RUN_ENV, config['run_id'])
        monkeypatch.setenv(hook.ENV, json.dumps(config))
        args=SimpleNamespace(debug_train_only=True,use_pytorch_profiler=False,save=None,debug_disable_optimizer=False,
            load_debug_rollout_data=opts.rollout_path,hf_checkpoint='/models/hf',ref_load='/models/release',
            tensor_model_parallel_size=1,pipeline_model_parallel_size=1,context_parallel_size=1,expert_model_parallel_size=4)
        hook.install(args)
        data = iterator()
        for step in range(4):
            result = model_module.train_one_step(args,0,step,[data],[],object(),None,3,512,None,0)
            assert result[1] == 2.0
        assert called == [0,1,2,3]
        output = Path(opts.output_root)/f'rank{rank}'
        assert len(list(output.glob('step*-timing.json'))) == 4
        assert len(list(output.glob('step*-packing.json'))) == 4
        assert json.loads((output/'step2-timing.json').read_text())['profiled_rank0'] is False
        assert len(captured) == (1 if rank==0 else 0)
        if rank == 0:
            assert captured[0]['with_stack'] is False
            assert captured[0]['record_shapes'] is False
            assert captured[0]['profile_memory'] is False
            receipt=json.loads((output/'receipt.json').read_text())
            assert receipt['status'] == 'COMPLETE'
            assert receipt['capture_window']=='single_forward_backward_microbatch'
            assert receipt['selected_microbatch']['local_indices']==[4]
            assert receipt['schedule_observation']['forward_calls']==3
            assert receipt['optimizer'] is None
            assert history.index('export') > history.index('update_return1')
            assert history.index('profile_stop') < history.index('optimizer1')
            raw=json.loads((output/'actor-update.json').read_text())['traceEvents']
            assert sum(e['name']=='actor.forward_step' for e in raw)==1
            assert sum(e['name']=='actor.backward_step' for e in raw)==1
            assert not any(e['name']=='actor.optimizer_step' for e in raw)
        else:
            assert not (output/'receipt.json').exists()

if __name__ == "__main__":
    unittest.main()
