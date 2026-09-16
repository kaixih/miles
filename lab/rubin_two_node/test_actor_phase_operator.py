"""CPU-only identity, lifecycle and recipe regressions; never calls SSH/Docker."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab.rubin_two_node import actor_phase_operator as op
from lab.rubin_two_node import plan_actor_rollout_batch as planner


class ActorPhaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.c = dict(node='gb300-node', job_id=123, node_ip='10.0.0.1', platform='gb300',
                      lease_deadline='2099-01-01T00:00:00Z', phase_id='actor-profile-test',
                      submission_id='actor-profile-test', container='actor-profile-test',
                      image='registry/img@sha256:'+'1'*64, image_id='sha256:'+'2'*64,
                      durable_phase=str(self.root), node_phase='/raid/test/phase',
                      node_repo='/raid/repo', node_models='/raid/models', models='/models',
                      node_inputs='/raid/inputs', node_helpers='/raid/helpers',
                      node_output='/raid/test/phase/run', node_cache='/raid/test/phase/cache',
                      megatron_path='/opt/Megatron-LM', dashboard='http://10.0.0.1:28265',
                      nccl_iface='eth0', max_runtime_seconds=1800, max_output_bytes=1024**3,
                      source_commit='a'*40)
        self.exact = '3'*64
        self.now = 1000.
        self.deadline = 2000.
        self.stack.enter_context(patch.object(op.time, 'time', return_value=self.now))
        self.identity = {'container_id': self.exact, 'guard': {'pid': 55, 'start_ticks': 678}}
        self.info = {'Id': self.exact, 'Name':'/'+self.c['container'], 'Image':self.c['image_id'],
                     'Config':{'Image':self.c['image'], 'User':'28644:30',
                               'Labels':{'miles.actor_phase':self.c['phase_id']}},
                     'State':{'Running':True},
                     'Mounts':[{'Source':src,'Destination':dst,'RW':rw,'Type':'bind'}
                               for src,dst,rw in op.mounts(self.c)]}
        self.snapshot = {'state': {'state':'armed', 'container_id':self.exact,
                         'deadline':self.deadline, 'config_sha256':op.config_sha(self.c),
                         'pid':55, 'start_ticks':678, 'heartbeat_epoch':self.now},
                         'process':{'pid':55,'start_ticks':678,'uid':28644,'process_state':'S',
                         'argv':['python3','-u',self.c['node_phase']+'/guard.py']}}

    def records(self):
        op.write(self.root/'phase-plan.json', {'config':self.c,'config_sha256':op.config_sha(self.c),
                                             'absolute_deadline':self.deadline})
        op.write(self.root/'container-identity.json', self.identity)
        op.write(self.root/'ready.json', {'container_id':self.exact,'deadline':self.deadline,
                'state':'RAY_READY_NO_SUBMISSION','config_sha256':op.config_sha(self.c),
                'normal_uid_megatron_import':{'uid':28644,'gid':30}})
        plan={'submission_id':self.c['submission_id'],
              'entrypoint':'python3 /opt/miles/train.py --debug-rollout-only',
              'runtime_env':{'env_vars':{'RUBIN_RUN_ID':self.c['phase_id']}}}
        op.write(self.root/'plan.json', plan)
        return self.root/'plan.json'

    def test_mounts_normal_uid_and_exact_container(self):
        op.validate_container(self.c,self.info,exact_id=self.exact)
        with self.assertRaisesRegex(ValueError,'ID mismatch'):
            op.validate_container(self.c,self.info,exact_id='4'*64)
        self.info['Mounts'].append({'Destination':'/evil','Source':'/','RW':True,'Type':'bind'})
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            op.validate_container(self.c,self.info,exact_id=self.exact)
        argv=op.docker_argv(self.c)
        self.assertEqual(argv[argv.index('--user')+1],'28644:30')
        self.assertIn('type=bind,src=/raid/repo,dst=/opt/miles,readonly',argv)

    def test_guard_rejects_replacement_stale_heartbeat_changed_deadline_and_pid(self):
        for target,key,value in [('state','container_id','4'*64),('state','deadline',3000),
                ('state','heartbeat_epoch',900),('state','config_sha256','bad'),
                ('process','start_ticks',679),('process','uid',0),('process','process_state','Z')]:
            snap=copy.deepcopy(self.snapshot);snap[target][key]=value
            with self.subTest(key=key), patch.object(op,'node_python',return_value=snap):
                with self.assertRaisesRegex(ValueError,'guard identity'):
                    op.phase_guard_snapshot(self.c,self.identity,self.deadline)
        with patch.object(op,'node_python',return_value=self.snapshot):
            op.phase_guard_snapshot(self.c,self.identity,self.deadline)

    def test_original_config_cannot_change(self):
        self.records(); altered={**self.c,'max_runtime_seconds':2700}
        with self.assertRaisesRegex(ValueError,'config changed'):
            op.pinned_phase(altered)

    def test_submit_rechecks_identity_immediately_before_post(self):
        path=self.records()
        with patch.object(op,'allocation'), patch.object(op,'submission_ready') as readiness, \
             patch.object(op,'api',side_effect=[[],{'submission_id':self.c['submission_id']}]) as api:
            op.submit(self.c,path)
        self.assertEqual(readiness.call_count,2)
        self.assertEqual([call.args[1] for call in api.call_args_list],['GET','POST'])

    def test_submit_fails_if_guard_dies_during_preflight(self):
        path=self.records()
        with patch.object(op,'allocation'), patch.object(op,'submission_ready',side_effect=[{},ValueError('guard died')]), \
             patch.object(op,'api',return_value=[]) as api:
            with self.assertRaisesRegex(ValueError,'guard died'):op.submit(self.c,path)
        self.assertEqual(api.call_count,1)
        self.assertFalse((self.root/'submitted.json').exists())

    def test_submit_requires_successful_normal_uid_import(self):
        path=self.records(); ready=json.loads((self.root/'ready.json').read_text())
        ready['normal_uid_megatron_import']['uid']=0
        (self.root/'ready.json').write_text(json.dumps(ready))
        with patch.object(op,'allocation'), patch.object(op,'api') as api:
            with self.assertRaisesRegex(ValueError,'readiness/import'):op.submit(self.c,path)
        api.assert_not_called()

    def test_submit_refuses_shell_tail_or_wrong_run_environment(self):
        path=self.records()
        original=json.loads(path.read_text())
        for plan in [{**original,'entrypoint':original['entrypoint']+'; touch /tmp/x'},
                     {**original,'runtime_env':{'env_vars':{'RUBIN_RUN_ID':'other'}}}]:
            path.write_text(json.dumps(plan))
            with patch.object(op,'allocation'), patch.object(op,'submission_ready'), patch.object(op,'api') as api:
                with self.assertRaisesRegex(ValueError,'submission identity'):op.submit(self.c,path)
                api.assert_not_called()

    def test_output_atomic_rename_race_and_symlink(self):
        (self.root/'stable').write_bytes(b'123')
        missing=self.root/'disappears'
        real=Path.lstat
        def race(path):
            if path==missing:raise FileNotFoundError()
            return real(path)
        with patch.object(op.os,'walk',return_value=[(str(self.root),[],['stable','disappears'])]), \
             patch.object(Path,'lstat',race):
            self.assertEqual(op.output_size(self.root),3)
        (self.root/'link').symlink_to(self.root/'stable')
        with self.assertRaisesRegex(ValueError,'symlink'):op.output_size(self.root)

    def test_guard_transient_inspect_failure_keeps_loop_then_exact_stops(self):
        c={**self.c,'node_phase':str(self.root),'node_output':str(self.root)}
        stopped=copy.deepcopy(self.info);stopped['State']['Running']=False
        with patch.object(op,'proc_identity',return_value={'start_ticks':678}), \
             patch.object(op.subprocess,'check_output',side_effect=[subprocess.CalledProcessError(1,'docker'),json.dumps([stopped])]), \
             patch.object(op,'validate_container'), patch.object(op.time,'sleep'), \
             patch.object(op.subprocess,'run') as mutation:
            op.guard(c,self.exact,self.deadline)
        mutation.assert_not_called()
        self.assertEqual(json.loads((self.root/'guard-state.json').read_text())['state'],'container_stopped')
        with patch.object(op,'proc_identity',return_value={'start_ticks':678}), \
             patch.object(op.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,'docker')), \
             patch.object(op.time,'sleep'), patch.object(op.subprocess,'run') as mutation:
            op.guard(c,self.exact,self.deadline)
        self.assertEqual(mutation.call_args.args[0],['docker','stop','--time','15',self.exact])

    def test_stopped_container_retains_guard_on_failed_copy_with_shared_budget(self):
        self.records(); info=copy.deepcopy(self.info);info['State']['Running']=False
        events=[]
        def evidence(c,root,label,deadline):
            events.append((label,deadline));op.write(root/('guard-evidence-'+label+'.json'),{'saved':True})
        with patch.object(op,'allocation',return_value={'lease_timestamp':1200}), \
             patch.object(op,'remote',return_value=json.dumps([info])) as remote, \
             patch.object(op,'retain_guard_evidence',side_effect=evidence), \
             patch.object(op,'node_python',return_value={'part':{'bytes':1,'sha256':'a'*64}}), \
             patch.object(op,'verify_retention_destination'), \
             patch.object(op.subprocess,'run',side_effect=subprocess.TimeoutExpired('rsync',1)) as transfer, \
             patch.object(op,'api') as api:
            with self.assertRaises(subprocess.TimeoutExpired):op.stop_retain(self.c)
        self.assertEqual(events,[('before',1140),('after',1140)])
        argv=transfer.call_args.args[0]
        self.assertEqual(argv[:5],['rsync','-rt','--no-perms','--no-owner','--no-group'])
        self.assertEqual(argv[-2:],[self.c['node']+':'+self.c['node_output']+'/',str(self.root/'node-output')+'/'])
        self.assertIn('ssh -o BatchMode=yes -o ConnectTimeout=10',argv)
        self.assertIn('--rsync-path=timeout --signal=TERM --kill-after=5s 135s rsync',argv)
        self.assertEqual(transfer.call_args.kwargs['timeout'],140)
        self.assertNotIn('--delete',argv)

        api.assert_not_called()
        self.assertTrue((self.root/'node-output').exists())
        self.assertFalse((self.root/'retention.json').exists())
        self.assertFalse(any('stop' in call.args[1] for call in remote.call_args_list))

    def test_retention_destination_requires_normal_uid_owned_non_symlink_directory(self):
        target=SimpleNamespace(lstat=lambda:SimpleNamespace(st_mode=stat.S_IFDIR|0o700,st_uid=28644,st_gid=30))
        with patch.object(op.os,'getuid',return_value=28644),patch.object(op.os,'getgid',return_value=30):
            op.verify_retention_destination(target)
            for mode,owner in [(stat.S_IFLNK,28644),(stat.S_IFDIR,0),(stat.S_IFDIR,28686)]:
                wrong=SimpleNamespace(lstat=lambda:SimpleNamespace(st_mode=mode,st_uid=owner,st_gid=30))
                with self.assertRaisesRegex(ValueError,'owned directory'):op.verify_retention_destination(wrong)
        with patch.object(op.os,'getuid',return_value=0),patch.object(op.os,'getgid',return_value=0):
            with self.assertRaisesRegex(ValueError,'normal dl3'):op.verify_retention_destination(target)

    def test_retention_rejects_new_active_job_before_stop(self):
        self.records()
        with patch.object(op,'allocation',return_value={'lease_timestamp':5000}), \
             patch.object(op,'remote',return_value=json.dumps([self.info])) as remote, \
             patch.object(op,'retain_guard_evidence'), \
             patch.object(op,'api',side_effect=[[],[{'status':'RUNNING'}]]):
            with self.assertRaisesRegex(ValueError,'became active'):op.stop_retain(self.c)
        self.assertFalse(any('stop' in call.args[1] for call in remote.call_args_list))

    def test_expired_retention_cannot_mutate(self):
        self.records()
        with patch.object(op,'allocation',return_value={'lease_timestamp':1059}), \
             patch.object(op,'remote') as remote:
            with self.assertRaises(TimeoutError):op.stop_retain(self.c)
        remote.assert_not_called()

    def test_embedded_guard_and_manifest_compile_without_execution(self):
        # start_guard emits executable source; capture its launch payload only.
        state={**self.snapshot['state'],'pid':55}
        with patch.object(op,'node_python',return_value={'pid':55}) as launch, \
             patch.object(op,'remote',return_value=json.dumps(state)):
            op.start_guard(self.c,self.exact,self.deadline)
        compile(launch.call_args.args[1],'<guard launcher>','exec')
        compile(op.manifest_code('/tmp/output'),'<manifest>','exec')


class RolloutPlanTests(unittest.TestCase):
    def source(self):
        flags={'--rollout-batch-size':'256','--n-samples-per-prompt':'8','--global-batch-size':'512',
               '--num-steps-per-rollout':'4','--max-tokens-per-gpu':'4096','--rollout-max-prompt-len':'512',
               '--rollout-max-response-len':'1024','--num-rollout':'50','--prompt-data':'/original/train.jsonl',
               '--save':'/original/checkpoints','--eval-interval':'5','--rollout-temperature':'1.0',
               '--custom-rm-path':'lab.rubin_two_node.gsm8k_verl_reward.reward_func',
               '--hf-checkpoint':'/models/Qwen3-30B-A3B'}
        command=['python3','/opt/miles/train.py',*[x for pair in flags.items() for x in pair],
                 '--sglang-disable-piecewise-cuda-graph']
        return {'entrypoint':shlex.join(command),'runtime_env':{'env_vars':{'RUBIN_RUN_ID':'old','SEED':'1'}}}

    def test_one_true_rollout_preserves_model_reward_sampling_and_scheduler(self):
        source=self.source(); original=copy.deepcopy(source)
        config={'phase_id':'actor-profile-generate','node_ip':'10.0.0.1','nccl_iface':'eth0',
                'megatron_path':'/opt/Megatron-LM','submission_id':'actor-profile-generate','source_commit':'a'*40}
        plan=planner.build(source,config,'b'*64)
        groups={g[0]:g[1:] for g in planner.groups(shlex.split(plan['entrypoint'])[2:])}
        self.assertEqual(groups['--num-rollout'],['50'])
        self.assertEqual(groups['--debug-exit-after-rollout'],['1'])
        self.assertEqual(groups['--prompt-data'],['/inputs/train.jsonl'])
        self.assertEqual(groups['--custom-rm-path'],['lab.rubin_two_node.gsm8k_verl_reward.reward_func'])
        self.assertEqual(groups['--rollout-temperature'],['1.0'])
        self.assertIn('--debug-rollout-only',groups)
        self.assertIn('--sglang-disable-piecewise-cuda-graph',groups)
        self.assertNotIn('--save',groups);self.assertNotIn('--eval-interval',groups)
        self.assertEqual(source,original)

    def test_refuses_resume_graph_off_and_wrong_batch(self):
        c={'phase_id':'actor-profile-generate','node_ip':'10.0.0.1','nccl_iface':'eth0',
           'megatron_path':'/opt/Megatron-LM','submission_id':'actor-profile-generate','source_commit':'a'*40}
        for extra in [' --load /checkpoint',' --sglang-disable-cuda-graph',' --rollout-batch-size 1']:
            source=self.source();source['entrypoint']+=extra
            with self.assertRaises(ValueError):planner.build(source,c,'b'*64)


if __name__=='__main__':unittest.main()
