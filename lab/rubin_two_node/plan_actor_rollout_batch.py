#!/usr/bin/env python3
"""Create a plan for one real initial-policy rollout used by both actor replays."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def groups(argv):
    result = []
    for token in argv:
        if token.startswith('--'):
            result.append(token.split('=', 1))
        elif result:
            result[-1].append(token)
        else:
            raise ValueError('Expected train flag')
    return result


def build(source, config, source_sha256):
    job = source.get('ray_job', source)
    command = shlex.split(job['entrypoint'])
    if command[:2] != ['python3', '/opt/miles/train.py']:
        raise ValueError('Expected exact direct frozen Miles entrypoint')
    original = groups(command[2:])
    required = {'--rollout-batch-size': '256', '--n-samples-per-prompt': '8',
                '--global-batch-size': '512', '--num-steps-per-rollout': '4',
                '--max-tokens-per-gpu': '4096', '--rollout-max-prompt-len': '512',
                '--rollout-max-response-len': '1024', '--num-rollout': '50'}
    for flag, value in required.items():
        if [x[1:] for x in original if x[0] == flag] != [[value]]:
            raise ValueError('Original recipe mismatch: ' + flag)
    forbidden = {'--load', '--ckpt-step', '--debug-train-only', '--debug-rollout-only',
                 '--load-debug-rollout-data', '--sglang-disable-cuda-graph'}
    if any(g[0] in forbidden for g in original):
        raise ValueError('Source must be the fresh-policy graph-ON learning run')
    omitted = {'--skip-eval-before-train', '--use-wandb', '--use-tensorboard',
               '--tensorboard-dir', '--profile', '--use-pytorch-profiler',
               '--record-memory-history', '--debug-exit-after-rollout', '--start-rollout-id'}
    kept, removed = [], []
    for g in original:
        if g[0] in omitted or g[0].startswith(('--save', '--eval', '--wandb-', '--profile-')):
            removed.append(g)
        else:
            kept.extend(['--prompt-data', '/inputs/train.jsonl'] if g[0] == '--prompt-data' else g)
    kept += ['--start-rollout-id', '0', '--debug-exit-after-rollout', '1',
             '--debug-rollout-only', '--save-debug-rollout-data', '/run-output/frozen/rollout-{rollout_id}.pt']
    environment = dict(job['runtime_env']['env_vars'])
    environment.update({'RUBIN_RUN_ID': config['phase_id'], 'MASTER_ADDR': config['node_ip'],
                        'NCCL_SOCKET_IFNAME': config['nccl_iface'], 'GLOO_SOCKET_IFNAME': config['nccl_iface'],
                        'PYTHONPATH': '/opt/actor-profile:/opt/miles:' + config['megatron_path'],
                        'no_proxy': 'localhost,127.0.0.1,' + config['node_ip']})
    return {'schema': 'actor-frozen-rollout-plan-v1', 'source_plan_sha256': source_sha256,
            'submission_id': config['submission_id'], 'source_commit': config['source_commit'],
            'phase_id': config['phase_id'], 'entrypoint': shlex.join(command[:2] + kept),
            'runtime_env': {'env_vars': environment}, 'removed_source_flag_groups': removed,
            'expected_samples': 2048, 'output_file': '/run-output/frozen/rollout-0.pt',
            'scope': 'One real initial-policy GSM8K batch, same recorded generation recipe. No training or evaluation. Both actor replays must consume its retained SHA-bound file.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.source_plan.read_bytes()
    plan = build(json.loads(raw), json.loads(a.config.read_text()), hashlib.sha256(raw).hexdigest())
    with a.output.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'plan': str(a.output), 'submission_id': plan['submission_id'],
                      'expected_samples': plan['expected_samples']}))


if __name__ == '__main__':
    main()
