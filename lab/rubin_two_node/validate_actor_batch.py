#!/usr/bin/env python3
"""Validate a trusted native Miles debug rollout dump on CPU; print JSON only.

Pickle loading is intentional and restricted to the user's own generated dump.
No GPU initialization, generation, training, mutation or file rewriting occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_samples(samples, *, count=2048, samples_per_prompt=8, reward_key='reward',
                     max_prompt=512, max_response=1024, max_context=4096, scorer=None):
    require(count > 0 and count % samples_per_prompt == 0, 'Invalid expected grouping')
    require(len(samples) == count, f'Expected {count} samples; found {len(samples)}')
    seen, group_records, sample_records = set(), {}, []
    statuses, masks, lengths, rewards = Counter(), Counter(), [], []
    for position, sample in enumerate(samples):
        sample.validate()  # Same structural validator as the installed Miles Sample.
        prefix = f'sample[{position}]'
        require(isinstance(sample.index, Integral) and not isinstance(sample.index, bool), prefix+' invalid index')
        require(sample.index not in seen, prefix+' duplicate sample index')
        seen.add(sample.index)
        require(isinstance(sample.group_index, Integral) and not isinstance(sample.group_index, bool), prefix+' invalid group')
        require(sample.rollout_id is None, prefix+' compact/subagent rollout is outside this recipe')
        status = sample.status.value
        require(status in {'completed', 'truncated'}, prefix+' invalid terminal status '+status)
        require(not sample.remove_sample, prefix+' removed sample would change loss/packing')
        tokens = sample.tokens
        require(isinstance(tokens, (list, tuple)) and all(isinstance(t, Integral) and not isinstance(t, bool) and t >= 0 for t in tokens),
                prefix+' invalid token IDs')
        response = sample.response_length
        require(isinstance(response, Integral) and not isinstance(response, bool) and 0 < response <= max_response,
                prefix+' invalid response length')
        prompt_length = len(tokens)-response
        require(0 < prompt_length <= max_prompt and len(tokens) <= max_context, prefix+' prompt/context length outside recipe')
        require(isinstance(sample.response, str) and isinstance(sample.label, str), prefix+' response/label must be text')
        logprobs = sample.rollout_log_probs
        require(isinstance(logprobs, (list, tuple)) and len(logprobs) == response, prefix+' missing/misaligned rollout logprobs')
        require(all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(float(x)) and float(x) <= 1e-4 for x in logprobs),
                prefix+' invalid log probability')
        mask = sample.loss_mask
        if mask is None:
            masks['implicit_all_ones'] += 1
            effective_mask = [1]*response
        else:
            require(isinstance(mask, (list, tuple)) and len(mask) == response and all(x in (0,1) for x in mask),
                    prefix+' invalid explicit loss mask')
            effective_mask = [int(x) for x in mask]
            masks['explicit'] += 1
        require(sum(effective_mask) > 0, prefix+' has no trainable response tokens')
        raw_reward = sample.reward
        if isinstance(raw_reward, dict):
            require(reward_key in raw_reward, prefix+' missing reward key '+reward_key)
            reward = raw_reward[reward_key]
        else:
            require(not reward_key, prefix+' recipe expects a reward dictionary')
            reward = raw_reward
        require(isinstance(reward, Real) and not isinstance(reward, bool) and math.isfinite(float(reward)) and reward in (0,1),
                prefix+' invalid binary GSM8K reward')
        if scorer is not None:
            expected = scorer(sample.response, sample.label)['reward']
            require(float(reward) == float(expected), prefix+' saved reward disagrees with frozen GSM8K scorer')
        group = int(sample.group_index)
        group_signature = canonical_sha({'prompt':sample.prompt,'label':sample.label,
                                        'prompt_tokens':[int(x) for x in tokens[:prompt_length]]})
        item = group_records.setdefault(group, {'positions':[], 'prompt_sha256':group_signature})
        require(item['prompt_sha256'] == group_signature, prefix+' group prompt/label/tokens mismatch')
        item['positions'].append(position)
        sample_records.append({'index':int(sample.index),'group':group,'status':status,
            'tokens_sha256':canonical_sha([int(x) for x in tokens]),'response_length':int(response),
            'mask_sha256':canonical_sha(effective_mask),'logprobs_sha256':canonical_sha([float(x) for x in logprobs]),
            'reward':float(reward)})
        statuses[status] += 1
        lengths.append((prompt_length, int(response)))
        rewards.append(float(reward))
    require(seen == set(range(count)), 'Fresh rollout must have sample indices 0..count-1')
    require(set(group_records) == set(range(count//samples_per_prompt)), 'Fresh rollout group IDs differ from 0..groups-1')
    for group, item in group_records.items():
        positions = item['positions']
        require(len(positions) == samples_per_prompt, f'group {group} size differs from {samples_per_prompt}')
        require(positions == list(range(positions[0], positions[0]+samples_per_prompt)), f'group {group} is not contiguous')
    return {'sample_count':count, 'prompt_groups':len(group_records), 'samples_per_prompt':samples_per_prompt,
            'status_counts':dict(statuses), 'mask_counts':dict(masks), 'reward_mean':sum(rewards)/count,
            'prompt_tokens':sum(x for x,_ in lengths), 'response_tokens':sum(y for _,y in lengths),
            'prompt_length_min_max':[min(x for x,_ in lengths),max(x for x,_ in lengths)],
            'response_length_min_max':[min(y for _,y in lengths),max(y for _,y in lengths)],
            'ordered_sample_fingerprint':canonical_sha(sample_records),
            'group_fingerprint':canonical_sha(group_records), 'native_sample_validate':True,
            'gsm8k_reward_recomputed':scorer is not None, 'mask_semantics':'None -> all ones, matching native conversion',
            'checks':['count','group_size_and_contiguity','unique_fresh_indices','same_prompt_per_group',
                      'terminal_status','token_and_response_lengths','finite_logprobs','effective_loss_masks','binary_rewards']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--count', type=int, default=2048)
    parser.add_argument('--samples-per-prompt', type=int, default=8)
    parser.add_argument('--reward-key', default='reward')
    parser.add_argument('--max-prompt', type=int, default=512)
    parser.add_argument('--max-response', type=int, default=1024)
    parser.add_argument('--max-context', type=int, default=4096)
    parser.add_argument('--max-bytes', type=int, default=512*1024**2)
    args = parser.parse_args()
    try:
        path = args.path.absolute()
        require(path.is_file() and not path.is_symlink(), 'Require a regular, non-symlink trusted dump')
        before = path.stat()
        require(0 < before.st_size <= args.max_bytes, 'Dump size exceeds validation bound')
        digest = sha_file(path)
        require(not args.expected_sha256 or args.expected_sha256 == digest, 'Input SHA mismatch')
        import torch
        from miles.utils.types import Sample
        from lab.rubin_two_node.gsm8k_verl_reward import score_response
        payload = torch.load(path, map_location='cpu', weights_only=False)
        require(isinstance(payload, dict) and payload.get('rollout_id') == 0, 'Require native rollout 0 payload')
        require(isinstance(payload.get('metadata', {}), dict), 'Invalid native metadata')
        samples = [Sample.from_dict(raw) for raw in payload['samples']]
        result = validate_samples(samples, count=args.count, samples_per_prompt=args.samples_per_prompt,
            reward_key=args.reward_key, max_prompt=args.max_prompt, max_response=args.max_response,
            max_context=args.max_context, scorer=score_response)
        after = path.stat()
        require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
                and sha_file(path)==digest, 'Dump changed during validation')
        result.update(status='PASS',path=str(path),bytes=before.st_size,sha256=digest,
                      validator_sha256=sha_file(Path(__file__)), torch_version=torch.__version__,
                      native_sample_source_sha256=sha_file(Path(sys.modules[Sample.__module__].__file__)),
                      metadata_keys=sorted(payload.get('metadata', {})), rollout_id=0)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except Exception as error:
        print(json.dumps({'status':'FAILED','path':str(args.path),'error':repr(error)}, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
