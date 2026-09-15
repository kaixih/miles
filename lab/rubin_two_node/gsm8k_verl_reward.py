"""Pinned VeRL GSM8K strict scorer for Miles training and evaluation.

Use --custom-rm-path lab.rubin_two_node.gsm8k_verl_reward.reward_func
    --reward-key reward --eval-reward-key reward
Single Sample calls return a numeric diagnostics dict; list[Sample] calls return
an aligned list of dicts. Both Miles custom reward call contracts are supported.

Provenance: the functions below are copied unchanged from official VeRL commit
152c599303dd4364aa8d581d405a84922dc8c713, verl/utils/reward_score/gsm8k.py.
The source Git blob is 98a8c24dc8c66922ec0518ee31072691db81d4e5; source SHA256 is
497c6e32b708d0bc8803b5a048ca5983de7756eb9c146abdc56e6f5d28767189.
https://github.com/verl-project/verl/blob/152c599303dd4364aa8d581d405a84922dc8c713/verl/utils/reward_score/gsm8k.py

The old GB300 log records c423ebcdba576a65cfd4a9027bffcfe24e013b70. That
commit is unavailable in the local official checkout and the official GitHub
commit API returned 422/no commit found. Exact parity with that historical
image is therefore UNVERIFIED; its floating image tag is not a source pin.
The pinned source matches the local official checkout byte-for-byte.

Adapter differences: adds numeric reward/accuracy/format diagnostics and an
async single/batch API, without importing VeRL or Miles. Scores Sample.response
verbatim, never changes labels, and never zeros a reward based on truncation.
The old VeRL naive manager decoded with skip_special_tokens=True upstream of
this scorer; callers control decoding. No token-marker cleanup is added here.
Strict scoring checks only the last 300 characters and the last '#### number'
match, removes commas from the extracted answer only, and compares strings.
The format diagnostic is parser-match presence, not a format bonus or proof
that the matched text is a valid mathematical number.
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


UPSTREAM_COMMIT = "152c599303dd4364aa8d581d405a84922dc8c713"
UPSTREAM_BLOB = "98a8c24dc8c66922ec0518ee31072691db81d4e5"
UPSTREAM_SHA256 = "497c6e32b708d0bc8803b5a048ca5983de7756eb9c146abdc56e6f5d28767189"
HISTORICAL_RECORDED_COMMIT = "c423ebcdba576a65cfd4a9027bffcfe24e013b70"
HISTORICAL_EXACT_SOURCE_VERIFIED = False


def score_response(response, ground_truth):
    """Shared pure scoring entry point for offline evaluation and the Miles hook."""
    answer = extract_solution(response, method="strict")
    reward = compute_score(response, ground_truth, method="strict")
    return {"reward": reward, "accuracy": reward, "format": int(answer is not None)}


async def reward_func(args, sample_or_samples, **kwargs):
    """Miles async_rm and batched_async_rm adapter; does not mutate samples."""
    if isinstance(sample_or_samples, list):
        return [score_response(sample.response, sample.label) for sample in sample_or_samples]
    return score_response(sample_or_samples.response, sample_or_samples.label)
