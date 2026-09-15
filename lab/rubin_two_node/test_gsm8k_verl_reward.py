"""CPU tests for strict formatting and Miles single/batched custom reward calls."""

import asyncio
import unittest
from types import SimpleNamespace

import gsm8k_verl_reward as scorer


class StrictGsm8kRewardTests(unittest.TestCase):
    def test_strict_answer_and_format_boundaries(self):
        cases = [
            ("Reasoning\n#### 7", "7", 1, 1),
            ("#### 1,234", "1234", 1, 1),
            ("#### -1,234.50", "-1234.50", 1, 1),
            ("#### 7", "8", 0, 1),
            ("Answer: \\boxed{7}", "7", 0, 0),
            ("The answer is 7", "7", 0, 0),
            ("####7", "7", 0, 0),
            ("####  7", "7", 0, 0),
            ("####\n7", "7", 0, 0),
            ("#### $7", "7", 0, 0),
            ("#### +7", "7", 0, 0),
            ("#### 7.0", "7", 0, 1),
            ("#### 1,234", "1,234", 0, 1),
            ("", "7", 0, 0),
        ]
        for response, label, reward, format_match in cases:
            with self.subTest(response=response, label=label):
                self.assertEqual(
                    scorer.score_response(response, label),
                    {"reward": reward, "accuracy": reward, "format": format_match},
                )

    def test_last_matching_answer_wins(self):
        response = "#### 7\nRevised answer: #### 8"
        self.assertEqual(scorer.score_response(response, "7")["reward"], 0)
        self.assertEqual(scorer.score_response(response, "8")["reward"], 1)
        # A later malformed marker is not a match and does not replace the prior answer.
        self.assertEqual(scorer.score_response("#### 7\n#### unknown", "7")["reward"], 1)

    def test_exact_300_character_window(self):
        self.assertEqual(scorer.score_response("#### 7" + "x" * 294, "7")["reward"], 1)
        self.assertEqual(scorer.score_response("#### 7" + "x" * 295, "7")["format"], 0)
        self.assertEqual(scorer.score_response("x" * 400 + "#### 7", "7")["reward"], 1)

    def test_labels_are_not_coerced_or_repaired(self):
        self.assertEqual(scorer.score_response("#### 7", 7)["reward"], 0)
        self.assertEqual(scorer.score_response("#### 7", " 7")["reward"], 0)
        # Preserve the upstream regex's permissiveness; format is match presence only.
        self.assertEqual(scorer.score_response("#### .", "7"), {"reward": 0.0, "accuracy": 0.0, "format": 1})

    def test_async_single_truncated_sample_is_not_zeroed_or_mutated(self):
        sample = SimpleNamespace(response="#### 7", label="7", status="truncated", reward=None)
        original = vars(sample).copy()
        result = asyncio.run(scorer.reward_func(SimpleNamespace(), sample, evaluation=True))
        self.assertEqual(result, {"reward": 1.0, "accuracy": 1.0, "format": 1})
        self.assertEqual(vars(sample), original)

    def test_async_batch_preserves_alignment_and_matches_single(self):
        samples = [SimpleNamespace(response="#### 7", label="7"), SimpleNamespace(response="7", label="7")]
        args = SimpleNamespace(reward_key="reward", eval_reward_key="reward")
        batch = asyncio.run(scorer.reward_func(args, samples))
        singles = [asyncio.run(scorer.reward_func(args, sample)) for sample in samples]
        self.assertEqual(batch, singles)
        self.assertEqual([result[args.reward_key] for result in batch], [1, 0])
        self.assertEqual(asyncio.run(scorer.reward_func(args, [])), [])


if __name__ == "__main__":
    unittest.main()
