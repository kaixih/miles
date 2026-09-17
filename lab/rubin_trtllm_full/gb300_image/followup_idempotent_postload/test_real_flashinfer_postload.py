"""Real FlashInfer CPU layout tests for PR33743 plus final Miles post-load.

The model shell is lightweight; FlashInfer permutation and BlockMajorK helpers
and SGLang's actual process/restore/repack methods are used without mocking.
No GPU kernel is launched. Geometry matches Qwen3-30B-A3B TP1 experts.
"""
import argparse
import json
import sys
import unittest
from types import SimpleNamespace

import torch
from flashinfer.fused_moe import core as real_flashinfer
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod


class Layer(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.num_local_experts = 2
        self.hidden_size = 2048
        self.intermediate_size_per_partition = 768
        self.moe_runner_config = SimpleNamespace(is_gated=True)
        self.quant_method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.w13_weight = torch.nn.Parameter(torch.randn(
            2, 1536, 2048, generator=generator, dtype=torch.bfloat16), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.randn(
            2, 2048, 768, generator=generator, dtype=torch.bfloat16), requires_grad=False)


NAMES = ("w13_weight", "w2_weight")


def cold(seed):
    layer = Layer(seed)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def restore(layer, name):
    layer.quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        layer, getattr(layer, name), f"model.layers.0.mlp.experts.{name}")


def snapshot(layer):
    return {name: getattr(layer, name).detach().clone() for name in NAMES}


class RealFlashInferPostLoad(unittest.TestCase):
    def assert_weights(self, layer, expected):
        for name in NAMES:
            actual = getattr(layer, name).detach()
            self.assertEqual(tuple(actual.shape), tuple(expected[name].shape), name)
            self.assertTrue(torch.equal(actual, expected[name]), f"{name} bytes changed")

    def test_real_pack_restore_round_trip(self):
        layer = Layer(3)
        expected = snapshot(layer)
        layer.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.ndim, 4)
        for name in NAMES:
            restore(layer, name)
        self.assert_weights(layer, expected)

    def test_repeated_final_postload_is_noop(self):
        layer = cold(0)
        expected = snapshot(layer)
        layer.quant_method.process_weights_after_loading(layer)
        layer.quant_method.process_weights_after_loading(layer)
        self.assert_weights(layer, expected)

    def test_bucketed_refit_then_final_postload_matches_cold(self):
        layer, updated = cold(0), Layer(1)
        expected = snapshot(cold(1))
        for name, expert in [("w13_weight", 0), ("w2_weight", None), ("w13_weight", 1)]:
            restore(layer, name)
            dst, src = getattr(layer, name).data, getattr(updated, name).data
            if expert is None:
                dst.copy_(src)
            else:
                dst[expert].copy_(src[expert])
            # Each RPC's finally hook repacks the canonical weight independently.
            layer.quant_method.repack_weights_after_hot_update(layer)
        self.assert_weights(layer, expected)
        # Miles end_weight_update(run_post_load=True) invokes this again.
        layer.quant_method.process_weights_after_loading(layer)
        self.assert_weights(layer, expected)

    def test_mixed_canonical_and_packed_postload(self):
        layer, updated, reference = cold(0), Layer(1), Layer(0)
        reference.w13_weight.data.copy_(updated.w13_weight.data)
        reference.quant_method.process_weights_after_loading(reference)
        restore(layer, "w13_weight")
        layer.w13_weight.data.copy_(updated.w13_weight.data)
        # W13 needs packing; W2 is already packed and must remain untouched.
        layer.quant_method.process_weights_after_loading(layer)
        self.assert_weights(layer, snapshot(reference))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-double-pack-failure", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU layout regression must not see CUDA devices")
    print("REAL_FLASHINFER_SOURCE", real_flashinfer.__file__, flush=True)
    if args.expect_double_pack_failure:
        suite = unittest.TestSuite([RealFlashInferPostLoad("test_repeated_final_postload_is_noop")])
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RealFlashInferPostLoad)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.expect_double_pack_failure:
        matched = (result.testsRun == 1 and len(result.failures) == 1
                   and not result.errors and not result.skipped
                   and "x should be a 2D tensor, not 3" in result.failures[0][1])
        print(json.dumps({"expected_double_pack_failure_reproduced": matched}))
        sys.exit(0 if matched else 1)
    passed = result.wasSuccessful() and result.testsRun == 4 and not result.skipped
    print(json.dumps({"real_flashinfer_tests": result.testsRun, "all_passed": passed}))
    sys.exit(0 if passed else 1)
