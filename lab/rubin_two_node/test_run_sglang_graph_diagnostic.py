"""Focused CPU-only checks for ownership, timing labels and diagnostic bounds."""

import datetime as dt
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

DIRECTORY = Path(__file__).parent
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("diagnostic", DIRECTORY / "run_sglang_graph_diagnostic.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def identity():
    return {"run_id": "test-diagnostic", "container_name": "miles-graph-diagnostic-test",
            "container_id": "a" * 64, "image_id": "sha256:" + "b" * 64,
            "image_reference": "repo@sha256:" + sorted(D.IMAGE_DIGESTS)[0],
            "uid": 28644, "gid": 30, "labels": {"miles.graph_diagnostic": "test-diagnostic"},
            "docker_inspect_verified_at": "2026-09-16T06:00:00Z", "main_terminal_verified": True,
            "source_commits": {"sglang": D.capture.SGLANG_COMMIT}}


def member(pid=90, start=100, run="run"):
    return {"pid": pid, "pgid": 90, "start_ticks": start, "run_id": run, "uid": 123,
            "rss_bytes": 12, "state": "S"}


class DiagnosticTests(unittest.TestCase):
    def test_identity_rejects_main_container_or_wrong_runtime(self):
        valid = identity()
        self.assertEqual(D.validate_identity(valid), valid["run_id"])
        for key, value in [("container_name", "miles-main"), ("main_terminal_verified", False),
                           ("uid", 0), ("image_reference", "repo:latest")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                D.validate_identity({**valid, key: value})
        with patch.object(D.os, "getuid", return_value=28644), patch.object(D.os, "getgid", return_value=30):
            with patch.dict(D.os.environ, {D.RUN_ENV: valid["run_id"], "CUDA_VISIBLE_DEVICES": "0"}):
                D.validate_identity(valid, runtime=True)
            with patch.dict(D.os.environ, {D.RUN_ENV: valid["run_id"], "CUDA_VISIBLE_DEVICES": "0,1"}):
                with self.assertRaises(ValueError):
                    D.validate_identity(valid, runtime=True)

    def test_same_engine_recipe_and_explicit_decode_only_difference(self):
        args = SimpleNamespace(model_path=Path("/models/Qwen3"), host="10.0.0.1", port=30123)
        off, on = D.engine_argv(args, "off"), D.engine_argv(args, "on")
        self.assertEqual(off[:-1], on[:-2])
        self.assertEqual(off[-1], "--disable-cuda-graph")
        self.assertIn("--disable-piecewise-cuda-graph", on)
        self.assertEqual(json.loads(on[-1]), {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}})
        self.assertEqual(on[on.index("--tp-size") + 1], "1")

    def test_freeze_filters_without_truncation_or_thinking_override(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"]
        tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [1] * (513 if text == "long" else 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps({"prompt": text}) for text in ["long"] + ["short"] * 128))
            frozen = D.freeze_inputs(path, tokenizer, "/models/Qwen3", "a" * 64)
        self.assertEqual(len(frozen["input_ids"]), 128)
        self.assertEqual(frozen["source"]["selected_row_indices"], list(range(1, 129)))
        self.assertEqual(frozen["source"]["chat_template_kwargs"], {})
        for call in tokenizer.apply_chat_template.call_args_list:
            self.assertEqual(call.kwargs, {"tokenize": False, "add_generation_prompt": True})

    def test_flush_requires_api_success_content_not_only_http200(self):
        for response in [{"status": 200, "body": "False"}, {"status": 400, "body": "Cache flushed.\n"}]:
            with patch.object(D, "http", return_value=response), self.assertRaises(ValueError):
                D.flush_cache("http://engine")
        success = {"status": 200, "body": "Cache flushed.\nPlease check backend logs"}
        with patch.object(D, "http", return_value=success) as call:
            self.assertEqual(D.flush_cache("http://engine"), success)
            self.assertEqual(call.call_args.args[1:3], ("/flush_cache?timeout=10", {}))

    def test_generation_wall_time_never_labeled_decode(self):
        response = {"status": 200, "body": json.dumps([{"meta_info": {
            "prompt_tokens": 100, "completion_tokens": 64, "cached_tokens": 0, "e2e_latency": 1.2}}] * 2)}
        result = D.batch_metrics(response, 2, 2)
        self.assertEqual(result["generation_request_seconds"], 2)
        self.assertEqual(result["output_tokens_per_second"], 64)
        self.assertEqual(result["input_plus_output_tokens_per_second"], 164)
        self.assertNotIn("decode_ms", result)
        self.assertIn("not decode-only", result["scope"])
        self.assertEqual(result["raw_timing_fields_per_request"], [{"e2e_latency": 1.2}] * 2)
        with self.assertRaises(ValueError):
            D.batch_metrics(response, 128, 2)
        short = {"status": 200, "body": json.dumps([{"meta_info": {"prompt_tokens": 1, "completion_tokens": 1}}])}
        with self.assertRaises(ValueError):
            D.batch_metrics(short, 1, 2)

    def test_owned_group_rejects_reused_or_foreign_pid(self):
        leader = member()
        D.verify_owned_group([leader, member(pid=91, start=101)], leader, "run", 123)
        for bad in [member(start=101), member(run="main"), {**member(), "uid": 0},
                    {**member(), "pgid": 80}, member(pid=91, start=90)]:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                D.verify_owned_group([bad], leader, "run", 123)

    def test_stop_only_own_group_and_refuses_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = D.Runtime(Path(directory), "run", 9999999999, 1024)
            runtime.child = Mock(pid=90, returncode=0)
            runtime.leader = member()
            with patch.object(D, "group_members", return_value=[member(run="main")]), \
                    patch.object(D.os, "getuid", return_value=123), patch.object(D.os, "killpg") as kill:
                with self.assertRaises(RuntimeError):
                    runtime.stop()
                kill.assert_not_called()
            with patch.object(D, "group_members", side_effect=[[member()], [], [], []]), \
                    patch.object(D.os, "getuid", return_value=123), patch.object(D.os, "killpg") as kill:
                runtime.stop()
                kill.assert_called_once_with(90, signal.SIGTERM)
            self.assertIsNone(runtime.child)

    def test_guard_stops_on_original_deadline_even_without_http_return(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = D.Runtime(Path(directory), "run", 1, 1024)
            self.assertEqual(runtime.inspect_limits(), "absolute_deadline")
            with patch.object(runtime.done, "wait", return_value=False), \
                    patch.object(runtime, "stop") as stop, \
                    patch.object(D.os, "_exit", side_effect=SystemExit(124)):
                with self.assertRaises(SystemExit):
                    runtime.guard()
                stop.assert_called_once()
            self.assertEqual(json.loads((Path(directory) / "terminal.json").read_text())["status"], "BOUNDED_STOP")

    def test_default_plan_does_not_create_output_or_import_transformers(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "identity.json"
            p.write_text(json.dumps(identity()))
            deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15)).isoformat()
            result = subprocess.run([sys.executable, str(DIRECTORY / "run_sglang_graph_diagnostic.py"),
                "--identity-file", str(p), "--model-path", "/models/absent", "--dataset", "/inputs/absent",
                "--output-dir", "/run-output/cpu-only-plan", "--host", "10.0.0.1", "--port", "30123",
                "--order", "off,on", "--deadline-utc", deadline], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["order"], ["off", "on"])
            self.assertFalse(Path("/run-output/cpu-only-plan").exists())


if __name__ == "__main__":
    unittest.main()
