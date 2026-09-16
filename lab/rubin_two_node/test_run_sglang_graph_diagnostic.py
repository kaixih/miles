"""Focused CPU-only checks for ownership, timing labels and diagnostic bounds."""

import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import signal
import select
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
    return {"pid": pid, "ppid": 89, "sid": 90, "pgid": 90, "start_ticks": start,
            "run_id": run, "run_env_present": True, "uid": 123,
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

    def test_run_mode_records_warmup_and_three_measured_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = D.Runtime(root, "run", 9999999999, 1024)
            runtime.leader = member()
            args = SimpleNamespace(model_path=Path("/models/Qwen3"), host="127.0.0.1", port=0,
                                   cache_dir=root / "cache")
            response = {"status": 200, "body": json.dumps([{"meta_info": {
                "prompt_tokens": 3, "completion_tokens": 64}}] * 128)}
            def api(origin, path, payload=None, **kwargs):
                if path == "/server_info":
                    return {"status": 200, "body": json.dumps({"model_path": str(args.model_path),
                        "tp_size": 1, "pp_size": 1, "base_gpu_id": 0,
                        "cuda_graph_config": {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}}})}
                if path == "/start_profile":
                    for name in ("EXTEND", "DECODE"):
                        (root / "on/traces" / (name + ".trace.json.gz")).write_bytes(b"fixture")
                    return {"status": 200, "body": "armed"}
                self.assertEqual(path, "/generate")
                return response
            evidence = {"prefill_forward_observed": True, "decode_graph_replay_proven": True,
                        "scheduler_forward_counts": {"DECODE": 2}, "cpu_and_gpu_activity_observed": True}
            with patch.object(runtime, "start"), patch.object(runtime, "stop"), patch.object(D.socket, "socket"), \
                    patch.object(D, "wait_ready"), patch.object(D, "flush_cache", return_value={"status": 200}), \
                    patch.object(D, "http", side_effect=api), patch.object(D.time, "sleep"), \
                    patch.object(D.capture, "stage_payload", return_value={"fixture": True}), \
                    patch.object(D.capture, "inspect_trace", return_value=evidence):
                result = D.run_mode(args, runtime, "on", {"input_ids": [[1, 2, 3]] * 128,
                    "source": {"dataset_sha256": "a" * 64, "tokenizer_model": str(args.model_path),
                               "chat_template_kwargs": {}, "selected_row_indices": list(range(128))}})
            events = json.loads((root / "events.json").read_text())
            self.assertEqual([e["kind"] for e in events], ["warmup"] + ["measured"] * 3)
            self.assertTrue(all(e["event"] == "batch_complete" for e in events))
            self.assertEqual(len(result["measurements"]), 3)
            self.assertTrue(result["capture"]["four_decode_forwards_verified"])

    def test_process_environ_permission_race_only_excuses_zombie_or_exit(self):
        def stat(state):
            return "90 (python) " + " ".join([state, "89", "90", "90"] + ["0"] * 15 + ["100"])
        for final, allowed in [(stat("Z"), True), (FileNotFoundError(), True), (stat("S"), False)]:
            states = iter([stat("S"), final])
            def read_text(path, *args, **kwargs):
                if path.name == "status": return "Uid:\t123\t123\t123\t123\nVmRSS:\t1 kB\n"
                value = next(states)
                if isinstance(value, Exception): raise value
                return value
            with patch.object(Path, "read_text", read_text), \
                    patch.object(Path, "read_bytes", side_effect=PermissionError(13, "denied")):
                if allowed: self.assertIsNone(D.process_info(90))
                else:
                    with self.assertRaises(PermissionError): D.process_info(90)

    def test_owned_group_rejects_reused_or_foreign_pid(self):
        leader = member()
        D.verify_owned_group([leader, member(pid=91, start=101)], leader, "run", 123)
        for bad in [member(start=101), member(run="main"), {**member(), "uid": 0},
                    {**member(), "pgid": 80}, {**member(), "sid": 80}, member(pid=91, start=90)]:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                D.verify_owned_group([bad], leader, "run", 123)

    def test_title_rewritten_environment_requires_verified_exact_session(self):
        leader = member()
        absent = {**member(pid=91, start=101), "run_id": "", "run_env_present": False}
        D.verify_owned_group([absent], leader, "run", 123)
        for changed in [{**absent, "sid": 91}, {**absent, "run_env_present": True},
                        {**absent, "run_id": "different"}, {**absent, "start_ticks": 99}]:
            with self.assertRaises(D.OwnershipError) as raised:
                D.verify_owned_group([changed], leader, "run", 123)
            self.assertEqual(raised.exception.detail["member"]["pid"], 91)
        with self.assertRaises(D.OwnershipError):
            D.verify_owned_group([absent], {**leader, "run_env_present": False}, "run", 123)

    @unittest.skipUnless(sys.platform == "linux" and importlib.util.find_spec("setproctitle"),
                         "Native Linux /proc + setproctitle CPU integration test")
    def test_native_linux_title_rewrite_keeps_owned_session(self):
        code = "import sys,setproctitle;print('ready',flush=True);sys.stdin.readline();setproctitle.setproctitle('sglang::scheduler');print('renamed',flush=True);sys.stdin.readline()"
        env = {**os.environ, D.RUN_ENV: "native-title-check"}; env.pop("SPT_NOENV", None)
        proc = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=env, start_new_session=True)
        try:
            self.assertTrue(select.select([proc.stdout], [], [], 5)[0], "child startup deadline")
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            leader = D.process_info(proc.pid)
            D.verify_owned_group([leader], leader, "native-title-check", os.getuid())
            proc.stdin.write("rename\n"); proc.stdin.flush()
            self.assertTrue(select.select([proc.stdout], [], [], 5)[0], "title rewrite deadline")
            self.assertEqual(proc.stdout.readline().strip(), "renamed")
            renamed = D.process_info(proc.pid)
            self.assertFalse(renamed["run_env_present"])
            self.assertEqual(renamed["sid"], leader["pid"])
            self.assertEqual(renamed["pgid"], leader["pid"])
            D.verify_owned_group([renamed], leader, "native-title-check", os.getuid())
        finally:
            proc.terminate(); proc.wait(timeout=5)
            proc.stdin.close(); proc.stdout.close(); proc.stderr.close()

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

    def test_guard_records_terminal_and_offending_member_when_cleanup_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = D.Runtime(Path(directory), "run", 1, 1024)
            failure = D.OwnershipError("explicit_run_environment_mismatch", member(run="other"), member())
            with patch.object(runtime.done, "wait", return_value=False), \
                    patch.object(runtime, "inspect_limits", side_effect=failure), \
                    patch.object(runtime, "stop", side_effect=failure), \
                    patch.object(D.os, "_exit", side_effect=SystemExit(124)):
                with self.assertRaises(SystemExit): runtime.guard()
            result = json.loads((Path(directory) / "terminal.json").read_text())
            self.assertEqual(result["engine_cleanup"], "FAILED")
            self.assertEqual(result["cleanup_ownership_error"]["member"]["run_id"], "other")
            self.assertEqual(result["ownership_error"]["original_leader"]["sid"], 90)
            event = json.loads((Path(directory) / "events.json").read_text())[0]
            self.assertEqual(event["ownership_error"]["member"]["pid"], 90)

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
