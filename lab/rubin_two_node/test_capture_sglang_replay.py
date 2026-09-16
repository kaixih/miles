"""CPU-only capture contracts; no Docker, network, Ray or GPU calls."""

from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("capture_under_test", Path(__file__).with_name("capture_sglang_replay.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"run_id": "test-one", "submission_id": "qwen3-profile-test-one",
                     "entrypoint_sha256": C.sha(b"python3 /opt/miles/train.py --real"),
                     "armed_timestamp": 100, "deadline_timestamp": 500}
        self.job = {"type": "SUBMISSION", "status": "RUNNING", "submission_id": self.plan["submission_id"],
                    "entrypoint": "python3 /opt/miles/train.py --real",
                    "runtime_env": {"env_vars": {C.RUN_ENV: "test-one"}}}
        self.args = SimpleNamespace(container="miles-profile-test-one", image="example@sha256:" + "a" * 64)
        self.info = {"Name": "/miles-profile-test-one", "Image": "sha256:actual",
                     "State": {"Running": True}, "Config": {"Labels": {"miles.profile_run": "test-one"},
                     "Image": self.args.image, "User": "28644:30"},
                     "Mounts": [{"Destination": "/run-output", "Source": "/tmp/profile/run", "Type": "bind", "RW": True}]}

    def test_exact_job_and_container(self):
        C.assert_job(self.plan, self.job)
        self.assertEqual(C.assert_container(self.info, self.args, self.plan, "sha256:actual"), Path("/tmp/profile/run"))

    def test_main_or_replaced_job_rejected(self):
        for key, value in [("submission_id", "raysubmit_MAIN"), ("entrypoint", "python3 train.py --different"),
                           ("type", "DRIVER"), ("status", "UNKNOWN")]:
            job = deepcopy(self.job)
            job[key] = value
            with self.subTest(key=key), self.assertRaises(C.IdentityError):
                C.assert_job(self.plan, job)
        job = deepcopy(self.job)
        job["runtime_env"]["env_vars"][C.RUN_ENV] = "old-main"
        with self.assertRaises(C.IdentityError):
            C.assert_job(self.plan, job)

    def test_wrong_image_label_uid_container_rejected(self):
        for key, value in [("Image", "other@sha256:" + "b" * 64), ("Labels", {"miles.profile_run": "main"}),
                           ("User", "0:0")]:
            info = deepcopy(self.info)
            info["Config"][key] = value
            with self.subTest(key=key), self.assertRaises(C.IdentityError):
                C.assert_container(info, self.args, self.plan, "sha256:actual")
        info = deepcopy(self.info)
        info["Name"] = "/miles-rubin-main"
        with self.assertRaises(C.IdentityError):
            C.assert_container(info, self.args, self.plan, "sha256:actual")

    def test_deadline_never_extended(self):
        self.assertEqual(C.remaining_capture(self.plan, 200, 180), 380)
        self.assertEqual(C.remaining_capture(self.plan, 450, 180), 500)
        self.assertEqual(C.remaining_capture(self.plan, 600, 180), 500)
        for timeout in (0, 181):
            with self.assertRaises(ValueError):
                C.remaining_capture(self.plan, 200, timeout)
        with self.assertRaises(ValueError):
            C.remaining_capture({**self.plan, "deadline_timestamp": float("inf")}, 200, 180)

    def test_cpu_only_trace_is_not_gpu_capture(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump({"traceEvents": [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 1, "dur": 2}]}, stream)
            with self.assertRaisesRegex(ValueError, "CUDA kernels"):
                C.trace_evidence(path)

    def test_real_categories_and_hash_required(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump({"traceEvents": [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 1, "dur": 2},
                    {"ph": "X", "cat": "kernel", "name": "decode_attention", "ts": 1, "dur": 2}]}, stream)
            result = C.trace_evidence(path)
            self.assertEqual(result["event_count"], 2)
            self.assertEqual(result["complete_event_categories"], {"cpu_op": 1, "kernel": 1})
            self.assertEqual(result["sha256"], C.sha(path.read_bytes()))
            self.assertEqual(result["stage_name_hints"], ["decode_attention"])
            link = Path(d) / "linked.gz"
            link.symlink_to(path)
            with self.assertRaises(C.IdentityError):
                C.trace_evidence(link)

    def test_credentials_or_wrong_url_path_rejected(self):
        for url in ("http://user:pass@localhost:20000", "http://localhost:20000/start_profile", "http://localhost"):
            with self.assertRaises(ValueError):
                C.origin(url)
        self.assertEqual(C.origin("http://10.102.74.82:20000/"), "http://10.102.74.82:20000")

    def test_ambiguous_single_arm_times_out_and_stops_only_exact_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "replay/traces").mkdir(parents=True)
            plan = {**self.plan, "output_dir": "/run-output/replay", "ray_address": "http://127.0.0.1:29265"}
            (output / "replay/profile-plan.json").write_text(json.dumps(plan))
            args = SimpleNamespace(**vars(self.args), engine_pid=123, engine_start_ticks=456,
                                   engine_url="http://127.0.0.1:20000", capture_timeout=15)
            info = deepcopy(self.info)
            info["Id"] = "exact-container-id"
            info["State"]["StartedAt"] = "1970-01-01T00:00:50+00:00"
            info["Mounts"][0]["Source"] = directory
            engine = {"pid": 123, "start_ticks": 456, "started_timestamp": 80,
                      "host": "127.0.0.1", "port": 20000, "tp_size": 1, "pp_size": 1,
                      "base_gpu_id": 0, "model_path": "/model"}
            calls = []
            clock = [200]

            def command(argv):
                if argv[:3] == ["docker", "image", "inspect"]:
                    return json.dumps([{"Id": "sha256:actual"}])
                if argv[:2] == ["docker", "inspect"]:
                    return json.dumps([info])
                if argv[:2] == ["docker", "exec"]:
                    return json.dumps(engine)
                self.fail("Unexpected process operation: " + repr(argv))

            def request(url, method="GET", payload=None):
                calls.append((url, method))
                if url.endswith("/server_info"):
                    return {"body": json.dumps(engine), "status": 200}
                if url.endswith("/start_profile"):
                    self.assertFalse(payload["profile_by_stage"])
                    self.assertEqual(payload["num_steps"], 4)
                    raise TimeoutError("Start could have been armed")
                if url.endswith("/stop_profile"):
                    raise TimeoutError("Stop result unknown")
                if url.endswith("/api/jobs/qwen3-profile-test-one/stop"):
                    return {"status": 200, "body": '{"stopped":true}'}
                if url.endswith("/api/jobs/qwen3-profile-test-one"):
                    return {"body": json.dumps(self.job), "status": 200}
                self.fail("Unexpected endpoint: " + url)

            with patch.object(C, "command", side_effect=command), patch.object(C, "request", side_effect=request), \
                    patch.object(C.os, "getuid", return_value=28644), patch.object(C.os, "getgid", return_value=30), \
                    patch.object(C.time, "time", side_effect=lambda: clock[0]), \
                    patch.object(C.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
                C.execute(args, plan)
            state = json.loads((output / "replay/sglang-capture.json").read_text())
            self.assertEqual(state["state"], "incomplete_stop_requested")
            self.assertEqual(state["start_requests"], 1)
            self.assertEqual(state["stop_profile_requests"], 1)
            self.assertEqual(state["stop_job_requests"], 1)
            self.assertEqual(sum(url.endswith("/start_profile") for url, _ in calls), 1)
            self.assertEqual([url for url, method in calls if method == "POST" and "/api/jobs/" in url],
                             ["http://127.0.0.1:29265/api/jobs/qwen3-profile-test-one/stop"])


if __name__ == "__main__":
    unittest.main()
