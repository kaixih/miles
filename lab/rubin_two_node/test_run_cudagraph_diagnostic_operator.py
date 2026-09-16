"""CPU-only refusal and exact-container deadline-guard tests."""
import copy
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
SPEC = importlib.util.spec_from_file_location("diagnostic_operator", DIR / "run_cudagraph_diagnostic_operator.py")
O = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(O)


def config():
    c = {"job_id": "123", "node": "compute-1", "node_ip": "10.0.0.1", "run_id": "main-run",
         "image": "repo@sha256:" + "a" * 64, "image_id": "sha256:image", "container": "main-container", "models": "/models",
         "node_models": "/raid/models", "node_repo": "/raid/repo", "node_run_dir": "/raid/main/run",
         "diagnostic_name": "miles-graph-diagnostic-test", "diagnostic_run": "diag-run", "node_base": "/raid/main/diag"}
    c.update(lease_timestamp=10000, soft_timestamp=4600, hard_timestamp=6400)
    return c


def info(c, diagnostic=False):
    pairs = [("/opt/diagnostic", c["node_base"] + "/source", False),
             ("/models/Qwen3-30B-A3B", c["node_models"] + "/Qwen3-30B-A3B", False),
             ("/inputs/train.jsonl", c["node_run_dir"] + "/inputs/train.jsonl", False),
             ("/run-output", c["node_base"] + "/run", True), ("/cache", c["node_base"] + "/cache", True)] if diagnostic else [
             ("/opt/miles", c["node_repo"], False), (c["models"], c["node_models"], False),
             ("/run-output", c["node_run_dir"], True)]
    return {"Id": "d" * 64, "Name": "/" + c["diagnostic_name" if diagnostic else "container"],
            "Image": "sha256:image", "State": {"Running": True}, "Config": {"Image": c["image"], "User": "28644:30",
            "Labels": {"miles.graph_diagnostic": c["diagnostic_run"]}},
            "Mounts": [{"Destination": d, "Source": s, "RW": rw, "Type": "bind"} for d,s,rw in pairs]}


def completed(c):
    state = {"run_id": c["run_id"], "state": "finished", "terminal_status": "SUCCEEDED", "stop_request_count": 0,
             "expected_rollouts": 50, "submission_id": "exact-submission", "soft_deadline_at": O.utc(c["soft_timestamp"]),
             "deadline_at": O.utc(c["hard_timestamp"]), "lease_deadline_at": O.utc(c["lease_timestamp"])}
    job = {"submission_id": "exact-submission", "status": "SUCCEEDED", "type": "SUBMISSION",
           "runtime_env": {"env_vars": {"RUBIN_RUN_ID": c["run_id"]}}, "entrypoint": "python3 train.py --num-rollout 50"}
    driver, train = {"exit_code": 0, "run_id": c["run_id"]}, {"exit_code": 0}
    summary = {"partial": False, "completed_training_rollouts": list(range(50)),
               "rows": [{"train_steps": [{"logged_id": i} for i in range(200)]}]}
    return state, [job], driver, train, summary


class OperatorTests(unittest.TestCase):
    def exercise_execution(self, directory, incomplete=False, unstable=False):
        """Exercise real execute control flow with filesystem receipts and mocked remote boundaries."""
        directory = str(Path(directory).resolve())
        c = {**config(), "run_dir": directory, "durable": directory + "/diagnostics/diag",
             "dashboard": "http://10.0.0.1:28265", "platform": "rubin", "source_commit": "c" * 40}
        c["lease_timestamp"] = int(time.time()) + 7200
        c["soft_timestamp"], c["hard_timestamp"] = c["lease_timestamp"] - 5400, c["lease_timestamp"] - 3600
        state, jobs, driver, train, summary = completed(c)
        if incomplete: jobs[0]["status"] = "RUNNING"
        main_info, diag_info = info(c), info(c, True)
        main_info["Id"], diag_info["Id"] = "a" * 64, "d" * 64
        plan = {"run_id": c["run_id"], "git_commit": c["source_commit"],
                "preflight": {"container_id": main_info["Id"], "image_id": c["image_id"]}}
        for name, value in [("train-driver-exit.json", driver), ("cudagraph-main-plan.json", plan), ("train-launch.json", {})]:
            Path(directory, name).write_text(json.dumps(value))
        events, guard, manifests = [], {}, 0
        manifest = {"partial.json": {"bytes": 2, "sha256": "x"}}
        def fake_node(conf, action, **kwargs):
            nonlocal manifests
            events.append("node:" + action)
            if action == "read": return {"watchdog": state, "train_exit": train}
            if action == "identity": return kwargs["identity"]
            if action == "guard":
                guard.update(state="armed", container_id=diag_info["Id"], deadline=kwargs["deadline"])
                return {"pid": 123, "start_ticks": 456, "deadline": kwargs["deadline"]}
            if action == "manifest":
                manifests += 1
                return manifest if not unstable or manifests == 1 else {"changed": {"bytes": 3, "sha256": "y"}}
            return {}
        def fake_remote(conf, argv, timeout=60):
            if argv[:3] == ["docker", "image", "inspect"]: return json.dumps([{"Id": c["image_id"]}])
            if argv[:2] == ["docker", "inspect"]:
                return json.dumps([main_info if argv[2] == c["container"] else diag_info])
            if argv[:2] == ["docker", "stop"]:
                events.append("stop:" + argv[-1]); return argv[-1]
            if argv[:2] == ["docker", "run"]: return diag_info["Id"]
            if argv[0] == "cat": return json.dumps(guard)
            return ""
        def fake_retain(conf, child, files, target, timeout=300):
            events.append("retain:" + child); target.mkdir()
            if child == "run": (target / "partial.json").write_text("{}")
        def fake_subprocess(argv, **kwargs):
            if argv[0] == "ssh":
                events.append("wrapper-timeout"); raise subprocess.TimeoutExpired(argv, 1)
            return subprocess.CompletedProcess(argv, 0)
        end = dt.datetime.fromtimestamp(c["lease_timestamp"], dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        slurm = f"JobId=123 JobState=RUNNING NodeList=compute-1 NumNodes=1 UserId=user(28644) EndTime={end}"
        original_stat = Path.stat
        def owned_stat(path, *args, **kwargs):
            actual = list(original_stat(path, *args, **kwargs)); actual[4] = 28644
            return os.stat_result(actual)
        with ExitStack() as stack:
            for target, name, value in [(O.os, "getuid", lambda: 28644), (O.os, "getgid", lambda: 30),
                (O, "run", lambda *args, **kwargs: slurm), (O, "node", fake_node), (O, "remote", fake_remote),
                (O, "retain", fake_retain), (O.subprocess, "run", fake_subprocess), (Path, "stat", owned_stat),
                (O, "JobsAPI", lambda address: Mock(list_jobs=lambda: jobs)),
                (O.metrics, "summarize_run", lambda *args, **kwargs: summary)]:
                stack.enter_context(patch.object(target, name, value))
            with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                O.execute(c, "off,on", 31081)
        return events, Path(c["durable"]), main_info["Id"], diag_info["Id"]

    def test_incomplete_main_refuses_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            events, root, _, _ = self.exercise_execution(directory, incomplete=True)
            self.assertEqual(events, ["node:read"])
            self.assertFalse(root.exists())

    def test_wrapper_timeout_still_retains_then_stops_exact_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            events, root, main_id, diag_id = self.exercise_execution(directory)
            self.assertLess(events.index("retain:main-before"), events.index("stop:" + main_id))
            self.assertLess(events.index("wrapper-timeout"), events.index("retain:run"))
            self.assertLess(events.index("retain:run"), events.index("stop:" + diag_id))
            self.assertTrue((root / "diagnostic-retention.json").is_file())
            self.assertIsNone(json.loads((root / "wrapper-exit.json").read_text())["exit_code"])

    def test_changed_output_retains_explicit_partial_failure_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            events, root, _, diag_id = self.exercise_execution(directory, unstable=True)
            self.assertIn("stop:" + diag_id, events)
            failure = json.loads((root / "diagnostic-retention-failure.json").read_text())
            self.assertEqual(failure["partial_destination_file_sizes"], {"partial.json": 2})
            self.assertFalse((root / "diagnostic-retention.json").exists())

    def test_original_lease_and_exact_allocation_required(self):
        c = config()
        end = dt.datetime.fromtimestamp(10000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        raw = f"JobId=123 JobState=RUNNING NodeList=compute-1 NumNodes=1 UserId=user(28644) EndTime={end}"
        O.allocation(c, raw, 7000)
        for changed in [raw.replace("123", "456"), raw.replace("RUNNING", "COMPLETED"),
                        raw.replace("compute-1", "compute-2"), raw.replace("28644", "0")]:
            with self.assertRaises(ValueError): O.allocation(c, changed, 7000)
        with self.assertRaises(ValueError): O.allocation(c, raw, 7600)
        with self.assertRaises(ValueError): O.allocation({**c, "lease_timestamp": 10010}, raw, 7000)

    def test_exact_main_identity_and_readonly_mounts(self):
        c = config(); original = info(c)
        O.container_identity(c, original, "sha256:image")
        for field, value in [("Image", "sha256:other"), ("Name", "/other")]:
            with self.assertRaises(ValueError): O.container_identity(c, {**original, field: value}, "sha256:image")
        bad = copy.deepcopy(original); bad["Mounts"][0]["RW"] = True
        with self.assertRaises(ValueError): O.container_identity(c, bad, "sha256:image")
        bad = copy.deepcopy(original); bad["Config"]["User"] = "0"
        with self.assertRaises(ValueError): O.container_identity(c, bad, "sha256:image")

    def test_completion_refuses_wrong_locked_job_run_or_active_job(self):
        c = config(); values = completed(c)
        O.completion(c, *values)
        for key, value in [("submission_id", "other"), ("state", "monitoring"), ("stop_request_count", 1),
                           ("deadline_at", O.utc(c["hard_timestamp"] + 60))]:
            changed = list(copy.deepcopy(values)); changed[0][key] = value
            with self.assertRaises(ValueError): O.completion(c, *changed)
        changed = list(copy.deepcopy(values)); changed[1][0]["runtime_env"]["env_vars"]["RUBIN_RUN_ID"] = "other"
        with self.assertRaises(ValueError): O.completion(c, *changed)
        changed = list(copy.deepcopy(values)); changed[1].append({"status": "RUNNING"})
        with self.assertRaises(ValueError): O.completion(c, *changed)

    def test_completion_requires_exact_50_200_and_zero_exits(self):
        c = config(); values = completed(c)
        for index, field, value in [(2, "exit_code", 1), (3, "exit_code", 1),
                                     (4, "completed_training_rollouts", list(range(49))), (4, "partial", True)]:
            changed = list(copy.deepcopy(values)); changed[index][field] = value
            with self.assertRaises(ValueError): O.completion(c, *changed)
        changed = list(copy.deepcopy(values)); changed[4]["rows"][0]["train_steps"].pop()
        with self.assertRaises(ValueError): O.completion(c, *changed)

    def test_diagnostic_single_gpu_label_and_no_ray(self):
        c = config(); argv = O.docker_command(c)
        self.assertEqual(argv[argv.index("--gpus") + 1], "device=0")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", argv)
        self.assertIn("miles.graph_diagnostic=diag-run", argv)
        self.assertFalse(any(value == "ray" for value in argv))
        valid = info(c, True)
        O.container_identity(c, valid, "sha256:image", True)
        valid["Config"]["Labels"]["miles.graph_diagnostic"] = "main-run"
        with self.assertRaises(ValueError): O.container_identity(c, valid, "sha256:image", True)

    def test_independent_guard_only_stops_exact_created_container(self):
        with tempfile.TemporaryDirectory() as directory:
            c = {**config(), "node_base": directory}; actual = info(c, True)
            source = O.guard_source(c, actual["Id"], time.time() - 1)
            with patch.object(subprocess, "check_output", return_value=json.dumps([actual])), \
                    patch.object(subprocess, "run") as action:
                exec(compile(source, "guard", "exec"), {})
                action.assert_called_once_with(["docker", "stop", "--time", "10", actual["Id"]], check=True, timeout=25)
            actual["Name"] = "/" + c["container"]
            with patch.object(subprocess, "check_output", return_value=json.dumps([actual])), \
                    patch.object(subprocess, "run") as action:
                with self.assertRaises(AssertionError): exec(compile(source, "guard", "exec"), {})
                action.assert_not_called()


if __name__ == "__main__": unittest.main()
