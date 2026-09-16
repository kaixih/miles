"""CPU contracts for the graph-on driver; never launch SSH, Ray or Docker."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from lab.rubin_two_node import run_cudagraph_main as D


def fixture(root):
    run = root / "new-run"
    source = run / "source"
    (run / "logs").mkdir(parents=True)
    hashes = {}
    for relative in D.SOURCE_FILES:
        p = source / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture " + relative)
        hashes[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = run / "source-manifest.json"
    manifest.write_text(json.dumps({"git_commit": "a" * 40, "source_sha256": hashes}))
    gate = run / "gate.log"
    gate.write_text("ALLREDUCE_OK world=4\n")
    return {"platform": "gb300", "run_id": "new-graph-run", "job_id": "9999999",
            "node": "new-node", "node_ip": "10.1.2.3", "image": "example/image@sha256:" + "b" * 64,
            "repo": str(source), "node_repo": "/readonly/new-source", "models": "/models",
            "node_models": "/raid/verified-models", "run_dir": str(run), "node_run_dir": "/raid/new-run",
            "cache_dir": "/raid/new-cache", "container_prefix": "miles-new-graph",
            "ray_port": 26379, "dashboard_port": 28265, "megatron_path": "/root/Megatron-LM",
            "nccl_iface": "eth-test", "lease_deadline": D._utc(time.time() + 6 * 3600),
            "gate_log": str(gate), "source_commit": "a" * 40,
            "source_manifest": str(manifest), "train_sha256": "c" * 64, "eval_sha256": "d" * 64,
            "uid": os.getuid(), "gid": os.getgid()}


def armed(c):
    return {"state": "armed", "run_id": c["run_id"], "submission_id": None,
            "ray_address": c["dashboard"], "soft_deadline_at": D._utc(c["soft_timestamp"]),
            "deadline_at": D._utc(c["hard_timestamp"]), "lease_deadline_at": D._utc(c["lease_timestamp"]),
            "sentinel": "/run-output/checkpoint-now", "save_dir": "/run-output/checkpoints",
            "expected_rollouts": 50, "stop_request_count": 0, "heartbeat_at": D._utc(time.time()), "pid": 123}


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = fixture(Path(self.temp.name))
        self.c = D._config(self.raw)

    def test_plan_is_local_pure_and_keeps_logical_recipe(self):
        with patch.object(D, "_run", side_effect=AssertionError("No subprocess in plan")):
            plan = D._plan(self.c)
        self.assertEqual(plan["mode"], "PLAN_ONLY_NO_REMOTE_CALLS")
        args = D._launcher_args(self.c)
        for flag, value in {"--max-tokens-per-gpu": "4096", "--save-interval": "50",
                            "--save-retain-interval": "1000000",
                            "--rollout-batch-size": "256", "--n-samples-per-prompt": "8",
                            "--global-batch-size": "512", "--rollout-max-prompt-len": "512",
                            "--rollout-max-response-len": "1024"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--sglang-enable-cuda-graph", args)
        self.assertNotIn("--no-enable-eval", args)
        self.assertNotIn("--no-save-optim", args)
        self.assertNotIn("--dynamic-sampling-filter-path", args)
        self.assertEqual(shlex.split(plan["common_orchestrator_args"][-1]), args)
        self.assertEqual(plan["recipe"]["optimizer_updates"], 200)
        self.assertEqual(plan["recipe"]["save_retain_interval"], 1000000)
        self.assertEqual(plan["train_command"][3], "train")
        self.assertIn("NCCL_SOCKET_IFNAME=eth-test", plan["common_orchestrator_args"])
        self.assertIn("GLOO_SOCKET_IFNAME=eth-test", plan["common_orchestrator_args"])
        self.assertIn("PYTHONPATH=/opt/miles:/root/Megatron-LM", plan["common_orchestrator_args"])

    def test_deadlines_are_absolute_offsets_not_arming_time(self):
        c = self.c
        self.assertEqual(c["lease_timestamp"] - c["soft_timestamp"], 5400)
        self.assertEqual(c["lease_timestamp"] - c["hard_timestamp"], 3600)
        one = D._plan(c)["watchdog_command"]
        with patch.object(D.time, "time", return_value=time.time() + 1000):
            self.assertEqual(D._plan(c)["watchdog_command"], one)
        self.assertNotIn("--log", one)  # The canonical training log lives on login/NFS.

    def test_identifiers_image_ports_and_timezone_fail_closed(self):
        for change in ({"node_ip": "127.0.0.1"}, {"image": "repo:mutable"},
                       {"node": "node; touch bad"}, {"ray_port": 28265},
                       {"lease_deadline": "2026-10-01T10:00:00"}, {"source_commit": "main"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D._config({**self.raw, **change})

    def test_snapshot_inside_durable_root_supported_but_outputs_inside_inputs_rejected(self):
        self.assertEqual(D._config(self.raw)["repo"], self.raw["repo"])
        with self.assertRaises(ValueError):
            D._config({**self.raw, "node_run_dir": "/raid/verified-models/output"})

    def test_snapshot_manifest_binds_four_runtime_files_and_separate_driver(self):
        source = D._source(self.c)
        self.assertEqual(set(source["source_sha256"]), set(D.SOURCE_FILES))
        self.assertEqual(len(source["driver_sha256"]), 64)
        changed = Path(self.c["repo"], D.SOURCE_FILES[1])
        changed.write_text("changed launcher")
        with self.assertRaisesRegex(ValueError, "Snapshot source changed"):
            D._source(self.c)

    def test_incomplete_or_wrong_snapshot_manifest_rejected(self):
        p = Path(self.c["source_manifest"])
        m = json.loads(p.read_text())
        m["source_sha256"].pop(D.SOURCE_FILES[0]);p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "omits"):
            D._source(self.c)

    def test_armed_guard_requires_exact_fresh_identity_and_immutable_budget(self):
        state = armed(self.c)
        D._armed(self.c, state, time.time())
        for change in ({"run_id": "old-run"}, {"state": "monitoring"}, {"submission_id": "already-used"},
                       {"deadline_at": D._utc(self.c["hard_timestamp"] + 1)},
                       {"heartbeat_at": D._utc(time.time() - 25)}, {"stop_request_count": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D._armed(self.c, {**state, **change}, time.time())

    def test_existing_driver_evidence_is_not_reused_or_reset(self):
        p = Path(self.c["run_dir"], "train-launch.json");p.write_text("old")
        with self.assertRaisesRegex(ValueError, "existing run evidence"):
            D._execute(self.c, D._plan(self.c))
        self.assertEqual(p.read_text(), "old")

    def test_expired_soft_budget_refuses_before_any_work(self):
        c = {**self.c, "soft_timestamp": time.time() - 1}
        with patch.object(D, "_source", side_effect=AssertionError("Must reject before source work")):
            with self.assertRaisesRegex(ValueError, "90-minute"):
                D._execute(c, D._plan(c))

    def test_bootstrap_mismatch_refuses(self):
        with self.assertRaises(ValueError):
            D._bootstrap(self.c, {"state": "ray_ready_training_not_started"})

    def _fake_orchestrator(self, events):
        def load(module):
            module._parser = lambda: types.SimpleNamespace(parse_args=lambda argv: argv)
            module._check_allocation = lambda args: events.append("allocation")
            module._cluster_status = lambda args: types.SimpleNamespace(returncode=0)
        return types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=load))

    def test_watchdog_is_armed_and_rechecked_before_train(self):
        events = []
        fake = self._fake_orchestrator(events)
        (Path(self.c["run_dir"]) / "bootstrap.json").write_text("{}")
        plan = D._plan(self.c)
        launched = {"state": armed(self.c), "pid": 123, "start_ticks": 456}
        def start(c, p):
            events.append("arm")
            return launched
        def recheck(c, p, l):
            events.append("fresh-process-check")
            return l
        def train(argv, **kwargs):
            events.append("train")
            self.assertEqual(argv, plan["train_command"])
            self.assertEqual(events[-3:], ["arm", "fresh-process-check", "train"])
            return types.SimpleNamespace(returncode=0)
        with patch.object(D, "_bootstrap"), patch.object(D, "_preflight", return_value={}), \
                patch.object(D.importlib.util, "spec_from_file_location", return_value=fake), \
                patch.object(D.importlib.util, "module_from_spec", return_value=types.SimpleNamespace()), \
                patch.object(D, "_start_watcher", side_effect=start), patch.object(D, "_check_watcher", side_effect=recheck), \
                patch.object(D.signal, "signal"), patch.object(D.subprocess, "run", side_effect=train):
            self.assertEqual(D._execute(self.c, plan), 0)
        record = json.loads(Path(self.c["run_dir"], "train-driver-exit.json").read_text())
        self.assertEqual(record["reason"], "orchestrator_exit")
        self.assertNotIn("SUCCEEDED", json.dumps(record))

    def test_generated_remote_programs_compile_without_execution(self):
        captured = []
        def collect(c, code):
            compile(code, "remote-probe", "exec")
            captured.append(code)
            return {"state": armed(c), "pid": 123, "start_ticks": 456}
        with patch.object(D, "_node_python", side_effect=collect):
            D._preflight(self.c, D._source(self.c))
            D._start_watcher(self.c, D._plan(self.c))
        self.assertEqual(len(captured), 2)
        self.assertNotIn("docker stop", "\n".join(captured))


if __name__ == "__main__":
    unittest.main()
