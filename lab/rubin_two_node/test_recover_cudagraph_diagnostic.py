"""Recovery evidence/deadline tests with real split login/node retained files."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import recover_cudagraph_diagnostic as R
from test_run_cudagraph_diagnostic_operator import config, info, completed


class RecoveryTests(unittest.TestCase):
    def owned(self):
        original = Path.stat
        def stat(path, *args, **kwargs):
            result = list(original(path, *args, **kwargs)); result[4] = 28644
            return os.stat_result(result)
        return patch.object(Path, "stat", stat)

    def fixture(self, directory, missing_update=False):
        root = Path(directory).resolve(); prior = root / "diagnostics/v1"; prior.mkdir(parents=True)
        now = time.time()
        old = {**config(), "platform": "rubin", "run_dir": str(root), "repo": str(root / "source"),
               "source_commit": "c"*40, "train_sha256": "a"*64, "eval_sha256": "b"*64,
               "durable": str(prior), "diagnostic_run": "diagnostic-v1",
               "diagnostic_name": "miles-graph-diagnostic-v1", "lease_timestamp": now+7200,
               "soft_timestamp": now+1800, "hard_timestamp": now+3600}
        current = {**old, "diagnostic_run": "diagnostic-v2", "diagnostic_name": "miles-graph-diagnostic-v2",
                   "node_base": "/raid/main/v2", "durable": str(root / "diagnostics/v2")}
        state, jobs, driver, train, _ = completed(old)
        source = root / "source/lab/launcher.py";source.parent.mkdir(parents=True);source.write_text("# frozen\n")
        main = info(old);main["Id"] = "a"*64;main["State"]["Running"] = False
        diag = info(old, True);diag["State"]["Running"] = False
        plan = {"run_id": old["run_id"], "git_commit": old["source_commit"],
                "preflight": {"container_id": main["Id"], "image_id": old["image_id"]},
                "source_sha256": {"lab/launcher.py": R.O.sha(source)}}
        login = prior / "main-login";login.mkdir()
        texts = {"train_exit.json": json.dumps(train), "train-driver-exit.json": json.dumps(driver),
                 "cudagraph-main-plan.json": json.dumps(plan), "source/lab/launcher.py": source.read_text()}
        lines = []
        for rollout in range(50):
            for step in range(4*rollout, 4*rollout+4):
                if missing_update and step == 199:continue
                lines.append(f"[2026-09-16 09:00:00] step {step}: {{'train/grad_norm': 1.25}}\n")
            lines.append(f"[2026-09-16 09:00:00] perf {rollout}: {{'perf/train_time': 2.0, 'perf/step_time': 3.0}}\n")
        texts["logs/qwen3_train.log"] = "".join(lines)
        files = {}
        for name, text in texts.items():
            p = login / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text(text)
            original = root / name;original.parent.mkdir(parents=True, exist_ok=True)
            original.write_text(text)
            files[name] = {"source": str(original), "bytes": p.stat().st_size, "sha256": R.O.sha(p)}
        before = prior / "main-before";before.mkdir();(before / "watchdog.json").write_text(json.dumps(state))
        node_files = {"watchdog.json": {"bytes": (before / "watchdog.json").stat().st_size,
                                      "sha256": R.O.sha(before / "watchdog.json")}}
        deadline = now+1500
        records = {
            "operator-plan.json": {"config": old, "order": "on,off", "main_container": main, "ray_job": jobs[0]},
            "main-driver-evidence.json": {"original_main_plan": plan, "driver_exit": driver, "train_exit": train},
            "main-login-retention.json": {"stable_source_and_destination_sha256_verified": True, "files": files},
            "main-retention.json": {"source_prefix_and_destination_sha256_verified": True, "login_files": files,
                                    "files": node_files, "completed_rollouts": list(range(50)), "optimizer_updates": 200},
            "main-stopped.json": {"container_id": main["Id"], "retention_verified_before_stop": True},
            "diagnostic-identity.json": {"container_id": diag["Id"], "container_name": old["diagnostic_name"],
                "run_id": old["diagnostic_run"], "image_id": old["image_id"], "image_reference": old["image"],
                "main_submission_id": jobs[0]["submission_id"], "initial_model_inventory": {"config.json": {"bytes": 2}},
                "initial_model_id": "e"*64},
            "diagnostic-stopped.json": {"container_id": diag["Id"]},
            "guard-armed.json": {"state": "armed", "container_id": diag["Id"], "deadline": deadline},
            "diagnostic-after-stop-retention.json": {"container_id": diag["Id"], "stopped_before_retention": True,
                                                     "source_before_after_and_destination_verified": True},
        }
        saved = prior / "node-output-after-stop";saved.mkdir()
        (saved / "diagnostic-identity.json").write_text(json.dumps(records["diagnostic-identity.json"]))
        records["diagnostic-after-stop-retention.json"]["files"] = {"diagnostic-identity.json": {
            "bytes": (saved / "diagnostic-identity.json").stat().st_size,
            "sha256": R.O.sha(saved / "diagnostic-identity.json")}}
        return current, prior, records, deadline, now, main, diag

    def test_real_split_login_log_50_200_passes_without_node_log(self):
        with tempfile.TemporaryDirectory() as tmp, self.owned():
            c, prior, records, deadline, now, _, _ = self.fixture(tmp)
            self.assertFalse((prior / "main-before/logs/qwen3_train.log").exists())
            proof = R.validate_prior(c, prior, records, deadline, now)
            self.assertEqual(proof["summary"]["completed_training_rollouts"], list(range(50)))
            self.assertEqual(proof["job"]["status"], "SUCCEEDED")

    def test_recovery_inherits_both_valid_orders_and_refuses_other_orders(self):
        with tempfile.TemporaryDirectory() as tmp, self.owned():
            c, prior, records, deadline, now, _, _ = self.fixture(tmp)
            for order in ("off,on", "on,off"):
                records["operator-plan.json"]["order"] = order
                self.assertEqual(R.validate_prior(c, prior, records, deadline, now)["order"], order)
                self.assertEqual(R.inherited_order(records), order)
            for order in (None, "on", "off,off", "on,off,on", ["off", "on"]):
                records["operator-plan.json"]["order"] = order
                with self.assertRaises(ValueError): R.validate_prior(c, prior, records, deadline, now)

    def test_missing_real_update_rejected_despite_200_receipt_claim(self):
        with tempfile.TemporaryDirectory() as tmp, self.owned():
            c, prior, records, deadline, now, _, _ = self.fixture(tmp, missing_update=True)
            with self.assertRaisesRegex(ValueError, "200 updates"):
                R.validate_prior(c, prior, records, deadline, now)

    def test_changed_retained_content_or_source_fails(self):
        for changed in ["retained", "source"]:
            with tempfile.TemporaryDirectory() as tmp, self.owned():
                c, prior, records, deadline, now, _, _ = self.fixture(tmp)
                target = prior / "main-login/logs/qwen3_train.log" if changed == "retained" else Path(c["repo"]) / "lab/launcher.py"
                target.write_text("changed")
                with self.assertRaises(ValueError):R.validate_prior(c, prior, records, deadline, now)

    def test_original_lease_ray_and_zero_exits_required(self):
        with tempfile.TemporaryDirectory() as tmp, self.owned():
            c, prior, records, deadline, now, _, _ = self.fixture(tmp)
            with self.assertRaises(ValueError):R.validate_prior({**c, "lease_timestamp": c["lease_timestamp"]+1}, prior, records, deadline, now)
            broken = copy.deepcopy(records);broken["operator-plan.json"]["ray_job"]["status"] = "RUNNING"
            with self.assertRaises(ValueError):R.validate_prior(c, prior, broken, deadline, now)
            broken = copy.deepcopy(records);broken["main-driver-evidence.json"]["driver_exit"]["exit_code"] = 1
            with self.assertRaises(ValueError):R.validate_prior(c, prior, broken, deadline, now)

    def test_deadline_cannot_reset_extend_or_start_too_late(self):
        guard = {"state": "armed", "deadline": 2000}
        R.deadline_check(2000, guard, 4000, 1000)
        for deadline, now in [(2001,1000), (1999,1000), (2000,2000), (2000,1401)]:
            with self.assertRaises(ValueError):R.deadline_check(deadline, guard, 4000, now)

    def test_exact_stopped_identity_label_mount_image_required(self):
        with tempfile.TemporaryDirectory() as tmp, self.owned():
            c, _, records, _, _, main, diag = self.fixture(tmp)
            old = records["operator-plan.json"]["config"]
            R.stopped(c, main, main["Id"]);R.stopped(old, diag, diag["Id"], True)
            for key in ["running", "image", "id", "label", "mount"]:
                bad = copy.deepcopy(diag)
                if key == "running":bad["State"]["Running"] = True
                if key == "image":bad["Image"] = "different"
                if key == "id":bad["Id"] = "b"*64
                if key == "label":bad["Config"]["Labels"]["miles.graph_diagnostic"] = "wrong"
                if key == "mount":bad["Mounts"][0]["RW"] = True
                with self.assertRaises(ValueError):R.stopped(old, bad, diag["Id"], True)

    def test_manifest_hash_binding_detects_receipt_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve();hashes = {}
            for name in R.RECEIPTS:
                (root / name).write_text("{}");hashes[name] = R.O.sha(root / name)
            manifest = root / "manifest.json";manifest.write_text(json.dumps({"files": hashes}))
            records, _ = R.evidence(root, manifest);self.assertEqual(set(records), R.RECEIPTS)
            (root / "operator-plan.json").write_text('{"changed": true}')
            with self.assertRaises(ValueError):R.evidence(root, manifest)

    def test_failed_v1_artifact_requires_stable_after_stop_hashes(self):
        with tempfile.TemporaryDirectory() as tmp, self.owned():
            c, prior, records, deadline, now, _, _ = self.fixture(tmp)
            records["diagnostic-after-stop-retention.json"]["stopped_before_retention"] = False
            with self.assertRaises(ValueError):R.validate_prior(c, prior, records, deadline, now)
            records["diagnostic-after-stop-retention.json"]["stopped_before_retention"] = True
            (prior / "node-output-after-stop/diagnostic-identity.json").write_text("changed")
            with self.assertRaises(ValueError):R.validate_prior(c, prior, records, deadline, now)


if __name__ == "__main__": unittest.main()
