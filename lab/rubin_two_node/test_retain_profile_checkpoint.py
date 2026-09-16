"""Local fixture checks only; no SSH, Docker, Ray, copy subprocess or GPU calls."""

from contextlib import contextmanager, redirect_stdout
import ast
import importlib.util
import inspect
import io
import json
from pathlib import Path
import pickle
import signal
import subprocess
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load("retention_under_test", "retain_profile_checkpoint.py")
O = load("retention_operator_contract", "profile_operator.py")
from lab.rubin_two_node.test_profile_qwen3_replay import P as REPLAY


@contextmanager
def expected_ownership():
    """Keep real inode/mtime/link/file-type semantics; emulate only remote UID."""
    original = Path.stat
    original_lstat = Path.lstat

    def rewrite(result):
        values = {name: getattr(result, name) for name in dir(result) if name.startswith("st_")}
        return SimpleNamespace(**{**values, "st_uid": R.UID})

    def owned(path, *args, **kwargs):
        return rewrite(original(path, *args, **kwargs))

    def owned_lstat(path, *args, **kwargs):
        return rewrite(original_lstat(path, *args, **kwargs))

    with patch.object(Path, "stat", new=owned), patch.object(Path, "lstat", new=owned_lstat):
        yield


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.pin = self.root / "pin"
        self.contents = {
            "latest_checkpointed_iteration.txt": b"9\n",
            "iter_0000009/common.pt": b"optimizer metadata",
            "iter_0000009/.metadata": b"distributed metadata",
            "iter_0000009/__0_0.distcp": b"tensor shard 0",
            "iter_0000009/__1_0.distcp": b"tensor shard 1",
            "rollout/global_dataset_state_dict_9.pt": b"dataset cursor",
        }
        for name, data in self.contents.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def source_bytes(self):
        return {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}

    def embedded(self, key="common_state/shard_0_1", byte_type="BytesStorageMetadata"):
        """Pickle real object/dict/memo structure, without requiring/importing Torch."""
        modules = {name: ModuleType(name) for name in (
            "torch", "torch.distributed", "torch.distributed.checkpoint", "torch.distributed.checkpoint.metadata")}
        module = modules["torch.distributed.checkpoint.metadata"]
        for name in ("Metadata", "BytesStorageMetadata", "TensorStorageMetadata"):
            setattr(module, name, type(name, (), {"__module__": module.__name__}))
        obj = module.Metadata()
        obj.state_dict_metadata = {key: getattr(module, byte_type)()}
        obj.planner_data = {key: (key,)}
        obj.storage_data = {}
        with patch.dict("sys.modules", modules):
            raw = pickle.dumps(obj, protocol=4)
        directory = self.source / "iter_0000009"
        (directory / "common.pt").unlink()
        (directory / ".metadata").write_bytes(raw)
        (directory / "metadata.json").write_text(json.dumps({"sharded_backend": "torch_dist", "sharded_backend_version": 1}))

    def test_tracker_is_independent_and_unpin_leaves_original_files(self):
        with expected_ownership():
            manifest = R.inventory(self.source)
            files, dirs = R.pin_files(self.source, self.pin, manifest)
            tracker = "latest_checkpointed_iteration.txt"
            shard = "iter_0000009/__0_0.distcp"
            self.assertNotEqual((self.pin / tracker).stat().st_ino, (self.source / tracker).stat().st_ino)
            self.assertEqual((self.pin / shard).stat().st_ino, (self.source / shard).stat().st_ino)
            (self.source / tracker).write_text("19\n")
            self.assertEqual((self.pin / tracker).read_text(), "9\n")
            self.assertEqual(R.inventory(self.pin)["id"], manifest["id"])
            expected = {**self.contents, tracker: b"19\n"}
            R.unpin(files, dirs)
            self.assertFalse(self.pin.exists())
            self.assertEqual(self.source_bytes(), expected)

    def test_hardlink_survives_native_source_unlink(self):
        with expected_ownership():
            files, dirs = R.pin_files(self.source, self.pin, R.inventory(self.source))
            shard = "iter_0000009/__0_0.distcp"
            (self.source / shard).unlink()  # Simulated producer retention, not retention helper action.
            self.assertEqual((self.pin / shard).read_bytes(), self.contents[shard])
            R.unpin(files, dirs)
            self.assertTrue((self.source / "iter_0000009/__1_0.distcp").exists())

    def test_partial_pin_failure_removes_only_created_links(self):
        with expected_ownership():
            manifest = R.inventory(self.source)
            original_link = R.os.link
            calls = [0]

            def fail_second(source, target):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("simulated link failure")
                return original_link(source, target)

            with patch.object(R.os, "link", side_effect=fail_second), self.assertRaises(OSError):
                R.pin_files(self.source, self.pin, manifest)
            self.assertFalse(self.pin.exists())
            self.assertEqual(self.source_bytes(), self.contents)

    def test_missing_rollout_state_cannot_be_declared_complete(self):
        (self.source / "rollout/global_dataset_state_dict_9.pt").unlink()
        with expected_ownership(), self.assertRaises(FileNotFoundError):
            R.inventory(self.source)

    def test_fingerprint_and_inventory_equal_operator_contract(self):
        with expected_ownership():
            retained = R.inventory(self.source)
            operator = O._checkpoint_inventory(str(self.source), 9)
            self.assertEqual(retained, operator)
            self.assertEqual(retained["id"], REPLAY._checkpoint_manifest(self.source, 9)["id"])
            self.assertNotIn("sha256", retained["files"]["iter_0000009/__0_0.distcp"])
            files, dirs = R.pin_files(self.source, self.pin, retained)
            self.assertEqual(R.inventory(self.pin), O._checkpoint_inventory(str(self.pin), 9))
            R.unpin(files, dirs)

    def test_embedded_common_parity_pinning_and_no_pickle_execution(self):
        self.embedded()
        with expected_ownership(), patch.object(pickle, "loads", side_effect=AssertionError("Never execute pickle")):
            retained = R.inventory(self.source)
            self.assertEqual(retained, O._checkpoint_inventory(str(self.source), 9))
            self.assertEqual(retained["id"], REPLAY._checkpoint_manifest(self.source, 9)["id"])
            self.assertNotIn("iter_0000009/common.pt", retained["files"])
            files, dirs = R.pin_files(self.source, self.pin, retained)
            self.assertEqual(R.inventory(self.pin), retained)
            R.unpin(files, dirs)

    def test_missing_common_requires_actual_bytes_storage_entry(self):
        self.embedded(key="unrelated_model_parameter")
        with expected_ownership(), self.assertRaisesRegex(ValueError, "embedded common_state"):
            R.inventory(self.source)

    def test_common_state_tensor_entry_does_not_prove_common_object(self):
        self.embedded(byte_type="TensorStorageMetadata")
        with expected_ownership(), self.assertRaisesRegex(ValueError, "BytesStorageMetadata"):
            R.inventory(self.source)

    def test_embedded_format_requires_exact_backend_and_rollout(self):
        self.embedded()
        config = self.source / "iter_0000009/metadata.json"
        config.write_text('{"sharded_backend":"zarr","sharded_backend_version":1}')
        with expected_ownership(), self.assertRaisesRegex(ValueError, "format"):
            R.inventory(self.source)
        config.write_text('{"sharded_backend":"torch_dist","sharded_backend_version":1}')
        (self.source / "rollout/global_dataset_state_dict_9.pt").unlink()
        with expected_ownership(), self.assertRaises(FileNotFoundError):
            R.inventory(self.source)

    def test_persistent_pickle_ids_rejected_without_resolution(self):
        self.embedded()
        (self.source / "iter_0000009/.metadata").write_bytes(b"Punsafe-persistent-reference\n.")
        with expected_ownership(), self.assertRaisesRegex(ValueError, "Unsupported.*PERSID"):
            R.inventory(self.source)

    def test_embedded_remote_sources_are_self_contained(self):
        self.embedded()
        with expected_ownership():
            expected = R.inventory(self.source)

            def local_remote_fixture(argv, **kwargs):
                self.assertEqual(argv[:3], ["ssh", "-o", "BatchMode=yes"])
                source = kwargs["input"]
                imports = [n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)]
                self.assertNotIn("checkpoint_metadata", imports)
                self.assertNotIn("lab.rubin_two_node.checkpoint_metadata", imports)
                with redirect_stdout(io.StringIO()) as output:
                    exec(compile(source, "<authored retention source fixture>", "exec"), {"__name__": "fixture"})
                return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr="")

            with patch.object(R.subprocess, "run", side_effect=local_remote_fixture):
                actual = R.remote_code("never-contacted", f"print(json.dumps(inventory({str(self.source)!r})))", 10)
            self.assertEqual(actual["id"], expected["id"])
            namespace = {}
            operator_source = inspect.getsource(O.checkpoint_metadata) + "\n" + inspect.getsource(O._checkpoint_inventory)
            exec(compile(operator_source, "<authored operator source fixture>", "exec"), namespace)
            self.assertEqual(namespace["_checkpoint_inventory"](str(self.source), 9), expected)

    def test_existing_pin_is_not_reused_or_removed(self):
        self.pin.mkdir()
        marker = self.pin / "unrelated.txt"
        marker.write_text("keep")
        with expected_ownership(), self.assertRaises(FileExistsError):
            R.pin_files(self.source, self.pin, R.inventory(self.source))
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.source_bytes(), self.contents)

    def test_expired_copy_deadline_starts_no_process(self):
        with patch.object(R.time, "time", return_value=200), \
                patch.object(R.subprocess, "Popen", side_effect=AssertionError("Must not start")), \
                self.assertRaises(TimeoutError):
            R.copy_process(["rsync", "source", "new-destination"], 199, self.root / "copy.log")
        self.assertFalse((self.root / "copy.log").exists())

    def test_copy_timeout_signals_only_new_copy_process_group(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("rsync", 5), -15]
        with patch.object(R.time, "time", return_value=200), \
                patch.object(R.subprocess, "Popen", return_value=process) as popen, \
                patch.object(R.os, "killpg") as killpg, self.assertRaises(TimeoutError):
            R.copy_process(["rsync", "source", "new-destination"], 205, self.root / "copy.log")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(123456, signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list[0].kwargs["timeout"], 5)


if __name__ == "__main__":
    unittest.main()
