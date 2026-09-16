"""Diagnostic-only Miles custom init hook; capture one rank-0 forward/backward microbatch.

The caller must supply an external exact-job deadline guard as well. This local
watchdog exits only its own diagnostic actor, never another PID or main run.
No torch import occurs until install() is called inside the training actor.
"""
from __future__ import annotations

import contextlib
import functools
import gzip
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time

ENV = "MILES_ACTOR_PROFILE_CONFIG"
RUN_ENV = "MILES_ACTOR_PROFILE_RUN_ID"
MIB = 1024 ** 2


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def validate_config(config, env, now):
    if not config["run_id"].startswith("actor-profile-") or env.get(RUN_ENV) != config["run_id"]:
        raise ValueError("Diagnostic run identity mismatch")
    if not now < config["deadline_epoch"] <= now + 3600:
        raise ValueError("Require an original absolute deadline within one hour")
    if config.get("target") != [0, 1, 0] or config.get("world_size") != 4:
        raise ValueError("Only rollout 0 / second update / first attempt, world size 4 is supported")
    if config.get("capture_window") != "single_forward_backward_microbatch" or config.get("microbatch_index") != 1:
        raise ValueError("Require exactly the second forward/backward microbatch")
    for key, ceiling in (("capture_seconds", 180), ("export_seconds", 120),
                         ("max_trace_bytes", 512 * MIB), ("max_rss_growth_bytes", 16 * 1024 * MIB)):
        if not 0 < config[key] <= ceiling:
            raise ValueError("Unsafe limit: " + key)
    if not Path(config["output_root"]).is_absolute():
        raise ValueError("Output root must be absolute")
    if "{" in config["rollout_path"]:
        raise ValueError("Use one explicit immutable rollout file, not a template")


def packing_record(iterators, count):
    """CPU metadata and selected token hashes; performed before profiling starts."""
    records = []
    for iterator in iterators:
        if iterator.micro_batch_indices is not None:
            batches = iterator.micro_batch_indices[iterator.offset:iterator.offset + count]
        else:
            width = iterator.micro_batch_size
            batches = [list(range(iterator.offset + i * width, iterator.offset + (i + 1) * width))
                       for i in range(count)]
        if len(batches) != count or any(not batch for batch in batches):
            raise ValueError("Incomplete target microbatch schedule")
        data = iterator.rollout_data
        indices = sorted({index for batch in batches for index in batch})
        samples = []
        for index in indices:
            tokens = data["tokens"][index]
            if hasattr(tokens, "detach"):
                tokens = tokens.detach().cpu().tolist()
            token_sha = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
            samples.append({"local_index": index, "length": int(data["total_lengths"][index]),
                            "response_length": int(data["response_lengths"][index]), "tokens_sha256": token_sha,
                            "global_sample_index": int(data["sample_indices"][index]) if "sample_indices" in data else None})
        records.append({"offset": iterator.offset, "micro_batch_indices": batches, "samples": samples})
    value = {"num_microbatches": count, "iterators": records}
    return {**value, "fingerprint": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}


def output_bytes(output):
    size = 0
    for path in output.iterdir():
        try:
            if path.is_file():
                size += path.stat().st_size
        except FileNotFoundError:
            # A small atomic receipt may be renamed between listing and stat.
            continue
    return size


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


class Budget:
    def __init__(self, config, output, exit_fn=os._exit):
        self.config, self.output, self.exit_fn = config, output, exit_fn
        self.baseline = rss_bytes()
        self.phase, self.until = "capture", min(config["deadline_epoch"], time.time() + config["capture_seconds"])
        self.done = threading.Event()

    def reason(self, now, rss, size):
        if now >= self.until:
            return self.phase + "_deadline"
        if rss > self.baseline + self.config["max_rss_growth_bytes"]:
            return "rss_growth"
        if size > self.config["max_trace_bytes"]:
            return "output_size"
        return None

    def watch(self):
        while not self.done.wait(0.25):
            try:
                size = output_bytes(self.output)
                reason = self.reason(time.time(), rss_bytes(), size)
            except Exception as error:
                reason = "guard_read_error:" + repr(error)
            if reason:
                try:
                    write_record(self.output / "budget-exit.json", {"reason": reason, "phase": self.phase,
                                 "pid": os.getpid(), "run_id": self.config["run_id"], "time": time.time()})
                finally:
                    self.exit_fn(124)
                return

    def __enter__(self):
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def exporting(self):
        self.phase = "export"
        self.until = min(self.config["deadline_epoch"], time.time() + self.config["export_seconds"])

    def __exit__(self, *unused):
        self.done.set()
        self.thread.join(timeout=1)


@contextlib.contextmanager
def annotations(torch, module, optimizer, models):
    """Only scoped Python call ranges; native autograd/c10d/CUDA events remain intact."""
    restored, names = [], []

    def patch(owner, name, label):
        if not hasattr(owner, name):
            return
        raw = inspect.getattr_static(owner, name)
        original = getattr(owner, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        restored.append((owner, name, raw, name in vars(owner)))
        setattr(owner, name, staticmethod(wrapped) if isinstance(raw, staticmethod) else wrapped)
        names.append(label)

    try:
        for name in ("all_to_all_single", "all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor"):
            patch(torch.distributed, name, "actor.comm_enqueue." + name)
        schedule = importlib.import_module("megatron.core.pipeline_parallel.schedules")
        for name in ("forward_step", "backward_step"):
            patch(schedule, name, "actor." + name)
        # Checkpoint backward includes recompute AND backward; do not call it pure recompute.
        random = importlib.import_module("megatron.core.tensor_parallel.random")
        checkpoint = getattr(random, "CheckpointFunction", None)
        if checkpoint:
            patch(checkpoint, "backward", "actor.checkpoint_recompute_and_backward")
        classes = {type(part) for model in models for part in model.modules()}
        for cls in classes:
            if cls.__name__ == "MoELayer":
                patch(cls, "forward", "actor.moe_forward")
            if "_checkpointed_forward" in cls.__dict__:
                patch(cls, "_checkpointed_forward", "actor.checkpointed_forward")
        yield names
    finally:
        for owner, name, original, owned in reversed(restored):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)



def validate_native_schedule(args, module, schedule, expected_sha):
    expected = {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1, "expert_model_parallel_size": 4}
    if any(getattr(args, name, None) != value for name, value in expected.items()):
        raise ValueError("Microbatch capture requires TP1/PP1/CP1/EP4")
    if getattr(args, "overlap_moe_expert_parallel_comm", False):
        raise ValueError("Overlapping MoE combined schedule is outside the capture ABI")
    if module.get_forward_backward_func() is not schedule.forward_backward_no_pipelining:
        raise ValueError("Expected the exact native no-pipeline schedule")
    if digest_file(schedule.__file__) != expected_sha:
        raise ValueError("Installed no-pipeline schedule source SHA mismatch")


def selected_packing(packing, index):
    if len(packing["iterators"]) != 1 or packing["num_microbatches"] <= index:
        raise ValueError("Require one model chunk and at least two microbatches")
    item = packing["iterators"][0]
    indices = item["micro_batch_indices"][index]
    records = {s["local_index"]: s for s in item["samples"]}
    return {"index": index, "local_indices": indices, "samples": [records[i] for i in indices],
            "total_tokens": sum(records[i]["length"] for i in indices),
            "response_tokens": sum(records[i]["response_length"] for i in indices)}


class MicrobatchCapture:
    """One continuous profile/range spanning verified F_i then B_i calls.

    Native PP1 source calls F0/B0, F1/B1, ... sequentially. Other schedule
    variants are refused before entry; observed calls/paired tensor identity are
    checked independently. No trace export happens in either wrapped function.
    """
    def __init__(self, torch, schedule, iterator, count, index=1):
        if count <= index:
            raise ValueError("Target microbatch does not exist")
        self.torch, self.schedule, self.iterator = torch, schedule, iterator
        self.count, self.index = count, index
        self.forward_calls = self.backward_calls = 0
        self.active = self.complete = False
        self.stack, self.profiler, self.cleanup_error = None, None, None
        self.last_output = None
        self.initial_offset = iterator.offset
        self.selected_indices = None

    def close(self, exc=(None, None, None)):
        if self.stack is not None:
            stack, self.stack = self.stack, None
            self.active = False
            stack.__exit__(*exc)

    def __enter__(self):
        self.original_forward = self.schedule.forward_step
        self.original_backward = self.schedule.backward_step
        forward_signature = inspect.signature(self.original_forward)
        backward_signature = inspect.signature(self.original_backward)

        @functools.wraps(self.original_forward)
        def forward(*values, **keywords):
            bound = forward_signature.bind(*values, **keywords).arguments
            current = bound.get("current_microbatch")
            if current != self.forward_calls or self.forward_calls != self.backward_calls or self.forward_calls >= self.count:
                raise RuntimeError("Native forward/backward order differs from sequential PP1 ABI")
            if bound.get("data_iterator") is not self.iterator or bound.get("num_microbatches") != self.count:
                raise RuntimeError("Native forward iterator/count does not match selected update")
            if getattr(bound["config"], "overlap_moe_expert_parallel_comm", False):
                raise RuntimeError("Combined overlapping MoE schedule is unsupported")
            stride = 1 if self.iterator.micro_batch_indices is not None else self.iterator.micro_batch_size
            if self.iterator.offset != self.initial_offset + current * stride:
                raise RuntimeError("Observed native iterator offset differs from the recorded microbatch packing")
            if current == self.index:
                if self.iterator.micro_batch_indices is not None:
                    self.selected_indices = list(self.iterator.micro_batch_indices[self.iterator.offset])
                else:
                    self.selected_indices = list(range(self.iterator.offset, self.iterator.offset + stride))
                self.torch.cuda.synchronize()  # Drain previous MB before recording.
                self.profiler = self.torch.profiler.profile(
                    activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False, with_flops=False)
                self.stack = contextlib.ExitStack()
                self.stack.enter_context(self.profiler)
                self.stack.enter_context(self.torch.profiler.record_function("actor.selected_microbatch"))
                self.active = True
            result = self.original_forward(*values, **keywords)
            self.last_output = result[0]
            self.forward_calls += 1
            return result

        @functools.wraps(self.original_backward)
        def backward(*values, **keywords):
            bound = backward_signature.bind(*values, **keywords).arguments
            if self.forward_calls != self.backward_calls + 1 or bound.get("output_tensor") is not self.last_output:
                raise RuntimeError("Native backward is not paired with its immediately preceding forward")
            result = self.original_backward(*values, **keywords)
            if self.backward_calls == self.index:
                if not self.active:
                    raise RuntimeError("Selected backward has no active profiler")
                self.torch.cuda.synchronize()  # Complete only the selected F/B window.
                self.close()
                self.complete = True
            self.backward_calls += 1
            self.last_output = None
            return result

        self.schedule.forward_step, self.schedule.backward_step = forward, backward
        return self

    def __exit__(self, typ, value, tb):
        try:
            self.close((typ, value, tb))
        except BaseException as cleanup:
            self.cleanup_error = repr(cleanup)
            if value is None:
                raise
            if hasattr(value, "add_note"):
                value.add_note("Profiler cleanup also failed: " + repr(cleanup))
        finally:
            self.schedule.forward_step, self.schedule.backward_step = self.original_forward, self.original_backward
        if value is None and (not self.complete or self.forward_calls != self.count or self.backward_calls != self.count):
            raise RuntimeError("Incomplete native F/B capture or unexpected observed microbatch count")
        return False

    def observation(self):
        return {"forward_calls": self.forward_calls, "backward_calls": self.backward_calls,
                "expected_calls_each": self.count, "captured_forward_calls": int(self.complete),
                "captured_backward_calls": int(self.complete), "capture_complete": self.complete,
                "cleanup_error": self.cleanup_error, "selected_local_indices": self.selected_indices}

def install(args):
    import torch
    from miles.backends.megatron_utils import model as module

    config = json.loads(os.environ[ENV])
    validate_config(config, os.environ, time.time())
    if (os.getuid(), os.getgid()) != (28644, 30):
        raise ValueError("Expected ordinary diagnostic UID 28644:GID 30")
    if torch.distributed.get_world_size() != config["world_size"]:
        raise ValueError("Unexpected training world size")
    if not args.debug_train_only or args.use_pytorch_profiler or args.save or args.debug_disable_optimizer:
        raise ValueError("Require debug replay, real optimizer, no native profiler or checkpoint saving")
    if args.load_debug_rollout_data != config["rollout_path"] or digest_file(config["rollout_path"]) != config["rollout_sha256"]:
        raise ValueError("Frozen rollout identity mismatch")
    if digest_file(module.__file__) != config["train_step_source_sha256"]:
        raise ValueError("Miles train_one_step source mismatch")
    if (args.hf_checkpoint, args.ref_load) != (config["hf_checkpoint"], config["ref_load"]):
        raise ValueError("Initial policy paths differ from the planned recipe")
    schedule = importlib.import_module("megatron.core.pipeline_parallel.schedules")
    validate_native_schedule(args, module, schedule, config["schedule_source_sha256"])
    rank = torch.distributed.get_rank()
    output = Path(config["output_root"]) / ("rank" + str(rank))
    output.mkdir(parents=True, exist_ok=False)
    write_record(output / "installed.json", {"config": config, "rank": rank, "pid": os.getpid(),
                 "hook_sha256": digest_file(__file__), "torch": torch.__version__})
    original = module.train_one_step
    signature = inspect.signature(original)
    if not {"rollout_id", "step_id", "attempt", "data_iterator", "num_microbatches", "optimizer", "model"} <= set(signature.parameters):
        raise ValueError("Unsupported train_one_step signature")
    seen = False

    @functools.wraps(original)
    def train_step(*values, **keywords):
        nonlocal seen
        bound = signature.bind(*values, **keywords).arguments
        identity = [bound["rollout_id"], bound["step_id"], bound["attempt"]]
        if identity[0] != 0 or identity[1] not in range(4) or identity[2] != 0:
            raise ValueError("Unexpected diagnostic update identity")
        selected = identity == config["target"] and rank == 0
        if selected and seen:
            raise RuntimeError("Target update encountered more than once")
        seen = seen or selected
        packing = packing_record(bound["data_iterator"], bound["num_microbatches"])
        prefix = "step" + str(identity[1])
        write_record(output / (prefix + "-packing.json"), packing)
        # All participants align outside the timed/profiled window. In particular,
        # peers must not charge rank-0 trace export to the next unprofiled update.
        torch.distributed.barrier()
        torch.cuda.synchronize()
        started, clock_start = time.time(), time.perf_counter()

        def timed_result(result, finished=None):
            if finished is None:
                torch.cuda.synchronize()
                finished = (time.time(), time.perf_counter())
            elapsed = finished[1] - clock_start
            grad = result[1]
            if hasattr(grad, "item"):
                grad = grad.item()
            write_record(output / (prefix + "-timing.json"), {
                "identity": identity, "rank": rank, "profiled_rank0": identity == config["target"],
                "start_epoch": started, "end_epoch": finished[0], "update_seconds": elapsed,
                "timing_basis": "host perf_counter bracketed by whole-update CUDA synchronize; preceding diagnostic barrier excluded",
                "warmup": identity[1] == 0, "grad_norm": float(grad),
                "outcome": getattr(result[2], "name", str(result[2])),
                "packing_fingerprint": packing["fingerprint"]})
            if not math.isfinite(float(grad)) or float(grad) <= 0 or getattr(result[2], "name", str(result[2])) != "NORMAL":
                raise RuntimeError("Selected diagnostic update did not produce a positive finite NORMAL gradient")
            return result

        if not selected:
            return timed_result(original(*values, **keywords))
        capture = None
        try:
            chosen = selected_packing(packing, config["microbatch_index"])
            validate_native_schedule(args, module, schedule, config["schedule_source_sha256"])
            with Budget(config, output) as budget:
                with annotations(torch, module, bound["optimizer"], bound["model"]) as labels:
                    with MicrobatchCapture(torch, schedule, bound["data_iterator"][0],
                                           bound["num_microbatches"], config["microbatch_index"]) as capture:
                        result = original(*values, **keywords)
                        torch.cuda.synchronize()
                        finished = (time.time(), time.perf_counter())
                timed_result(result, finished)
                if capture.selected_indices != chosen["local_indices"]:
                    raise RuntimeError("Captured microbatch indices do not match the recorded packing")
                # The full update/optimizer has returned. Peers wait at the next
                # update boundary barrier, not inside a forward/backward call.
                budget.exporting()
                raw = output / "actor-update.json"
                capture.profiler.export_chrome_trace(str(raw))
                if raw.stat().st_size > config["max_trace_bytes"] // 2:
                    raise RuntimeError("Raw trace exceeds reserved raw+gzip budget")
                # Size is capped above and RSS remains guarded during validation/compression.
                parsed = json.loads(raw.read_bytes())
                events = parsed["traceEvents"]
                counts = {kind: sum(e.get("cat") == kind for e in events)
                          for kind in ("kernel", "cpu_op", "python_function")}
                if not counts["kernel"] or not any(e.get("name") == "actor.selected_microbatch" for e in events):
                    raise RuntimeError("Trace lacks GPU kernels or the selected microbatch range")
                if counts["python_function"]:
                    raise RuntimeError("Unexpected Python stack events in stackless capture")
                del events, parsed
                compressed = output / "actor-update.json.gz"
                with raw.open("rb") as source, gzip.open(compressed, "wb", compresslevel=1) as target:
                    for chunk in iter(lambda: source.read(MIB), b""):
                        target.write(chunk)
                record = {"status": "COMPLETE", "rank": 0, "target": config["target"], "start_epoch": started,
                          "end_epoch": time.time(), "annotations": labels, "event_counts": counts,
                          "packing_fingerprint": packing["fingerprint"],
                          "capture_window": config["capture_window"], "microbatch_index": config["microbatch_index"],
                          "selected_microbatch": chosen, "schedule_observation": capture.observation(),
                          "schedule_source_sha256": config["schedule_source_sha256"],
                          "optimizer": None, "final_gradient_sync": None,
                          "trace": {"path": str(compressed), "bytes": compressed.stat().st_size,
                                    "sha256": digest_file(compressed)},
                          "raw": {"bytes": raw.stat().st_size, "sha256": digest_file(raw)},
                          "scope": "one sequential forward/backward microbatch including recompute/MoE enqueue; optimizer and final gradient synchronization excluded",
                          "profiler_options": {"with_stack": False, "record_shapes": False, "profile_memory": False}}
                write_record(output / "receipt.json", record)
            return result
        except BaseException as error:
            write_record(output / "failure.json", {"error": repr(error), "time": time.time(), "target": config["target"],
                         "schedule_observation": capture.observation() if capture is not None else None})
            raise

    module.train_one_step = train_step
