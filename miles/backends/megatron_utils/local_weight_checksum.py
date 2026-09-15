"""Per-rank per-step weight checksum dumper for cross-replica consistency verification.

Design principle: fail fast. This module must never silently produce partial results.
If any parameter, buffer, master weight, or optimizer state cannot be hashed, it should
raise an error rather than skip it — incomplete checksums defeat the purpose of
cross-replica consistency verification.
"""

import hashlib
import logging
from argparse import Namespace
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes

if TYPE_CHECKING:
    from miles.utils.audit_utils.event_logger.models import OptimizerStateInfo, TrainEngineLocalWeightChecksumState

logger = logging.getLogger(__name__)


class _MainParamId(NamedTuple):
    tensor_id: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "_MainParamId":
        return cls(tensor_id=id(tensor))


def dump_local_weight_checksums(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
) -> None:
    """Compute and dump weight checksums if enabled."""

    if not args.save_local_weight_checksum:
        return

    # Local imports to break circular dependency:
    # logger.py → models.py → model.py → local_weight_checksum.py → logger.py
    from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
    from miles.utils.audit_utils.event_logger.models import TrainEngineLocalWeightChecksumEvent

    assert is_event_logger_initialized(), "save_local_weight_checksum is enabled but EventLogger is not initialized"

    event_logger = get_event_logger()
    state = _compute_weight_checksum_state(
        model=model,
        optimizer=optimizer,
    )
    event_logger.log(
        TrainEngineLocalWeightChecksumEvent,
        dict(state=state),
        print_log=False,
    )


def _compute_weight_checksum_state(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
) -> "TrainEngineLocalWeightChecksumState":
    from miles.utils.audit_utils.event_logger.models import TrainEngineLocalWeightChecksumState

    param_hashes = _hash_named_tensors(model, accessor="named_parameters")
    assert param_hashes, "No parameters found in model"
    buffer_hashes = _hash_named_tensors(model, accessor="named_buffers")

    optimizer_hashes = _collect_optimizer_hashes(model=model, optimizer=optimizer)
    assert optimizer_hashes, "No sub-optimizers found"

    return TrainEngineLocalWeightChecksumState(
        param_hashes=param_hashes,
        buffer_hashes=buffer_hashes,
        optimizer_hashes=optimizer_hashes,
    )


def _hash_named_tensors(model: Sequence[DDP], *, accessor: str) -> dict[str, str]:
    """Hash all named tensors from model chunks using the given accessor method.

    Witness params (_is_witness_param) are intentionally included: cross-replica
    checksum should verify that witness weights are consistent across replicas.
    """
    hashes: dict[str, str] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, tensor in sorted(getattr(model_chunk, accessor)(), key=lambda x: x[0]):
            assert tensor is not None, f"pp{pp_idx}.{name}: tensor is None"
            hashes[f"pp{pp_idx}.{name}"] = _hash_tensor_sha256(tensor)
    return hashes


def _collect_optimizer_hashes(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
) -> list["OptimizerStateInfo"]:
    """Collect optimizer state snapshots with tensors replaced by hashes."""
    from miles.utils.audit_utils.event_logger.models import OptimizerStateInfo

    name_by_tensor_id = _build_name_by_tensor_id(model, optimizer=optimizer)
    result: list[OptimizerStateInfo] = []

    for sub_opt in _iter_sub_optimizers(optimizer):
        inner = sub_opt.optimizer
        assert isinstance(inner, torch.optim.Optimizer), f"Expected torch.optim.Optimizer, got {type(inner)}"

        param_names = _build_param_names_for_optimizer(inner, name_by_tensor_id=name_by_tensor_id)
        sd = inner.state_dict()
        hashed_sd = _transform_tensor_to_hash(sd)

        result.append(
            OptimizerStateInfo(
                param_names=param_names,
                state_dict=hashed_sd,
            )
        )

    return result


def _build_name_by_tensor_id(
    model: Sequence[DDP], optimizer: MegatronOptimizer | None = None
) -> dict[_MainParamId, str]:
    """Map FP32 master copies, native FP32 params, and owned FP32 shards to names."""
    name_map: dict[_MainParamId, str] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            assert param is not None, f"pp{pp_idx}.{name}: param is None"
            main_param = getattr(param, "main_param", None)
            if main_param is None and param.dtype == torch.float32:
                # Float16OptimizerWithFloat16Params keeps native FP32 params
                # themselves in the inner optimizer; they need no master copy.
                main_param = param
            if main_param is None:
                assert getattr(param, "main_param_sharded", False), (
                    f"pp{pp_idx}.{name}: main_param is None but main_param_sharded is not set. "
                    "Expected only for distributed optimizer params not owned by this DP rank."
                )
                continue
            name_map[_MainParamId.from_tensor(main_param)] = f"pp{pp_idx}.{name}"

    if optimizer is not None:
        for sub_opt in _iter_sub_optimizers(optimizer):
            param_map = getattr(sub_opt, "model_param_group_index_map", None)
            if param_map is None:
                continue
            # DistributedOptimizer uses detached native-FP32 shard views, whose
            # IDs differ from the model parameter, without setting main_param.
            # Its ownership map points to the actual inner optimizer tensor;
            # FP32 parameters not owned by this DP rank have no entry here.
            for param, (group_index, group_order) in param_map.items():
                if param.dtype != torch.float32:
                    continue
                name = name_map.get(_MainParamId.from_tensor(param))
                assert name is not None, "Native FP32 optimizer parameter not found in model name mapping"
                shard = sub_opt.optimizer.param_groups[group_index]["params"][group_order]
                assert shard.dtype == torch.float32, f"{name}: native FP32 optimizer shard has dtype {shard.dtype}"
                name_map[_MainParamId.from_tensor(shard)] = name
    return name_map


def _build_param_names_for_optimizer(
    inner: torch.optim.Optimizer,
    name_by_tensor_id: dict[_MainParamId, str],
) -> dict[int, str]:
    """Build state_dict index → name mapping by walking optimizer's param_groups."""
    param_names: dict[int, str] = {}
    idx = 0
    for group in inner.param_groups:
        for fp32_param in group["params"]:
            key = _MainParamId.from_tensor(fp32_param)
            name = name_by_tensor_id.get(key)
            assert name is not None, f"fp32 param {key} not found in model name mapping"
            param_names[idx] = name
            idx += 1
    return param_names


def _transform_tensor_to_hash(obj: Any) -> Any:
    """Recursively replace all tensors in a nested structure with their SHA-256 hashes."""
    if isinstance(obj, torch.Tensor):
        return _hash_tensor_sha256(obj)
    if isinstance(obj, dict):
        return {k: _transform_tensor_to_hash(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_transform_tensor_to_hash(v) for v in obj)
    return obj


def _iter_sub_optimizers(optimizer: MegatronOptimizer) -> Iterator[MegatronOptimizer]:
    """Flatten ChainedOptimizer into individual sub-optimizers."""
    if hasattr(optimizer, "chained_optimizers"):
        for sub in optimizer.chained_optimizers:
            yield from _iter_sub_optimizers(sub)
    else:
        yield optimizer


def _hash_tensor_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = _hash_tensor_bytes(tensor)
    return hashlib.sha256(raw_bytes).hexdigest()
