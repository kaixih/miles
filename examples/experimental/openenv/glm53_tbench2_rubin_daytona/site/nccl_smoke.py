#!/usr/bin/env python3
"""CUDA NCCL smoke across all nodes of the run: one all-reduce and every directed cross-node rank pair.

Launched by run_nccl_node.py under torchrun on every node at once (4 ranks per node). XOR matchings
over the world size cover each directed cross-node edge exactly once; the world size must be a power
of two (2 nodes -> 8 ranks -> 32 edges; 8 nodes -> 32 ranks -> 896 edges). Transport is read from the
NCCL debug log the launcher captures, not from these data checks.
"""

import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    rank, local_rank, node_rank = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "GROUP_RANK"))
    world = int(os.environ["WORLD_SIZE"])
    assert int(os.environ["LOCAL_WORLD_SIZE"]) == 4 and rank == node_rank * 4 + local_rank
    assert world & (world - 1) == 0 and world >= 8, "XOR matchings need a power-of-two world size with at least two nodes"
    assert os.environ["NCCL_CUMEM_ENABLE"] == "1" and torch.cuda.is_available() and torch.cuda.device_count() == 4
    torch.cuda.set_device(local_rank)
    probe_dir = Path(os.environ["SMOKE_OUTPUT_DIR"])
    node = socket.gethostname().split(".")[0]
    identity = {"probe": probe_dir.name, "node": node, "rank": rank, "local_rank": local_rank}
    with (probe_dir / f"rank-{rank:02d}.jsonl").open("x", buffering=1) as records:
        def record(event, **values):
            records.write(json.dumps({"event": event, **identity, **values}, sort_keys=True) + "\n")

        record("RANK_READY", torch_version=torch.__version__, cuda_version=torch.version.cuda,
               gpu_name=torch.cuda.get_device_name(local_rank))
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        try:
            tensor = torch.full((65536,), rank + 1, device="cuda", dtype=torch.float32)
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            expected_sum = world * (world + 1) / 2
            assert torch.equal(tensor, torch.full_like(tensor, expected_sum)), tensor[0].item()
            record("ALLREDUCE_OK", value=tensor[0].item())
            edges = 0
            for pairing in range(4, world):
                peer = rank ^ pairing
                assert peer // 4 != node_rank
                sent = torch.full_like(tensor, rank * world + peer + 1)
                received = torch.empty_like(tensor)
                for request in dist.batch_isend_irecv([dist.P2POp(dist.isend, sent, peer), dist.P2POp(dist.irecv, received, peer)]):
                    request.wait()
                torch.cuda.synchronize()
                assert torch.equal(received, torch.full_like(received, peer * world + rank + 1)), (peer, received[0].item())
                edges += 1
                record("P2P_OK", source_rank=peer, target_rank=rank)
            dist.barrier()
            torch.cuda.synchronize()
        finally:
            dist.destroy_process_group()
        record("RANK_EXIT_OK", received_cross_node_edges=edges)


if __name__ == "__main__":
    main()
