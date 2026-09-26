#!/usr/bin/env python
"""NCCL collective bandwidth on one node (launch with torchrun --nproc_per_node=N).

Times all_reduce / reduce_scatter / all_gather at several sizes and dtypes, plus
a replay of ZeRO-2's per-microstep gradient traffic: the full gradient volume in
reduce_bucket_size chunks (default 2e8 elements, ds_zero2_config.json), each
reduce-scattered with a synchronize in between, as with overlap_comm=false.
Reports algorithm bandwidth (bytes / time) and bus bandwidth (nccl-tests
convention). Run with NCCL_DEBUG=INFO to see the chosen transports.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-elements", type=float, default=6.0e9, help="gradient elements per microstep (model size)")
    parser.add_argument("--bucket", type=float, default=2e8, help="ZeRO reduce_bucket_size in elements")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda")
    out = []

    def report(name, nbytes, seconds, busfactor):
        if rank == 0:
            algbw = nbytes / seconds / 1e9
            line = (f"{name:44s} {nbytes / 2**20:10.0f} MiB  {seconds * 1e3:9.2f} ms  "
                    f"algbw {algbw:7.1f} GB/s  busbw {algbw * busfactor:7.1f} GB/s")
            print(line, flush=True)
            out.append(line)

    for dtype in (torch.bfloat16, torch.float32):
        for mib in (64, 256, 1024, 4096):
            n = mib * 2**20 // torch.tensor([], dtype=dtype).element_size()
            n -= n % world
            x = torch.randn(n, device=dev, dtype=dtype)
            shard = torch.empty(n // world, device=dev, dtype=dtype)
            nbytes = x.numel() * x.element_size()
            report(f"all_reduce      {str(dtype)[6:]}", nbytes,
                   bench(lambda: dist.all_reduce(x), args.iters), 2 * (world - 1) / world)
            report(f"reduce_scatter  {str(dtype)[6:]}", nbytes,
                   bench(lambda: dist.reduce_scatter_tensor(shard, x), args.iters), (world - 1) / world)
            report(f"all_gather      {str(dtype)[6:]}", nbytes,
                   bench(lambda: dist.all_gather_into_tensor(x, shard), args.iters), (world - 1) / world)
            del x, shard
            torch.cuda.empty_cache()

    # ZeRO-2 microstep replay: grad volume in buckets, synchronous reduce-scatter per bucket.
    bucket = int(args.bucket) - int(args.bucket) % world
    buckets = int(args.grad_elements // bucket)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(bucket, device=dev, dtype=dtype)
        shard = torch.empty(bucket // world, device=dev, dtype=dtype)

        def microstep():
            for _ in range(buckets):
                dist.reduce_scatter_tensor(shard, x)
                torch.cuda.synchronize()

        seconds = bench(microstep, max(2, args.iters // 5), warmup=1)
        report(f"zero2 replay {buckets}x{bucket:.0e} {str(dtype)[6:]} (per microstep)",
               buckets * bucket * x.element_size(), seconds, (world - 1) / world)
        del x, shard
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
