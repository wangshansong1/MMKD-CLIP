# Copyright (c) Meta Platforms, Inc. and affiliates

import os

import torch

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):

    import os
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0

    if getattr(args, "horovod", False):
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        args.local_rank = int(hvd.local_rank())
        args.rank = int(hvd.rank())
        args.world_size = int(hvd.size())
        args.distributed = True
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)

    # ---------- 2) PyTorch DDP ----------
    elif is_using_distributed():

        args.local_rank = int(os.environ.get("LOCAL_RANK",
                               os.environ.get("SLURM_LOCALID", 0)))

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
        elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
            args.rank = int(os.environ["SLURM_PROCID"])
            args.world_size = int(os.environ["SLURM_NTASKS"])

            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            os.environ["LOCAL_RANK"] = str(args.local_rank)

        if not dist.is_initialized():
            timeout = timedelta(seconds=3600)

            dist.init_process_group(
                backend=getattr(args, "dist_backend", "nccl"),
                init_method=getattr(args, "dist_url", "env://"),
                world_size=args.world_size,
                rank=args.rank,
                timeout=timeout,
            )
        args.distributed = True

    if torch.cuda.is_available():

        if args.distributed and not getattr(args, "no_set_device_rank", False):
            local = int(args.local_rank)
        else:
            local = 0

        ndev = torch.cuda.device_count()
        if ndev == 0:
            raise RuntimeError("CUDA is available but no device visible (device_count=0). "
                               "Check Slurm --gres=gpu:X / cgroup.")
        if not (0 <= local < ndev):
            raise RuntimeError(
                f"LOCAL_RANK={local} out of range for visible devices={ndev}. "
                "通常是 --nproc_per_node 与分到的 GPU 数不一致，或手动设了 CUDA_VISIBLE_DEVICES 造成冲突。"
            )

        torch.cuda.set_device(local)                
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")

    args.device = device
    return device
