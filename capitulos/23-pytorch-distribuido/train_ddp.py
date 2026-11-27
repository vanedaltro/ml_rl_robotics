"""MLP sintético em PyTorch DDP; inicialize o programa com torchrun."""
import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

N_SAMPLES = 2048
N_FEATURES = 16
N_CLASSES = 4
SEED = 7


def dataset():
    rng = torch.Generator().manual_seed(SEED)
    x = torch.randn(N_SAMPLES, N_FEATURES, generator=rng)
    teacher = torch.randn(N_FEATURES, N_CLASSES, generator=rng)
    return TensorDataset(x, (x @ teacher).argmax(dim=1))


def model():
    return nn.Sequential(nn.Linear(N_FEATURES, 64), nn.ReLU(), nn.Linear(64, N_CLASSES))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32, help="amostras por processo")
    parser.add_argument("--checkpoint", type=Path, default=Path("ddp.pt"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("--epochs e --batch-size devem ser positivos")
    return args


def main():
    args = parse_args()
    if not all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise SystemExit("Execute via torchrun ou python -m torch.distributed.run.")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if N_SAMPLES % (world_size * args.batch_size):
        raise SystemExit("Use 2.048 divisível por WORLD_SIZE × --batch-size neste lab.")

    if args.device == "cuda":
        if not dist.is_nccl_available() or local_rank >= torch.cuda.device_count():
            raise SystemExit("CUDA exige PyTorch com NCCL e uma GPU visível por processo.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
        device_ids = [local_rank]
    else:
        if not dist.is_gloo_available():
            raise SystemExit("CPU exige PyTorch com suporte ao backend Gloo.")
        torch.set_num_threads(1)
        device = torch.device("cpu")
        backend = "gloo"
        device_ids = None

    dist.init_process_group(backend=backend, timeout=timedelta(seconds=90))
    try:
        rank = dist.get_rank()
        data = dataset()  # Mesmos exemplos e mesma ordem inicial em todos os processos.
        sampler = DistributedSampler(data, shuffle=True, seed=SEED)
        batches = DataLoader(data, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        torch.manual_seed(SEED)
        ddp = DDP(model().to(device), device_ids=device_ids)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        print(f"rank={rank} local_rank={local_rank} world_size={world_size} "
              f"backend={backend} shard={len(sampler)}", flush=True)

        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            # Soma das perdas, número de acertos e número de exemplos.
            stats = torch.zeros(3, device=device, dtype=torch.float64)
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = ddp(x)
                loss = criterion(logits, y)
                loss.backward()  # DDP sincroniza os gradientes neste ponto.
                optimizer.step()  # Cada rank atualiza sua própria cópia.
                with torch.no_grad():
                    stats[0] += loss.detach().double() * y.numel()
                    stats[1] += (logits.argmax(1) == y).sum()
                    stats[2] += y.numel()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, correct, count = stats.tolist()
            if count != len(data):
                raise RuntimeError(f"Época processou {count}, esperado {len(data)}.")
            if rank == 0:
                print(f"epoch={epoch + 1:02d} train_loss={total_loss / count:.4f} "
                      f"train_acc={correct / count:.2%} examples={int(count)}", flush=True)

        # Verificação didática: compare os parâmetros locais com os do rank 0.
        with torch.no_grad():
            flat = nn.utils.parameters_to_vector(ddp.module.parameters()).detach()
            ref = flat.clone()
            dist.broadcast(ref, src=0)
            gap = (flat - ref).abs().max()
            dist.all_reduce(gap, op=dist.ReduceOp.MAX)
            if not torch.isfinite(gap).item() or gap.item() > 1e-6:
                raise RuntimeError(f"Réplicas divergiram: {gap.item():.3e}")
        if rank == 0:
            print(f"max_replica_diff={gap.item():.3e}", flush=True)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            weights = {key: value.detach().cpu() for key, value in ddp.module.state_dict().items()}
            tmp = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            torch.save({"model_state": weights, "epoch": args.epochs}, tmp)
            tmp.replace(args.checkpoint)
            print(f"checkpoint={args.checkpoint}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
