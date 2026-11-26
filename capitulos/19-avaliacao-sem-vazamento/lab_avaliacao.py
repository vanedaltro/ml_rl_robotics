"""Experiment: time windows with shared episodes and machines."""
import argparse
import json
import platform
import statistics
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

N_MACHINES, EPISODES, WINDOWS = 24, 8, 12
WIDTH, STRIDE = 24, 2
MODES = ("linhas", "episodios", "maquinas")


class Dataset(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    machine: torch.Tensor
    episode: torch.Tensor
    start: torch.Tensor


def dataset(seed=2026):
    g = torch.Generator().manual_seed(seed)
    n_episodes = N_MACHINES * EPISODES
    machine = torch.arange(N_MACHINES).repeat_interleave(EPISODES)
    episode = torch.arange(n_episodes)
    # Stable machine and episode signatures do not include IDs as feature columns.
    machine_signature = torch.randn(N_MACHINES, 8, generator=g)
    episode_signature = torch.randn(n_episodes, 8, generator=g)
    # Moderate fault signal plus random calibration and inter-episode offset.
    prevalence = 0.15 + 0.45 * torch.rand(N_MACHINES, generator=g)
    y = (torch.rand(n_episodes, generator=g) < prevalence[machine]).long()
    calibration = 0.45 * torch.randn(N_MACHINES, 2, generator=g)[machine]
    episode_variation = 0.7 * torch.randn(n_episodes, 2, generator=g)
    signal = 0.80 * (y.float()[:, None] - 0.5) + calibration + episode_variation
    level = torch.cat([signal, machine_signature[machine], episode_signature], dim=1)
    samples = WIDTH + STRIDE * (WINDOWS - 1)
    raw = level[:, None] + 0.22 * torch.randn(n_episodes, samples, level.shape[1], generator=g)
    # [episodes, window count, channels, window width]
    x = raw.unfold(1, WIDTH, STRIDE).mean(-1).flatten(0, 1)
    return Dataset(x, y.repeat_interleave(WINDOWS), machine.repeat_interleave(WINDOWS),
                   episode.repeat_interleave(WINDOWS), (torch.arange(WINDOWS) * STRIDE).repeat(n_episodes))


def split(data, mode, seed):
    g = torch.Generator().manual_seed(seed)
    if mode == "linhas":
        order = torch.randperm(len(data.y), generator=g)
        return order[len(order)//4:], order[:len(order)//4]
    if mode == "episodios":
        # Select two complete episodes from each of the 24 machines.
        groups = torch.tensor([EPISODES * m + i for m in range(N_MACHINES)
                               for i in torch.randperm(EPISODES, generator=g)[:2].tolist()])
        held_out = torch.isin(data.episode, groups)
    elif mode == "maquinas":
        groups = torch.randperm(N_MACHINES, generator=g)[:N_MACHINES//4]
        held_out = torch.isin(data.machine, groups)
    else:
        raise ValueError(mode)
    return torch.where(~held_out)[0], torch.where(held_out)[0]


def preprocess(train_x):
    """Return one pair of statistics fitted only on the training rows."""
    return train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)


def fit(data, train, seed):
    torch.manual_seed(seed)
    mu, sd = preprocess(data.x[train])
    x = (data.x[train] - mu) / sd
    y = data.y[train]
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    g = torch.Generator().manual_seed(seed + 100000)
    model.train()
    for _ in range(24):
        for i in torch.randperm(len(y), generator=g).split(128):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[i]), y[i]).backward()
            opt.step()
    return model.eval(), mu, sd


def score(y, pred):
    hit = y == pred
    recall = [float(hit[y==c].double().mean()) if (y==c).any() else None for c in (0,1)]
    return {"n": len(y), "positive": int((y==1).sum()),
            "accuracy_pct": 100 * float(hit.double().mean()),
            "balanced_accuracy_pct": 50 * sum(recall) if None not in recall else None,
            "false_positive": int(((y==0)&(pred==1)).sum()),
            "false_negative": int(((y==1)&(pred==0)).sum())}


def run(data, mode, seed):
    train, test = split(data, mode, seed)
    model, mu, sd = fit(data, train, seed)
    y = data.y[test]
    pred = model((data.x[test] - mu) / sd).detach().argmax(1)
    majority = int(torch.bincount(data.y[train], minlength=2).argmax())
    base = torch.full_like(y, majority)
    common_ep = torch.isin(data.episode[test], data.episode[train])
    common_machine = torch.isin(data.machine[test], data.machine[train])
    dist = (data.start[test][:, None] - data.start[train][None, :]).abs()
    raw_overlap = ((dist < WIDTH) & (data.episode[test][:,None] == data.episode[train][None,:])).any(dim=1)
    per_machine = []
    for machine in data.machine[test].unique().tolist():
        s = data.machine[test] == machine
        per_machine.append({"machine": machine, "mlp": score(y[s], pred[s]),
                            "majority": score(y[s], base[s])})
    return {"protocol": mode, "seed": seed, "train_n": len(train), "test_n": len(test),
            "train_majority_class": majority, "test_episode_seen_pct": 100 * float(common_ep.double().mean()),
            "test_machine_seen_pct": 100 * float(common_machine.double().mean()),
            "test_overlapping_window_pct": 100 * float(raw_overlap.double().mean()),
            "mlp": score(y,pred), "majority": score(y,base), "per_machine": per_machine}


def summarize(rows):
    result = []
    for mode in MODES:
        a = [r for r in rows if r["protocol"] == mode]
        m = {"protocol": mode}
        for field in ("test_episode_seen_pct","test_machine_seen_pct","test_overlapping_window_pct"):
            m[field] = statistics.mean(r[field] for r in a)
        for algo in ("mlp","majority"):
            m[algo] = {}
            for met in ("accuracy_pct","balanced_accuracy_pct"):
                nums = [r[algo][met] for r in a]
                m[algo][met] = {"mean": statistics.mean(nums),
                                "sample_sd": statistics.stdev(nums) if len(nums)>1 else None}
        result.append(m)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("resultados.json"))
    p.add_argument("--seeds", nargs="+", type=int, default=[7,23,42])
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = dataset()
    rows = [run(data, mode, seed) for mode in MODES for seed in args.seeds]
    summary = summarize(rows)
    body = {"metadata": {"pytorch": torch.__version__, "platform": platform.platform(),
                         "data_seed": 2026, "machines": N_MACHINES, "episodes_per_machine": EPISODES,
                         "windows_per_episode": WINDOWS, "window_width": WIDTH, "stride": STRIDE,
                         "epochs":24, "batch_size":128, "optimizer":"AdamW", "lr":0.003,
                         "threads":1, "script_seeds":args.seeds},
            "summary": summary, "runs": rows}
    args.out.write_text(json.dumps(body, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
