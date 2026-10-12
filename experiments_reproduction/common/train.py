"""The one training loop every experiment runs, and the run cache behind it.

A run is identified by its resolved config (minus bookkeeping keys): the name
is `<dataset>_<arch>_<method>_<hash>_seed<k>`, so two experiments that need
the same run - Table 1 and Table 4 of the 2024 paper share their FC / RigL /
MEST rows, for instance - train it once. Results go to
`experiments_reproduction/results/runs/<name>.json` with the config, a
per-epoch log and a summary; a finished JSON is never retrained.

Per epoch: train, run the topology round (if any), evaluate on the test split.
The summary records what the papers report:

  best_acc   best test accuracy over the epochs (the ACC columns)
  aae        area across the epochs, ICLR 2024 Eq 12 (the AAE columns)
  anp        active neuron post-percolation rate at the end (utils.topology)
  time_to_best / total_time  wall-clock seconds (Figure 10B)
"""

import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .data import make_loaders
from .methods import build, lr_at, resolve, scheduled_density
from .metrics import aae

# REPRO_RESULTS redirects everything (runs and reports), e.g. for smoke tests
RESULTS = os.environ.get("REPRO_RESULTS", "experiments_reproduction/results")
RUNS = os.path.join(RESULTS, "runs")

# keys that change how a run is stored or executed, not what it computes
BOOKKEEPING = {"save_masks_every", "checkpoint_every", "eval_batch", "track_itop", "paper_epochs"}


def run_name(cfg):
    cfg = resolve(cfg)
    ident = {k: v for k, v in cfg.items() if k not in BOOKKEEPING and k != "seed"}
    digest = hashlib.sha1(json.dumps(ident, sort_keys=True, default=list).encode()).hexdigest()[:10]
    return f"{cfg['dataset']}_{cfg['arch']}_{cfg['method']}_{digest}_seed{cfg['seed']}"


def group_key(cfg):
    """Identity of a config across seeds: the run name without the seed."""
    return run_name(cfg).rsplit("_seed", 1)[0]


def result_path(cfg):
    return os.path.join(RUNS, run_name(cfg) + ".json")


def load_result(cfg):
    path = result_path(cfg)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def is_done(cfg):
    r = load_result(cfg)
    return r is not None and r.get("done", False)


def _to(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _to(batch, device)
            logits = model(x)
            loss_sum += loss_fn(logits, y) * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum()
            total += y.size(0)
    return loss_sum.item() / total, correct.item() / total


def _mask_snapshot(sparse):
    """Each sparse layer's mask as flat int32 indices of its links, plus shape."""
    return [
        {"shape": list(m.shape), "idx": torch.nonzero(m.reshape(-1)).squeeze(1).int().cpu()}
        for m in sparse.masks()
    ]


def load_masks(cfg):
    """{epoch: [dense bool (out, in) mask per layer]} saved by a run with save_masks_every."""
    path = result_path(cfg).replace(".json", "_masks.pt")
    snaps = torch.load(path)
    out = {}
    for epoch, layers in snaps.items():
        masks = []
        for layer in layers:
            m = torch.zeros(int(np.prod(layer["shape"])), dtype=torch.bool)
            m[layer["idx"].long()] = True
            masks.append(m.view(*layer["shape"]))
        out[epoch] = masks
    return out


def _layer_mean(stats, key):
    vals = [s[key] for s in stats.values() if isinstance(s, dict) and key in s]
    return float(np.mean(vals)) if vals else None


def train(cfg, device="cuda", data_root="data", imagenet_root=None, workers=8, verbose=True):
    cfg = resolve(cfg)
    name = run_name(cfg)
    os.makedirs(RUNS, exist_ok=True)
    json_path = os.path.join(RUNS, name + ".json")
    ckpt_path = os.path.join(RUNS, name + ".ckpt")
    mask_path = os.path.join(RUNS, name + "_masks.pt")

    previous = load_result(cfg)
    if previous is not None and previous.get("done"):
        return previous

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = make_loaders(
        cfg["dataset"], cfg["batch_size"], cfg["eval_batch"], device=device, root=data_root,
        imagenet_root=imagenet_root, workers=workers,
    )
    model, sparse, evolver = build(cfg, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    log, snapshots, start_epoch, elapsed = [], {}, 0, 0.0
    if sparse is not None and cfg["save_masks_every"]:
        snapshots[0] = _mask_snapshot(sparse)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if evolver is not None:
            evolver.load_state_dict(ckpt["evolver"])
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        log, start_epoch, elapsed = ckpt["log"], ckpt["epoch"], ckpt["elapsed"]
        if os.path.exists(mask_path):
            snapshots = torch.load(mask_path)
        if verbose:
            print(f"[{name}] resuming at epoch {start_epoch}", flush=True)

    if verbose:
        links = sparse.link_counts() if sparse is not None else "dense"
        print(f"[{name}] {cfg['method']} on {cfg['dataset']} / {cfg['arch']}; links {links}",
              flush=True)

    iters = len(train_loader)
    best = max((r["test_acc"] for r in log), default=0.0)
    early_stop_epoch = next((r["epoch"] for r in log if r.get("converged")), None)
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        lr = lr_at(cfg, epoch, 0, iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        warm = cfg["lr_schedule"] == "multistep" and epoch < cfg["warmup_epochs"]

        model.train()
        run_sum = torch.zeros((), device=device)
        seen = 0
        batch = None
        for it, batch in enumerate(train_loader):
            x, y = _to(batch, device)
            if warm:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(cfg, epoch, it, iters)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            run_sum += loss.detach() * y.size(0)
            seen += y.size(0)
        if device != "cpu":
            torch.cuda.synchronize()
        t_train = time.time() - t0

        t1 = time.time()
        stats = {}
        if evolver is not None:
            stats = evolver.step(epoch + 1, optimizer, _to(batch, device), loss_fn)
        if device != "cpu":
            torch.cuda.synchronize()
        t_evolve = time.time() - t1
        elapsed += t_train + t_evolve

        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        best = max(best, test_acc)

        record = {
            "epoch": epoch, "lr": lr,
            "train_loss": run_sum.item() / max(seen, 1),
            "test_loss": test_loss, "test_acc": test_acc, "best_acc": best,
            "time_train": t_train, "time_evolve": t_evolve, "elapsed": elapsed,
        }
        if sparse is not None:
            links = sparse.link_counts()
            total = sum(m.numel() for m in sparse.masks())
            record |= {
                "links": links,
                "density": sum(links) / total,
                "scheduled_density": scheduled_density(cfg, epoch + 1),
                "anp": sparse.anp(),
                "itop": sparse.itop_rate(),
                "overlap": _layer_mean(stats, "overlap"),
                "turnover": _layer_mean(stats, "turnover"),
                "gamma_in": _layer_mean(stats, "gamma_in"),
                "converged": bool(sparse.converged),
                "layers": stats,
            }
            if early_stop_epoch is None and sparse.converged:
                early_stop_epoch = epoch
        log.append(record)

        if verbose:
            extra = ""
            if sparse is not None:
                extra = (f"  density {100 * record['density']:.2f}%  ANP {100 * record['anp']:.1f}%"
                         + (f"  overlap {100 * record['overlap']:.0f}%"
                            if record["overlap"] is not None else ""))
            print(f"[{name}] epoch {epoch:3d}  lr {lr:.5f}  loss {record['train_loss']:.4f}  "
                  f"test {100 * test_acc:.2f}%  best {100 * best:.2f}%{extra}  "
                  f"{elapsed / 60:.1f} min", flush=True)

        every = cfg["save_masks_every"]
        if sparse is not None and every and ((epoch + 1) % every == 0 or epoch + 1 == cfg["epochs"]):
            snapshots[epoch + 1] = _mask_snapshot(sparse)
            torch.save(snapshots, mask_path)

        _write(json_path, cfg, name, log, done=False, early_stop_epoch=early_stop_epoch)

        if cfg["checkpoint_every"] and (epoch + 1) % cfg["checkpoint_every"] == 0:
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "evolver": evolver.state_dict() if evolver is not None else {},
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "log": log, "epoch": epoch + 1, "elapsed": elapsed,
            }, ckpt_path)

    result = _write(json_path, cfg, name, log, done=True, early_stop_epoch=early_stop_epoch)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if verbose:
        s = result["summary"]
        print(f"[{name}] done: best {100 * s['best_acc']:.2f}%  AAE {100 * s['aae']:.2f}  "
              + (f"ANP {100 * s['anp']:.1f}%" if s["anp"] is not None else ""), flush=True)
    return result


def _write(path, cfg, name, log, done, early_stop_epoch=None):
    accs = [r["test_acc"] for r in log]
    best_epoch = int(np.argmax(accs)) if accs else None
    summary = {
        "best_acc": max(accs) if accs else None,
        "final_acc": accs[-1] if accs else None,
        "best_epoch": best_epoch,
        # Eq 12 divides by E - 1; a single epoch's AAE is just its accuracy
        "aae": aae(accs) if len(accs) > 1 else (accs[0] if accs else None),
        "anp": log[-1].get("anp") if log else None,
        "density": log[-1].get("density") if log else None,
        "total_time": log[-1]["elapsed"] if log else None,
        "time_to_best": log[best_epoch]["elapsed"] if accs else None,
        "early_stop_epoch": early_stop_epoch,
        "epochs_run": len(log),
    }
    result = {"name": name, "config": cfg, "done": done, "summary": summary, "log": log}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, default=list)
    os.replace(tmp, path)
    return result
