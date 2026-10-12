"""Flags and the grid runner shared by every experiment script.

    --seeds 0 1 2         the papers average three seeds
    --epochs N            override every run's epoch count (for smoke tests;
                          it changes the run identity, so full runs are untouched)
    --dry-run             list the runs and whether each is done, train nothing
    --report-only         skip training, rebuild the report from what is finished
    --shard i/n           train only runs i, i+n, i+2n, ... (split over processes)
"""

import argparse

from .train import is_done, run_name, train


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None, help="override the paper's epoch count")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th config starting at i")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data")
    p.add_argument("--imagenet-root", default=None,
                   help="ImageNet folder with train/ and val/ class subfolders")
    p.add_argument("--workers", type=int, default=8, help="ImageNet loader workers")
    return p


def apply_overrides(cfg, args):
    cfg = dict(cfg)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
        # milestone schedules are given in epochs; scale them with the run
        if cfg.get("milestones") and cfg.get("paper_epochs"):
            f = args.epochs / cfg["paper_epochs"]
            cfg["milestones"] = [max(1, round(m * f)) for m in cfg["milestones"]]
    return cfg


def unique(cfgs):
    seen, out = set(), []
    for cfg in cfgs:
        name = run_name(cfg)
        if name not in seen:
            seen.add(name)
            out.append(cfg)
    return out


def run_grid(cfgs, args):
    """Train every config in the grid (skipping finished ones) unless told not to."""
    cfgs = unique(cfgs)
    shard, nshards = (int(x) for x in args.shard.split("/"))
    todo = [c for c in cfgs if not is_done(c)]
    print(f"{len(cfgs)} runs, {len(cfgs) - len(todo)} finished, {len(todo)} to go")
    if args.dry_run:
        for c in cfgs:
            print(("  done  " if is_done(c) else "  todo  ") + run_name(c))
        return
    if args.report_only:
        return
    # shard over the full grid, not the to-do list, which shrinks as other
    #shards finish runs
    for i, cfg in enumerate(cfgs):
        if i % nshards != shard or is_done(cfg):
            continue
        train(cfg, device=args.device, data_root=args.data_root,
              imagenet_root=args.imagenet_root, workers=args.workers)
