"""Aggregate finished runs into the papers' tables: mean +- standard error over seeds.

An experiment lays its runs out as `cells`: {(row, column): [cfg per seed]}.
`cell_stats` reads each finished run's summary and reduces a metric over the
seeds; `markdown_table` puts it next to the paper's number for the same cell.
The paper's own tables are pasted into the scripts as text and read with
`parse_paper_table`.
"""

import os
import re

import numpy as np

from .train import RESULTS, load_result

PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s*[±+]\s*(\d+(?:\.\d+)?)")


def parse_paper_table(text):
    """{row label: [(value, error), ...]} from rows like 'CHT1×(CSTI) 98.05 ±0.04 96.9 ±0.06'.

    The label is everything before the first 'value ±error' pair; '*' marks
    and bold are ignored. A '+' in place of '±' (a typo in Table 4) is accepted.
    """
    out = {}
    for line in text.strip().splitlines():
        line = line.strip()
        first = PAIR.search(line)
        if not first:
            continue
        label = line[: first.start()].strip()
        out[label] = [(float(v), float(e)) for v, e in PAIR.findall(line)]
    return out


def mean_se(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return None, None, 0
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return float(v.mean()), float(se), int(v.size)


def summary_values(cfgs, metric, scale=100.0, finished_only=True):
    vals = []
    for cfg in cfgs:
        r = load_result(cfg)
        if r is None or (finished_only and not r.get("done")):
            continue
        v = r["summary"].get(metric)
        if v is not None:
            vals.append(scale * v)
    return vals


def cell_stats(cells, metric, scale=100.0):
    """{(row, col): (mean, se, n)} over the finished runs of each cell."""
    return {key: mean_se(summary_values(cfgs, metric, scale)) for key, cfgs in cells.items()}


def fmt(stat, digits=2):
    mean, se, n = stat
    if mean is None:
        return "-"
    return f"{mean:.{digits}f}±{se:.{digits}f}" + ("" if n >= 3 else f" (n={n})")


def markdown_table(rows, cols, stats, paper=None, col_names=None, title=None):
    """One line per row: our mean±se, then the paper's value in brackets."""
    col_names = col_names or cols
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("| | " + " | ".join(col_names) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for row in rows:
        cells = []
        for col in cols:
            ours = fmt(stats.get((row, col), (None, None, 0)))
            ref = (paper or {}).get((row, col))
            cells.append(ours if ref is None else f"{ours} [{ref}]")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def report_dir(experiment):
    path = os.path.join(RESULTS, experiment)
    os.makedirs(path, exist_ok=True)
    return path


def write_report(experiment, text):
    path = os.path.join(report_dir(experiment), "report.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")
    return path


def epoch_curve(cfgs, key="test_acc", scale=100.0):
    """(epochs, mean curve, se curve) of a per-epoch log value over finished seeds."""
    curves = []
    for cfg in cfgs:
        r = load_result(cfg)
        if r is None or not r.get("done"):
            continue
        curves.append([scale * (rec.get(key) or 0.0) for rec in r["log"]])
    if not curves:
        return None
    n = min(len(c) for c in curves)
    arr = np.asarray([c[:n] for c in curves])
    se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(n)
    return np.arange(1, n + 1), arr.mean(axis=0), se


def paper_lookup(parsed, tasks, index, per_task=2):
    """{(row, task): 'v±e'} for metric `index` (0 = ACC, 1 = AAE) of a parsed paper table."""
    out = {}
    for row, pairs in parsed.items():
        for i, task in enumerate(tasks):
            k = per_task * i + index
            if k < len(pairs):
                v, e = pairs[k]
                out[(row, task)] = f"{v:g}±{e:g}"
    return out
