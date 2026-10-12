"""The run grid and markdown report for Table 1 of the ICLR 2024 paper.

Table 1 is tasks (columns) x rows (methods, per 1x/2x scale) x seeds. The
report has one markdown table per metric - ours as mean±SE over the seeds, the
paper's value in brackets - for ACC (best test accuracy) and AAE.
"""

from experiments_reproduction.common import cli
from experiments_reproduction.common.report import (cell_stats, markdown_table, paper_lookup,
                                                    parse_paper_table, write_report)


def build_cells(tasks, rows_fn, scales, args, extra=None):
    """({(label, task): [cfg per seed]}, row labels in order)."""
    cells, labels = {}, []
    for scale in scales:
        for label, overrides in rows_fn(scale):
            if label not in labels:
                labels.append(label)
            for task, base in tasks.items():
                for seed in args.seeds:
                    cfg = dict(base, scale=scale, seed=seed, **overrides, **(extra or {}))
                    cells.setdefault((label, task), []).append(cli.apply_overrides(cfg, args))
    return cells, labels


def all_cfgs(cells):
    return [cfg for cfgs in cells.values() for cfg in cfgs]


METRICS = (("best_acc", "ACC (%)", 0), ("aae", "AAE (%)", 1))


def report(experiment, title, cells, labels, tasks, paper_text=None, paper_tasks=None,
           notes=""):
    parsed = parse_paper_table(paper_text) if paper_text else {}
    text = f"# {title}\n\nOurs: mean±SE over finished seeds; [paper].\n\n{notes}\n"
    for metric, name, index in METRICS:
        stats = cell_stats(cells, metric)
        paper = paper_lookup(parsed, paper_tasks, index) if parsed else None
        text += "\n" + markdown_table(labels, list(tasks), stats, paper=paper, title=name)
    return write_report(experiment, text)
