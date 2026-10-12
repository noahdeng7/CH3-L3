"""2026 CHTs/CHTss paper, Table 1: sparse training of an MLP on CIFAR-10 at 99% sparsity.

"Brain network science modelling of sparse neural networks enables
Transformers and LLMs to perform as fully connected". Eight rows, in two
groups:

  fixed sparsity   FC, SET, RigL, CHT, CHTs
  density decay    GMP, GraNet, CHTss   (all starting from 50% sparsity)

Protocol: Table 4 (common.protocols.MLP_2026_CIFAR10): 3 hidden layers of
3072, dense classifier, batch 32, 100 epochs, SGD momentum 0.9, weight decay
5e-4, LR linear 0.025 -> 2.5e-4, zeta 0.3, one update per epoch. CHT, CHTs and
CHTss start from CSTI; CHTs and CHTss regrow with CH2-L3n.

Reported per method: ACC (best test accuracy), its change against FC, and ANP.

    PYTHONPATH=. python experiments_reproduction/chts2026/table1_cifar10.py
    PYTHONPATH=. python experiments_reproduction/chts2026/table1_cifar10.py --methods chtss granet --seeds 0
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments_reproduction.common import cli
from experiments_reproduction.common.protocols import MLP_2026_CIFAR10
from experiments_reproduction.common.report import (cell_stats, epoch_curve, fmt, report_dir,
                                                    write_report)

EXPERIMENT = "chts2026_table1_cifar10"

ROWS = {  # label -> method preset
    "FC": "fc",
    "CHTs": "chts",
    "CHT": "cht",
    "RigL": "rigl",
    "SET": "set",
    "CHTss": "chtss",
    "GraNet": "granet",
    "GMP": "gmp",
}
# Table 1: ACC, change vs FC, ANP
PAPER = {
    "FC": ("62.85±0.16", "-", "-"),
    "CHTs": ("69.97±0.06", "+11.33%", "54%"),
    "CHT": ("59.10±0.06", "-5.97%", "96%"),
    "RigL": ("63.90±0.19", "+1.67%", "59%"),
    "SET": ("62.70±0.11", "-0.24%", "100%"),
    "CHTss": ("71.29±0.14", "+13.43%", "63%"),
    "GraNet": ("69.31±0.17", "+10.28%", "61%"),
    "GMP": ("65.11±0.11", "+3.60%", "75%"),
}


def main():
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--methods", nargs="+", choices=list(ROWS.values()), default=list(ROWS.values()))
    args = p.parse_args()

    rows = {label: m for label, m in ROWS.items() if m in args.methods}
    cells = {label: [cli.apply_overrides(dict(MLP_2026_CIFAR10, method=m, seed=s), args)
                     for s in args.seeds]
             for label, m in rows.items()}
    cli.run_grid([c for cfgs in cells.values() for c in cfgs], args)

    acc = cell_stats(cells, "best_acc")
    anp = cell_stats(cells, "anp")
    fc = acc.get("FC", (None,))[0]
    text = ("# 2026 Table 1 - MLP on CIFAR-10 at 99% sparsity\n\n"
            "Ours: mean±SE over finished seeds; [paper].\n\n"
            "| method | ACC (%) | vs FC | ANP (%) |\n|---|---|---|---|\n")
    for label in rows:
        mean = acc[label][0]
        change = "-" if fc is None or mean is None or label == "FC" else f"{100 * (mean - fc) / fc:+.2f}%"
        pa, pc, pn = PAPER[label]
        anp_s = "-" if label == "FC" else fmt(anp[label], 1)
        text += f"| {label} | {fmt(acc[label])} [{pa}] | {change} [{pc}] | {anp_s} [{pn}] |\n"

    fig, (ax_acc, ax_den) = plt.subplots(1, 2, figsize=(13, 4.5))
    for label in rows:
        curve = epoch_curve(cells[label])
        if curve is not None:
            x, m, se = curve
            ax_acc.plot(x, m, label=label)
            ax_acc.fill_between(x, m - se, m + se, alpha=0.2)
        if label != "FC":
            den = epoch_curve(cells[label], key="density")
            if den is not None:
                ax_den.plot(den[0], den[1], label=label)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.legend(frameon=False, fontsize=8)
    ax_acc.grid(alpha=0.3)
    ax_den.set_yscale("log")
    ax_den.set_xlabel("epoch")
    ax_den.set_ylabel("density (%)")
    ax_den.legend(frameon=False, fontsize=8)
    ax_den.grid(alpha=0.3)
    fig.suptitle("2026 Table 1: MLP 3072x3 on CIFAR-10, 99% final sparsity")
    fig.tight_layout()
    path = os.path.join(report_dir(EXPERIMENT), "curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    write_report(EXPERIMENT, text + "\n![curves](curves.png)\n")


if __name__ == "__main__":
    main()
