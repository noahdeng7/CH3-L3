"""ICLR 2024, Table 1: CHT vs RigL, MEST and FC on CNN heads, ACC and AAE.

Six tasks (VGG16 on Tiny-ImageNet; GoogLeNet on CIFAR-100 and Tiny-ImageNet;
ResNet50 on CIFAR-100 and ImageNet; ResNet152 on CIFAR-100). The backbone is
dense; the three fully connected layers before the classifier are the sparse
sandwich layers, at 1x and 2x the backbone's feature width, 99% sparsity.
CHT starts from ER here (Appendix H). Schedules are Table 5's
(common.protocols.CNN_2024).

ImageNet has to be downloaded by hand; its task only runs with --imagenet-root.

    PYTHONPATH=. python experiments_reproduction/cht2024/table1_cnn.py
    PYTHONPATH=. python experiments_reproduction/cht2024/table1_cnn.py --tasks ResNet50-CIFAR100 --scales 1
"""

from experiments_reproduction.cht2024 import paper_tables, rows
from experiments_reproduction.cht2024.table_grid import all_cfgs, build_cells, report
from experiments_reproduction.common import cli
from experiments_reproduction.common.protocols import CNN_2024

EXPERIMENT = "cht2024_table1_cnn"


def default_tasks(args):
    names = args.tasks or [t for t in CNN_2024 if t != "ResNet50-ImageNet" or args.imagenet_root]
    return {t: CNN_2024[t] for t in names}


def main():
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--tasks", nargs="+", choices=list(CNN_2024), default=None)
    p.add_argument("--scales", type=int, nargs="+", default=[1, 2])
    p.add_argument("--rows", nargs="*", help="only rows whose label contains one of these")
    args = p.parse_args()

    tasks = default_tasks(args)
    cells, labels = build_cells(tasks, lambda s: rows.select(rows.table1_rows(s), args.rows),
                                args.scales, args)
    cli.run_grid(all_cfgs(cells), args)
    report(EXPERIMENT, "ICLR 2024 Table 1 - CNNs with sparse heads at 99% sparsity", cells,
           labels, tasks, paper_tables.TABLE1, paper_tables.TABLE1_TASKS)


if __name__ == "__main__":
    main()
