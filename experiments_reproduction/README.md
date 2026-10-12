# Paper reproductions

Scripts reproducing Table 1 of the ICLR 2024 CHT paper (`CH3-L3.pdf`,
*Epitopological learning and Cannistraci-Hebb network shape intelligence
brain-inspired theory for ultra-sparse advantage in deep learning*) and the
CIFAR-10 experiment of the 2026 CHTs/CHTss paper (`CHT_in_Transformers.pdf`,
Table 1). The Transformer and LLaMA experiments of the 2026 paper are out of
scope.

Everything is built on `utils/`: the masked MLP and `CHT` model, CSTI and ER
masks, the CH link predictors, percolation, the density-decay and LR
schedules, and the GPU loader are all imported from there. `common/` adds only
what `utils/` lacks (see its `__init__.py`).

## Running

Run from the repository root:

```bash
PYTHONPATH=. python experiments_reproduction/<paper>/<script>.py [flags]
```

Every training script shares these flags (`common/cli.py`):

| flag | |
|---|---|
| `--seeds 0 1 2` | seeds to run; the papers average three |
| `--epochs N` | override the epoch count, for smoke tests (the runs are stored separately from full runs) |
| `--dry-run` | list the runs and whether each is finished |
| `--report-only` | rebuild the report and figures from finished runs, train nothing |
| `--shard i/n` | train every n-th run starting at i, to split a grid over processes or GPUs |
| `--imagenet-root DIR` | ImageNet folder with `train/` and `val/` class subfolders |

Most scripts also take `--rows`, `--tasks`, `--datasets` or `--scales` to run
part of a grid.

Each run is identified by a hash of its full config and stored once in
`results/runs/<dataset>_<arch>_<method>_<hash>_seed<k>.json`, holding the
config, a per-epoch log and a summary. A finished run is never retrained, and
two experiments that need the same run share it. CNN runs checkpoint every epoch
and resume after an interruption. Each script writes `results/<experiment>/report.md`
(ours as mean ± SE over seeds, the paper's value in brackets) . Set `REPRO_RESULTS=<dir>` to put all of it somewhere else.

## Experiments

### ICLR 2024 (`cht2024/`)

| paper | script | what |
|---|---|---|
| Table 1 | `table1_cnn.py` | FC / RigL / MEST / CHT, sparse 3-layer heads on VGG16, GoogLeNet, ResNet50, ResNet152 (Table 5 schedules), 1x and 2x; ACC and AAE |

### 2026 CHTs / CHTss (`chts2026/`)

| paper | script | what |
|---|---|---|
| Table 1 | `table1_cifar10.py` | MLP 3072x3 on CIFAR-10 at 99% sparsity: FC, SET, RigL, CHT, CHTs; GMP, GraNet, CHTss decaying from 50% sparsity |

### Cost

The CNN runs are 200 epochs each. Table 1 is 5 tasks (6 with ImageNet) × 2
scales × 4 methods × 3 seeds = 120 runs. Use `--rows` / `--tasks` / `--seeds`
to prioritize, and `--shard` to spread the work.

## Methods (`common/methods.py`)

| method | init | weights | removal | regrowth | other |
|---|---|---|---|---|---|
| `fc` | dense | Kaiming | - | - | |
| `set` | ER | Kaiming | weight magnitude | random | |
| `rigl` | ER | Kaiming | weight magnitude | top-k dense gradient | ζ cosine-annealed to 0 at 75% of training |
| `mest` | ER | Kaiming | \|w\| + 0.01\|grad\| | random | EM&S: from 50% sparsity in two discrete stages |
| `gmp` | magnitude | Kaiming | - | - (pruned weights can reappear) | cubic decay 50% → target over 75% of training |
| `granet` | ER | Kaiming | weight magnitude | top-k dense gradient | GMP's cubic prune stage + RigL rewiring |
| `static` | BA / WS | SWI | - | - | |
| `dynamic_ba` / `dynamic_ws` | BSF / BSW | SWI | weight magnitude | preferential attachment / random | |
| `esml`, `set_esml` | ER | Kaiming | weight magnitude | CH3-L3p / random | percolation; Section 4.1's framework |
| `cht` | CSTI (MLP) / ER (CNN) | SWI | weight magnitude | CH3-L3p top-k | percolation, 90%-overlap early stop |
| `chts` | CSTI | SWI | weight magnitude | CH2-L3n, soft | percolation |
| `chtss` | CSTI at 50% sparsity | SWI | soft weight magnitude, exponent 1→9 | CH2-L3n top-k | sigmoid decay pruning by relative importance; ζ annealed to 0 at 75% |

The CHTs and CHTss presets are the ones `experiments/CHTss/` found to
reproduce the 2026 Table 1 numbers (see its README for why they depart from
the paper's prose). Any preset value can be overridden per run. In a check
against that folder's best run (seed 1, 71.03%), the `chtss` preset here
follows the same density trajectory to five decimals over the first six
epochs. Its test accuracy stays within ±2 points of that run. The two are not
bit-identical, which is ordinary GPU nondeterminism.

## Choices the papers leave open

Each of these is a config value that can be overridden. Where a choice is
also documented in code, it sits next to that value.

| what | choice here | where |
|---|---|---|
| weight decay of GoogLeNet / ResNet | the model zoo defaults: 5e-4 (CIFAR), 1e-4 (ImageNet) | `protocols.CNN_2024` |
| CNN backbones | the CIFAR model-zoo variants that Table 5's schedule comes from (3x3 stem, no max-pool); torchvision ResNet50 for ImageNet; global average pooling in front of the head | `common/cnn.py` |
| MEST EM&S stages | 50% sparsity, then the geometric midpoint at 25% of training, then the target at 50% | `methods.METHODS["mest"]`, `evolvers.EMSchedule` |
| RigL / GraNet ζ decay end | 75% of training (RigL's default) | `methods.METHODS` |
| GMP / GraNet decay window | 0 → 75% of training, as CHTss | `methods.METHODS` |
| dynamic BA / WS weight init | SWI (they are the paper's own models) | `methods.METHODS` |
| BSF surplus frustrations | one endpoint is re-attached by preferential attachment, so the link count is kept (Appendix M says both nodes are rewired) | `masks.bipartite_scale_free` |
| ACC | best test accuracy over the epochs (the reference convention); `final_acc` is also stored | `train.py` |

## Not reproduced exactly

- **ImageNet** has to be downloaded by hand. The ImageNet task runs only
  with `--imagenet-root`.
- **Tiny-ImageNet** reports accuracy on the labelled validation split (the
  test split has no labels). It is downloaded and decoded to
  `data/tiny-imagenet-200/*_uint8.pt` on first use.
