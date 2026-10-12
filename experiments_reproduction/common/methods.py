"""One preset per method, and the builder that turns a run config into a model.

A run config is a flat dict. Values an experiment sets explicitly win over the
method's preset, which wins over DEFAULTS. The presets say how each method
removes, regrows and initializes; the experiment scripts set the paper-level
protocol (dataset, network, optimizer, schedule, epochs).

  fc         the dense control network
  static     a fixed sparse topology (the "BA/WS static" rows of Tables 3-4)
  set        SET: ER, Kaiming, weight-magnitude removal, random regrowth
  rigl       RigL: as SET but gradient regrowth and zeta cosine-annealed to 0
  mest       MEST with EM&S: |w| + 0.01|grad| removal, random regrowth,
             starting at 50% sparsity and dropping to the target in two stages
  gmp        gradual magnitude pruning, cubic decay from 50% sparsity
  granet     GraNet: GMP's cubic prune stage + RigL-style rewiring
  dynamic_ba the dynamic BSF baseline: BSF init, preferential-attachment regrowth
  dynamic_ws the dynamic BSW baseline: BSW init, uniform regrowth
  esml       Section 4.1's ESML: ER, Kaiming, magnitude removal, percolation,
             CH3-L3 regrowth - the bare link-prediction rule, none of CHT's
             other three steps
  set_esml   Section 4.1's SET arm: the same framework with random regrowth
  cht        CHT (ICLR 2024): CSTI + SWI + ESML + the 90%-overlap early stop
  chts       CHTs (2026): soft CH2-L3n regrowth, no early stop
  chtss      CHTss (2026): CHTs + the sigmoid density decay from 50% sparsity

The CHTs / CHTss presets are the ones experiments/CHTss/run_chtss.py found to
reproduce Table 1 of the 2026 paper (its PRESETS and README explain why they
differ from the paper's prose in places).
"""

import torch
import torch.nn as nn

from utils.data import DATASETS, input_dim
from utils.model_defs import CHT, MLP
from utils.schedules import DensityDecay

from . import cnn
from .evolvers import GMP, CHTEvolver, EMSchedule, PruneRegrow, Static
from .masks import layer_masks

DEFAULTS = dict(
    # network
    arch="mlp",             # "mlp" or a key of cnn.ARCHS
    hidden=None,            # hidden width; None means scale x the input width
    scale=1,                # the papers' 1x / 2x
    n_sparse=3,             # sandwich layers before the dense classifier
    # topology
    init="er", beta=1.0, weight_init="swi",
    sparsity=0.99, init_sparsity=None,
    zeta=0.3, update_interval=1,
    # optimization
    epochs=100, batch_size=32, eval_batch=2048,
    lr=0.01, lr_schedule="constant", lr_end=None,
    milestones=(), gamma=0.1, warmup_epochs=0,
    momentum=0.9, weight_decay=1e-5,
    seed=0,
    # bookkeeping; not part of a run's identity
    save_masks_every=0, checkpoint_every=0, track_itop=False,
)

METHODS = {
    "fc": dict(evolver=None),
    "static": dict(evolver="static"),
    "set": dict(evolver="prune_regrow", removal="wm", regrow="random", zeta_schedule="fixed",
                init="er", weight_init="kaiming"),
    "rigl": dict(evolver="prune_regrow", removal="wm", regrow="gradient", zeta_schedule="cosine",
                 zeta_end=0.75, init="er", weight_init="kaiming"),
    "mest": dict(evolver="prune_regrow", removal="mest", regrow="random", zeta_schedule="fixed",
                 ems=True, init_sparsity=0.5, ems_milestones=(0.25, 0.5), mest_lambda=0.01,
                 init="er", weight_init="kaiming"),
    "gmp": dict(evolver="gmp", decay="cubic", init_sparsity=0.5, decay_end=0.75,
                weight_init="kaiming_dense"),
    "granet": dict(evolver="prune_regrow", removal="wm", regrow="gradient", zeta_schedule="cosine",
                   zeta_end=0.75, decay="cubic", init_sparsity=0.5, decay_end=0.75,
                   init="er", weight_init="kaiming"),
    "dynamic_ba": dict(evolver="prune_regrow", removal="wm", regrow="pa", zeta_schedule="fixed",
                       init="ba", weight_init="swi"),
    "dynamic_ws": dict(evolver="prune_regrow", removal="wm", regrow="random", zeta_schedule="fixed",
                       init="ws", weight_init="swi"),
    "esml": dict(evolver="cht", scorer="ch3_l3p", early_stop=None, remove="wm", soft_regrow=False,
                 init="er", weight_init="kaiming"),
    "set_esml": dict(evolver="cht", scorer="random", early_stop=None, remove="wm", soft_regrow=False,
                     init="er", weight_init="kaiming"),
    "cht": dict(evolver="cht", scorer="ch3_l3p", early_stop=0.9, remove="wm", soft_regrow=False,
                init="csti", weight_init="swi"),
    "chts": dict(evolver="cht", scorer="ch2_l3n", early_stop=None, remove="wm", soft_regrow=True,
                 adaptive_zeta=False, end_temperature=3.0, init="csti", weight_init="swi",
                 track_itop=True),
    "chtss": dict(evolver="cht", scorer="ch2_l3n", early_stop=None, remove="wm_soft",
                  soft_regrow=False, adaptive_zeta=True, zeta_end=0.75, end_temperature=9.0,
                  decay="sigmoid", init_sparsity=0.5, decay_end=0.75, prune="ri",
                  init="csti", weight_init="swi", track_itop=True),
}

# removal criterion -> (alpha, soft); alpha=1 weight magnitude, alpha=0 relative importance
CRITERIA = {"wm": (1.0, False), "wm_soft": (1.0, True), "ri": (0.0, False), "ri_soft": (0.0, True)}


def resolve(cfg):
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; pick one of {sorted(METHODS)}")
    out = dict(DEFAULTS)
    out.update(METHODS[method])
    out.update(cfg)
    if out.get("init_sparsity") is None:
        out["init_sparsity"] = out["sparsity"]
    return out


def feature_dim(cfg):
    return input_dim(cfg["dataset"]) if cfg["arch"] == "mlp" else cnn.ARCHS_OUT_DIM[cfg["arch"]]


def hidden_width(cfg):
    return cfg["hidden"] or int(round(cfg["scale"] * feature_dim(cfg)))


def kaiming_sparse_(sparse):
    """Kaiming init of the kept weights, the DST baselines' choice (ICLR 2024, Figure 9)."""
    for layer in sparse.sparse_layers():
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        layer.weight.data.mul_(layer.mask)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def build(cfg, device="cuda"):
    """(model, sparse head or None, evolver or None) for a resolved config."""
    dataset, arch = cfg["dataset"], cfg["arch"]
    num_classes = DATASETS[dataset][0]
    features = None if arch == "mlp" else cnn.backbone(arch)
    in_dim = feature_dim(cfg)
    hidden = hidden_width(cfg)
    dims = [in_dim] + [hidden] * cfg["n_sparse"]

    sparse, evolver = None, None
    if cfg["method"] == "fc":
        if arch == "mlp" and cfg["n_sparse"] == 3:
            head = MLP(dataset, hidden, num_classes)
        else:
            head = cnn.DenseHead(dims, num_classes)
    else:
        decaying = cfg.get("decay") or cfg.get("ems")
        start_density = 1.0 - (cfg["init_sparsity"] if decaying else cfg["sparsity"])
        if cfg["init"] == "csti" and arch != "mlp":
            raise ValueError("CSTI reads input pixels; the CNN heads use er / ws / ba")
        masks = layer_masks(dims, start_density, cfg["init"], cfg["seed"], beta=cfg["beta"],
                            dataset=dataset if arch == "mlp" else None)
        alpha, soft_remove = CRITERIA[cfg.get("remove", "wm")]
        head = CHT(
            masks,
            num_classes,
            scorer=cfg.get("scorer", "random"),
            early_stop=cfg.get("early_stop"),
            soft_regrow=cfg.get("soft_regrow", False),
            soft_remove=soft_remove,
            alpha=alpha,
            adaptive_zeta=cfg.get("adaptive_zeta", False),
            end_temperature=cfg.get("end_temperature", 3.0),
            track_itop=cfg["track_itop"],
        )
        if cfg["weight_init"] == "kaiming":
            kaiming_sparse_(head)
        elif cfg["weight_init"] == "kaiming_dense":
            GMP.initialize(head, start_density)
        elif cfg["weight_init"] != "swi":
            raise ValueError(f"unknown weight_init {cfg['weight_init']!r}")
        sparse = head

    model = head if features is None else cnn.BackboneWithHead(features, head)
    model = model.to(device)

    if sparse is not None:
        rounds, ui = cfg["epochs"], cfg["update_interval"]
        decay = None
        if cfg.get("decay"):
            decay = DensityDecay(cfg["init_sparsity"], cfg["sparsity"], rounds,
                                 shape=cfg["decay"], end_fraction=cfg["decay_end"])
        kind = cfg["evolver"]
        if kind == "static":
            evolver = Static(sparse, rounds)
        elif kind == "cht":
            evolver = CHTEvolver(sparse, rounds, ui, cfg["zeta"],
                                 adaptive_zeta=cfg.get("adaptive_zeta", False),
                                 zeta_end=cfg.get("zeta_end", 0.75), decay=decay,
                                 prune_alpha=0.0 if cfg.get("prune", "ri") == "ri" else 1.0)
        elif kind == "prune_regrow":
            ems = (EMSchedule(cfg["init_sparsity"], cfg["sparsity"], rounds,
                              tuple(cfg["ems_milestones"])) if cfg.get("ems") else None)
            evolver = PruneRegrow(model, sparse, rounds, ui, cfg["zeta"],
                                  removal=cfg["removal"], regrow=cfg["regrow"],
                                  zeta_schedule=cfg["zeta_schedule"],
                                  zeta_end=cfg.get("zeta_end", 0.75), decay=decay, ems=ems,
                                  mest_lambda=cfg.get("mest_lambda", 0.01))
        elif kind == "gmp":
            evolver = GMP(sparse, rounds, ui, decay)
        else:
            raise ValueError(f"unknown evolver {kind!r}")
    return model, sparse, evolver


def scheduled_density(cfg, rnd):
    """The density the method's schedule targets after round `rnd` (for logging)."""
    if cfg["method"] == "fc":
        return 1.0
    if cfg.get("decay"):
        d = DensityDecay(cfg["init_sparsity"], cfg["sparsity"], cfg["epochs"],
                         shape=cfg["decay"], end_fraction=cfg["decay_end"])
        return d.density(rnd)
    if cfg.get("ems"):
        return EMSchedule(cfg["init_sparsity"], cfg["sparsity"], cfg["epochs"],
                          tuple(cfg["ems_milestones"])).density(rnd)
    return 1.0 - cfg["sparsity"]


def lr_at(cfg, epoch, it=0, iters_per_epoch=1):
    """Learning rate for epoch `epoch` (0-based), iteration `it` inside it.

    constant   Appendix P.1 of the ICLR 2024 paper: "a fixed rate for all tasks"
    linear     2026 Table 4: linear from lr to lr_end, via utils.schedules
    multistep  2024 Table 5: x gamma at each milestone epoch, with a per-iteration
               linear warm-up over the first `warmup_epochs`
    """
    from utils.schedules import linear_lr

    kind = cfg["lr_schedule"]
    if kind == "constant":
        return cfg["lr"]
    if kind == "linear":
        return linear_lr(epoch, cfg["lr"], cfg["lr_end"], cfg["epochs"])
    if kind == "multistep":
        if epoch < cfg["warmup_epochs"]:
            total = cfg["warmup_epochs"] * iters_per_epoch
            return cfg["lr"] * (epoch * iters_per_epoch + it + 1) / total
        passed = sum(epoch >= m for m in cfg["milestones"])
        return cfg["lr"] * cfg["gamma"] ** passed
    raise ValueError(f"unknown lr_schedule {kind!r}")

