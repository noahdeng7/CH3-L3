"""The topology update of every method the two papers compare.

One round runs at the end of every `update_interval`-th epoch, before that
epoch's evaluation (Table 4 / Appendix P: update interval 1 epoch). Rounds are
counted from 1. Three families:

  CHTEvolver   CHT, CHTs, CHTss and the ESML-framework arms of Section 4.1.
               A thin driver around utils.model_defs.CHT.evolve, which already
               does removal, percolation, CH regrowth, the early stop and the
               sigmoid-decay prune stage.
  PruneRegrow  SET, RigL, MEST (with EM&S), GraNet, and the dynamic BSF/BSW
               baselines of Tables 3-4: drop the weakest links, regrow the same
               number by a non-topological rule, no percolation.
  GMP          gradual magnitude pruning: no regrowth, just a cubic density
               ramp, with pruned weights kept so they can come back.

Invariant kept by the first two: a link that is not in the mask has weight 0.
That is what lets `dense_grads` read RigL's gradient for missing links by
briefly opening the mask.
"""

import torch

from utils import topology


class Evolver:
    def __init__(self, sparse, total_rounds, update_interval=1, last_round=None):
        self.sparse = sparse
        self.total_rounds = total_rounds
        self.update_interval = update_interval
        # the reference skips the round that would land after the final epoch
        self.last_round = total_rounds - 1 if last_round is None else last_round

    def active(self, rnd):
        return rnd <= self.last_round and rnd % self.update_interval == 0

    def step(self, rnd, optimizer, batch, loss_fn):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


Static = Evolver


def keep_top(mask, score, n_keep):
    """Keep the `n_keep` highest-scoring links of `mask`, in place."""
    n_keep = max(1, int(n_keep))
    flat = score.masked_fill(mask == 0, float("-inf")).reshape(-1)
    if n_keep >= int(mask.sum().item()):
        return
    keep = torch.zeros_like(flat)
    keep[torch.topk(flat, n_keep, sorted=False).indices] = 1.0
    mask.copy_(keep.view_as(mask))


def pa_scores(mask, device=None):
    """Preferential attachment for the dynamic BSF baseline (Appendix M).

    "Shooting from random nodes the same number of links using preferential
    attachment probability": a node picked uniformly on one side links to the
    other side with probability proportional to degree (+1, so a node that lost
    every link can be picked again). Either side can shoot, so the two
    directions are averaged. Masks are (out, in).
    """
    a = mask.float()
    rows = a.sum(dim=1) + 1.0
    cols = a.sum(dim=0) + 1.0
    return (rows / rows.sum())[:, None] / a.shape[1] + (cols / cols.sum())[None, :] / a.shape[0]


def grow(mask, budget, how, grad=None):
    if budget <= 0:
        return
    if how == "random":
        topology.regrow_links(mask, budget, topology.random_scores, soft=False)
    elif how == "pa":
        topology.regrow_links(mask, budget, pa_scores, soft=True)
    elif how == "gradient":
        score = grad.abs().masked_fill(mask != 0, float("-inf")).reshape(-1)
        mask.view(-1)[torch.topk(score, int(budget), sorted=False).indices] = 1.0
    else:
        raise ValueError(f"unknown regrowth {how!r}")


def dense_grads(model, layers, batch, loss_fn):
    """Gradient of the loss w.r.t. every position of each sparse layer, present or not.

    MaskedLinear multiplies by its mask, so weight.grad is zero off the mask.
    Opening the masks for one forward/backward pass gives the dense gradient
    RigL and GraNet regrow by; since off-mask weights are zero, the forward
    pass itself is unchanged.
    """
    saved = [layer.mask.clone() for layer in layers]
    for layer in layers:
        layer.weight.data.mul_(layer.mask)
        layer.mask.fill_(1.0)
    model.zero_grad(set_to_none=True)
    x, y = batch
    loss_fn(model(x), y).backward()
    grads = [layer.weight.grad.detach().clone() for layer in layers]
    for layer, m in zip(layers, saved):
        layer.mask.copy_(m)
    model.zero_grad(set_to_none=True)
    return grads


class EMSchedule:
    """MEST's EM&S: start denser and drop to the target density in discrete stages.

    The 2026 paper describes it as "a two-stage density decrease strategy as
    described in the original work"; neither paper gives the stage epochs, so
    they are parameters here (fractions of training), and the intermediate
    density is the geometric midpoint between the start and the target.
    """

    def __init__(self, start_sparsity, final_sparsity, total_rounds, milestones=(0.25, 0.5)):
        self.start = 1.0 - start_sparsity
        final = 1.0 - final_sparsity
        self.rounds = [max(1, round(total_rounds * f)) for f in milestones]
        n = len(milestones)
        self.levels = [self.start * (final / self.start) ** ((i + 1) / n) for i in range(n)]
        self.tf = self.rounds[-1]

    def density(self, rnd):
        d = self.start
        for r, level in zip(self.rounds, self.levels):
            if rnd >= r:
                d = level
        return d

    def sparsity(self, rnd):
        return 1.0 - self.density(rnd)

    def prunes_at(self, rnd):
        return rnd in self.rounds


class CHTEvolver(Evolver):
    """Drives utils.model_defs.CHT.evolve, following experiments/CHTss/run_chtss.py.

    With adaptive zeta, zeta is cosine-annealed to 0 over the first `zeta_end`
    of training and no round runs after it, so the topology is frozen for the
    rest of the run (the reference's T_end). Otherwise rounds run to the end.
    """

    def __init__(self, sparse, total_rounds, update_interval, zeta, adaptive_zeta=False,
                 zeta_end=0.75, decay=None, prune_alpha=0.0):
        if adaptive_zeta:
            self.t_end = max(round(total_rounds * zeta_end), 1)
            last = self.t_end
        else:
            self.t_end = total_rounds
            last = total_rounds - 1
        super().__init__(sparse, total_rounds, update_interval, last)
        self.zeta = zeta
        self.decay = decay
        self.prune_alpha = prune_alpha

    def step(self, rnd, optimizer, batch, loss_fn):
        if not self.active(rnd):
            return {}
        prune_density = (
            self.decay.density(rnd)
            if self.decay is not None and self.decay.prunes_at(rnd)
            else None
        )
        if self.sparse.converged and prune_density is None:
            # every layer passed the early-stop overlap: nothing left to evolve
            return {"converged": True}
        return self.sparse.evolve(
            self.zeta,
            optimizer,
            progress=rnd / self.t_end,
            prune_density=prune_density,
            prune_alpha=self.prune_alpha,
        )

    def state_dict(self):
        return {"frozen": list(self.sparse.frozen)}

    def load_state_dict(self, state):
        self.sparse.frozen = list(state["frozen"])


class PruneRegrow(Evolver):
    """Remove-and-regrow without link prediction or percolation.

    removal  'wm'    weight magnitude (SET, RigL, GraNet, dynamic BSF/BSW)
             'mest'  |w| + lambda |grad| of the existing links (MEST, lambda
                     = 0.01 per the 2026 paper's Appendix G)
    regrow   'random' (SET, MEST, dynamic BSW), 'gradient' (RigL, GraNet),
             'pa' (dynamic BSF)
    zeta     'fixed', or 'cosine': RigL's cosine decay of the update fraction
             to zero at `zeta_end` of training, after which the mask is fixed
    decay    a utils.schedules.DensityDecay: GraNet's gradual prune stage,
             magnitude pruning to the scheduled density before rewiring
    ems      an EMSchedule: MEST's elastic mutation, removing the excess
             density on top of the usual zeta at each stage
    """

    def __init__(self, model, sparse, total_rounds, update_interval, zeta, removal="wm",
                 regrow="random", zeta_schedule="fixed", zeta_end=0.75, decay=None, ems=None,
                 mest_lambda=0.01):
        if zeta_schedule == "cosine":
            self.t_end = max(round(total_rounds * zeta_end), 1)
            last = self.t_end
        else:
            self.t_end = total_rounds
            last = total_rounds - 1
        for schedule in (decay, ems):
            if schedule is not None:
                last = max(last, schedule.tf)
        super().__init__(sparse, total_rounds, update_interval, last)
        self.model = model
        self.zeta = zeta
        self.removal = removal
        self.regrow = regrow
        self.zeta_schedule = zeta_schedule
        self.decay = decay
        self.ems = ems
        self.mest_lambda = mest_lambda

    def zeta_at(self, rnd):
        if self.zeta_schedule == "cosine":
            if rnd > self.t_end:
                return 0.0
            return topology.cosine_annealed_zeta(self.zeta, rnd, self.t_end)
        return self.zeta

    def step(self, rnd, optimizer, batch, loss_fn):
        if not self.active(rnd):
            return {}
        layers = self.sparse.sparse_layers()
        zeta = self.zeta_at(rnd)
        need_grad = self.regrow == "gradient" or self.removal == "mest"
        grads = (dense_grads(self.model, layers, batch, loss_fn) if need_grad
                 else [None] * len(layers))

        stats = {}
        for i, layer in enumerate(layers):
            mask, w = layer.mask, layer.weight.data
            before = mask.clone()
            n = int(mask.sum().item())

            # GraNet's prune stage: magnitude prune to this round's density
            if self.decay is not None and self.decay.prunes_at(rnd):
                keep_top(mask, w.abs(), round(mask.numel() * self.decay.density(rnd)))
            pruned = n - int(mask.sum().item())
            n -= pruned

            # elastic mutation: at an EM&S stage the round ends at the stage's
            #density, so it removes the excess on top of the zeta it rewires
            n_final = n
            if self.ems is not None and self.ems.prunes_at(rnd):
                n_final = min(n, round(mask.numel() * self.ems.density(rnd)))
            n_grow = int(n_final * zeta)
            n_remove = n - n_final + n_grow

            score = w.abs()
            if self.removal == "mest":
                score = score + self.mest_lambda * grads[i].abs()
            keep_top(mask, score, n - n_remove)
            mid = mask.clone()
            grow(mask, n_grow, self.regrow, grads[i])

            changed, layer_stats = topology.rewire_stats(before, mid, mask, n_grow)
            self.sparse.reset_opt_state(optimizer, layer, changed)
            w[changed] = 0.0
            layer_stats |= {"pruned": pruned, "zeta": zeta,
                            "density": layer_stats["links"] / mask.numel()}
            stats[f"layer{i + 1}"] = layer_stats
        return stats


class GMP(Evolver):
    """Gradual magnitude pruning (Zhu & Gupta), as the 2026 paper runs it.

    Appendix G.2: GMP "stores historical weights and allows pruned weights to
    reappear during training, since, during training, the pruning threshold
    might change." So pruned weights are masked, not zeroed, and each prune
    re-ranks every position of the layer - kept and previously pruned - by
    |w|. There is no regrowth. (A pruned weight receives no gradient, but
    SGD's weight decay still shrinks it.)
    """

    def __init__(self, sparse, total_rounds, update_interval, decay):
        super().__init__(sparse, total_rounds, update_interval, decay.tf)
        self.decay = decay

    @staticmethod
    def initialize(sparse, density):
        """Kaiming weights everywhere, masked down to the |w|-largest `density`."""
        for layer in sparse.sparse_layers():
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            n_keep = round(layer.mask.numel() * density)
            flat = layer.weight.data.abs().reshape(-1)
            keep = torch.zeros_like(flat)
            keep[torch.topk(flat, n_keep, sorted=False).indices] = 1.0
            layer.mask.copy_(keep.view_as(layer.mask))

    def step(self, rnd, optimizer, batch, loss_fn):
        if not (self.active(rnd) and self.decay.prunes_at(rnd)):
            return {}
        stats = {}
        for i, layer in enumerate(self.sparse.sparse_layers()):
            before = layer.mask.clone()
            n_keep = round(layer.mask.numel() * self.decay.density(rnd))
            flat = layer.weight.data.abs().reshape(-1)
            keep = torch.zeros_like(flat)
            keep[torch.topk(flat, n_keep, sorted=False).indices] = 1.0
            layer.mask.copy_(keep.view_as(layer.mask))
            links = int(layer.mask.sum().item())
            stats[f"layer{i + 1}"] = {
                "links": links,
                "density": links / layer.mask.numel(),
                "removed": int(((before != 0) & (layer.mask == 0)).sum().item()),
                "reappeared": int(((before == 0) & (layer.mask != 0)).sum().item()),
            }
        return stats
