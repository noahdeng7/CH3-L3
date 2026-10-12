"""The MLP of Table 4: 3 hidden layers, plain Linear + ReLU, dense classifier head.

No BatchNorm or dropout by default - Table 4 lists neither, and the CHT papers'
MLP is a plain stack. `reg` switches on the regularizers from the CIFAR ablation
harness; with it empty the network is exactly the paper's. The sparse arm masks
the three hidden Linears ("sandwich layers"); the classifier head stays dense.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import topology
from .data import input_dim

HIDDEN_LAYERS = 3


class MaskedLinear(nn.Linear):
    """A Linear gated by a binary mask held as a buffer.

    The forward pass multiplies rather than indexes, so a masked-out weight
    already receives zero gradient and evolution only has to touch the mask.
    """

    def __init__(self, mask, bias=True):
        out_features, in_features = mask.shape
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


def _stack(hidden_linears, num_classes, reg):
    """Flatten -> N x [Linear, BatchNorm?, ReLU, Dropout?] -> dense head."""
    layers = [nn.Flatten(start_dim=1)]
    for linear in hidden_linears:
        layers.append(linear)
        if reg.get("batchnorm"):
            layers.append(nn.BatchNorm1d(linear.out_features))
        layers.append(nn.ReLU())
        if reg.get("dropout"):
            layers.append(nn.Dropout(reg["dropout_rate"]))
    layers.append(nn.Linear(hidden_linears[-1].out_features, num_classes))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """The fully connected control network (the FC row of Table 7)."""

    def __init__(self, dataset, hidden, num_classes, reg=None):
        super().__init__()
        reg = reg or {}
        # BatchNorm supplies the bias, so the Linear in front of it goes without
        bias = not reg.get("batchnorm")
        dims = [input_dim(dataset)] + [hidden] * HIDDEN_LAYERS
        linears = [
            nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(HIDDEN_LAYERS)
        ]
        self.layers = _stack(linears, num_classes, reg)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


class CHT(nn.Module):
    """Cannistraci-Hebb Training, following the reference implementation.

    The variants differ only in how links leave and enter:

      CHT   hard weight-magnitude removal, hard CH3-L3p regrowth, cosine-annealed
            zeta, and the 90%-overlap topological early stop.
      CHTs  hard removal, *soft* (multinomial) CH3-L3n regrowth, the EM_S density
            schedule, and no early stop.
      CHTss as CHTs but with a sigmoid density decay in place of EM_S.
    """

    def __init__(
        self,
        masks,
        num_classes,
        scorer="ch3_l3p",
        early_stop=0.9,
        soft_regrow=False,
        soft_remove=False,
        alpha=1.0,
        adaptive_zeta=False,
        end_temperature=3.0,
        track_itop=False,
        reg=None,
    ):
        super().__init__()
        reg = reg or {}
        bias = not reg.get("batchnorm")
        # masks arrive as (in, out); nn.Linear.weight is (out, in)
        linears = [MaskedLinear(m.T.contiguous(), bias=bias) for m in masks]
        self.layers = _stack(linears, num_classes, reg)

        self.scorer = topology.SCORERS[scorer]
        self.early_stop = early_stop
        self.soft_regrow = soft_regrow
        self.soft_remove = soft_remove
        self.alpha = alpha
        self.adaptive_zeta = adaptive_zeta
        self.end_temperature = end_temperature
        self.frozen = [False] * len(linears)
        # in-time over-parameterization: the union of every mask the run has
        #held. Figure 3 reads the soft rule off this - a rule that keeps
        #exploring keeps adding links it has never used before, a rule trapped
        #in an epitopological local minimum flattens out. Held as buffers so
        #.to(device) and state_dict() carry them like the masks themselves.
        self.track_itop = track_itop
        if track_itop:
            for i, layer in enumerate(linears):
                self.register_buffer(f"itop_record{i}", layer.mask.bool().clone())

        nn.init.kaiming_normal_(self.layers[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.layers[-1].bias)
        self._swi()

    def _swi(self):
        """Sparse Weight Initialization.

        Kaiming and Xavier assume a dense fan-in, so at 1% density they leave
        the surviving weights far too small; SWI rescales by realised density.
        """
        gain = math.sqrt(2.0)
        for layer in self.sparse_layers():
            density = layer.mask.mean().item()
            std = gain / math.sqrt(max(density * layer.in_features, 1.0))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            layer.weight.data *= layer.mask
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def sparse_layers(self):
        return [m for m in self.layers if isinstance(m, MaskedLinear)]

    def masks(self):
        return [layer.mask for layer in self.sparse_layers()]

    def forward(self, x):
        return self.layers(x)

    def link_counts(self):
        return [int(m.sum().item()) for m in self.masks()]

    def anp(self):
        return topology.active_neuron_rate(self.masks())

    @property
    def record(self):
        if not self.track_itop:
            return None
        return [
            getattr(self, f"itop_record{i}") for i in range(len(self.sparse_layers()))
        ]

    def itop_rate(self):
        """Mean over layers of the fraction of positions ever carrying a link."""
        if not self.track_itop:
            return None
        rates = [r.sum().item() / r.numel() for r in self.record]
        return sum(rates) / len(rates)

    @property
    def converged(self):
        return all(self.frozen)

    def evolve(
        self,
        zeta,
        optimizer=None,
        progress=0.0,
        densities=None,
        prune_density=None,
        prune_alpha=0.0,
    ):
        """One round of topological evolution.

        `progress` is the fraction of training elapsed, used for the removal
        temperature and the cosine zeta. `densities` is the per-layer target
        density for this round; under it the round removes down to that density
        and regrows back to the round's link budget. With it None the layer
        keeps its current link count, which is plain fixed-sparsity DST.

        `prune_density` is the CHTss stage: the density this round's gradual
        decay puts the layer at, reached by a hard prune under `prune_alpha`
        (0 = relative importance, which is what Section 3.3 substitutes for the
        weight magnitude GMP and GraNet prune with) *before* the round rewires.
        The rewiring then conserves the post-prune count, so the decay alone
        sets the density and zeta alone sets the turnover, exactly as the
        reference runs its pruning stage ahead of removal and regrowth.
        """
        layers = self.sparse_layers()
        start = [layer.mask.clone() for layer in layers]

        pruned = [0] * len(layers)
        if prune_density is not None:
            # a frozen layer is still pruned: the early stop halts exploration,
            #while the decay is a density schedule that has to reach its target
            for i, layer in enumerate(layers):
                n_active = int(layer.mask.sum().item())
                target = round(layer.mask.numel() * prune_density)
                if n_active <= target:
                    continue
                topology.prune_hard(layer.weight.data, layer.mask, target, prune_alpha)
                pruned[i] = n_active - int(layer.mask.sum().item())

        before = [layer.mask.clone() for layer in layers]
        temperature = topology.removal_temperature(progress, 1.0, self.end_temperature)
        drop = (
            topology.cosine_annealed_zeta(zeta, progress, 1.0)
            if self.adaptive_zeta
            else zeta
        )

        budgets = []
        for i, layer in enumerate(layers):
            n_active = int(layer.mask.sum().item())
            if densities is None:
                n_prune = int(n_active * drop)
                budgets.append(None)          # regrow back to where we started
            else:
                # EM_S / sigmoid decay: strip down to the target density, then
                #regrow up to this round's budget
                target = round(layer.mask.numel() * densities[i]["target"])
                n_prune = max(0, n_active - target)
                budgets.append(round(layer.mask.numel() * densities[i]["regrow"]))
            n_keep = n_active - n_prune

            if self.frozen[i] or n_prune <= 0:
                continue
            if self.soft_remove:
                topology.prune_soft(
                    layer.weight.data, layer.mask, n_keep, self.alpha, temperature
                )
            else:
                topology.prune_hard(layer.weight.data, layer.mask, n_keep, self.alpha)

        percolated = topology.percolate(self.masks())
        mid = [layer.mask.clone() for layer in layers]

        stats = {}
        for i, layer in enumerate(layers):
            if budgets[i] is None:
                # conserve the link count, replacing percolated links too
                budget = int((before[i] - mid[i]).sum().item())
            else:
                budget = max(0, budgets[i] - int(mid[i].sum().item()))

            if not self.frozen[i]:
                topology.regrow_links(layer.mask, budget, self.scorer, self.soft_regrow)

            changed, layer_stats = topology.rewire_stats(
                before[i], mid[i], layer.mask, budget
            )
            # links the decay stage took are removals too: one regrown later in
            #this same round would otherwise come back carrying its old weight
            changed |= (start[i] != 0) & (mid[i] == 0)
            if optimizer is not None:
                self.reset_opt_state(optimizer, layer, changed)
            # a pruned link must not come back carrying the magnitude it was
            #pruned for, and a regrown one starts from zero
            layer.weight.data[changed] = 0.0

            if self.track_itop:
                getattr(self, f"itop_record{i}").logical_or_(layer.mask.bool())

            layer_stats |= {
                "percolated": percolated[i],
                "frozen": self.frozen[i],
                "pruned": pruned[i],
                "density": layer_stats["links"] / layer.mask.numel(),
            }
            stats[f"layer{i + 1}"] = layer_stats

            if (
                self.early_stop is not None
                and not self.frozen[i]
                and layer_stats["removed"] > 0
                and layer_stats["overlap"] > self.early_stop
            ):
                self.frozen[i] = True

        return stats

    def reset_opt_state(self, optimizer, layer, changed):
        #SGD momentum must be zeroed before a weight is reintroduced, so it does
        #not inherit the state it carried before being pruned
        state = optimizer.state.get(layer.weight)
        if not state:
            return
        for key in ("momentum_buffer", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state and state[key] is not None:
                state[key][changed] = 0.0
