import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import topology


class sparse_MNIST(nn.Module):

    def __init__(self, mask_path, scorer="ch3", early_stop=0.9):
        super().__init__()
        self.flatten = nn.Flatten()

        masks = np.load(mask_path)
        # npz masks are (in, out); nn.Linear.weight is (out, in)
        m1 = torch.from_numpy(masks["mask1"]).T.contiguous()
        m2 = torch.from_numpy(masks["mask2"]).T.contiguous()

        self.L1 = nn.Linear(m1.shape[1], m1.shape[0])
        self.L2 = nn.Linear(m2.shape[1], m2.shape[0])
        self.L3 = nn.Linear(m2.shape[0], 10)

        self.register_buffer("mask1", m1.float())
        self.register_buffer("mask2", m2.float())

        self.scorer = topology.SCORERS[scorer]
        self.early_stop = early_stop
        self.frozen = [False, False]
        self._swi()

    def _swi(self):
        #since we mask out 99% of the initialized weights, we need to rescale
        #the remaining weights back to their proper distribution

        gain = math.sqrt(2.0)
        for layer, mask in self.sparse_layers():
            density = mask.mean().item()
            std = gain / math.sqrt(max(density * layer.in_features, 1.0))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            layer.weight.data *= mask
            nn.init.zeros_(layer.bias)

    def sparse_layers(self):
        return ((self.L1, self.mask1), (self.L2, self.mask2))

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(F.linear(x, self.L1.weight * self.mask1, self.L1.bias))
        x = F.relu(F.linear(x, self.L2.weight * self.mask2, self.L2.bias))
        return self.L3(x)

    def evolve(self, zeta, optimizer=None):
        before = [m.clone() for _, m in self.sparse_layers()]

        for i, (layer, mask) in enumerate(self.sparse_layers()):
            if not self.frozen[i]:
                topology.prune_smallest(layer.weight.data, mask, zeta)

        # percolation runs on every layer, frozen or not: a frozen layer can
        #still lose links to a neuron that died upstream of it
        percolated = topology.percolate(self.mask1, self.mask2)
        mid = [m.clone() for _, m in self.sparse_layers()]

        # after percolation, percolated links are replaced rather than
        #just deleted and the total link count is conserved
        budgets = [int((b - m).sum().item()) for b, m in zip(before, mid)]

        for i, (_, mask) in enumerate(self.sparse_layers()):
            if not self.frozen[i]:
                topology.regrow(mask, budgets[i], self.scorer)

        stats = {}
        for i, (layer, mask) in enumerate(self.sparse_layers()):
            changed, layer_stats = topology.rewire_stats(
                before[i], mid[i], mask, budgets[i]
            )

            if optimizer is not None:
                self.reset_opt_state(optimizer, layer, changed)
            # both directions start from zero: a pruned link must not come back
            #carrying the magnitude it was pruned for
            layer.weight.data[changed] = 0.0

            layer_stats |= {"percolated": percolated[i], "frozen": self.frozen[i]}
            stats[f"layer{i + 1}"] = layer_stats

            if (
                not self.frozen[i]
                and layer_stats["removed"] > 0
                and layer_stats["overlap"] >= self.early_stop
            ):
                self.frozen[i] = True

        return stats

    def reset_opt_state(self, optimizer, layer, changed):
        #Adam momentum must be zeroed before a weight is reintroduced, so it does not
        #inherit its previous optimizer state
        state = optimizer.state.get(layer.weight)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
            if key in state and state[key] is not None:
                state[key][changed] = 0.0

    def anp(self):
        return [
            topology.active_neuron_rate(self.mask1.T, self.mask2.T),
            topology.active_neuron_rate(self.mask2.T, None),
        ]


class MNIST(nn.Module):
    """The dense 784-1000-1000-10 control network, selected by `model: Baseline`.

    Same depth and width as the sparse model, no masks and no evolution - the
    reference point for what the architecture reaches at 100% density.
    """

    def __init__(self, hidden=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)
