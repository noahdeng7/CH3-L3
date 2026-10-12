"""The two schedules a CHTss run is driven by: density decay and learning rate.

Section 3.3 of the CHTs paper adds a gradual density decay on top of CHTs, so
the network starts at a sparsity of 50% and is pruned down to its target 99%
while the rewiring runs. The decay is the only difference between CHTs and
CHTss, which is why it lives behind one object that the CHTs arm just leaves
switched off.

Both schedules are indexed by *round*, where one round is one topological update
(`update_interval` of 1 epoch, Table 4), counting from the round at the end of
the first epoch. The reference indexes them by optimizer step and divides by the
steps per epoch, which comes to the same thing at an interval of one epoch.
"""

from . import topology

# Table 4's LR decays over `DECAY_FACTOR` of the run and is held at the end
# value for the remainder: LinearLR(total_iters=int(epochs * decay_factor)) with
# run.py's default --decay_factor 0.9. So CIFAR10's 0.025 reaches 2.5e-4 at
# epoch 90 and stays there for the last ten.
DECAY_FACTOR = 0.9

# Density decay ends at three quarters of training, leaving the last quarter to
# rewire at the final sparsity. The paper does not state the window for the MLP;
# this is the reference default (`pruning_T_end = 0.75 * num_training_steps`,
# and the same 0.75 the MLP code uses for EM&S).
DECAY_END = 0.75

SHAPES = {
    "sigmoid": topology.sigmoid_density_decay,  # Eq 4, CHTss
    "cubic": topology.cubic_density_decay,      # Eq 9, GMP / GraNet
}


class DensityDecay:
    """Sparsity per round, from `start_sparsity` down to `final_sparsity`.

    Rounds after the decay window return the final sparsity, and `prunes_at`
    reports whether a round is still inside the window - past it the arm is
    plain fixed-sparsity CHTs and the prune stage is skipped entirely rather
    than re-pruning to a density the network is already at.
    """

    def __init__(
        self,
        start_sparsity,
        final_sparsity,
        total_rounds,
        shape="sigmoid",
        k=6.0,
        start_round=0,
        end_fraction=DECAY_END,
    ):
        if shape not in SHAPES:
            raise ValueError(f"unknown decay shape {shape!r}; pick one of {sorted(SHAPES)}")
        self.start_sparsity = start_sparsity
        self.final_sparsity = final_sparsity
        self.shape = shape
        self.k = k
        self.t0 = start_round
        self.tf = max(round(total_rounds * end_fraction), start_round + 1)

    def sparsity(self, round_idx):
        if self.shape == "sigmoid":
            return SHAPES[self.shape](
                round_idx, self.start_sparsity, self.final_sparsity, self.t0, self.tf, self.k
            )
        return SHAPES[self.shape](
            round_idx, self.start_sparsity, self.final_sparsity, self.t0, self.tf
        )

    def density(self, round_idx):
        return 1.0 - self.sparsity(round_idx)

    def prunes_at(self, round_idx):
        return self.t0 < round_idx <= self.tf

    def __repr__(self):
        return (
            f"DensityDecay({self.shape}, {self.start_sparsity:.2f} -> "
            f"{self.final_sparsity:.2f} over rounds {self.t0}..{self.tf}, k={self.k})"
        )


def linear_lr(epoch, start, end, epochs, decay_factor=DECAY_FACTOR):
    """Table 4's linear LR decay, as torch's LinearLR applies it per epoch."""
    total = max(int(epochs * decay_factor), 1)
    return start + (end - start) * min(epoch, total) / total
