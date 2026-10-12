"""Sparse topological initialization for the CHT MLP.

The input sandwich layer uses CSTI (Appendix B of the ICLR 2024 paper); the
remaining sandwich layers, which do not touch the input features, use
Erdos-Renyi. Both are produced at the same per-layer density.
"""

import numpy as np
import torch

from .data import input_dim, load_split


def generate_erdos_renyi(n, m, density, seed):
    rng = np.random.default_rng(seed)
    total = n * m
    k = round(total * density)
    idx = rng.choice(total, size=k, replace=False)
    mask = np.zeros(total, dtype=np.int8)
    mask[idx] = 1
    return mask.reshape(n, m)


def generate_csti(dataset, n_out, density, root="data", device="cuda"):
    """Correlated Sparse Topological Initialization for the input layer.

    Four steps, following Appendix B: vectorize the training images, drop the
    zero-variance features, score the remaining feature pairs by the absolute
    Pearson correlation, keep the top-k%, then tile that block out to the hidden width
    ("assembling topologically hubbed network blocks" - a 2x hidden layer is
    two copies of the same block side by side, a 1x hidden layer is one).

    The correlation is done on the GPU; CIFAR-10's 3072x3072 matrix over 50k
    samples is far too slow in numpy.
    """
    d = input_dim(dataset)
    images, _ = load_split(dataset, train=True, root=root, device=device)
    X = images.reshape(images.shape[0], -1).float()

    std = X.std(dim=0)
    valid = std > 1e-8
    Xv = X[:, valid]
    # Pearson correlation as a normalized Gram matrix of the centered columns
    Xv = Xv - Xv.mean(dim=0, keepdim=True)
    Xv = Xv / Xv.norm(dim=0, keepdim=True).clamp_min(1e-12)
    corr_valid = (Xv.T @ Xv).cpu().numpy().astype(np.float64)

    valid = valid.cpu().numpy()
    corrs = np.zeros((d, d))
    corrs[np.ix_(valid, valid)] = corr_valid
    np.fill_diagonal(corrs, 0.0)

    # as the reference's update_topology_scheduler: the always-zero pixels and
    #the diagonal are zeroed rather than excluded, pairs are ranked by
    #|correlation| so strongly anti-correlated pixels are linked too, and the
    #threshold is the (k+1)-th largest magnitude, kept with >=
    k = int(d * d * density)
    corrs = np.abs(corrs)
    flat = corrs.ravel()
    threshold = np.partition(flat, flat.size - 1 - k)[flat.size - 1 - k]
    block = (corrs >= threshold).astype(np.int8)

    reps = -(-n_out // d)  # ceil
    return np.tile(block, (1, reps))[:, :n_out]


def build_masks(dataset, hidden, density, seed, init="csti", root="data"):
    """Return the three sandwich-layer masks, each in (in, out) orientation."""
    d = input_dim(dataset)
    if init == "csti":
        mask1 = generate_csti(dataset, hidden, density, root=root)
    else:
        mask1 = generate_erdos_renyi(d, hidden, density, seed)
    mask2 = generate_erdos_renyi(hidden, hidden, density, seed + 1)
    mask3 = generate_erdos_renyi(hidden, hidden, density, seed + 2)

    masks = [mask1, mask2, mask3]
    for i, m in enumerate(masks):
        assert m.sum() > 0, f"mask{i + 1} has no links"
    return [torch.from_numpy(m) for m in masks]
