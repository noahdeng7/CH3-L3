import numpy as np
import torch


def find_hanging(mask, direction):
    if direction == "forward":
        mask = mask.T
    hanging_neurons = np.where(~mask.any(axis=1))[0]
    return hanging_neurons


def remove_hanging(m1, m2):
    before = (m1.sum(), m2.sum())

    for _ in range(2):
        # forward: H1 neurons with no incoming link cannot pass anything on
        m2[find_hanging(m1, "forward")] = 0
        # backward: H1 neurons with no outgoing link receive nothing useful
        m1[:, find_hanging(m2, "backward")] = 0

    return m1, m2, int(before[0] - m1.sum()), int(before[1] - m2.sum())


def percolate(mask1, mask2):
    n1, n2, r1, r2 = remove_hanging(
        mask1.cpu().numpy().T.copy(),
        mask2.cpu().numpy().T.copy(),
    )
    mask1.copy_(torch.from_numpy(n1.T).to(mask1))
    mask2.copy_(torch.from_numpy(n2.T).to(mask2))
    return [r1, r2]


def CH3_L3_scores(mask):
    #naive implementation of CH3-L3. is only used as verification for 
    #the faster implementation
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.float64)
    n, m = mask.shape
    scores = np.zeros((n, m))
    d_col = mask.sum(axis=0)
    d_row = mask.sum(axis=1)

    for u in range(n):
        z1_candidates = np.where(mask[u] != 0)[0]
        for v in range(m):
            if mask[u, v] != 0:
                continue
            z2_candidates = np.where(mask[:, v] != 0)[0]
            score = 0.0
            community_rows = set(z2_candidates) | {u}
            community_cols = set(z1_candidates) | {v}
            for z1 in z1_candidates:
                for z2 in z2_candidates:
                    if mask[z2, z1] != 0:
                        de_z1 = d_col[z1] - np.sum(mask[list(community_rows), z1])
                        de_z2 = d_row[z2] - np.sum(mask[z2, list(community_cols)])
                        score += 1.0 / np.sqrt((1 + de_z1) * (1 + de_z2))
            scores[u, v] = score

    return scores


def CH3_L3_scores_exact(mask, device=None):
    #exact CH3-L3, vectorize on GPU for massive performance increase
    A, device = _as_tensor(mask, device)
    n, m = A.shape

    d_row = A.sum(dim=1)
    d_col = A.sum(dim=0)
    C = A.T @ A
    G = (d_row[:, None] - A @ A.T).clamp_(min=1.0).rsqrt_()

    scores = torch.zeros((n, m), dtype=A.dtype, device=device)
    for v in range(m):
        Nv = torch.nonzero(A[:, v], as_tuple=True)[0]
        if Nv.numel() == 0:
            continue
        f_v = (d_col - C[:, v]).clamp_(min=1.0).rsqrt_()
        P = A @ (A[Nv, :] * f_v).T
        scores[:, v] = (P * G[Nv, :].T).sum(dim=1)

    scores[A != 0] = 0.0
    return scores


def RA_L3_scores(mask, device=None):
    #modification of CH3-L3 where we use the total node degree instead of 
    #only external connections. basically useless.
    A, _ = _as_tensor(mask, device)

    w_col = (A.sum(dim=0) + 1.0).rsqrt()
    w_row = (A.sum(dim=1) + 1.0).rsqrt()

    mid = (A * w_col) @ A.T
    mid *= w_row
    scores = mid @ A

    scores[A != 0] = 0.0
    return scores


def random_scores(mask, device=None):
    A, device = _as_tensor(mask, device)
    scores = torch.rand(A.shape, dtype=A.dtype, device=device)
    scores[A != 0] = 0.0
    return scores


SCORERS = {"ch3": CH3_L3_scores_exact, "ra": RA_L3_scores, "random": random_scores}


def _as_tensor(mask, device):
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.asarray(mask))
    device = device or mask.device
    return mask.to(device=device, dtype=torch.float32), device


def prune_smallest(weight, mask, zeta):
    #removes zeta fraction of links with smallest weights
    #not the same as sparse pruning, as the evolution stage is different,
    #and zeta is much higher than the actual sparsity rate

    active = torch.nonzero(mask.view(-1)).squeeze(1)
    k = int(round(zeta * active.numel()))
    if k <= 0:
        return 0

    w = weight.abs().view(-1)[active]
    mask.view(-1)[active[torch.topk(w, k, largest=False).indices]] = 0.0
    return k


def regrow(mask, budget, scorer):
    """Add `budget` absent links, the highest scoring ones under `scorer`.

    Mutates mask in place. This is the only point at which the ESML, RA and SET
    arms differ - everything else about a run is identical.
    """
    if budget <= 0:
        return 0
    scores = scorer(mask, device=mask.device)
    scores[mask != 0] = -float("inf")
    mask.view(-1)[torch.topk(scores.flatten(), budget).indices] = 1.0
    return budget


def rewire_stats(before, mid, after, budget):
    """Diagnostics for one evolution round on one layer.

    Takes the mask at the three points of the round: `before` it, after
    prune+percolate (`mid`), and `after` regrowth.

    `changed` is `removed | added`, not `before != after`. A link that is pruned
    and immediately regrown is identical at the endpoints but still has to
    restart from zero weight and zero optimizer state - the endpoint comparison
    would miss exactly the links the reset exists for.
    """
    removed = (before != 0) & (mid == 0)
    added = (after != 0) & (mid == 0)
    changed = removed | added

    n_before = int(before.sum().item())
    n_removed = int(removed.sum().item())
    # the fraction of this round's removals that regrowth immediately put back
    n_readded = int((removed & (after != 0)).sum().item())

    # masks are (out, in): a column is one source neuron's out-degree, a row is
    #one target neuron's in-degree
    stats = {
        "links": int(after.sum().item()),
        "removed": n_removed,
        # the budget, not the links actually added - the two differ only on a
        #frozen layer, which loses links to percolation but does not regrow them
        "regrown": budget,
        "turnover": n_removed / n_before if n_before else 0.0,
        "overlap": n_readded / n_removed if n_removed else 0.0,
        "gamma_in": power_law_gamma(after.sum(dim=1)),
        "gamma_out": power_law_gamma(after.sum(dim=0)),
    }
    return changed, stats


def active_neuron_rate(mask_in, mask_out=None):
    """ANP - the fraction of neurons carrying both an incoming and outgoing link.

    Both masks are in (in, out) orientation, so the layer being measured is the
    columns of `mask_in` and the rows of `mask_out`. `None` for either side means
    that side is unconstrained - the last hidden layer feeds the dense head and
    is therefore always connected downstream.
    """
    alive = torch.ones(mask_in.shape[1], dtype=torch.bool, device=mask_in.device)
    alive &= mask_in.sum(dim=0) > 0
    if mask_out is not None:
        alive &= mask_out.sum(dim=1) > 0
    return alive.float().mean().item()


def power_law_gamma(degrees, cap=10.0, min_tail=10):
    """Clauset-Shalizi-Newman power-law exponent of a degree sequence.

    k_min is chosen by minimising the KS distance between the empirical CDF and
    the fitted Pareto CDF; the sweep is not optional. With k_min pinned at 1 the
    MLE degenerates into a function of the mean log-degree, and a perfectly
    homogeneous layer (every node at degree 10) scores gamma = 1.43 - reading as
    *more* scale-free than a genuine Pareto sample. It is the sweep that lets a
    random topology saturate at the cap, reproducing the paper's gamma = 10 for
    random against ~2 for ESML.
    """
    if isinstance(degrees, torch.Tensor):
        degrees = degrees.detach().cpu().numpy()
    d = np.asarray(degrees, dtype=np.float64).ravel()
    d = d[d > 0]
    if d.size < min_tail:
        return cap

    d_sorted = np.sort(d)
    best_alpha, best_ks = cap, np.inf

    for k_min in np.unique(d_sorted):
        tail = d_sorted[d_sorted >= k_min]
        # np.unique is ascending, so every later tail is strictly smaller
        if tail.size < min_tail:
            break

        # (k_min - 0.5) is the standard discrete correction
        scale = k_min - 0.5
        s = np.log(tail / scale).sum()
        if s <= 0:
            continue
        alpha = 1.0 + tail.size / s

        empirical = np.arange(1, tail.size + 1) / tail.size
        fitted = 1.0 - (tail / scale) ** (1.0 - alpha)
        ks = np.abs(empirical - fitted).max()

        if ks < best_ks:
            best_alpha, best_ks = alpha, ks

    return float(min(best_alpha, cap))
