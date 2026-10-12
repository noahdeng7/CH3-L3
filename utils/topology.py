import math

import numpy as np
import torch


def percolate(masks):
    """Chain removal, as the reference implements it.

    One backward sweep from the last sandwich layer to the first, zeroing the
    incoming links of any neuron with no outgoing link, then one forward sweep
    in the other direction. Deliberately NOT iterated to a fixed point - the
    reference does exactly one pass each way, and iterating percolates strictly
    more of the network.

    Masks are (out, in) and are mutated in place. Returns links lost per layer.
    """
    before = [int(m.sum().item()) for m in masks]

    # backward: a neuron with no outgoing link cannot receive anything useful
    for i in reversed(range(len(masks) - 1)):
        outdeg = (masks[i + 1].sum(dim=0) > 0).to(masks[i].dtype)
        masks[i].mul_(outdeg.reshape(-1, 1))

    # forward: a neuron with no incoming link cannot pass anything on
    for i in range(1, len(masks)):
        indeg = (masks[i - 1].sum(dim=1) > 0).to(masks[i].dtype)
        masks[i].mul_(indeg.reshape(1, -1))

    return [b - int(m.sum().item()) for b, m in zip(before, masks)]


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


def CH_Ln_scores(mask, ch="CH3", device=None):
    """Node-based Cannistraci-Hebb scores (CH2-L3n / CH3-L3n), ported verbatim
    from the reference `_scores_for_regrowth_ch`.

    For each side, dd2 counts a candidate common neighbour's internal local
    community links and the eLCL term is what remains of its degree. CH3 uses
    only the indicator of an iLCL in the numerator, CH2 uses the count.

    The broadcast in the eLCL line is deliberate and matches the reference:
    `dt.sum(dim=1) - dd2` aligns the degree vector to dd2's *last* axis.
    """
    A, device = _as_tensor(mask, device)
    dt = A
    td = A.transpose(1, 0)

    dd2 = dt @ td
    tt2 = td @ dt
    bdd2 = (dd2 != 0).to(A.dtype)
    btt2 = (tt2 != 0).to(A.dtype)

    elcl_dt = (dt.sum(dim=1) - dd2) * bdd2
    elcl_td = (td.sum(dim=1) - tt2) * btt2
    elcl_dt = elcl_dt.clamp_min(1) - 1
    elcl_td = elcl_td.clamp_min(1) - 1

    if ch == "CH2":
        f_dt = (dd2 + bdd2) / (elcl_dt + 1)
        f_td = (tt2 + btt2) / (elcl_td + 1)
    elif ch == "CH3":
        f_dt = bdd2 / (elcl_dt + 1)
        f_td = btt2 / (elcl_td + 1)
    else:
        raise NotImplementedError(f"unsupported CH variant {ch!r}")

    scores = (f_dt @ dt) + (f_td @ td).transpose(1, 0)
    scores[A != 0] = 0.0
    return scores


def CH2_L3n_scores(mask, device=None):
    return CH_Ln_scores(mask, ch="CH2", device=device)


def CH3_L3n_scores(mask, device=None):
    return CH_Ln_scores(mask, ch="CH3", device=device)


SCORERS = {
    # path-based (the original CHT); accurate but O(N*d^3) and unusable at
    #the densities EM_S and the sigmoid decay operate at
    "ch3_l3p": CH3_L3_scores_exact,
    # node-based; what CHTs actually regrows with
    "ch3_l3n": CH3_L3n_scores,
    "ch2_l3n": CH2_L3n_scores,
    "ra": RA_L3_scores,
    "random": random_scores,
}


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
        # the regrowth budget, which is also the number of links actually
        #added: every layer regrows exactly what prune+percolate took from it
        "regrown": budget,
        "turnover": n_removed / n_before if n_before else 0.0,
        "overlap": n_readded / n_removed if n_removed else 0.0,
        "gamma_in": power_law_gamma(after.sum(dim=1)),
        "gamma_out": power_law_gamma(after.sum(dim=0)),
    }
    return changed, stats


def active_neuron_rate(masks):
    """Active Neuron Post-percolation rate, as `run.py --record_anp` computes it.

    For every sandwich layer it counts the input-side neurons that still carry
    at least one link, then adds the final layer's output-side neurons. Two
    things about this are easy to get wrong: the test is one-sided (a neuron
    needs a link on one side, not both), and the input layer is included in the
    denominator, which puts a floor under the achievable rate.
    """
    active = 0
    total = 0
    for m in masks:
        active += int((m.sum(dim=0) > 0).sum().item())
        total += m.shape[1]
    active += int((masks[-1].sum(dim=1) > 0).sum().item())
    total += masks[-1].shape[0]
    return active / total


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


# ---------------------------------------------------------------------------
# CHTs: soft removal and soft regrowth
#
# Both replace a deterministic top-k with a multinomial sample, so that a link
# the metric currently dislikes can still survive (or be born). This is what
# lets CHTs keep exploring instead of converging into an epitopological local
# minimum, which is why CHTs drops CHT's topological early stop entirely.
# ---------------------------------------------------------------------------


def removal_score(weight, alpha, eps=1e-5):
    """Link-removal criterion. alpha=1 is weight magnitude, alpha=0 is RI (Eq 1).

    The reference implements these as two discrete branches rather than as
    Equation 2's continuous interpolation, and normalises RI by summing
    |W| + eps elementwise, which is what is reproduced here.
    """
    a = weight.abs()
    if alpha >= 1.0:
        return a
    return a / (a + eps).sum(dim=0) + a / (a + eps).sum(dim=1).reshape(-1, 1)


def removal_temperature(step, total_steps, end=3.0):
    """T = 1 + (end - 1) * step / total: the exponent runs 1 -> `end`.

    The MLP reference hard-codes end=3, equivalent to the paper's delta going
    0.5 -> 0.75 through delta/(1-delta), ramped on the optimizer step rather
    than the epoch. The newer LLM reference makes both ends arguments, and its
    sweep script runs `--start_T 1.0 --end_T 9.0`, which is the paper's
    delta 0.5 -> 0.9 stated for the main article.
    """
    return 1.0 + step * ((end - 1.0) / max(total_steps, 1))


def _multinomial_indices(weights, k):
    """Sample k distinct indices proportional to `weights`, robustly."""
    k = max(1, min(int(k), weights.numel()))
    total = weights.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return torch.randperm(weights.numel(), device=weights.device)[:k]
    return torch.multinomial(weights / total, k, replacement=False)


def prune_hard(weight, mask, n_keep, alpha):
    """Keep the top n_keep links by the removal criterion, drop the rest."""
    score = removal_score(weight, alpha).view(-1)
    keep = torch.zeros_like(score)
    keep[torch.topk(score, max(1, int(n_keep)), largest=True, sorted=False).indices] = 1.0
    mask.copy_(keep.view_as(mask))


def prune_soft(weight, mask, n_keep, alpha, temperature):
    """Sample which links to KEEP, with probability proportional to score^T.

    Note the direction: the reference samples the survivors, not the casualties.
    Sampling removals proportional to the inverse score instead is far harsher
    on small weights and collapses the network onto a few hubs.
    """
    score = removal_score(weight, alpha).view(-1).clamp_min(0.0).pow(temperature)
    keep = torch.zeros_like(score)
    keep[_multinomial_indices(score, n_keep)] = 1.0
    mask.copy_(keep.view_as(mask))


def regrow_links(mask, budget, scorer, soft):
    """Add `budget` absent links, by top-k or by multinomial sample.

    Soft regrowth samples proportional to the raw link-prediction scores, with
    no temperature applied - that is the reference's behaviour.
    """
    if budget <= 0:
        return 0
    scores = scorer(mask, device=mask.device).clamp_min(0.0)
    scores = scores * (mask == 0)
    flat = scores.view(-1)

    if soft:
        idx = _multinomial_indices(flat, budget)
    else:
        idx = torch.topk(flat, max(1, int(budget)), largest=True, sorted=False).indices

    mask.view(-1)[idx] = 1.0
    return budget


def cosine_annealed_zeta(zeta, step, total_steps):
    """--adaptive_zeta: zeta decays from its initial value to 0 as a cosine."""
    return zeta / 2.0 * (1.0 + math.cos(math.pi * step / max(total_steps, 1)))


def sigmoid_density_decay(epoch, start_sparsity, final_sparsity, t0, tf, k=6.0):
    """Equation 4: a sigmoid ramp of the sparsity from `start` to `final`.

    The paper writes the curvature as k=6 with no normalisation, which would be
    a step function at epoch granularity; the argument is taken over the decay
    window so that k=6 spans sigmoid(-6)..sigmoid(6) across it, which is the
    smooth ramp Figure 11 shows.

    The ramp is then rescaled affinely so the window's endpoints land exactly on
    `start_sparsity` and `final_sparsity`. Without it a raw sigmoid(-6) leaves
    the first round 0.25% of the range short of the initial sparsity and the
    last round the same distance short of the target, so the run would never
    actually reach 99%. This is the `s_shape` branch of the reference's
    `_current_prune_rate`.
    """
    if epoch <= t0:
        return start_sparsity
    if epoch >= tf:
        return final_sparsity

    mid = (t0 + tf) / 2.0
    half = max((tf - t0) / 2.0, 1e-12)
    sig = 1.0 / (1.0 + math.exp(-k * (epoch - mid) / half))
    edge = 1.0 / (1.0 + math.exp(-k))
    shaped = (sig - 0.5) / max(2.0 * edge - 1.0, 1e-12) + 0.5
    return start_sparsity + (final_sparsity - start_sparsity) * shaped


def cubic_density_decay(epoch, start_sparsity, final_sparsity, t0, tf):
    """Equation 9: the cubic ramp GMP and GraNet use, for the decay ablation.

    Sparsity rises steeply at the start - which is the behaviour Section 3.3
    objects to, since it prunes most of the network before it has learned
    anything - and flattens as it approaches the target.
    """
    if epoch <= t0:
        return start_sparsity
    if epoch >= tf:
        return final_sparsity

    remaining = 1.0 - (epoch - t0) / max(tf - t0, 1e-12)
    return final_sparsity + (start_sparsity - final_sparsity) * remaining**3
