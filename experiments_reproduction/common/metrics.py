"""Learning-speed and network-science measures used by the ICLR 2024 paper.

  aae                      Area Across the Epochs, Appendix L Eq 12
  network_adjacency        the whole sparse MLP (all sandwich layers) as one
                           undirected graph, percolated nodes dropped
  characteristic_path_length, modularity, structural_consistency
                           the Figure 2B measures (Appendix C glossary)
  credit_assigned_paths    Appendix O: input-to-output paths, and the share
                           passing through the intermediate-layer hubs
  coalescent_embedding     Figure 2A's 2D hyperbolic embedding

The power-law exponent is utils.topology.power_law_gamma and the ANP rate is
utils.topology.active_neuron_rate. Everything here is torch so it runs on the
GPU without scipy or networkx.
"""

import math

import numpy as np
import torch

from utils.topology import power_law_gamma


def aae(accs):
    """Eq 12: the integral of accuracy over the epochs divided by (E - 1)."""
    a = np.asarray(accs, dtype=np.float64)
    return float(np.trapezoid(a) / (len(a) - 1))


def network_adjacency(masks, device="cuda"):
    """(A, layer) for the network made of the sandwich layers' masks.

    Masks are (out, in), layer by layer. Node order is input layer, hidden 1,
    ..., last hidden; `layer` gives each node's layer index. Nodes without any
    link (percolated) are removed, as the paper measures the network that is
    left.
    """
    sizes = [masks[0].shape[1]] + [m.shape[0] for m in masks]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n = int(offsets[-1])
    A = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i, m in enumerate(masks):
        rows = slice(offsets[i + 1], offsets[i + 2])
        cols = slice(offsets[i], offsets[i + 1])
        A[rows, cols] = m.to(device=device, dtype=torch.float32)
    A = torch.maximum(A, A.T)
    layer = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes)).to(device)
    keep = A.sum(dim=1) > 0
    return A[keep][:, keep], layer[keep]


def hop_distances(A):
    """All-pairs unweighted shortest-path lengths by breadth-first matrix products.

    Returns an (n, n) float tensor with inf for unreachable pairs.
    """
    n = A.shape[0]
    dist = torch.full((n, n), float("inf"), device=A.device)
    visited = torch.eye(n, dtype=torch.bool, device=A.device)
    dist[visited] = 0.0
    frontier = visited.float()
    d = 0
    while True:
        d += 1
        new = ((frontier @ A) > 0) & ~visited
        if not new.any():
            break
        dist[new] = float(d)
        visited |= new
        frontier = new.float()
    return dist


def characteristic_path_length(A):
    """Mean shortest-path length over the connected node pairs."""
    dist = hop_distances(A)
    finite = torch.isfinite(dist) & (dist > 0)
    return float(dist[finite].mean().item())


def largest_component(A):
    dist = hop_distances(A)
    reach = torch.isfinite(dist)
    sizes = reach.sum(dim=1)
    return reach[int(torch.argmax(sizes))]


def modularity(A, labels):
    """Newman's Q for a partition."""
    k = A.sum(dim=1)
    m2 = k.sum()
    same = labels[:, None] == labels[None, :]
    return float(((A - torch.outer(k, k) / m2) * same).sum().item() / m2.item())


def leading_eigenvector_communities(A, min_gain=1e-8):
    """Newman (2006)'s leading-eigenvector method, splitting recursively.

    The paper cites Newman 2006 for modularity; this is that paper's community
    detection, without the Kernighan-Lin refinement step.
    """
    A = A.double()
    k = A.sum(dim=1)
    m2 = k.sum()
    labels = torch.zeros(A.shape[0], dtype=torch.long, device=A.device)
    todo = [torch.arange(A.shape[0], device=A.device)]
    next_label = 1
    while todo:
        g = todo.pop()
        if g.numel() < 2:
            continue
        kg = k[g]
        Bg = A[g][:, g] - torch.outer(kg, kg) / m2
        Bg = Bg - torch.diag(Bg.sum(dim=1))
        vals, vecs = torch.linalg.eigh(Bg)
        if vals[-1] <= min_gain:
            continue
        s = torch.where(vecs[:, -1] >= 0, 1.0, -1.0).double()
        gain = float((s @ Bg @ s).item()) / (2 * m2.item())
        if gain <= min_gain:
            continue
        left, right = g[s > 0], g[s < 0]
        if left.numel() == 0 or right.numel() == 0:
            continue
        labels[right] = next_label
        next_label += 1
        todo += [left, right]
    return labels


def structural_consistency(A, fraction=0.1, trials=10, seed=0):
    """Lu et al. (2015)'s structural consistency via the structural perturbation method.

    Remove a random `fraction` of the links (the perturbation set), perturb
    the eigenvalues of the rest by first-order theory, rebuild the adjacency
    matrix from them, and count how many of the removed links land in the top
    |removed| of the non-observed pairs. Degenerate eigenvalues are treated as
    distinct (the method's special handling of them is omitted).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = A.double()
    n = A.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=A.device)
    upper = A[iu[0], iu[1]]
    edges = torch.nonzero(upper).squeeze(1)
    scores = []
    for _ in range(trials):
        perm = torch.randperm(edges.numel(), generator=gen).to(A.device)
        probe = edges[perm[: max(1, int(fraction * edges.numel()))]]
        dA = torch.zeros_like(A)
        dA[iu[0][probe], iu[1][probe]] = 1.0
        dA = dA + dA.T
        AR = A - dA
        vals, vecs = torch.linalg.eigh(AR)
        dvals = ((vecs * (dA @ vecs)).sum(dim=0))
        tilde = (vecs * (vals + dvals)) @ vecs.T
        cand = tilde[iu[0], iu[1]].masked_fill(AR[iu[0], iu[1]] != 0, float("-inf"))
        top = torch.topk(cand, probe.numel()).indices
        hit = torch.zeros(cand.numel(), dtype=torch.bool, device=A.device)
        hit[probe] = True
        scores.append(float(hit[top].float().mean().item()))
    return float(np.mean(scores))


def credit_assigned_paths(masks, hubs=80):
    """Appendix O: number of input->output paths, and the share through hubs.

    Masks are (out, in). A path picks one link per sandwich layer, so the count
    is 1^T M_L ... M_1 1. The hubs are the `hubs` highest-degree nodes of the
    intermediate layers (layers 2 and 3 of the node stack in the paper's
    784-1000-1000-1000 network); the hub share is the fraction of paths that
    visit at least one of them.
    """
    ms = [m.double() for m in masks]

    def count(ms):
        v = torch.ones(ms[0].shape[1], dtype=torch.float64, device=ms[0].device)
        for m in ms:
            v = m @ v
        return float(v.sum().item())

    total = count(ms)
    # intermediate node layers are the outputs of every mask but the last
    degrees = []
    for i in range(len(ms) - 1):
        degrees.append(ms[i].sum(dim=1) + ms[i + 1].sum(dim=0))
    flat = torch.cat(degrees)
    top = torch.topk(flat, min(hubs, flat.numel())).indices
    blocked = [m.clone() for m in ms]
    start = 0
    for i, d in enumerate(degrees):
        local = top[(top >= start) & (top < start + d.numel())] - start
        blocked[i][local, :] = 0.0     # no path enters a hub
        start += d.numel()
    avoiding = count(blocked)
    return {"caps": total, "hub_caps": total - avoiding,
            "hub_ratio": (total - avoiding) / total if total else 0.0}


def coalescent_embedding(A, gamma=None):
    """2D hyperbolic coalescent embedding (Muscoloni et al. 2017), Figure 2A.

    RA1 pre-weighting, Isomap (weighted shortest paths + classical MDS) to two
    dimensions, equidistant adjustment of the angles, and the popularity-
    similarity radial coordinate r_i = 2(beta ln i + (1 - beta) ln N) of the
    node with the i-th largest degree, beta = 1/(gamma - 1). Only the largest
    connected component is embedded. Returns (radius, angle, node index).
    """
    comp = largest_component(A)
    nodes = torch.nonzero(comp).squeeze(1)
    A = A[nodes][:, nodes].double()
    n = A.shape[0]
    deg = A.sum(dim=1)
    cn = A @ A
    W = A * (deg[:, None] + deg[None, :] + deg[:, None] * deg[None, :]) / (1.0 + cn)

    D = torch.where(A > 0, W, torch.full_like(W, float("inf")))
    D.fill_diagonal_(0.0)
    for k in range(n):
        D = torch.minimum(D, D[:, k : k + 1] + D[k : k + 1, :])

    J = torch.eye(n, dtype=D.dtype, device=D.device) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = torch.linalg.eigh(B)
    xy = vecs[:, -2:] * vals[-2:].clamp_min(0).sqrt()
    theta = torch.atan2(xy[:, 1], xy[:, 0])
    # equidistant adjustment: keep the angular order, space the nodes evenly
    order = torch.argsort(theta)
    theta = torch.empty_like(theta)
    theta[order] = torch.arange(n, dtype=D.dtype, device=D.device) * (2 * math.pi / n)

    if gamma is None:
        gamma = power_law_gamma(deg)
    beta = 1.0 / max(gamma - 1.0, 1e-6)
    rank = torch.empty(n, dtype=D.dtype, device=D.device)
    rank[torch.argsort(deg, descending=True)] = torch.arange(1, n + 1, dtype=D.dtype, device=D.device)
    radius = 2.0 * (beta * torch.log(rank) + (1.0 - beta) * math.log(n))
    return radius.cpu(), theta.cpu(), nodes.cpu()


def network_measures(masks, device="cuda", sc_trials=5):
    """The four Figure 2B measures of one network, plus gamma and ANP."""
    from utils.topology import active_neuron_rate

    A, _ = network_adjacency(masks, device=device)
    labels = leading_eigenvector_communities(A)
    return {
        "gamma": power_law_gamma(A.sum(dim=1)),
        "structural_consistency": structural_consistency(A, trials=sc_trials),
        "modularity": modularity(A.double(), labels),
        "char_path_length": characteristic_path_length(A),
        "anp": active_neuron_rate([m.float() for m in masks]),
        "nodes": int(A.shape[0]),
        "links": int(A.sum().item() // 2),
    }
