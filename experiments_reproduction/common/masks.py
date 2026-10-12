"""Sparse topological initialization for any stack of sandwich layers.

utils.masks covers CSTI and Erdos-Renyi for the 3-layer MLP. The ICLR 2024
paper also initializes with the bipartite versions of Watts-Strogatz and
Barabasi-Albert (Appendix M), and the CNN experiments put the sparse layers on
top of a convolutional backbone, where the input width is the backbone's
feature dimension rather than a dataset's pixel count. `layer_masks` handles
both.

Every mask is returned as an int8 (in, out) array, the orientation
utils.model_defs.CHT expects.
"""

import numpy as np
import torch

from utils.masks import generate_csti, generate_erdos_renyi

INITS = ("csti", "er", "ws", "ba")


def degree_sequence(n, total, rng):
    """n integer degrees summing to `total`, as near-equal as possible.

    Appendix M: "Given a certain m, approximate its value to the first decimal
    ... 10-FD nodes have degree [m] and FD nodes have degree [m]+1." Spreading
    the remainder over randomly chosen nodes is the same rule without the
    rounding to a tenth, and hits the link budget exactly.
    """
    base, extra = divmod(int(total), n)
    degrees = np.full(n, base, dtype=np.int64)
    degrees[rng.choice(n, size=extra, replace=False)] += 1
    return degrees


def bipartite_small_world(n_in, n_out, density, beta, seed):
    """Bipartite Watts-Strogatz (BSW), Appendix M.

    The two layers are interleaved on one ring in proportion to their sizes,
    each output node is linked to its nearest input nodes on the ring, then
    each link is rewired with probability beta to an input node chosen
    uniformly at random (no duplicates). beta=0 is the regular lattice and
    beta=1 is Erdos-Renyi.
    """
    rng = np.random.default_rng(seed)
    total = round(n_in * n_out * density)
    degrees = np.minimum(degree_sequence(n_out, total, rng), n_in)
    mask = np.zeros((n_in, n_out), dtype=np.int8)
    everyone = np.arange(n_in)

    for j in range(n_out):
        k = int(degrees[j])
        if k == 0:
            continue
        centre = (j + 0.5) * n_in / n_out
        start = int(np.floor(centre - k / 2.0))
        neighbours = np.arange(start, start + k) % n_in

        rewire = rng.random(k) < beta
        if rewire.any():
            keep = neighbours[~rewire]
            free = np.setdiff1d(everyone, keep, assume_unique=False)
            new = rng.choice(free, size=int(rewire.sum()), replace=False)
            neighbours = np.concatenate([keep, new])
        mask[neighbours, j] = 1
    return mask


def _barabasi_albert(n, m_seq, rng):
    """Monopartite BA graph where arriving node t attaches m_seq[t] links."""
    m0 = int(max(m_seq.max(), 1)) + 1
    edges = [(i, i + 1) for i in range(m0 - 1)]         # a seed path
    # every endpoint of every edge: sampling from it is preferential attachment
    ends = [v for e in edges for v in e]
    for t in range(m0, n):
        m = int(min(m_seq[t], t))
        targets = set()
        while len(targets) < m:
            targets.add(ends[rng.integers(len(ends))])
        for v in targets:
            edges.append((t, v))
            ends.extend((t, v))
    return np.asarray(edges, dtype=np.int64)


def bipartite_scale_free(n_in, n_out, density, seed):
    """Bipartite Barabasi-Albert (BSF), Appendix M.

    1. a BA graph on n_in + n_out nodes with the link budget of the layer;
    2. nodes randomly split into the two layers;
    3. 'frustrations' (links inside one layer) are paired across the layers and
       each pair resolved by one Maslov-Sneppen swap, (a,b),(c,d) -> (a,c),(b,d).
       The unpaired surplus in one layer is resolved by keeping one endpoint
       and re-attaching the other end in the opposite layer by preferential
       attachment. (The appendix says both nodes are rewired; doing that would
       grow the link count, so here one end moves and the budget is kept.)
    The result keeps BA's power-law degree distribution.
    """
    rng = np.random.default_rng(seed)
    n = n_in + n_out
    total = round(n_in * n_out * density)
    edges = _barabasi_albert(n, degree_sequence(n, total, rng), rng)

    perm = rng.permutation(n)
    side = np.zeros(n, dtype=np.int64)            # 0 = input layer, 1 = output layer
    side[perm[n_in:]] = 1
    index = np.empty(n, dtype=np.int64)            # position inside its layer
    index[perm[:n_in]] = np.arange(n_in)
    index[perm[n_in:]] = np.arange(n_out)

    s = side[edges]
    cross = edges[s[:, 0] != s[:, 1]]
    f1 = edges[(s[:, 0] == 0) & (s[:, 1] == 0)]
    f2 = edges[(s[:, 0] == 1) & (s[:, 1] == 1)]
    rng.shuffle(f1)
    rng.shuffle(f2)

    pairs = min(len(f1), len(f2))
    swapped = []
    for (a, b), (c, d) in zip(f1[:pairs], f2[:pairs]):
        swapped += [(a, c), (b, d)]
    surplus = np.concatenate([f1[pairs:], f2[pairs:]])

    mask = np.zeros((n_in, n_out), dtype=np.int8)

    def put(u, v):
        # u, v are graph nodes on opposite sides
        if side[u] == 1:
            u, v = v, u
        mask[index[u], index[v]] = 1

    for u, v in cross:
        put(u, v)
    for u, v in swapped:
        put(u, v)

    deg_in = mask.sum(axis=1).astype(np.float64)
    deg_out = mask.sum(axis=0).astype(np.float64)
    for u, v in surplus:
        keep = u if rng.random() < 0.5 else v
        if side[keep] == 0:
            p = (deg_out + 1.0) / (deg_out + 1.0).sum()
            j = rng.choice(n_out, p=p)
            i = index[keep]
        else:
            p = (deg_in + 1.0) / (deg_in + 1.0).sum()
            i = rng.choice(n_in, p=p)
            j = index[keep]
        if not mask[i, j]:
            mask[i, j] = 1
            deg_in[i] += 1
            deg_out[j] += 1

    # duplicates collapsed by the swaps leave the count a little short (or the
    #surplus pass a little long); top up / trim to the exact budget, again by
    #preferential attachment so the degree distribution is not flattened
    current = int(mask.sum())
    if current > total:
        on = np.flatnonzero(mask)
        mask.flat[rng.choice(on, size=current - total, replace=False)] = 0
    elif current < total:
        deg_in = mask.sum(axis=1).astype(np.float64) + 1.0
        deg_out = mask.sum(axis=0).astype(np.float64) + 1.0
        weight = np.outer(deg_in, deg_out) * (mask == 0)
        weight = weight.ravel() / weight.sum()
        mask.flat[rng.choice(weight.size, size=total - current, replace=False, p=weight)] = 1
    return mask


def layer_masks(dims, density, init="er", seed=0, beta=1.0, dataset=None, root="data"):
    """Masks for the sandwich layers dims[0] -> dims[1] -> ... -> dims[-1].

    `csti` only makes sense for a layer that reads the input pixels, so, as in
    utils.masks.build_masks, it initializes the first layer from `dataset` and
    Erdos-Renyi the rest. `ws` and `ba` apply to every layer.
    """
    masks = []
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
        layer_seed = seed + i
        if init == "csti" and i == 0:
            if dataset is None:
                raise ValueError("CSTI needs the dataset whose pixels the first layer reads")
            m = generate_csti(dataset, n_out, density, root=root)
        elif init in ("csti", "er"):
            m = generate_erdos_renyi(n_in, n_out, density, layer_seed)
        elif init == "ws":
            m = bipartite_small_world(n_in, n_out, density, beta, layer_seed)
        elif init == "ba":
            m = bipartite_scale_free(n_in, n_out, density, layer_seed)
        else:
            raise ValueError(f"unknown init {init!r}; pick one of {INITS}")
        assert m.sum() > 0, f"layer {i + 1} has no links"
        masks.append(torch.from_numpy(np.ascontiguousarray(m)))
    return masks
