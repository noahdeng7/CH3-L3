from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import numpy as np
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)


def generate_corrs(n_out, density):
    #CSTI correlation initialization as described in the paper
    trainset = MNIST(root=".", download=True, train=True, transform=ToTensor())
    X = trainset.data.view(60000, -1).float()

    std = X.std(axis=0)
    valid = std > 1e-8
    corrs = np.full((784, 784), np.nan)
    corrs[np.ix_(valid, valid)] = np.corrcoef(X[:, valid], rowvar=False)
    np.fill_diagonal(corrs, np.nan)

    # NaN entries (diagonal + always-zero pixels) can never carry a link, so the
    #threshold is taken over the valid pairs only but the budget k is a fraction
    #of the full matrix, to match the density of the Erdos-Renyi masks.
    k = round(784 * 784 * density)
    finite = ~np.isnan(corrs)
    threshold = np.nanpercentile(corrs, 100.0 * (1.0 - k / finite.sum()))
    block = (finite & (corrs >= threshold)).astype(np.int8)

    reps = -(-n_out // 784)  # ceil
    return np.tile(block, (1, reps))[:, :n_out]


def generate_erdos_renyi(n, m, density, seed=42):
    rng = np.random.default_rng(seed)
    total = n * m
    k = round(total * density)
    idx = rng.choice(total, size=k, replace=False)
    mask = np.zeros(total, dtype=np.int8)
    mask[idx] = 1
    return mask.reshape(n, m)


if __name__ == "__main__":
    density = 1.0 - cfg["sparsity"] / 100
    hidden = cfg["hidden"]
    seed = cfg["seed"]

    if cfg["init"] == "csti":
        mask1 = generate_corrs(hidden, density)
    else:
        mask1 = generate_erdos_renyi(784, hidden, density, seed)
    mask2 = generate_erdos_renyi(hidden, hidden, density, seed + 1)

    for name, m in (("mask1", mask1), ("mask2", mask2)):
        assert m.sum() > 0, f"{name} has no links"
        print(f"{name}: {m.shape} {int(m.sum())} links, density {m.mean():.4f}")

    np.savez("masks.npz", mask1=mask1, mask2=mask2)
    print(f"masks written: {cfg['init']} init, {cfg['sparsity']}% sparsity")
