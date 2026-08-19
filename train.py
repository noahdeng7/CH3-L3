import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.datasets import MNIST

from model_defs import MNIST as MNIST_model
from model_defs import sparse_MNIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# both MNIST splits are on the device the whole time so the sweep
#reuses them across runs instead of re-reading the tensors ten times
_SPLITS = {}


def load_split(train, device=DEVICE):
    #load everything to device
    #MNIST is tiny so we dont need dataloaders
    key = (train, device)
    if key not in _SPLITS:
        ds = MNIST(root=".", download=True, train=train)
        x = ds.data.unsqueeze(1).float().div_(255.0)
        _SPLITS[key] = (x.to(device), ds.targets.to(device))
    return _SPLITS[key]


def build_model(cfg, mask_path="masks.npz", device=DEVICE):
    if cfg["model"] == "Baseline":
        return MNIST_model(cfg["hidden"]).to(device)
    return sparse_MNIST(
        mask_path, scorer=cfg["scorer"], early_stop=cfg["early_stop"]
    ).to(device)


def run(cfg, mask_path="masks.npz", wandb_run=None, device=DEVICE, verbose=True):
    """Train one arm for cfg['epochs'] and return the per-epoch history.

    `wandb_run` is an already-initialised run, or None to skip logging entirely -
    the sweep drives ten runs and records them locally instead.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    batch_size = cfg["batch_size"]
    train_x, train_y = load_split(train=True, device=device)
    test_x, test_y = load_split(train=False, device=device)
    n_train, n_test = train_y.size(0), test_y.size(0)

    sparse = cfg["model"] == "Sparse"
    model = build_model(cfg, mask_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    train_step = 0
    accuracy_history = []
    history = {}

    for epoch in range(cfg["epochs"]):

        stats = {}
        if sparse and epoch > 0:
            stats = model.evolve(cfg["zeta"], optimizer)

        model.train()
        train_correct = train_seen = 0

        #the epoch order is drawn on the CPU generator so that it depends only on
        #the seen chosen and not on whether the run landed on GPU or CPU
        perm = torch.randperm(n_train).to(device)

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            images, labels = train_x[idx], train_y[idx]

            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_correct += (output.argmax(dim=1) == labels).sum().item()
            train_seen += labels.size(0)
            if wandb_run is not None:
                wandb_run.log({"train_loss": loss.item(), "train_step": train_step})
            train_step += 1

        model.eval()
        test_correct = test_seen = 0
        test_loss_sum = 0.0

        with torch.no_grad():
            for start in range(0, n_test, batch_size):
                images = test_x[start : start + batch_size]
                labels = test_y[start : start + batch_size]
                output = model(images)
                test_loss_sum += loss_fn(output, labels).item() * labels.size(0)
                test_correct += (output.argmax(dim=1) == labels).sum().item()
                test_seen += labels.size(0)

        test_accuracy = test_correct / test_seen
        accuracy_history.append(test_accuracy)

        log = {
            "epoch": epoch,
            "train_accuracy_epoch": train_correct / train_seen,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss_sum / test_seen,
            # area across the epochs (Appendix L): the running mean of test
            #accuracy, the paper's learning-*speed* metric
            "aae": float(np.mean(accuracy_history)),
        }

        if sparse:
            #measured after the epoch, so no metric is read off a topology whose
            #newest links have had zero gradient steps
            anp1, anp2 = model.anp()
            log |= {"anp1": anp1, "anp2": anp2}
            for layer, layer_stats in stats.items():
                for k, v in layer_stats.items():
                    if k != "frozen":
                        log[f"{layer}/{k}"] = v

        if wandb_run is not None:
            wandb_run.log(log)
        for k, v in log.items():
            # the per-layer stats only exist from epoch 1 (no evolution at epoch
            #0), so pad with None to keep every series indexed by epoch
            history.setdefault(k, [None] * epoch).append(v)

        if verbose:
            print(
                f"epoch {epoch:3d}  test_acc {test_accuracy:.4f}  "
                + (f"anp {log['anp1']:.3f}/{log['anp2']:.3f}" if sparse else "")
            )

    return history


def main():
    import wandb

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.login()
    wandb_run = wandb.init(project=cfg["project"], name=cfg["run_name"], config=cfg)

    wandb.define_metric("train_loss", step_metric="train_step")
    for metric in ("test_accuracy", "test_loss", "train_accuracy_epoch", "aae", "anp1", "anp2"):
        wandb.define_metric(metric, step_metric="epoch")
    for layer in ("layer1", "layer2"):
        for metric in ("links", "turnover", "overlap", "percolated", "gamma_in", "gamma_out"):
            wandb.define_metric(f"{layer}/{metric}", step_metric="epoch")

    run(cfg, wandb_run=wandb_run)
    wandb.finish()

if __name__ == "__main__":
    main()
