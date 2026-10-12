"""Datasets beyond the four in utils.data, and a loader factory for all of them.

MNIST, Fashion-MNIST, EMNIST and CIFAR-10 go straight through
`utils.data.load_split` and `utils.data.GPULoader`. The CNN experiments of the
ICLR 2024 paper also need CIFAR-100, Tiny-ImageNet and ImageNet:

  cifar100      torchvision download, held on the GPU like CIFAR-10
  tinyimagenet  cs231n zip, decoded once to a uint8 cache; 1.2 GB, so it is
                held in host memory and each batch is moved over before the
                same crop/flip augmentation runs on the GPU
  imagenet      too large to hold anywhere; a standard ImageFolder pipeline
                over a local copy (`--imagenet-root`), which has to be
                downloaded by hand from image-net.org

The augmentation is Appendix P.1's for every dataset: random crop and
horizontal flip, then normalization.
"""

import os
import urllib.request
import zipfile

import torch
from torchvision import datasets
from torchvision.io import ImageReadMode, decode_image

from utils.data import DATASETS, GPULoader
from utils.data import load_split as load_builtin

BUILTIN = ("mnist", "fmnist", "emnist", "cifar10")

# name -> (num_classes, mean, std, image_size, channels), as utils.data.DATASETS
EXTRA = {
    "cifar100": (100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 32, 3),
    "tinyimagenet": (200, (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821), 64, 3),
    "imagenet": (1000, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224, 3),
}
for _name, _spec in EXTRA.items():
    DATASETS.setdefault(_name, _spec)

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

_CACHE = {}


def num_classes(name):
    return DATASETS[name][0]


def load_split(name, train, root="data", device="cuda"):
    """uint8 NCHW images and int64 labels, on `device` (host for Tiny-ImageNet)."""
    if name in BUILTIN:
        return load_builtin(name, train, root=root, device=device)

    key = (name, train, device)
    if key in _CACHE:
        return _CACHE[key]
    if name == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, download=True)
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        out = (images.to(device), torch.tensor(ds.targets).to(device))
    elif name == "tinyimagenet":
        out = _tinyimagenet(root, train)
    else:
        raise ValueError(f"{name!r} has no tensor split; use make_loaders")
    _CACHE[key] = out
    return out


def _tinyimagenet(root, train):
    base = os.path.join(root, "tiny-imagenet-200")
    split = "train" if train else "val"
    cache = os.path.join(base, f"{split}_uint8.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    if not os.path.isdir(base):
        os.makedirs(root, exist_ok=True)
        archive = os.path.join(root, "tiny-imagenet-200.zip")
        if not os.path.exists(archive):
            print(f"downloading {TINY_URL}", flush=True)
            urllib.request.urlretrieve(TINY_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)

    with open(os.path.join(base, "wnids.txt")) as f:
        classes = {wnid: i for i, wnid in enumerate(line.strip() for line in f)}

    files, labels = [], []
    if train:
        for wnid, label in classes.items():
            folder = os.path.join(base, "train", wnid, "images")
            for fname in sorted(os.listdir(folder)):
                files.append(os.path.join(folder, fname))
                labels.append(label)
    else:
        # the test split has no labels; the paper's accuracies are on val
        with open(os.path.join(base, "val", "val_annotations.txt")) as f:
            for line in f:
                fname, wnid = line.split("\t")[:2]
                files.append(os.path.join(base, "val", "images", fname))
                labels.append(classes[wnid])

    images = torch.empty((len(files), 3, 64, 64), dtype=torch.uint8)
    for i, path in enumerate(files):
        with open(path, "rb") as f:
            data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
        images[i] = decode_image(data, mode=ImageReadMode.RGB)
    out = (images, torch.tensor(labels))
    torch.save(out, cache)
    return out


class HostLoader(GPULoader):
    """GPULoader over a host-resident dataset: each batch moves to the device first."""

    def __init__(self, images, labels, batch_size, train, dataset, augment=True, device="cuda"):
        super().__init__(images, labels, batch_size, train, dataset, augment=augment)
        self.device = device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if device != "cpu":
            self.images = self.images.pin_memory()

    def __iter__(self):
        n = len(self.images)
        order = torch.randperm(n) if self.train else torch.arange(n)
        for i in range(len(self)):
            batch = order[i * self.batch_size : (i + 1) * self.batch_size]
            x = self.images[batch].to(self.device, non_blocking=True).float().div_(255)
            if self.do_augment:
                x = self.augment(x)
            yield (x - self.mean) / self.std, self.labels[batch].to(self.device)


def _imagenet_loaders(root, batch_size, eval_batch, workers):
    from torchvision import transforms

    _, mean, std, _, _ = DATASETS["imagenet"]
    normalize = transforms.Normalize(mean, std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    val = datasets.ImageFolder(os.path.join(root, "val"), test_tf)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
        torch.utils.data.DataLoader(val, eval_batch, shuffle=False, **kw),
    )


def make_loaders(name, batch_size, eval_batch=1024, device="cuda", root="data",
                 imagenet_root=None, workers=8):
    """(train_loader, test_loader) yielding normalized float batches on `device`."""
    if name == "imagenet":
        if imagenet_root is None:
            raise ValueError("ImageNet has to be downloaded by hand; pass --imagenet-root "
                             "pointing at a folder with train/ and val/ class subfolders")
        return _imagenet_loaders(imagenet_root, batch_size, eval_batch, workers)

    if name == "tinyimagenet":
        tx, ty = load_split(name, True, root=root)
        vx, vy = load_split(name, False, root=root)
        return (
            HostLoader(tx, ty, batch_size, True, name, device=device),
            HostLoader(vx, vy, eval_batch, False, name, augment=False, device=device),
        )

    tx, ty = load_split(name, True, root=root, device=device)
    vx, vy = load_split(name, False, root=root, device=device)
    return (
        GPULoader(tx, ty, batch_size, True, name, augment=True),
        GPULoader(vx, vy, eval_batch, False, name, augment=False),
    )
