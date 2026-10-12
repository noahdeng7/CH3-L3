"""Dataset loading and the augmentation described in the CH3-L3 paper.

Appendix P.1 of the ICLR 2024 paper: "All the experiments in this article
involve image augmentation, which includes random cropping and horizontal
flipping, followed by a standard normalization process."
"""

import math

import torch
from torch import nn
from torchvision import datasets

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# name -> (num_classes, mean, std, image_size, channels)
DATASETS = {
    "mnist": (10, (0.1307,), (0.3081,), 28, 1),
    "fmnist": (10, (0.2860,), (0.3530,), 28, 1),
    "emnist": (47, (0.1751,), (0.3332,), 28, 1),
    "cifar10": (10, CIFAR10_MEAN, CIFAR10_STD, 32, 3),
}

PAD = 4

_CACHE = {}


def input_dim(name):
    _, _, _, size, channels = DATASETS[name]
    return size * size * channels


def load_split(name, train, root="data", device="cuda"):
    """Load a split onto the device as uint8 NCHW images and int64 labels."""
    key = (name, train, device)
    if key in _CACHE:
        return _CACHE[key]

    if name == "mnist":
        ds = datasets.MNIST(root=root, train=train, download=True)
        images = ds.data.unsqueeze(1)
        labels = ds.targets
    elif name == "fmnist":
        ds = datasets.FashionMNIST(root=root, train=train, download=True)
        images = ds.data.unsqueeze(1)
        labels = ds.targets
    elif name == "emnist":
        # 'balanced' is the 47-class split; its 112.8k/18.8k sizes and ~87%
        #fully connected accuracy are what Table 7 reports
        ds = datasets.EMNIST(root=root, split="balanced", train=train, download=True)
        # EMNIST ships its images transposed relative to MNIST and torchvision
        #does not correct it, so crop and flip would act on the wrong axis
        images = ds.data.transpose(-2, -1).unsqueeze(1)
        labels = ds.targets
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=train, download=True)
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
        labels = torch.tensor(ds.targets)
    else:
        raise ValueError(f"unknown dataset {name!r}")

    out = (images.contiguous().to(device), labels.to(device))
    _CACHE[key] = out
    return out


class GPULoader:
    """Batches a device-resident dataset, augmenting and normalizing on device."""

    def __init__(self, images, labels, batch_size, train, dataset, augment=True):
        _, mean, std, size, channels = DATASETS[dataset]
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.train = train
        self.do_augment = train and augment
        self.size = size
        self.channels = channels
        dev = images.device
        self.mean = torch.tensor(mean, device=dev).view(1, channels, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, channels, 1, 1)

    def __len__(self):
        n = len(self.images)
        return n // self.batch_size if self.train else math.ceil(n / self.batch_size)

    def augment(self, x):
        """Random crop with 4px padding, then random horizontal flip."""
        n, c, size = x.shape[0], self.channels, self.size
        padded = nn.functional.pad(x, (PAD, PAD, PAD, PAD))
        offsets = torch.randint(0, 2 * PAD + 1, (n, 2), device=x.device)
        span = torch.arange(size, device=x.device)
        rows = (offsets[:, 0:1] + span)[:, None, :, None]
        cols = (offsets[:, 1:2] + span)[:, None, None, :]
        idx = torch.arange(n, device=x.device)[:, None, None, None]
        chans = torch.arange(c, device=x.device)[None, :, None, None]
        x = padded[idx, chans, rows, cols]
        flip = torch.rand(n, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def __iter__(self):
        n = len(self.images)
        device = self.images.device
        order = (
            torch.randperm(n, device=device)
            if self.train
            else torch.arange(n, device=device)
        )
        for i in range(len(self)):
            batch = order[i * self.batch_size : (i + 1) * self.batch_size]
            x = self.images[batch].float().div_(255)
            if self.do_augment:
                x = self.augment(x)
            yield (x - self.mean) / self.std, self.labels[batch]
