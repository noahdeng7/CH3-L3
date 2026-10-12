"""CNN backbones for the ICLR 2024 CNN experiments, each ending in a feature vector.

Appendix H: "The models employing dynamic sparse training ... adopt the final
three fully connected layers (excluding the last layer) of all the convolution
neural networks ... the default configuration for GoogLeNet and ResNet
comprises just one fully connected layer. We expanded this to three layers."
So every network is `backbone -> 3 sandwich layers -> dense classifier`, and
only the sandwich layers are sparse; the convolutions train densely for every
method. The head is a utils.model_defs.CHT (sparse) or a plain dense stack
(FC); it reads the backbone's pooled features, whose width fixes the hidden
dimension (1x = that width, 2x = twice it).

The backbones follow the CIFAR variants Table 5's schedule comes from (lr 0.1,
milestones 60/120/160, gamma 0.2, one warm-up epoch, batch 128): a 3x3 stem
and no initial max-pool, global average pooling at the end. ImageNet ResNet50
uses torchvision's model with its fc removed.
"""

import torch
import torch.nn as nn

from utils.model_defs import _stack

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def _conv_bn(cin, cout, k, stride=1, padding=0):
    return [nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class VGG16Features(nn.Module):
    """VGG16 with batch norm; its 4096-wide classifier is replaced by the head."""

    out_dim = 512

    def __init__(self):
        super().__init__()
        layers, cin = [], 3
        for v in VGG16:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += _conv_bn(cin, v, 3, padding=1)
                cin = v
        self.features = nn.Sequential(*layers)
        # 32x32 inputs already end at 1x1; Tiny-ImageNet's 64x64 end at 2x2 and
        #are pooled to the same 512 features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


class Inception(nn.Module):
    def __init__(self, cin, n1, r3, n3, r5, n5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(*_conv_bn(cin, n1, 1))
        self.b2 = nn.Sequential(*_conv_bn(cin, r3, 1), *_conv_bn(r3, n3, 3, padding=1))
        # the 5x5 branch as two stacked 3x3s
        self.b3 = nn.Sequential(*_conv_bn(cin, r5, 1), *_conv_bn(r5, n5, 3, padding=1),
                                *_conv_bn(n5, n5, 3, padding=1))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), *_conv_bn(cin, pool_proj, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNetFeatures(nn.Module):
    out_dim = 1024

    def __init__(self):
        super().__init__()
        self.prelayer = nn.Sequential(*_conv_bn(3, 64, 3, padding=1), *_conv_bn(64, 64, 3, padding=1),
                                      *_conv_bn(64, 192, 3, padding=1))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.prelayer(x))
        x = self.pool(self.b3(self.a3(x)))
        x = self.pool(self.e4(self.d4(self.c4(self.b4(self.a4(x))))))
        x = self.b5(self.a5(x))
        return self.dropout(torch.flatten(self.avgpool(x), 1))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, width, stride=1):
        super().__init__()
        cout = width * self.expansion
        self.residual = nn.Sequential(
            *_conv_bn(cin, width, 1),
            *_conv_bn(width, width, 3, stride=stride, padding=1),
            nn.Conv2d(width, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual(x) + self.shortcut(x))


class ResNetFeatures(nn.Module):
    out_dim = 2048

    def __init__(self, blocks):
        super().__init__()
        self.stem = nn.Sequential(*_conv_bn(3, 64, 3, padding=1))
        stages, cin = [], 64
        for width, n, stride in zip((64, 128, 256, 512), blocks, (1, 2, 2, 2)):
            for s in [stride] + [1] * (n - 1):
                stages.append(Bottleneck(cin, width, s))
                cin = width * Bottleneck.expansion
        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.avgpool(self.stages(self.stem(x))), 1)


class TorchvisionResNet50(nn.Module):
    out_dim = 2048

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        self.net = resnet50(weights=None)
        self.net.fc = nn.Identity()

    def forward(self, x):
        return self.net(x)


ARCHS = {
    "vgg16": VGG16Features,
    "googlenet": GoogLeNetFeatures,
    "resnet50": lambda: ResNetFeatures((3, 4, 6, 3)),
    "resnet152": lambda: ResNetFeatures((3, 8, 36, 3)),
    "resnet50_imagenet": TorchvisionResNet50,
}

# pooled feature width of each backbone, i.e. the head's input and its 1x hidden size
ARCHS_OUT_DIM = {"vgg16": 512, "googlenet": 1024, "resnet50": 2048, "resnet152": 2048,
                 "resnet50_imagenet": 2048}


def backbone(arch):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; pick one of {sorted(ARCHS)}")
    net = ARCHS[arch]()
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class DenseHead(nn.Module):
    """The fully connected control head: dims[0] -> ... -> dims[-1] -> classes.

    With a single entry in `dims` this is the backbone's original lone
    classifier (Table 2's "Original" row).
    """

    def __init__(self, dims, num_classes):
        super().__init__()
        if len(dims) == 1:
            self.layers = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(dims[0], num_classes))
        else:
            linears = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
            self.layers = _stack(linears, num_classes, {})
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


class BackboneWithHead(nn.Module):
    def __init__(self, features, head):
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x):
        return self.head(self.features(x))
