"""The papers' training protocols, as base run configs.

Values the papers state are cited. Values they leave open are marked
UNSTATED, with the choice made here; every one of them can be overridden per
run, and the README lists them together.
"""

# ---------------------------------------------------------------------------
# ICLR 2024, "Epitopological learning and Cannistraci-Hebb network shape
# intelligence brain-inspired theory for ultra-sparse advantage in deep learning"
# ---------------------------------------------------------------------------

# Appendix P.1: sparsity 0.99, zeta 0.3, weight decay 1e-5, batch 32, momentum
# 0.9 "for all the MLP and VGG16 tasks", and "a fixed rate for all tasks".
# Figure 7: "we keep ui=1 and lr=0.01".
MLP_2024 = dict(
    arch="mlp", sparsity=0.99, zeta=0.3, update_interval=1,
    batch_size=32, momentum=0.9, weight_decay=1e-5,
    lr=0.01, lr_schedule="constant",
    # UNSTATED: the epoch count of the CHT comparisons (Tables 3, Figures 7,
    #8, 10). Section 4.1 trains 250 epochs; Figure 2's ANP settles "after
    #approximately 50 epochs" and CHT's early stop fires in the first ~20.
    epochs=100,
)

# Section 4.1 / Figure 1: ESML vs SET "adopt the same network structure as
# outlined in SET ... the hidden dimension is set to 1000", "a lower learning
# rate of 0.0001", and accuracy is marked "at the 250th epoch".
ESML_2024 = dict(MLP_2024, hidden=1000, lr=1e-4, epochs=250)

# Section 4.1 also runs ESML vs SET on VGG16 (CIFAR-10, CIFAR-100) under the
# same P.1 settings ("all the MLP and VGG16 tasks").
ESML_VGG_2024 = dict(ESML_2024, arch="vgg16", hidden=None, scale=1, eval_batch=1000,
                     checkpoint_every=1)

# Table 5, one entry per CNN task. P.1: the models "adhere to their default
# hyperparameter configurations" apart from the 4-layer head, so weight decay
# is each model's default (5e-4 in the CIFAR model zoo the schedule comes from,
# 1e-4 for torchvision's ImageNet ResNet50) except VGG16, which P.1 puts at
# 1e-5 with the MLPs.
_CNN = dict(sparsity=0.99, zeta=0.3, update_interval=1, momentum=0.9,
            lr_schedule="multistep", eval_batch=1000, checkpoint_every=1)

CNN_2024 = {
    "VGG16-TinyImageNet": dict(_CNN, arch="vgg16", dataset="tinyimagenet", epochs=200, lr=0.1,
                               milestones=[61, 121, 161], gamma=0.2, batch_size=256,
                               warmup_epochs=0, weight_decay=1e-5),
    "GoogLeNet-CIFAR100": dict(_CNN, arch="googlenet", dataset="cifar100", epochs=200, lr=0.1,
                               milestones=[60, 120, 160], gamma=0.2, batch_size=128,
                               warmup_epochs=1, weight_decay=5e-4),
    "GoogLeNet-TinyImageNet": dict(_CNN, arch="googlenet", dataset="tinyimagenet", epochs=200,
                                   lr=0.1, milestones=[61, 121, 161], gamma=0.2, batch_size=256,
                                   warmup_epochs=0, weight_decay=5e-4),
    "ResNet50-CIFAR100": dict(_CNN, arch="resnet50", dataset="cifar100", epochs=200, lr=0.1,
                              milestones=[60, 120, 160], gamma=0.2, batch_size=128,
                              warmup_epochs=1, weight_decay=5e-4),
    "ResNet50-ImageNet": dict(_CNN, arch="resnet50_imagenet", dataset="imagenet", epochs=90,
                              lr=0.1, milestones=[32, 62], gamma=0.1, batch_size=256,
                              warmup_epochs=0, weight_decay=1e-4, eval_batch=256),
    "ResNet152-CIFAR100": dict(_CNN, arch="resnet152", dataset="cifar100", epochs=200, lr=0.1,
                               milestones=[60, 120, 160], gamma=0.2, batch_size=128,
                               warmup_epochs=1, weight_decay=5e-4),
}
for _cfg in CNN_2024.values():
    _cfg["paper_epochs"] = _cfg["epochs"]

# Appendix H: "For the sake of simplicity, ER (WS, beta = 1) initialization is
# utilized as the default approach in the experiments presented in the main
# text" - CHT on the CNN heads starts from ER, not CSTI.
CNN_CHT_INIT = "er"

# ---------------------------------------------------------------------------
# 2026, "Brain network science modelling of sparse neural networks enables
# Transformers and LLMs to perform as fully connected" (CHTs / CHTss)
# ---------------------------------------------------------------------------

# Table 4 (MLP on image classification), with CIFAR-10's 3072 hidden width.
MLP_2026_CIFAR10 = dict(
    arch="mlp", dataset="cifar10", hidden=3072,
    batch_size=32, epochs=100,
    lr=0.025, lr_end=2.5e-4, lr_schedule="linear",
    zeta=0.3, update_interval=1,
    momentum=0.9, weight_decay=5e-4,
    sparsity=0.99,
)
