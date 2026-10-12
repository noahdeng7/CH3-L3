"""Shared machinery for the paper reproductions, built on top of `utils/`.

Everything `utils/` already provides (the masked MLP and CHT model, CSTI and
Erdos-Renyi masks, the CH link predictors, percolation, the density-decay and
LR schedules, the on-GPU loader) is imported from there. What lives here is only
what the reproductions need beyond it:

  data.py      CIFAR-100, Tiny-ImageNet and ImageNet next to utils.data's four
  masks.py     bipartite small-world (BSW) and scale-free (BSF) initialization
  cnn.py       VGG16 / GoogLeNet / ResNet50 / ResNet152 backbones with a sparse head
  evolvers.py  the topology update of every method compared in the two papers
  methods.py   one preset per method (SET, RigL, MEST, GMP, GraNet, CHT, CHTs, ...)
  train.py     the shared training loop, run cache and result format
  metrics.py   AAE and the network-science measures of Figure 2 / Appendix O
  report.py    mean +- standard error tables against the paper's numbers
  cli.py       the flags and grid runner every experiment script shares
"""
