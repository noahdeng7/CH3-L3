# CH3-L3

This repository is a reimplementation of the ESML paper (Zhang et al. 2024), where we focus on examining the CH3-L3 regrowth rule.
Specifically, this repo aims to recreate the CH3-L3 vs SET result on MNIST with a sparse MLP.

We focus on 2 methods to create ultra-sparse networks: CH3-L3, and SET (random regrowth)

Both methods train a network whose weights are masked at 1% density and periodically rewire those links; useless links are dropped, and new ones are formed.
Every other aspect of training is held the same for the ablation, so we can see the difference from a deliberate calculation of which links might be most useful to regrow (CH3-L3) and randomly creating links (SET)

# HOW TO USE THIS REPO

Using this repository is very simple. First, install dependencies in requirements.txt.
Then, create the masks using the sparse initializations outlined by Zhang et al. by running create_masks.py. 
Log in to wandb to track run stats
Then, configure config.yaml to run what specific test you want, and then run train.py.

# FILE STRUCTURE

`utils/topology.py` - pure topology functions
`utils/model_defs.py` - sparse model definitions, application of topology functions to models
`utils/masks.py` - sparse mask generation (CSTI / Erdos-Renyi)
`utils/data.py` - dataset loading and the on-GPU `GPULoader`
`utils/schedules.py` - density decay (sigmoid / cubic) and the Table 4 LR ramp
`experiments/CHTss/` - CHTss on CIFAR-10, the Table 1 replication (see its README)
`CIFAR_baseline.py` - FC overfitting check on CIFAR-10 (run from the repo root)
`table7_repro/run_table7.py` - Table 7 reproduction harness
`config.yaml` - all hyperparameters and model details

