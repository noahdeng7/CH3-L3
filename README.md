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

`topology.py` - pure topology functions
`model_defs.py` - sparse model definitions, application of topology functions to models
`create_masks.py` - sparse mask generation
`train.py` - data loading, training loop, and wandb logging
`config.yaml` - all hyperparameters and model details
`sweep.py` - multi-seed CH3-L3 vs SET ablation and the aggregate plot

