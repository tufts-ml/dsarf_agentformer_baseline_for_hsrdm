#!/bin/sh

# Load the right conda environment
source activate agentformer


# Pass along all ENV variables as arguments to my Python script
python train.py --cfg user_bball_agentformer_large_pre --set_type large
python train.py --cfg user_bball_agentformer_large --set_type large
python test.py --cfg user_bball_agentformer_large

source deactivate