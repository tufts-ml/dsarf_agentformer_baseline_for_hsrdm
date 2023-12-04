#!/bin/sh

# Load the right conda environment
source activate agentformer


# Pass along all ENV variables as arguments to my Python script
python train.py --cfg user_bball_agentformer_medium_pre --set_type medium
python train.py --cfg user_bball_agentformer_medium --set_type medium
python test.py --cfg user_bball_agentformer_medium

source deactivate