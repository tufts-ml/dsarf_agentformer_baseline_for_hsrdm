#!/bin/sh

# Load the right conda environment
source activate agentformer


# Pass along all ENV variables as arguments to my Python script
python train.py --cfg user_bball_agentformer_small_pre --set_type small
python train.py --cfg user_bball_agentformer_small --set_type small
python test.py --cfg user_bball_agentformer_small

source deactivate