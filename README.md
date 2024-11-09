# DSARF on Figure 8, MarchingBand, and Agentformer on Basketball
Baselines on Figure 8 and Basketball data for paper titled 'Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models'


## Environment
* **Tested OS:** MacOS, Linux
* Python >= 3.7
* PyTorch == 1.8.0

### Dependencies:
1. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Cloning the anonymous repo 
Please follow the instructions [here](https://github.com/fedebotu/clone-anonymous-github) to clone an anonymous repo. (Credit : Clone Anonymous Github created by fedebotu)

## DSARF on Figure 8
For reproducing the experiments for complete independence, multi channel and complete pooling, see the notebook provided in [dsarf_on_figure_8/DSARF_on_figure_8.ipynb](dsarf_on_figure_8/DSARF_on_figure_8.ipynb)

## DSARF on MarchingBand
For reproducing the experiments for classifying system-states, run the model [here](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/dsarf_on_marching/run_marchingband.py) and plot the results [here](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/dsarf_on_marching/plot_results.ipynb)

## AgentFormer on Basketball
For reproducing the experiments for predicting basketball player trajectories, follow the readme in [agentformer_on_bball/](agentformer_on_bball/)

