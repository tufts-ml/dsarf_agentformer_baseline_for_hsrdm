from dsarf import DSARF
from dsarf import compute_NRMSE
from marchingband_data.run_sim import generate_training_data
import numpy as np
import torch
import os

#Directories 
home_dir = os.path.expanduser("~")
losses_dir = f"{home_dir}/dsarf_agentformer_baseline_for_hsrdm/dsarf_on_marching/losses/"
predictions_dir = f"{home_dir}/dsarf_agentformer_baseline_for_hsrdm/dsarf_on_marching/predictions/"
results_dir = f"{home_dir}/dsarf_agentformer_baseline_for_hsrdm/dsarf_on_marching/results/"

#Run DSARF on MarchingBand Data 
def ind_run(N, J, T, seed, factor_dim, L, total_time): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    GLOBAL_MSG = "LAUGH" * N
    total_data = generate_training_data(GLOBAL_MSG, J, T, 0)
    data = total_data[0]
    
    end_times = [0, 1000, 2050, 3100, 4100, 5100, 6100, 7150, 8200, 9200, 10300]
    data = data.reshape(total_time, J*D)
    segments = []
    for i in range(len(end_times) - 1):
        start = end_times[i]
        end = end_times[i+1]
        segment = data[start:end] 
        segments.append(segment)

    # Create and train the model
    dsarf = DSARF(J * D, factor_dim=factor_dim, L=L, S=6, batch_size=1)
    model = dsarf.fit(segments, epoch_num=200)
    model_train = model[0]
    loss = model[1]
    stats = model_train.report_stats(segments)

    # Extract posterior summary, losses, prediction stats
    posterior_summary = model_train.q_s
    return posterior_summary, loss, stats

def save_results(results, seed, factor_dim, L):
    filename = results_dir + f"run_{seed}_{factor_dim}_{L[-1]}.pth"
    torch.save(results, filename)

def save_loss(loss, seed, factor_dim, L):
    filename = losses_dir + f"run_{seed}_{factor_dim}_{L[-1]}.pth"
    torch.save(loss, filename)

def save_predictions(prediction, seed, factor_dim, L):
    filename = predictions_dir + f"run_{seed}_{factor_dim}_{L[-1]}.pth"
    torch.save(prediction, filename)

if __name__ == '__main__':
    D = 2
    N = 10
    J = 64
    T = 200
    total_time = 10300
    seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    factor_dims = [25]
    lags = [list(range(1, i + 1)) for i in [200]] 


    for seed in seeds:
        for L in lags:
            for factor_dim in factor_dims:
                outcome = ind_run(N, J, T, seed, factor_dim, L, total_time)
                results = outcome[0]
                loss = outcome[1]
                prediction = outcome[2]
                save_results(results, seed, factor_dim, L)          
                save_loss(loss, seed, factor_dim, L)   
                save_predictions(prediction, seed, factor_dim, L)       