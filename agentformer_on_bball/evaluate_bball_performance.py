import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train configs
    parser.add_argument('--set_type', 
                        type=str, 
                        default='small', 
                        help="choose between small, medium or large")
    

    args = parser.parse_args()

    ## Load the data and event indices for the training set
    data_dir = os.path.join('datasets', 'basketball')    
    
    
    context_times = np.load(os.path.join(data_dir, 'random_context_times.npy'))
    
    test_x = np.load(os.path.join(data_dir, 'player_coords_test__with_5_games.npy'))
    event_idx_test = np.load(os.path.join(data_dir, 'example_stop_idxs_test__with_5_games.npy'))
    
    set_name=args.set_type
    
    # Use the saved model trajectories or the trajectories saved from your trained model to make predictions. 
    eval_dir = 'saved_results/trajectories/%s/test/samples'%set_name
    
    # OR 
    
    # Replace the eval_dir path to the directory where you have the agentformer predictions saved. 
    # They are saved in the results/user_bball_agentformer_{set_type}/results/epoch_*/test/samples/ 
    # directory after running the provided bash scripts for training
    
#     eval_dir = '/results/user_bball_agentformer_%s/results/epoch_0041/test/samples/'%set_name
    
    
    
    sample_ind = 15 # or loop over the 20 samples
    pred_t=30
    _, J, D = test_x.shape
    
    
    save_dir = 'saved_results/trajectories/%s/plots'%set_name
    
    if not os.path.exists(save_dir):
        # Create the directory
        os.makedirs(save_dir)
    
    print('Plotting trajectory, sample index : %d'%sample_ind)
    predicted_traj_list = []
    true_traj_list = []
    for event_id in range(len(context_times)):
        event_dir = os.path.join(eval_dir, 'player_event_%d_test'%event_id)
        frame_dirs = glob.glob(os.path.join(event_dir, 'frame_*'))
        J = test_x.shape[1]

        if np.isnan(context_times[event_id]):
            print('Skipping event %d due to lack of data'%event_id)
            continue

        # get the frames to predict for this event
        curr_event_t_start = event_idx_test[event_id]+ int(context_times[event_id])
        curr_event_t_end = curr_event_t_start+pred_t

        player_trajectories_dict_list = []
        for j in range(J):
            traj_curr_j = test_x[curr_event_t_start:curr_event_t_end, j, :]
            x_arr = traj_curr_j[:, 0]
            y_arr = traj_curr_j[:, 1]


        true_traj_list.append(test_x[curr_event_t_start:curr_event_t_end, :, :])
        start_frame_dirs = [ii for ii in frame_dirs if '%06d'%int(context_times[event_id]-1) in ii]
        start_frame_dir = start_frame_dirs[0]
        start_frame_dir_ind = frame_dirs.index(start_frame_dir)-1



        predicted_traj_TJD = np.zeros((pred_t, J, 2))
        D=2
        # load the predicted trajectory file

        predicted_traj_arr = np.loadtxt(os.path.join(frame_dirs[start_frame_dir_ind], 'sample_0%02d.txt'%sample_ind))[:pred_t*J, 2:]

        for j in range(J):
            preds = predicted_traj_arr[j*pred_t : j*pred_t+pred_t]
            predicted_traj_TJD[:, j, :] = preds
            predicted_traj_TJD[0, j, :] = test_x[curr_event_t_start, j, :]*10


        predicted_traj_list.append(predicted_traj_TJD/10)

        # plot samples for 10 events. To plot for all events and all samples, remove the if condition.
        if (event_id<10)&(sample_ind==15):
            f, axs = plt.subplots(2, 5, figsize=(40, 20), 
                                  sharex=True, 
                                  sharey=True
                                 )
            sns.set_context('notebook', font_scale=2.2)
            sns.set_style('whitegrid')
            axs_list = axs.flatten()
            for j in range(J):
                traj_curr_j = test_x[curr_event_t_start:curr_event_t_end, j, :]
                contxt_curr_j = test_x[curr_event_t_start-10:curr_event_t_start+1, j, :]

                t_arr = np.arange(0, curr_event_t_end-curr_event_t_start)*1.0
                x_arr = traj_curr_j[:, 0]
                y_arr = traj_curr_j[:, 1]

                x_c_arr = contxt_curr_j[:, 0]
                y_c_arr = contxt_curr_j[:, 1]

                x_pred_arr = predicted_traj_TJD[:, j, 0]/10
                y_pred_arr = predicted_traj_TJD[:, j, 1]/10

                axs_list[j].scatter(x_pred_arr[0], y_pred_arr[0], color='b', marker='x', s=200)
                axs_list[j].scatter(x_arr[0], y_arr[0], color='r', marker='x', s=200)

                axs_list[j].plot(x_pred_arr, y_pred_arr, color='b', label='predicted trajectory')
                axs_list[j].plot(x_arr, y_arr, color='r', label='true trajectory')
                axs_list[j].plot(x_c_arr, y_c_arr, color='g', label='context')

                axs_list[j].set_title('Player %d'%j)
                axs_list[j].set_xlabel('x')
                axs_list[j].set_ylabel('y')
                axs_list[j].legend()
                f.suptitle('True and Predicted Trajectories for event %d after training on %s set'%(event_id, set_name))


                fname = os.path.join(save_dir, 'predicted_traj_set_%s_event_%d_sample_%d.png'%(set_name, 
                                                                                               event_id, sample_ind))
                f.savefig(fname)
    
    
