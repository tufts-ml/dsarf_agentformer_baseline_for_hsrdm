import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    
    
    if args.set_type=='small':
        train_x = np.load(os.path.join(data_dir, 'player_coords_train__with_1_games.npy'))
        event_idx_train = np.load(os.path.join(data_dir, 'example_stop_idxs_train__with_1_games.npy'))
    elif args.set_type=='medium':
        train_x = np.load(os.path.join(data_dir, 'player_coords_train__with_5_games.npy'))
        event_idx_train = np.load(os.path.join(data_dir, 'example_stop_idxs_train__with_5_games.npy'))
    elif args.set_type=='large':
        train_x = np.load(os.path.join(data_dir, 'player_coords_train__with_20_games.npy'))
        event_idx_train = np.load(os.path.join(data_dir, 'example_stop_idxs_train__with_20_games.npy'))
        

    test_x = np.load(os.path.join(data_dir, 'player_coords_test__with_5_games.npy'))
    event_idx_test = np.load(os.path.join(data_dir, 'example_stop_idxs_test__with_5_games.npy'))
    
    T, J, D = train_x.shape
    
    # create the training data files
    print('Generating training data files that can be ingested by agentformer...')
    player_event_names_list = []
    for event_idx in range(len(event_idx_train)-1):
        curr_event_t_start = event_idx_train[event_idx]+1
        curr_event_t_end = event_idx_train[event_idx+1]

        curr_event_length = curr_event_t_end-curr_event_t_start

        if curr_event_length<40:
            print('Skipping event %d due to insufficient data to forecast 30 timepoints'%event_idx)
            continue

        player_trajectories_dict_list = []
        for j in range(J):
            traj_curr_j = train_x[curr_event_t_start:curr_event_t_end, j, :]
            t_arr = np.arange(0, curr_event_t_end-curr_event_t_start)*1.0
            player_id_arr = j*np.ones(len(traj_curr_j))
            x_arr = traj_curr_j[:, 0]*10
            y_arr = traj_curr_j[:, 1]*10

            curr_player_traj_dict = {'timestep' : t_arr,
                                    'player_id' : player_id_arr,
                                    'x' : x_arr.round(5),
                                    'y' : y_arr.round(5)}

            if j==0:
                player_trajectories_df = pd.DataFrame(curr_player_traj_dict)
            else:
                player_trajectories_df = player_trajectories_df.append(pd.DataFrame(curr_player_traj_dict), ignore_index=True)

        player_trajectories_df = player_trajectories_df.sort_values(by=['timestep', 'player_id']).reset_index(drop=True)

        if len(player_trajectories_df)>0:
            # do some more post-processing to set the contexts (context here means non-player trajectory related data like some ) as -1.0
            for ii in range(12):
                player_trajectories_df.loc[:, 'dummy_%d'%(ii+1)]=-1.0

            player_trajectories_df.loc[:, 'type'] = 'player'
            player_trajectories_df = player_trajectories_df[['timestep', 'player_id', 'type', 'dummy_1', 'dummy_2', 'dummy_3', 'dummy_4',
                                                            'dummy_5', 'dummy_6', 'dummy_7', 'dummy_8', 'dummy_9', 'dummy_10', 
                                                            'x', 'dummy_11', 'y', 'dummy_12']]

            # save to text file
            save_filename = os.path.join(data_dir, '%s_set'%args.set_type, 'player_event_%d_train.txt'%event_idx)
            player_trajectories_df.to_csv(save_filename, sep=' ', index=False, header=False)
            print('Done saving %s'%save_filename)
        else:
            print('Skipping event %d because of no data'%event_idx)

        player_event_names_list.append('player_event_%d'%event_idx)
        
        T, J, D = test_x.shape
        
        
    print('Generating test data files for evaluation by agentformer...')
    past_t = np.load(os.path.join(data_dir, 'random_context_times.npy'))

    pred_t = 30
    # event_idx = 0
    player_event_names_list = []
    for event_idx in range(len(event_idx_test)-1):
        if np.isnan(past_t[event_idx]):
            continue

        buffer=10
        context_time = past_t[event_idx]
        curr_event_t_start = int(event_idx_test[event_idx]+1+context_time-buffer)
        curr_event_t_end = int(curr_event_t_start+pred_t+buffer)#event_idx_test[event_idx+1]



        player_trajectories_dict_list = []
        for j in range(J):
            traj_curr_j = test_x[curr_event_t_start:curr_event_t_end, j, :]
    #         t_arr = np.arange(0, curr_event_t_end-curr_event_t_start)*1.0
            t_arr = np.arange(context_time-buffer, context_time+pred_t)*1.0
            player_id_arr = j*np.ones(len(traj_curr_j))
            x_arr = traj_curr_j[:, 0]*10
            y_arr = traj_curr_j[:, 1]*10

            curr_player_traj_dict = {'timestep' : t_arr,
                                    'player_id' : player_id_arr,
                                    'x' : x_arr.round(5),
                                    'y' : y_arr.round(5)}

            if j==0:
                player_trajectories_df_test = pd.DataFrame(curr_player_traj_dict)
            else:
                player_trajectories_df_test = player_trajectories_df_test.append(pd.DataFrame(curr_player_traj_dict), ignore_index=True)

        player_trajectories_df_test = player_trajectories_df_test.sort_values(by=['timestep', 'player_id']).reset_index(drop=True)

        if len(player_trajectories_df_test)>0:
            # do some more post-processing to set the contexts as -1.0
            for ii in range(12):
                player_trajectories_df_test.loc[:, 'dummy_%d'%(ii+1)]=-1.0

            player_trajectories_df_test.loc[:, 'type'] = 'player'
            player_trajectories_df_test = player_trajectories_df_test[['timestep', 'player_id', 'type', 'dummy_1', 'dummy_2', 'dummy_3', 'dummy_4',
                                                            'dummy_5', 'dummy_6', 'dummy_7', 'dummy_8', 'dummy_9', 'dummy_10', 
                                                            'x', 'dummy_11', 'y', 'dummy_12']]

            # save to text file
            save_filename = os.path.join(data_dir, '%s_set'%args.set_type, 'player_event_%d_test.txt'%event_idx)
            player_trajectories_df_test.to_csv(save_filename, sep=' ', index=False, header=False)


            player_event_names_list.append('player_event_%d_test'%event_idx)
            print('Done saving %s'%save_filename)
            print('Max timesteps for this event : %d'%player_trajectories_df_test.groupby('player_id').max()['timestep'].unique()[0])
        else:
            print('Skipping event %d because of no data'%event_idx)
        
        
        
    
    
