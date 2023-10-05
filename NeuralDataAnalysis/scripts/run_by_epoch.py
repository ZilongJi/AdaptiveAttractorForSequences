import logging
import os
import pdb
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loren_frank_data_processing import save_xarray
from loren_frank_data_processing.position import (EDGE_ORDER, EDGE_SPACING,
                                                  make_track_graph)
from replay_trajectory_classification import (ClusterlessClassifier,
                                              SortedSpikesClassifier)
from scipy.ndimage import label
from src.analysis import (get_place_field_max, get_replay_info,
                          reshape_to_segments, get_replay_traj)

from src.load_data import load_data, load_sleep_data
from src.parameters import (ANIMALS, FIGURE_DIR, PROBABILITY_THRESHOLD,
                            PROCESSED_DATA_DIR, SAMPLING_FREQUENCY,
                            TRANSITION_TO_CATEGORY,
                            sleep_duration_threshold, 
                            continuous_transition_types, discrete_diag,
                            knot_spacing, model, model_kwargs, movement_var,
                            place_bin_size, replay_speed, spike_model_penalty)

from src.parameters import (classifier_parameters_thetasweeps, discrete_state_transition)

from trajectory_analysis_tools import (get_ahead_behind_distance,
                                       get_trajectory_data)

from sklearn.model_selection import KFold
from src.visualization import plot_ripple_decode_1D
from tqdm.auto import tqdm

from ZilongCode.utils import find_sleep_intervals, get_sleep_ripples, detect_sleep_periods


FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
plt.switch_backend('agg')

def sorted_spikes_analysis_1D(epoch_key,
                              plot_ripple_figures=False,
                              exclude_interneuron_spikes=False,
                              use_multiunit_HSE=False,
                              brain_areas=None,
                              overwrite=False):
    animal, day, epoch = epoch_key
    data_type, dim = 'sorted_spikes', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)
    is_training = data['position_info'].speed > 4
    position = data['position_info'].loc[:, 'linear_position']
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    # Set up naming
    group = f'/{data_type}/{dim}/'
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'

    if exclude_interneuron_spikes:
        logging.info('Excluding interneuron spikes...')
        epoch_identifier += '_no_interneuron'
        group += 'no_interneuron/'

    if brain_areas is not None:
        area_str = '-'.join(brain_areas)
        epoch_identifier += f'_{area_str}'
        group += f'{area_str}/'

    if use_multiunit_HSE:
        epoch_identifier += '_multiunit_HSE'
        group += 'classifier/multiunit_HSE/'
        replay_times = data['multiunit_HSE_times']
    else:
        group += 'classifier/ripples/'
        replay_times = data['ripple_times']

    model_name = os.path.join(
        PROCESSED_DATA_DIR, epoch_identifier + '_model.pkl')

    try:
        if overwrite:
            raise FileNotFoundError
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
            group=group)
        logging.info('Found existing results. Loading...')
        ripple_spikes = reshape_to_segments(
            data['spikes'], replay_times.loc[:, ['start_time', 'end_time']])
        classifier = SortedSpikesClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        logging.info('Fitting classifier...')
        classifier = SortedSpikesClassifier(
            place_bin_size=place_bin_size, movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            spike_model_penalty=spike_model_penalty, knot_spacing=knot_spacing,
            continuous_transition_types=continuous_transition_types).fit(
                position, data['spikes'], is_training=is_training,
                track_graph=track_graph, center_well_id=center_well_id,
                edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING)
        classifier.save_model(model_name)
        logging.info(classifier)

        # Plot Place Fields
        g = (classifier.place_fields_ * data['sampling_frequency']).plot(
            x='position', col='neuron', col_wrap=4)
        arm_grouper = (data['position_info']
                       .groupby('arm_name')
                       .linear_position)
        max_df = arm_grouper.max()
        min_df = arm_grouper.min()
        plt.xlim((0, data['position_info'].linear_position.max()))
        max_rate = (classifier.place_fields_.values.max() *
                    data['sampling_frequency'])
        for ax in g.axes.flat:
            for arm_name, min_position in min_df.iteritems():
                ax.axvline(min_position, color='lightgrey', zorder=0,
                           linestyle='--')
                ax.text(min_position + 0.2, max_rate, arm_name,
                        color='lightgrey', horizontalalignment='left',
                        verticalalignment='top', fontsize=8)
            for arm_name, max_position in max_df.iteritems():
                ax.axvline(max_position, color='lightgrey', zorder=0,
                           linestyle='--')
        plt.suptitle(epoch_key, y=1.04, fontsize=16)
        fig_name = (
            f'{animal}_{day:02d}_{epoch:02d}_{data_type}_place_fields_1D.png')
        fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(g.fig)

        # Decode
        is_test = ~is_training

        test_groups = pd.DataFrame(
            {'test_groups': label(is_test.values)[0]}, index=is_test.index)
        immobility_results = []
        for _, df in tqdm(test_groups.loc[is_test].groupby('test_groups'),
                          desc='immobility'):
            start_time, end_time = df.iloc[0].name, df.iloc[-1].name
            test_spikes = data['spikes'].loc[start_time:end_time]
            immobility_results.append(
                classifier.predict(test_spikes, time=test_spikes.index))
        immobility_results = xr.concat(immobility_results, dim='time')

        results = [(immobility_results
                    .sel(time=slice(df.start_time, df.end_time))
                    .assign_coords(time=lambda ds: ds.time - ds.time[0]))
                   for _, df in replay_times.iterrows()]

        results = (xr.concat(results, dim=replay_times.index)
                   .assign_coords(state=lambda ds: ds.state.to_index()
                                  .map(TRANSITION_TO_CATEGORY)))

        logging.info('Saving results...')
        ripple_spikes = reshape_to_segments(
            data['spikes'], replay_times.loc[:, ['start_time', 'end_time']])
        save_xarray(PROCESSED_DATA_DIR, epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=f'/{data_type}/{dim}/classifier/ripples/')

    logging.info('Saving replay_info...')
    replay_info = get_replay_info(
        results, data['spikes'], replay_times, data['position_info'],
        track_graph, SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, epoch_key,
        classifier, data["ripple_consensus_trace_zscore"])
    prob = int(PROBABILITY_THRESHOLD * 100)
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_{prob:02d}.csv')
    replay_info.to_csv(replay_info_filename)

    if plot_ripple_figures:
        logging.info('Plotting ripple figures...')
        place_field_max = get_place_field_max(classifier)
        linear_position_order = place_field_max.argsort(axis=0).squeeze()
        ripple_position = reshape_to_segments(
            position, replay_times.loc[:, ['start_time', 'end_time']])

        for ripple_number in tqdm(replay_times.index, desc='ripple figures'):
            try:
                posterior = (
                    results
                    .acausal_posterior
                    .sel(ripple_number=ripple_number)
                    .dropna('time', how='all')
                    .assign_coords(
                        time=lambda ds: 1000 * ds.time /
                        np.timedelta64(1, 's')))
                plot_ripple_decode_1D(
                    posterior, ripple_position.loc[ripple_number],
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], classifier)
                plt.suptitle(
                    f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
                    f'{ripple_number:04d}')
                fig_name = (
                    f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...')


def clusterless_thetasweeps(epoch_key, 
                            exclude_interneuron_spikes=False,
                            brain_areas=None,
                            save_original_data=False):
    '''
    created by Zilong 29/08/2023
    Decoding theta sweeps using clusterless classifier
    Since we are using the state-space model, we need to cross validatd the classifier
    We do 5-fold cross validation here...
    '''
    animal, day, epoch = epoch_key
    
    #load data
    logging.info('Loading data...')
    data = load_data(epoch_key,
                     brain_areas=brain_areas,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)
 
    #get lfp info from data
    lfp_info = data['lfps']
    #save the ifp_info, which is pandas.core.series.Series
    lfp_info.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_lfp_info.pkl'))
            
    is_running = data["position_info"].speed > 4
    
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)
    
    cv = KFold()
    cv_classifier_clusterless_results = []

    logging.info('Cross validation the classifier...')
    for fold_ind, (train, test) in tqdm(enumerate(cv.split(data["position_info"].index))):
        
        print(f'Fold {fold_ind}')
        
        cv_classifier = ClusterlessClassifier(**classifier_parameters_thetasweeps)

        cv_classifier.fit(
            position=data["position_info"].iloc[train].linear_position,
            multiunits=data["multiunit"].isel(time=train),
            is_training=is_running.iloc[train],
            track_graph=track_graph,
            center_well_id=center_well_id,
            edge_order=EDGE_ORDER,
            edge_spacing=EDGE_SPACING,
        )
        cv_classifier.discrete_state_transition_ = discrete_state_transition

        cv_result = cv_classifier.predict(
                            data["multiunit"].isel(time=test),
                            time=data["position_info"].iloc[test].index / np.timedelta64(1, "s")
                            )
        
        #get the decoded trajectory in test data
        test_position_info = data["position_info"].iloc[test]
        posterior = cv_result.acausal_posterior
        trajectory_data = get_trajectory_data(
            posterior.sum('state'), track_graph, cv_classifier, test_position_info)
        
        #calculate the distance between actual position and mental position
        mental_distance_from_actual_position = np.abs(get_ahead_behind_distance(
            track_graph, *trajectory_data))
        #add mental_distance_from_actual_position to cv_result, which is a xarray.Dataset
        cv_result['mental_distance_from_actual_position'] = xr.DataArray(
            mental_distance_from_actual_position, dims=['time'])
        
        #calculate the distance between actual position and center well
        mental_distance_from_center_well = np.abs(get_ahead_behind_distance(
            track_graph, *trajectory_data, source=center_well_id))
        #add mental_distance_from_center_well to cv_result, which is a xarray.Dataset
        cv_result['mental_distance_from_center_well'] = xr.DataArray(
            mental_distance_from_center_well, dims=['time'])
        
        cv_classifier_clusterless_results.append(cv_result)
        
    cv_classifier_clusterless_results = xr.concat(cv_classifier_clusterless_results, dim="time")   
    
    logging.info('Saving results...')
    #save classification results
    cv_classifier_clusterless_results.to_netcdf(
        os.path.join(
            PROCESSED_DATA_DIR, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_cv_classifier_clusterless_results.nc'
        )
    )
    
    if save_original_data:
        #save data. Be careful, the data is too large. Only saved for animal bon for later visualization purpose
        with open(os.path.join(
                PROCESSED_DATA_DIR, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
    #get speed info from data
    speed_info = data['position_info'].speed
    #save the speed_info, which is pandas.core.series.Series
    speed_info.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_speed_info.pkl'))
    

def clusterless_sleep_replay(sleep_epoch_key,
                             prev_run_epoch_key, 
                             exclude_interneuron_spikes=True,
                             brain_areas=None):
    '''
    created by Zilong 11/09/2023
    Decoding sleep replay using clusterless classifier from the previous running epoch
    For example, if we want to decode sleep replay for bon, 3, 3, which is a sleep spoch
    We will use the classifier from bon, 3, 2, which is a running epoch. The classifier 
    has been saved to Processed-data
    '''
    DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'TrueSleepReplayTrajectories')

    animal, day, epoch = sleep_epoch_key
    data_type, dim = 'clusterless', '1D'

    logging.info('Loading sleeping data...')
    
    #try to load sleep data, if there is error in loading sleep data, then skip this epoch
    #for exampel, con,1,3 raise an error 'IndexError: Item wrong length 429 instead of 430'..
    #need to fix this later, but skip now
    try:
        data = load_sleep_data(sleep_epoch_key,
                               brain_areas=brain_areas,
                               exclude_interneuron_spikes=exclude_interneuron_spikes)
    except:
        logging.info('Error in loading sleep data. Skip this epoch...')
        return
    
    group = f'/{data_type}/{dim}/'
    if exclude_interneuron_spikes:
        group += 'no_interneuron/'
    group += 'classifier/ripples/'
    
    logging.info('Get the potential sleep intervals...')
    
    is_test, valid_durations, valid_intervals = detect_sleep_periods(data, sleep_epoch_key, 
                                                        lowspeed_thres=4, lowspeed_duration=60,
                                                        theta2alpha_thres=1.5, REM_duration=10,
                                                        sleep_duration=90, LIA_duration=5,
                                                        plot=True, figdir=DATA_DIR)
    
    #speed = data['position_info'].speed
    #is_test, valid_durations, valid_intervals = find_sleep_intervals(speed, sleep_duration_threshold=180)
    
    #if valid_durations is empty, then print out the message and skip this epoch
    #save valid_durations to a pickle file
    with open(os.path.join(DATA_DIR, f'{animal}_{day:02d}_{epoch:02d}_valid_durations.pkl'), 'wb') as f:
        pickle.dump(valid_durations, f)

    if not valid_durations:
        logging.info('No valid sleep intervals found. Skip this epoch...')
        return
    else:
        print(valid_durations)
        logging.info(f'Found {len(valid_durations)} valid sleep intervals...')
        
        #for data["ripple_times"], filter those intervals within valid_intervals
        sleep_ripple_times = get_sleep_ripples(data["ripple_times"], valid_intervals)
            
        logging.info('Loading classifier from the previous running epoch...')
        prev_animal, prev_day, prev_epoch = prev_run_epoch_key
        model_name = os.path.join(
        PROCESSED_DATA_DIR,
        "ReplayTrajectories",
        (f"{prev_animal}_{prev_day:02d}_{prev_epoch:02d}_clusterless_1D_no_interneuron_model.pkl"),
        )   
        
        classifier = ClusterlessClassifier.load_model(model_name)
        
        logging.info('Decoding sleep replay...')
        
        test_groups = pd.DataFrame(
            {"test_groups": label(is_test.values)[0]}, index=is_test.index
        )
        
        immobility_results = []
        for _, df in tqdm(test_groups.loc[is_test].groupby("test_groups"), desc="immobility"):
            start_time, end_time = df.iloc[0].name, df.iloc[-1].name
            test_multiunit = data["multiunit"].sel(time=slice(start_time, end_time))
            immobility_results.append(
                classifier.predict(test_multiunit, time=test_multiunit.time)
            )

        immobility_results = xr.concat(immobility_results, dim="time")
        
        results = [
            (
                immobility_results.sel(time=slice(df.start_time, df.end_time)).assign_coords(
                    time=lambda ds: ds.time - ds.time[0]
                )
            )
            for _, df in sleep_ripple_times.iterrows()
        ]
        
        results = xr.concat(results, dim=sleep_ripple_times.index).assign_coords(
            state=lambda ds: ds.state.to_index().map(TRANSITION_TO_CATEGORY)
        )
      
        #########################
        logging.info('Saving results...')
        save_xarray(DATA_DIR, sleep_epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=group)
        
        #get replay trajectory
        track_graph, center_well_id = make_track_graph(prev_run_epoch_key, ANIMALS)
        #add three zero columns to data["position_info"], 
        # projected_x_position, projected_y_position, track_segment_id
        # since it is an open field
        #so there is no projected position, but it is necessary for geting the replay trajectory
        
        data["position_info"].loc[:, "projected_x_position"] = 0.0
        data["position_info"].loc[:, "projected_y_position"] = 0.0
        data["position_info"].loc[:, "track_segment_id"] = 0.0
        
        replay_trajectories = get_replay_traj(
            results, sleep_ripple_times, data['position_info'],
            track_graph, classifier)
        
        with open(os.path.join(DATA_DIR, f'{animal}_{day:02d}_{epoch:02d}_traj.pkl'), 'wb') as f:
            pickle.dump(replay_trajectories, f)  

def clusterless_analysis_1D(epoch_key,
                            plot_ripple_figures=False,
                            exclude_interneuron_spikes=False,
                            use_multiunit_HSE=False,
                            brain_areas=None,
                            overwrite=False):
    
    DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'ReplayTrajectories')
    
    animal, day, epoch = epoch_key
    data_type, dim = 'clusterless', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     brain_areas=brain_areas,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)

    is_training = data['position_info'].speed > 4
    position = data['position_info'].loc[:, 'linear_position']
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    # Set up naming
    group = f'/{data_type}/{dim}/'
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'

    if exclude_interneuron_spikes:
        logging.info('Excluding interneuron spikes...')
        epoch_identifier += '_no_interneuron'
        group += 'no_interneuron/'

    if brain_areas is not None:
        area_str = '-'.join(brain_areas)
        epoch_identifier += f'_{area_str}'
        group += f'{area_str}/'

    if use_multiunit_HSE:
        epoch_identifier += '_multiunit_HSE'
        group += 'classifier/multiunit_HSE/'
        replay_times = data['multiunit_HSE_times']
    else:
        group += 'classifier/ripples/'
        replay_times = data['ripple_times']

    model_name = os.path.join(
        DATA_DIR, epoch_identifier + '_model.pkl')

    try:
        if overwrite:
            raise FileNotFoundError
        results = xr.open_dataset(
            os.path.join(
                DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
            group=group)
        logging.info('Found existing results. Loading...')
        spikes = (((data['multiunit'].sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(
            spikes, replay_times.loc[:, ['start_time', 'end_time']])
        classifier = ClusterlessClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        logging.info('Fitting classifier...')
        classifier = ClusterlessClassifier(
            place_bin_size=place_bin_size, movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            continuous_transition_types=continuous_transition_types,
            model=model, model_kwargs=model_kwargs).fit(
                position, data['multiunit'], is_training=is_training,
                track_graph=track_graph, center_well_id=center_well_id,
                edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING)
        classifier.save_model(model_name)
        logging.info(classifier)

        # Decode
        is_test = ~is_training

        test_groups = pd.DataFrame(
            {'test_groups': label(is_test.values)[0]}, index=is_test.index)
        immobility_results = []
        for _, df in tqdm(test_groups.loc[is_test].groupby('test_groups'),
                          desc='immobility'):
            start_time, end_time = df.iloc[0].name, df.iloc[-1].name
            test_multiunit = data['multiunit'].sel(
                time=slice(start_time, end_time))
            immobility_results.append(
                classifier.predict(test_multiunit, time=test_multiunit.time))

        immobility_results = xr.concat(immobility_results, dim='time')

        results = [(immobility_results
                    .sel(time=slice(df.start_time, df.end_time))
                    .assign_coords(time=lambda ds: ds.time - ds.time[0]))
                   for _, df in replay_times.iterrows()]

        results = (xr.concat(results, dim=replay_times.index)
                   .assign_coords(state=lambda ds: ds.state.to_index()
                                  .map(TRANSITION_TO_CATEGORY)))

        spikes = ((((~np.isnan(data['multiunit'])).sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(
            spikes, replay_times.loc[:, ['start_time', 'end_time']])

        logging.info('Saving results...')
        save_xarray(DATA_DIR, epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=group)
        
        #get replay trajectory
        replay_trajectories = get_replay_traj(
            results, replay_times, data['position_info'],
            track_graph, classifier)
        
        with open(os.path.join(DATA_DIR, f'{animal}_{day:02d}_{epoch:02d}_traj.pkl'), 'wb') as f:
            pickle.dump(replay_trajectories, f)  

    logging.info('Saving replay_info...')
    replay_info = get_replay_info(
        results, spikes, replay_times, data['position_info'],
        track_graph, SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, epoch_key,
        classifier, data["ripple_consensus_trace_zscore"])
    prob = int(PROBABILITY_THRESHOLD * 100)
    replay_info_filename = os.path.join(
        DATA_DIR, f'{epoch_identifier}_replay_info_{prob:02d}.csv')
    replay_info.to_csv(replay_info_filename)

    if plot_ripple_figures:
        logging.info('Plotting ripple figures...')
        place_field_max = get_place_field_max(classifier)
        linear_position_order = place_field_max.argsort(axis=0).squeeze()
        ripple_position = reshape_to_segments(
            position, replay_times.loc[:, ['start_time', 'end_time']])

        for ripple_number in tqdm(replay_times.index, desc='ripple figures'):
            try:
                posterior = (
                    results
                    .acausal_posterior
                    .sel(ripple_number=ripple_number)
                    .dropna('time', how='all')
                    .assign_coords(
                        time=lambda ds: 1000 * ds.time /
                        np.timedelta64(1, 's')))
                plot_ripple_decode_1D(
                    posterior, ripple_position.loc[ripple_number],
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], classifier, spike_label='Tetrodes')
                plt.suptitle(
                    f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
                    f'{ripple_number:04d}')
                fig_name = (
                    f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...\n')


run_analysis = {
    ('sorted_spikes', '1D'): sorted_spikes_analysis_1D,
    ('clusterless', '1D'): clusterless_analysis_1D,
}


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--data_type', type=str, default='sorted_spikes')
    parser.add_argument('--dim', type=str, default='1D')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--threads_per_worker', type=int, default=1)
    parser.add_argument('--plot_ripple_figures', action='store_true')
    parser.add_argument('--exclude_interneuron_spikes', action='store_true')
    parser.add_argument('--use_multiunit_HSE', action='store_true')
    parser.add_argument('--CA1', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()

def main():
    args = get_command_line_arguments()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)

    def _signal_handler(signal_code, frame):
        logging.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    logging.info(f'Data type: {args.data_type}, Dim: {args.dim}')
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    if args.CA1:
        brain_areas = ['CA1']
    else:
        brain_areas = None

    # Analysis Code
    run_analysis[(args.data_type, args.dim)](
        epoch_key,
        plot_ripple_figures=args.plot_ripple_figures,
        exclude_interneuron_spikes=args.exclude_interneuron_spikes,
        use_multiunit_HSE=args.use_multiunit_HSE,
        brain_areas=brain_areas,
        overwrite=args.overwrite)


if __name__ == '__main__':
    sys.exit(main())
