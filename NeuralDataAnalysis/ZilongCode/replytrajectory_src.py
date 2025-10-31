import logging
import os

import xarray as xr
from loren_frank_data_processing.position import make_track_graph
from replay_trajectory_classification import (ClusterlessClassifier, SortedSpikesClassifier)
from src.analysis import get_replay_traj
from src.load_data import load_data
from src.parameters import (ANIMALS, PROCESSED_DATA_DIR)


def get_replay_traj_from_sorted_spikes( epoch_key,
                                        exclude_interneuron_spikes=False,
                                        use_multiunit_HSE=False,
                                        brain_areas=None):
    animal, day, epoch = epoch_key
    data_type, dim = 'sorted_spikes', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)
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
    
    results = xr.open_dataset(
        os.path.join(
            PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
        group=group)
    
    logging.info('Found existing results. Loading...')
    classifier = SortedSpikesClassifier.load_model(model_name)
    logging.info(classifier)

    replay_trajectories = get_replay_traj(
        results, replay_times, data['position_info'],
        track_graph, classifier)


    return replay_trajectories
    
def get_replay_traj_from_clusterless(epoch_key,
                                     exclude_interneuron_spikes=False,
                                     use_multiunit_HSE=False,
                                     brain_areas=None):
    
    animal, day, epoch = epoch_key
    data_type, dim = 'clusterless', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     brain_areas=brain_areas,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)

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

    #load processed data
    results = xr.open_dataset(
        os.path.join(
            PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
        group=group)
    logging.info('Found existing results. Loading...')

    classifier = ClusterlessClassifier.load_model(model_name)
    logging.info(classifier)

    #get replay trajectory
    replay_trajectories = get_replay_traj(
        results, replay_times, data['position_info'],
        track_graph, classifier)

    return replay_trajectories