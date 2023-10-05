#%%
import os
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, _BRAIN_AREAS)
from src.load_data import get_sleep_and_prev_run_epochs

from scripts.run_by_epoch import clusterless_sleep_replay
from tqdm.auto import tqdm

def main():
    epoch_info = make_epochs_dataframe(ANIMALS)
    neuron_info = make_neuron_dataframe(ANIMALS)
    
    neuron_info = neuron_info.loc[
    (neuron_info.type == 'principal') &
    (neuron_info.numspikes > 100) &
    neuron_info.area.isin(_BRAIN_AREAS)]
    
    n_neurons = (neuron_info
                    .groupby(['animal', 'day', 'epoch'])
                    .neuron_id
                    .agg(len)
                    .rename('n_neurons')
                    .to_frame())

    epoch_info = epoch_info.join(n_neurons)
    
    # select only sleep epochs
    is_sleep = (epoch_info.type.isin(['sleep']))
    
    is_animal = epoch_info.index.isin(['bon', 'fra', 'gov', 'dud', 'con', 'dav', 'Cor', 'egy', 'cha'], level='animal')

    #get valid epochs with is_sleep and is_animal and n_neurons > MIN_N_NEURONS
    valid_epochs =  epoch_info.loc[is_sleep & 
                                   is_animal & 
                                   (epoch_info.n_neurons > MIN_N_NEURONS)]

    sleep_epoch_keys, prev_run_epoch_keys = get_sleep_and_prev_run_epochs()

    # get valid sleep epochs with keys only in sleep_epoch_keys
    valid_sleep_epochs = valid_epochs.loc[valid_epochs.index.isin(sleep_epoch_keys)]
    
    PROCESSED_DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    for sleep_epoch_key in tqdm(valid_sleep_epochs.index, desc='epochs'):
        #current sleep epoch
        animal, day, epoch = sleep_epoch_key    
        
        #if sleep_epoch_key is egy,11,7, then skip this epoch
        if sleep_epoch_key == ('egy', 11, 7):
            continue             
        
        #get revious run epoch
        prev_run_epoch_key = prev_run_epoch_keys[sleep_epoch_keys.index(sleep_epoch_key)]
        prev_animal, prev_day, prev_epoch = prev_run_epoch_key
        #if model_name do not exist, then print out the message and skip this epoch
        model_name = os.path.join(
        PROCESSED_DATA_DIR,
        "ReplayTrajectories",
        (f"{prev_animal}_{prev_day:02d}_{prev_epoch:02d}_clusterless_1D_no_interneuron_model.pkl"),
        )  
        #skip this epoch if model_name do not exist, this is because some filtering criteria are not met
        #for example, n_neurons < MIN_N_NEURONS during running trials
        if not os.path.isfile(model_name):
            continue
        
        # Check if this file has already been run
        replay_info_filename = os.path.join(
            PROCESSED_DATA_DIR,
            'TrueSleepReplayTrajectories',
            f'{animal}_{day:02d}_{epoch:02d}_valid_durations.pkl')
        
        if not os.path.isfile(replay_info_filename):
            animal, day, epoch = sleep_epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            clusterless_sleep_replay(sleep_epoch_key, 
                                     prev_run_epoch_key,
                                     exclude_interneuron_spikes=True,
                                     brain_areas=None)

if __name__ == '__main__':
    main()