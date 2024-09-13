#%%
import os
import pdb
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, _BRAIN_AREAS)
from src.load_data import get_sleep_and_prev_run_epochs

from scripts.run_by_epoch import get_fr_in_awake_subsequentsleep_ripples
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
    
    result_folder = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    for sleep_epoch_key in tqdm(valid_sleep_epochs.index, desc='epochs'):
        #current sleep epoch
        animal, day, epoch = sleep_epoch_key    
        
        #if sleep_epoch_key is egy,11,7, then skip this epoch
        if sleep_epoch_key == ('egy', 11, 7):
            continue             
        
        #get revious run epoch
        prev_run_epoch_key = prev_run_epoch_keys[sleep_epoch_keys.index(sleep_epoch_key)]
        prev_animal, prev_day, prev_epoch = prev_run_epoch_key
        
        # Check if this file has already been run
        processed_data_filename = os.path.join(
            result_folder,
            'awake_sleep_firingrate_in_ripples',
            f'{prev_animal}_{prev_day:02d}_{prev_epoch:02d}_firing_rate_awake_sleep.pkl')
        
        if not os.path.isfile(processed_data_filename):
            animal, day, epoch = sleep_epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            get_fr_in_awake_subsequentsleep_ripples(sleep_epoch_key, 
                                    prev_run_epoch_key,
                                    result_folder,
                                    exclude_interneuron_spikes=True,
                                    brain_areas=None)

if __name__ == '__main__':
    main()