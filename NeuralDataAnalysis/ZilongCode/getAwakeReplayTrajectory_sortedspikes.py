#%%
import os
import pdb
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MAX_N_EXPOSURES, MIN_N_NEURONS)

from scripts.run_by_epoch import clusterless_analysis_1D, sorted_spikes_analysis_1D
from tqdm.auto import tqdm

def main():
    epoch_info = make_epochs_dataframe(ANIMALS)
    neuron_info = make_neuron_dataframe(ANIMALS)
    n_neurons = (neuron_info
                    .groupby(['animal', 'day', 'epoch'])
                    .neuron_id
                    .agg(len)
                    .rename('n_neurons')
                    .to_frame())

    epoch_info = epoch_info.join(n_neurons)
    is_w_track = (epoch_info.environment
                    .isin(['TrackA', 'TrackB', 'WTrackA', 'WTrackB']))
    #is_animal = epoch_info.index.isin(['bon', 'fra', 'gov', 'dud', 'con'], level='animal')
    is_animal = epoch_info.index.isin(['bon'], level='animal')
    valid_epochs = (is_w_track &
                    (epoch_info.n_neurons > MIN_N_NEURONS) &
                    (epoch_info.exposure <= MAX_N_EXPOSURES) &
                    is_animal
                    )
    log_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_directory,  exist_ok=True)

    
    
    #%%
    PROCESSED_DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    for epoch_key in tqdm(epoch_info[valid_epochs].index, desc='epochs'):
        animal, day, epoch = epoch_key
        # Check if this file has already been run
        replay_info_filename = os.path.join(
            PROCESSED_DATA_DIR,
            f'{animal}_{day:02d}_{epoch:02d}_sorted_spikes_1D_replay_info_80.csv')
        if not os.path.isfile(replay_info_filename):
            #pdb.set_trace()
            animal, day, epoch = epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            
            sorted_spikes_analysis_1D(epoch_key, brain_areas=None) #set brain_areas to ['CA1'] if only care about CA1
    # %%

if __name__ == '__main__':
    main()