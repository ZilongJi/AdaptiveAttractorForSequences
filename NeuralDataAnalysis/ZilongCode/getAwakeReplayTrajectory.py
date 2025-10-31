#%%
import os
import pdb
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, _BRAIN_AREAS)

from scripts.run_by_epoch import clusterless_analysis_1D, sorted_spikes_analysis_1D
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
    
    is_w_track = (epoch_info.environment
                    .isin(['TrackA', 'TrackB', 'WTrackA', 'WTrackB']))
    
    is_animal = epoch_info.index.isin(['bon', 'fra', 'gov', 'dud', 'con', 'dav', 'Cor', 'egy', 'cha'], level='animal')

    valid_epochs = (is_w_track &
                    (epoch_info.n_neurons > MIN_N_NEURONS) &
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
            'ReplayTrajectories',
            f'{animal}_{day:02d}_{epoch:02d}_clusterless_1D_no_interneuron_replay_info_80.csv')
        if not os.path.isfile(replay_info_filename):
            #pdb.set_trace()
            animal, day, epoch = epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            #pdb.set_trace()
            clusterless_analysis_1D(epoch_key, 
                                    exclude_interneuron_spikes=True,
                                    brain_areas=None) #set brain_areas to ['CA1'] if only care about CA1
    # %%

if __name__ == '__main__':
    main()