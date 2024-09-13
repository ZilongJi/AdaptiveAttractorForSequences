#%%
import os
import pdb
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, _BRAIN_AREAS)

from scripts.run_by_epoch import clusterless_analysis_1D, sorted_spikes_analysis_1D
from tqdm.auto import tqdm

def main():
    #get epoch_key for remy, 35, 2; remy, 35, 4; remy, 36, 2; remy, 36, 4; remy, 36, 2; remy, 37, 2; remy, 37, 4; 
    #and put them in a list 
    epoch_keys = [('remy', 35, 2), ('remy', 35, 4), ('remy', 36, 2), ('remy', 36, 4), ('remy', 37, 2), ('remy', 37, 4)]
    
    #%%
    PROCESSED_DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    for epoch_key in tqdm(epoch_keys):
        animal, day, epoch = epoch_key                 
        
        # Check if this file has already been run
        replay_info_filename = os.path.join(
            PROCESSED_DATA_DIR,
            'ReplayTrajectories',
            f'{animal}_{day:02d}_{epoch:02d}_traj.pkl')
        if not os.path.isfile(replay_info_filename):
            animal, day, epoch = epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            clusterless_analysis_1D(epoch_key, 
                                    exclude_interneuron_spikes=True,
                                    brain_areas=None) #set brain_areas to ['CA1'] if only care about CA1

if __name__ == '__main__':
    main()