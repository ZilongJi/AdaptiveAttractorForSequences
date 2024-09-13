#%% This code is not working so far. Need to be debugged.
import os
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, _BRAIN_AREAS)
from src.load_data import get_sleep_and_prev_run_epochs

from scripts.run_by_epoch import clusterless_sleep_replay
from tqdm.auto import tqdm

def main():
    PROCESSED_DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    sleep_epoch_key = ('bon', 8, 5)  # Animal, day, epoch
    prev_run_epoch_key = ('bon', 8, 4)  # Animal, day, epoch
    #current sleep epoch
    animal, day, epoch = sleep_epoch_key    
        
    prev_animal, prev_day, prev_epoch = prev_run_epoch_key
    #if model_name do not exist, then print out the message and skip this epoch
    model_name = os.path.join(
    PROCESSED_DATA_DIR,
    "ReplayTrajectories",
    (f"{prev_animal}_{prev_day:02d}_{prev_epoch:02d}_clusterless_1D_no_interneuron_model.pkl"),
    )  
    
    # Check if this file has already been run
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR,
        'bon_8_5',
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