#%%
import os
import pdb
import pickle
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, _BRAIN_AREAS)

from src.load_data import load_data

from scripts.run_by_epoch import clusterless_thetasweeps
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
    is_animal = epoch_info.index.isin(['bon', 'fra', 'gov', 'dud', 'con', 'Cor', 'dav', 'egy', 'cha'], level='animal')
    
    valid_epochs = (is_w_track &
                    (epoch_info.n_neurons > MIN_N_NEURONS) &
                    is_animal
                    )
    
    #%%
    DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    for epoch_key in tqdm(epoch_info[valid_epochs].index, desc='epochs'):
        animal, day, epoch = epoch_key
              
        # Check if this file has already been run
        spike_info_filename = os.path.join(
            DATA_DIR,
            'ThetaSweepTrajectories',
            f'{animal}_{day:02d}_{epoch:02d}_spikeinfo.pkl')

        if not os.path.isfile(spike_info_filename):
            animal, day, epoch = epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            
            data = load_data(epoch_key, brain_areas=None, exclude_interneuron_spikes=True)
            spikes = data['spikes']
            multiunit = data['multiunit']
            multiunit_fr = data['multiunit_firing_rate']

            spike_info_filename = os.path.join(
                DATA_DIR,
                'ThetaSweepTrajectories',
                f'{animal}_{day:02d}_{epoch:02d}_spike_info.pkl')
            
            #save spikes, multiunit and multiunit_fr intop one pkl
            with open(spike_info_filename, 'wb') as f:
                pickle.dump([spikes, multiunit, multiunit_fr], f)

if __name__ == '__main__':
    main()