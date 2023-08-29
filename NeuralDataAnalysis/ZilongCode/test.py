import sys
from scripts.run_by_epoch import clusterless_analysis_1D, sorted_spikes_analysis_1D

def main():
    epoch_key = 'bon', 3, 4
    sorted_spikes_analysis_1D(epoch_key, brain_areas=None)
    
if __name__ == '__main__':
    sys.exit(main())