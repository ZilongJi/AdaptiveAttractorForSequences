import os
import pdb
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import matplotlib.gridspec as gridspec
import pickle
import xarray as xr

def get_zero_phase(filtered_lfp):
    """
    get the zero phase of the filtered lfp using hilbert transform
    """
    filtered_lfp = filtered_lfp.values.squeeze()
    analytic_signal = hilbert(filtered_lfp)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # wrap the instantaneous_phase to -pi and pi
    instantaneous_phase = np.mod(instantaneous_phase + np.pi, 2 * np.pi) - np.pi
    # find phase of 0 and mark all of them as a red point on the plot
    zero_phase = np.where(
        np.logical_and(instantaneous_phase > -3.15, instantaneous_phase < -3.0)
    )[0]

    # if successive zero phase is less than 10, only keep the first one
    splits = np.split(zero_phase, np.where(np.diff(zero_phase) > 10)[0] + 1)

    # rremove empty in splits
    splits = [split for split in splits if len(split) > 0]

    zero_phase_index = np.array([split[0] for split in splits])

    return zero_phase_index

def bandpassfilter(data, lowcut=5, highcut=11, fs=500):
    """
    band pass filter of the signal
    Created by Zilong, 30/08/2021
    """
    lowcut = 5
    highcut = 11
    order = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    filtereddata = signal.filtfilt(b, a, data)

    return filtereddata

def get_steps(Replay_traj, timeb=1):
    '''
    Get the levy exponent for replay trajectory
    Input 
        (dictionary): Replay_traj
        (int): time bin
    Output:
        (numpy array): all_steps for all ripple events
    
    '''
    #initialze an empty numpy array
    all_steps = []

    keys = Replay_traj.keys()
    for key in keys:
        #get the replay trajectory for each ripple events
        ripple_traj = Replay_traj[key]
        traj_step = np.abs(np.diff(ripple_traj[::timeb]))
        #concatenate the steps for all ripple events
        all_steps = np.concatenate((all_steps, traj_step))
    
    return all_steps

#log-log plot of the average step size against time bins
def get_diffusion_exponent(Replay_traj, plot=False, get_intercept=False): 
    
    #return plt
    all_timebins = np.arange(1,20,1)
    all_avg_steps = []
    all_std_steps = []
    for timeb in all_timebins:
        all_steps = get_steps(Replay_traj, timeb)
        #get the average step size for each time bin
        avg_step = np.mean(all_steps)
        std_step = np.std(all_steps)
        
        all_avg_steps.append(avg_step)
        all_std_steps.append(std_step)
    #get the slope of the log-log plot
    slope, intercept = np.polyfit(np.log(all_timebins), np.log(all_avg_steps), 1)

    #print('The slope of the log-log plot is %.2f'%slope)
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(3,3))
        #plot the log-log plot and add the slope as a lagend
        ax.plot(all_timebins, all_avg_steps, 'o', color='black')
        #add shadow plot of the std
        #ax.fill_between(all_timebins, np.array(all_avg_steps), np.array(all_avg_steps), color='black', alpha=0.2)
        ax.plot(all_timebins, np.exp(intercept)*all_timebins**slope, 'r-', label='slope = %.2f'%slope)  
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time bins')
        ax.set_ylabel('Average step size')
        ax.legend()
        #set both ticks to [0,10]
        ax.set_xticks([1,10])
        ax.set_yticks([1,10])
        if get_intercept:
            return fig, ax, slope, intercept
        else:
            return fig, ax, slope
    else:
        if get_intercept:
            return slope, intercept
        else:
            return slope
        
def get_sweep_and_awake_replay_results(epoch_key, datadir, sweep_speed_threshold=4):
    '''
    Get the average sweep length and replay exponent for one epoch
    '''
    animal, day, epoch = epoch_key
    
    #1, LOAD THETA SWEEPS RESULTS
    #load the classifier results
    cv_classifier_clusterless_results = xr.open_dataset(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_cv_classifier_clusterless_results.nc'))
    #load the data with pickle
    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_speed_info.pkl'), 'rb') as f:
        speed_dic = pickle.load(f)
        speed = speed_dic.values
        
    #from speed, get the moving distance by multiplying speed with the time which is the key of speed
    speedkey = speed_dic.keys()
    time = speedkey/np.timedelta64(1, "s")
    tifmediff = np.diff(time)
    #append zero to the beginning of tifmediff
    tifmediff = np.insert(tifmediff, 0, 0)
    distance  = {}
    dist = np.cumsum(speed*tifmediff)
    #downsampling the dist to 1/50, that is 0.1 second
    dist = dist[::50]
    distance[0] = dist
    behavior_slope =  get_diffusion_exponent(distance, plot=False, get_intercept=False)
    print(behavior_slope)
    
    #get the average sweep length
    #extract speed from data and threshold it with 10 cm/s
    is_running = speed > sweep_speed_threshold

    sweeps_dist_in_running = cv_classifier_clusterless_results.mental_distance_from_actual_position[is_running]
    
    #get mean speed with speed < 4 (mean speed udring immobile state)
    mean_immobile_speed = np.mean(speed[speed<4])
    mean_active_speed = np.mean(speed[speed>=4])
    
    mean_sweep_dist = np.mean(sweeps_dist_in_running).values
    
    #2, LOAD REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    with open(os.path.join(datadir, 'ReplayTrajectories', f'{animal}_{day:02d}_{epoch:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    #fig, ax, slope = get_exponent(Replay_traj, plot=True)
    replay_exponent, replay_intercept = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)
        
    return mean_immobile_speed, mean_active_speed, behavior_slope, mean_sweep_dist, replay_exponent, replay_intercept


def get_sweep_and_awake_replay_results_using_postive_negative_peak_in_LFP(epoch_key, datadir, sweep_speed_threshold=4):
    '''
    Get the average sweep length and replay exponent for one epoch
    '''
    animal, day, epoch = epoch_key
    
    #1, LOAD THETA SWEEPS RESULTS
    #load the classifier results
    cv_classifier_clusterless_results = xr.open_dataset(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_cv_classifier_clusterless_results.nc'))
    #load the data with pickle
    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_speed_info.pkl'), 'rb') as f:
        speed_dic = pickle.load(f)
        speed = speed_dic.values
    #load lfp info
    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_lfp_info.pkl'), 'rb') as f:
        lfp = pickle.load(f)
    
    lfp = (
    lfp.reset_index(drop=True).set_index(lfp.index / np.timedelta64(1, "s"))
    )
        
    #from speed, get the moving distance by multiplying speed with the time which is the key of speed
    speedkey = speed_dic.keys()
    time = speedkey/np.timedelta64(1, "s")
    tifmediff = np.diff(time)
    #append zero to the beginning of tifmediff
    tifmediff = np.insert(tifmediff, 0, 0)
    distance  = {}
    dist = np.cumsum(speed*tifmediff)
    #downsampling the dist to 1/50, that is 0.1 second
    dist = dist[::50]
    distance[0] = dist
    behavior_slope =  get_diffusion_exponent(distance, plot=False, get_intercept=False)
    
    #get the average sweep length
    #extract speed from data and threshold it with 10 cm/s
    is_running = speed > sweep_speed_threshold

    sweeps_dist_in_running = cv_classifier_clusterless_results.mental_distance_from_actual_position[is_running]

    
    #get mean speed with speed < 4 (mean speed udring immobile state)
    mean_immobile_speed = np.mean(speed[speed<4])
    mean_active_speed = np.mean(speed[speed>=4])
    
    mean_sweep_dist = np.mean(sweeps_dist_in_running).values
    
    
    # add band passs filter to lfp.iloc[:,0], using band between 5-11Hz
    lfp0 = bandpassfilter(lfp.iloc[:, 0], lowcut=5, highcut=11, fs=500)
    lfp0 = pd.DataFrame(lfp0, index=lfp.index)
    
    #filter the lfp0 using is_running
    lfp0_is_running = lfp0[is_running]
    
    zero_phase_index = get_zero_phase(lfp0_is_running)  
    
    sweeps_dist_in_running_ahead_behind = cv_classifier_clusterless_results.mental_distance_from_actual_position_ahead_behind[is_running]
    
    #for two successive zero phases, find the peak value from sweeps_dist_in_running_ahead_behind in that interval
    sweep_amp = []
    sweep_ahead = []
    sweep_behind = []
    for i in range(len(zero_phase_index)-1):
        #get the start and end index of the current zero phase
        start_index = zero_phase_index[i]
        end_index = zero_phase_index[i+1]
        #get the peak value in the current interval
        ahead_peak = np.max(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        behind_peak = np.min(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        sweep_amp.append(ahead_peak-behind_peak)
        sweep_ahead.append(ahead_peak)
        sweep_behind.append(behind_peak)
        
    mean_sweep_amp = np.mean(sweep_amp)
    mean_sweep_ahead = np.mean(sweep_ahead)
    mean_sweep_behind = np.mean(sweep_behind)
    
    sweep_mass = np.mean(sweeps_dist_in_running_ahead_behind).values
    
    #if successive zero phases are both in      
    
    # group the lfp0 into dataframe with time index
    lfp0 = pd.DataFrame(lfp0, index=lfp.index)
    
    #2, LOAD REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    with open(os.path.join(datadir, 'ReplayTrajectories', f'{animal}_{day:02d}_{epoch:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    #fig, ax, slope = get_exponent(Replay_traj, plot=True)
    replay_exponent, replay_intercept = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)
    
    #return mean_immobile_speed, mean_active_speed, behavior_slope, mean_sweep_dist, mean_sweep_amp, mean_sweep_ahead, mean_sweep_behind, sweep_mass, replay_exponent, replay_intercept in a list
    
    results = [mean_immobile_speed, 
               mean_active_speed, 
               behavior_slope, 
               mean_sweep_dist, 
               mean_sweep_amp, 
               mean_sweep_ahead, 
               mean_sweep_behind, 
               sweep_mass, 
               replay_exponent, 
               replay_intercept]
    
    return results

def get_sweep_and_shuffled_awake_replay_results_using_postive_negative_peak_in_LFP(epoch_key, sweepdatadir, replaydatadir, sweep_speed_threshold=4):
    '''
    Get the average sweep length and replay exponent for one epoch
    '''
    animal, day, epoch = epoch_key
    
    #1, LOAD THETA SWEEPS RESULTS
    #load the classifier results
    cv_classifier_clusterless_results = xr.open_dataset(os.path.join(sweepdatadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_cv_classifier_clusterless_results.nc'))
    #load the data with pickle
    with open(os.path.join(sweepdatadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_speed_info.pkl'), 'rb') as f:
        speed_dic = pickle.load(f)
        speed = speed_dic.values
    #load lfp info
    with open(os.path.join(sweepdatadir, 'ThetaSweepTrajectories', f'{animal}_{day:02d}_{epoch:02d}_lfp_info.pkl'), 'rb') as f:
        lfp = pickle.load(f)
    
    lfp = (
    lfp.reset_index(drop=True).set_index(lfp.index / np.timedelta64(1, "s"))
    )
        
    #from speed, get the moving distance by multiplying speed with the time which is the key of speed
    speedkey = speed_dic.keys()
    time = speedkey/np.timedelta64(1, "s")
    tifmediff = np.diff(time)
    #append zero to the beginning of tifmediff
    tifmediff = np.insert(tifmediff, 0, 0)
    distance  = {}
    dist = np.cumsum(speed*tifmediff)
    #downsampling the dist to 1/50, that is 0.1 second
    dist = dist[::50]
    distance[0] = dist
    behavior_slope =  get_diffusion_exponent(distance, plot=False, get_intercept=False)
    
    #get the average sweep length
    #extract speed from data and threshold it with 10 cm/s
    is_running = speed > sweep_speed_threshold

    sweeps_dist_in_running = cv_classifier_clusterless_results.mental_distance_from_actual_position[is_running]

    
    #get mean speed with speed < 4 (mean speed udring immobile state)
    mean_immobile_speed = np.mean(speed[speed<4])
    mean_active_speed = np.mean(speed[speed>=4])
    
    mean_sweep_dist = np.mean(sweeps_dist_in_running).values
    
    
    # add band passs filter to lfp.iloc[:,0], using band between 5-11Hz
    lfp0 = bandpassfilter(lfp.iloc[:, 0], lowcut=5, highcut=11, fs=500)
    lfp0 = pd.DataFrame(lfp0, index=lfp.index)
    
    #filter the lfp0 using is_running
    lfp0_is_running = lfp0[is_running]
    
    zero_phase_index = get_zero_phase(lfp0_is_running)  
    
    sweeps_dist_in_running_ahead_behind = cv_classifier_clusterless_results.mental_distance_from_actual_position_ahead_behind[is_running]
    
    #for two successive zero phases, find the peak value from sweeps_dist_in_running_ahead_behind in that interval
    sweep_amp = []
    sweep_ahead = []
    sweep_behind = []
    for i in range(len(zero_phase_index)-1):
        #get the start and end index of the current zero phase
        start_index = zero_phase_index[i]
        end_index = zero_phase_index[i+1]
        #get the peak value in the current interval
        ahead_peak = np.max(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        behind_peak = np.min(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        sweep_amp.append(ahead_peak-behind_peak)
        sweep_ahead.append(ahead_peak)
        sweep_behind.append(behind_peak)
        
    mean_sweep_amp = np.mean(sweep_amp)
    mean_sweep_ahead = np.mean(sweep_ahead)
    mean_sweep_behind = np.mean(sweep_behind)
    
    sweep_mass = np.mean(sweeps_dist_in_running_ahead_behind).values
    
    #if successive zero phases are both in      
    
    # group the lfp0 into dataframe with time index
    lfp0 = pd.DataFrame(lfp0, index=lfp.index)
    
    #2, LOAD REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    with open(os.path.join(replaydatadir, 'ReplayTrajectories', f'{animal}_{day:02d}_{epoch:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    #fig, ax, slope = get_exponent(Replay_traj, plot=True)
    replay_exponent, replay_intercept = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)
    
    #return mean_immobile_speed, mean_active_speed, behavior_slope, mean_sweep_dist, mean_sweep_amp, mean_sweep_ahead, mean_sweep_behind, sweep_mass, replay_exponent, replay_intercept in a list
    
    results = [mean_immobile_speed, 
               mean_active_speed, 
               behavior_slope, 
               mean_sweep_dist, 
               mean_sweep_amp, 
               mean_sweep_ahead, 
               mean_sweep_behind, 
               sweep_mass, 
               replay_exponent, 
               replay_intercept]
    
    return results

def get_sweep_awake_and_sleep_replay_results(run_epoch_key, sleep_epoch_key, datadir, sweep_speed_threshold=4):
    '''
    Get the average sweep length and replay exponent for one epoch
    '''
    animal_run, day_run, epoch_run = run_epoch_key
    
    #1, LOAD THETA SWEEPS RESULTS
    #load the classifier results
    cv_classifier_clusterless_results = xr.open_dataset(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_cv_classifier_clusterless_results.nc'))
    #load the data with pickle

    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_speed_info.pkl'), 'rb') as f:
        speed_dic = pickle.load(f)
        speed = speed_dic.values
    #load lfp info
    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_lfp_info.pkl'), 'rb') as f:
        lfp = pickle.load(f)
    
    lfp = (
    lfp.reset_index(drop=True).set_index(lfp.index / np.timedelta64(1, "s"))
    )

    #get the average sweep length
    #extract speed from data and threshold it with 10 cm/s
    is_running = speed > sweep_speed_threshold

    # add band passs filter to lfp.iloc[:,0], using band between 5-11Hz
    lfp0 = bandpassfilter(lfp.iloc[:, 0], lowcut=5, highcut=11, fs=500)
    lfp0 = pd.DataFrame(lfp0, index=lfp.index)
    
    #filter the lfp0 using is_running
    lfp0_is_running = lfp0[is_running]
    
    zero_phase_index = get_zero_phase(lfp0_is_running)  
    
    sweeps_dist_in_running_ahead_behind = cv_classifier_clusterless_results.mental_distance_from_actual_position_ahead_behind[is_running]
    
    #for two successive zero phases, find the peak value from sweeps_dist_in_running_ahead_behind in that interval
    sweep_amp = []
    for i in range(len(zero_phase_index)-1):
        #get the start and end index of the current zero phase
        start_index = zero_phase_index[i]
        end_index = zero_phase_index[i+1]
        #get the peak value in the current interval
        ahead_peak = np.max(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        behind_peak = np.min(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        sweep_amp.append(ahead_peak-behind_peak)
        
    mean_sweep_amp = np.mean(sweep_amp)
    
    #2, LOAD awake REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    with open(os.path.join(datadir, 'ReplayTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    awake_replay_exponent, _ = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)
    
    #3, LOAD Sleep REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    animal_sleep, day_sleep, epoch_sleep = sleep_epoch_key
    with open(os.path.join(datadir, 'TrueSleepReplayTrajectories', f'{animal_sleep}_{day_sleep:02d}_{epoch_sleep:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    sleep_replay_exponent, _ = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)

    results = [mean_sweep_amp, 
               awake_replay_exponent,
               sleep_replay_exponent]

    return results


def get_sweep_and_sleep_replay_results_using_postive_negative_peak_in_LFP(run_epoch_key, sleep_epoch_key, datadir, sweep_speed_threshold=4):
    '''
    Get the average sweep length and replay exponent for one epoch
    '''
    animal_run, day_run, epoch_run = run_epoch_key
    
    #1, LOAD THETA SWEEPS RESULTS
    #load the classifier results
    cv_classifier_clusterless_results = xr.open_dataset(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_cv_classifier_clusterless_results.nc'))
    #load the data with pickle

    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_speed_info.pkl'), 'rb') as f:
        speed_dic = pickle.load(f)
        speed = speed_dic.values
    #load lfp info
    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_lfp_info.pkl'), 'rb') as f:
        lfp = pickle.load(f)
    
    lfp = (
    lfp.reset_index(drop=True).set_index(lfp.index / np.timedelta64(1, "s"))
    )
        
    #from speed, get the moving distance by multiplying speed with the time which is the key of speed
    speedkey = speed_dic.keys()
    time = speedkey/np.timedelta64(1, "s")
    tifmediff = np.diff(time)
    #append zero to the beginning of tifmediff
    tifmediff = np.insert(tifmediff, 0, 0)
    distance  = {}
    dist = np.cumsum(speed*tifmediff)
    #downsampling the dist to 1/50, that is 0.1 second
    dist = dist[::50]
    distance[0] = dist
    behavior_slope =  get_diffusion_exponent(distance, plot=False, get_intercept=False)
    
    #get the average sweep length
    #extract speed from data and threshold it with 10 cm/s
    is_running = speed > sweep_speed_threshold

    sweeps_dist_in_running = cv_classifier_clusterless_results.mental_distance_from_actual_position[is_running]

    
    #get mean speed with speed < 4 (mean speed udring immobile state)
    mean_immobile_speed = np.mean(speed[speed<4])
    mean_active_speed = np.mean(speed[speed>=4])
    
    mean_sweep_dist = np.mean(sweeps_dist_in_running).values
    
    
    # add band passs filter to lfp.iloc[:,0], using band between 5-11Hz
    lfp0 = bandpassfilter(lfp.iloc[:, 0], lowcut=5, highcut=11, fs=500)
    lfp0 = pd.DataFrame(lfp0, index=lfp.index)
    
    #filter the lfp0 using is_running
    lfp0_is_running = lfp0[is_running]
    
    zero_phase_index = get_zero_phase(lfp0_is_running)  
    
    sweeps_dist_in_running_ahead_behind = cv_classifier_clusterless_results.mental_distance_from_actual_position_ahead_behind[is_running]
    
    #for two successive zero phases, find the peak value from sweeps_dist_in_running_ahead_behind in that interval
    sweep_amp = []
    sweep_ahead = []
    sweep_behind = []
    for i in range(len(zero_phase_index)-1):
        #get the start and end index of the current zero phase
        start_index = zero_phase_index[i]
        end_index = zero_phase_index[i+1]
        #get the peak value in the current interval
        ahead_peak = np.max(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        behind_peak = np.min(sweeps_dist_in_running_ahead_behind[start_index:end_index])
        sweep_amp.append(ahead_peak-behind_peak)
        sweep_ahead.append(ahead_peak)
        sweep_behind.append(behind_peak)
        
    mean_sweep_amp = np.mean(sweep_amp)
    mean_sweep_ahead = np.mean(sweep_ahead)
    mean_sweep_behind = np.mean(sweep_behind)
    
    sweep_mass = np.mean(sweeps_dist_in_running_ahead_behind).values
    
    #2, LOAD REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    animal_sleep, day_sleep, epoch_sleep = sleep_epoch_key
    with open(os.path.join(datadir, 'TrueSleepReplayTrajectories', f'{animal_sleep}_{day_sleep:02d}_{epoch_sleep:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    #fig, ax, slope = get_exponent(Replay_traj, plot=True)
    sleep_replay_exponent, sleep_replay_intercept = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)

    results = [mean_immobile_speed, 
               mean_active_speed, 
               behavior_slope, 
               mean_sweep_dist, 
               mean_sweep_amp, 
               mean_sweep_ahead, 
               mean_sweep_behind, 
               sweep_mass, 
               sleep_replay_exponent, 
               sleep_replay_intercept]

    return results


def get_sweep_and_sleep_replay_results(run_epoch_key, sleep_epoch_key, datadir, sweep_speed_threshold=4):
    '''
    Get the average sweep length and replay exponent for one epoch
    '''
    animal_run, day_run, epoch_run = run_epoch_key
    
    #1, LOAD THETA SWEEPS RESULTS
    #load the classifier results
    cv_classifier_clusterless_results = xr.open_dataset(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_cv_classifier_clusterless_results.nc'))
    #load the data with pickle
    with open(os.path.join(datadir, 'ThetaSweepTrajectories', f'{animal_run}_{day_run:02d}_{epoch_run:02d}_speed_info.pkl'), 'rb') as f:
        speed = pickle.load(f)
        speed = speed.values
    
    #get the average sweep length
    #extract speed from data and threshold it with 10 cm/s
    is_running = speed > sweep_speed_threshold

    sweeps_dist_in_running = cv_classifier_clusterless_results.mental_distance_from_actual_position[is_running]
    
    #get mean speed with speed < 4 (mean speed udring immobile state)
    mean_immobile_speed = np.mean(speed[speed<4])
    
    mean_sweep_dist = np.mean(sweeps_dist_in_running).values
    
    #2, LOAD REPLAY RESULTS
    #load the dictionary from the 'ReplayTrajectories' folder under PROCESSED_DATA_DIR using pickle
    animal_sleep, day_sleep, epoch_sleep = sleep_epoch_key
    with open(os.path.join(datadir, 'TrueSleepReplayTrajectories', f'{animal_sleep}_{day_sleep:02d}_{epoch_sleep:02d}_traj.pkl'), 'rb') as f:
        Replay_traj = pickle.load(f)
        
    #fig, ax, slope = get_exponent(Replay_traj, plot=True)
    sleep_replay_exponent, sleep_replay_intercept = get_diffusion_exponent(Replay_traj, plot=False, get_intercept=True)
        
    return mean_immobile_speed, mean_sweep_dist, sleep_replay_exponent, sleep_replay_intercept


def bandpassfilter(data, lowcut=5, highcut=11, fs=500):
    """
    band pass filter of the signal
    Created by Zilong, 30/08/2021
    Input:
        data: 1D array
        lowcut: low cut frequency
        highcut: high cut frequency
        fs: sampling frequency
    Output:
        filtereddata: 1D array
    """
    order = 3
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    filtereddata = signal.filtfilt(b, a, data)

    return filtereddata

def find_ones_intervals(data):
    '''
    find all contimuous 1s in x, seperate by 0s before and after
    
    Input:
        data: a list of 0s and 1s   
    return a list of tuples each tuple is a pair of index of 1s
    e.g. [(0, 3), (5, 7), (9, 10)...]
    '''
    ones = []
    start = 0
    end = 0
    for i in range(len(data)):
        if data[i] == 1:
            end = i
        else:
            if end > start:
                ones.append((start, end))
            start = i + 1
            end = start
    if end > start:
        ones.append((start, end))
    return ones
    
def find_sleep_intervals(speed, sleep_duration_threshold=60, lowspeed_threshold=4):
    '''
    find the sleep intervals from speed
    Input:  
        speed: a pandas series of speed
        sleep_duration_threshold: the threshold for sleep duration, default is 60 seconds
    Return:
        is_test: a pandas series of the same size as speed, with True for sleep and False for not sleep
        valid_durations: a list of sleep durations that are longer than sleep_duration_threshold
        valid_intervals: a list of sleep intervals
    '''

    lowspeed = speed <= lowspeed_threshold
    
    # change lowspeed.values from true false to 0 1
    flags = lowspeed.values.astype(int)
    intervals = find_ones_intervals(flags)

    # for each interval, find the value in lowspeed.index / np.timedelta64(1, "s")
    # and calculate the difference between the two values
    times = lowspeed.index / np.timedelta64(1, "s")
    # create a vector the same size as lowspeed.index / np.timedelta64(1, "s"), but set all values to false
    is_test = np.zeros_like(times, dtype=bool)

    valid_durations = []
    valid_intervals = [] 
    for interval in intervals:
        duration = times[interval[1]] - times[interval[0]]
        #store valid durations if it is more than sleep_duration_threshold seconds
        if duration > sleep_duration_threshold:
            valid_durations.append(duration)
            is_test[interval[0] : interval[1]] = True
            valid_intervals.append([lowspeed.index[interval[0]], lowspeed.index[interval[1]]])
            
    #make is_test a pandas series with the same index as lowspeed
    is_test = pd.Series(is_test, index=lowspeed.index)
    
    return is_test, valid_durations, valid_intervals

def find_SIA_intervals(lfp, SIA_threshold=-0.67):
    '''
    find the sleep intervals from speed
    Input:  
        speed: a pandas series of speed
        sleep_duration_threshold: the threshold for sleep duration, default is 60 seconds
    Return:
        is_test: a pandas series of the same size as speed, with True for sleep and False for not sleep
        valid_durations: a list of sleep durations that are longer than sleep_duration_threshold
        valid_intervals: a list of sleep intervals
    '''

    SIA = lfp <= SIA_threshold
    
    # change lowspeed.values from true false to 0 1
    flags = SIA.values.astype(int)
    intervals = find_ones_intervals(flags)

    # for each interval, find the value in lowspeed.index / np.timedelta64(1, "s")
    # and calculate the difference between the two values
    times = SIA.index / np.timedelta64(1, "s")
    # create a vector the same size as lowspeed.index / np.timedelta64(1, "s"), but set all values to false
    is_test = np.zeros_like(times, dtype=bool)

    valid_durations = []
    valid_intervals = [] 
    for interval in intervals:
        duration = times[interval[1]] - times[interval[0]]
        #store valid durations
        valid_durations.append(duration)
        is_test[interval[0] : interval[1]] = True
        valid_intervals.append([SIA.index[interval[0]], SIA.index[interval[1]]])
            
    #make is_test a pandas series with the same index as lowspeed
    is_test = pd.Series(is_test, index=SIA.index)
    
    return is_test, valid_durations, valid_intervals

def find_REM_interval(theta2alpharatio, REMduration=10, REMthreshold=1.5):
    '''
    find the REM intervals from theta2alpharatio
    Input:  
        theta2alpharatio: a pandas series of theta2alpharatio
        duration: the duration of REM, default is 10 seconds
        threshold: the threshold for theta2alpharatio, default is 1.5
    '''
    highratio = theta2alpharatio >= REMthreshold
    
    # change highratio.values from true false to 1 0 
    flags = highratio.values.astype(int)
    intervals = find_ones_intervals(flags)
    
    # for each interval, find the value in highratio.index and calculate the difference between the two values
    times = highratio.index
    #if time type is pandas._libs.tslibs.timedeltas.Timedelta, then convert it to seconds
    if type(times[0]) == pd._libs.tslibs.timedeltas.Timedelta:
        times = times / np.timedelta64(1, "s")
    
    # create a vector the same size as highratio.index, but set all values to false
    is_test = np.zeros_like(times, dtype=bool)
    
    valid_durations = []
    valid_intervals = []
    for interval in intervals:
        duration = times[interval[1]] - times[interval[0]]
        #store valid durations if it is more than duration seconds
        if duration > REMduration:
            valid_durations.append(duration)
            is_test[interval[0] : interval[1]] = True
            valid_intervals.append([highratio.index[interval[0]], highratio.index[interval[1]]])
    is_test = pd.Series(is_test, index=highratio.index)
    
    return is_test, valid_durations, valid_intervals

def get_intervals(logic_flags):
    '''
    get the intervals from logic_flags
    '''
    intervals = find_ones_intervals(logic_flags)
    
    valid_intervals = []
    for interval in intervals:
        #store valid durations if it is more than duration seconds
        valid_intervals.append([logic_flags.index[interval[0]], logic_flags.index[interval[1]]])
    return valid_intervals

def get_sleep_ripples(all_ripple_times, valid_intervals):
    #copy daa["ripple_times"] to a new dataframe
    sleep_ripple_times = all_ripple_times.copy()
    for _, df in sleep_ripple_times.iterrows():
        #if df.start_time and df.end_time are both in one of the valid_intervals, then keep this interval
        #otherwise, drop this interval
        #valid_intervals is list of list, each sublist has two elements, start_time and end_time
        #get the start_time and end_time of the current interval
        
        start_time, end_time = df.start_time, df.end_time
        flag = 0
        for valid_interval in valid_intervals:
            if start_time >= valid_interval[0] and end_time <= valid_interval[1]:
                #keep this interval
                flag=1
            else:
                pass
            
        if flag==0:
            #drop this interval
            sleep_ripple_times.drop(index=_, inplace=True)
            
    return sleep_ripple_times

def get_power(lfp, sampling_frequency=500, band=[5, 11]):
    """Returns filtered amplitude.
    Parameters
    ----------
    lfp : pandas.Series
    sampling_frequency : float, optional
    band : list [5, 11] is theta band; whereas [1, 4] is alpha band
    Returns
    -------
    filtered_amplitude : pandas.Series
    """
    lfp = lfp.dropna()
    
    #band pass filter the lfp signal 
    bandpass_lfp = bandpassfilter(lfp, lowcut=band[0], highcut=band[1], fs=sampling_frequency)

    analytic_signal = hilbert(bandpass_lfp)
    amplitude_envelope = np.abs(analytic_signal)

    #smooth the amplitude with a Gaussian kernel, sigma = 1 second, which equals to 1000 points
    amplitude_envelope = pd.DataFrame(amplitude_envelope, index=lfp.index)
    #filtered_amplitude = amplitude_envelope.rolling(1000, win_type='gaussian', center=True, min_periods=1).mean(std=100)
    
    return amplitude_envelope

def get_theta2alpha_ratio(lfp, thetaband=[5,11], alphaband=[1,4]):
    '''
    get the theta/alpha ratio    
    Input:
        lfp: the lfp signal
        thetaband: the theta band
        alphaband: the alpha band
    Output:
        theta/alpha ratio
    '''
    
    lfp = lfp.dropna()
    #get the theta power and alpha power
    theta_power = get_power(lfp, sampling_frequency=500, band=thetaband)
    alpha_power = get_power(lfp, sampling_frequency=500, band=alphaband)
    #get the theta/alpha ratio
    theta2alpha_ratio = theta_power/alpha_power
    
    #Gaussian smooth the ratio with std=500 using gaussian_filter
    theta2alpha_ratio = gaussian_filter(theta2alpha_ratio, sigma=500)
    
    ##to pd dataframe
    theta2alpha_ratio = pd.DataFrame(theta2alpha_ratio, index=lfp.index)
    return theta2alpha_ratio

def detect_sleep_periods(data, epoch_key, 
                         lowspeed_thres=4, lowspeed_duration=60,
                         theta2alpha_thres=1.5, REM_duration=10,
                         sleep_duration=90, LIA_duration=5,
                         plot=True, figdir=None, figsize=(8,3)):
    '''
    detect sleep periods from the data
    Input:
        data: a dictionary loaded from a epoch_key
        epoch_key: a tuple of (animal, day, epoch)
        lowspeed_thres: the threshold for low speed, default is 4 cm/s
        lowspeed_duration: the duration for low speed, default is 60 seconds
        theta2alpha_thres: the threshold for theta/alpha ratio, default is 1.5
        REM_duration: the duration for REM, default is 10 seconds
        sleep_duration: the duration for sleep, default is 90 seconds
        LIA_duration: the duration for LIA, default is 5 seconds
        plot: whether to plot the results, default is True
        figdir: the directory to save the figure, default is None
    Return:
        dictionsary containing sleep information
    '''
    
    #if plot==True, and figdir is None, through an error
    if plot==True and figdir==None:
        raise ValueError("figdir can not be None if plot is True")
    
    ############################################################################################################
    #1, get candidate sleep periods with speed < 4 cm/s, preceded by 60 s with no movement > 4 cm/s. 
    print('Get candidate sleep periods by thresholding speed...')
    speed = data['position_info'].speed
    is_lowspeed, lowspeed_durations, lowspeed_intervals = find_sleep_intervals(speed, 
                                                                               sleep_duration_threshold=lowspeed_duration, 
                                                                               lowspeed_threshold=lowspeed_thres)
    
    ############################################################################################################
    #2, get REM sleep periods with the averaged theta/alpha ratio > theta2alpha_ratio using only CA1 tetrodes
    print('Get REM sleep periods from all CA1 tetrodes signal by thresholding theta/alpha ratio...')
    lfps = data['lfps']

    tetrode_info = data['tetrode_info']
    is_CA1_areas = (tetrode_info.area.astype(str).str.upper().isin(['CA1']))
    #find lfps in CA1 according to ishilberthilbert_brain_areas
    CA1_lfps = lfps.loc[:, is_CA1_areas.values]

    #initial a panda dataframe to store the theta/alpha ratio for each interval
    all_theta2alpha_ratio = pd.DataFrame(index=CA1_lfps.index)
    #for each column in CA1_lfps, get the theta/alpha ratio
    for i in range(CA1_lfps.shape[1]):
        CA1_lfp = CA1_lfps.iloc[:, i]
        #get the theta/alpha ratio
        theta2alpha_ratio = get_theta2alpha_ratio(CA1_lfp, thetaband=[5,11], alphaband=[1,4])
        #add the ratio to the theta2alpha_ratio to i column in all_theta2alpha_ratio
        all_theta2alpha_ratio[i] = theta2alpha_ratio
    #get the mean ratio across all columns for each time point
    mean_theta2alpha_ratio = all_theta2alpha_ratio.mean(axis=1)

    #get the REM intervals
    is_REM, REM_durations, REM_intervals  = find_REM_interval(mean_theta2alpha_ratio, REMduration=REM_duration, REMthreshold=theta2alpha_thres)    

    ############################################################################################################
    #3, get the SIA threshod using aggregated hippocampal LFP within CA1 CA2 and CA3 tetrodes, and then find LIA periods
    print('Get SIA periods from aggregated LFP signal by thresholding LFP amplitude...')
    is_CA123_areas = (tetrode_info.area.astype(str).str.upper().isin(['CA1', 'CA2', 'CA3']))
    #find lfps in CA1 according to is_brain_areas
    CA123_lfps = lfps.loc[:, is_CA123_areas.values]
    #get the aggregated lfps
    for i in range(CA123_lfps.shape[1]):
        #square the lfp signal
        CA123_lfps.iloc[:, i] = CA123_lfps.iloc[:, i]**2
        #Gaussian smooth the lfp signal with sigma=150, i.e. 300 ms since sampling rate is 500 Hz
        CA123_lfps.iloc[:, i] = gaussian_filter(CA123_lfps.iloc[:, i], sigma=150)
        #take square root of the lfp signal
        CA123_lfps.iloc[:, i] = np.sqrt(CA123_lfps.iloc[:, i])
        #z-score the lfp signal
        CA123_lfps.iloc[:, i] = (CA123_lfps.iloc[:, i] - CA123_lfps.iloc[:, i].mean())/CA123_lfps.iloc[:, i].std()
    #sum the lfp over all columns
    CA123_lfps_sum = CA123_lfps.sum(axis=1)
    #z-score the sum lfp to get the aggregate hippocampal LFP signal
    aggregate_hpc_lfp = (CA123_lfps_sum - CA123_lfps_sum.mean())/CA123_lfps_sum.std()
    
    #get the SIA threshold
    try:
        #get the histogram of the aggregate hippocampal LFP signal
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Get the histogram values and bin edges
        hist_values, bin_edges, _ = ax.hist(aggregate_hpc_lfp, bins=100, density=True, color='k')

        #fitting an bimodal distribution to the histogram
        # Define the function to fit
        def bimodal(x, mu1, sigma1, mu2, sigma2, p):
            return p * norm.pdf(x, mu1, sigma1) + (1 - p) * norm.pdf(x, mu2, sigma2)

        # Fit the data using scipy.optimize.curve_fit
        p0 = [0, 1, 0, 1, 0.5]
        params, _ = curve_fit(bimodal, bin_edges[:-1], hist_values, p0=p0)
        # Plot the fitted curve on top of the histogram
        x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        #find the local minima of the fitted curve
        local_minima = argrelextrema(bimodal(x, *params), np.less)[0]
            
        SIA_threshold = x[local_minima][0]   
        
        #save the figure
        plt.savefig
        
        #close the figure
        plt.close(fig) 
    except:
        SIA_threshold = -0.67
            
    #get the SIA intervals
    is_SIA, SIA_durations, SIA_intervals  = find_SIA_intervals(aggregate_hpc_lfp, SIA_threshold=SIA_threshold)
    
    ############################################################################################################
    #4, get nonREM_interval from lowspeed_intervals, which equals to is_lowspeed = True and is_REM = False
    print('Get nonREM periods from low speed periods and REM periods...')
    is_nonREM = np.logical_and(is_lowspeed, np.logical_not(is_REM)) 

    nonREM_int = find_ones_intervals(is_nonREM)
    nonREM_intervals = []
    for interval in nonREM_int:
        #store valid durations if it is more than duration seconds
        nonREM_intervals.append([is_nonREM.index[interval[0]], is_nonREM.index[interval[1]]])
    
    ############################################################################################################
    #5, get LIA intervals as the intervals with is_lowspeed = True and is_REM = False and is_SIA = False
    print('Get LIA periods from low speed periods, REM periods and SIA periods...')
    is_LIA = np.logical_and(is_lowspeed, np.logical_not(is_REM))
    is_LIA = np.logical_and(is_LIA, np.logical_not(is_SIA))

    LIA_int = find_ones_intervals(is_LIA)
    LIA_intervals = []
    LIA_durations = []
    for interval in LIA_int:
        #store valid durations if it is more than duration seconds
        LIA_intervals.append([is_LIA.index[interval[0]], is_LIA.index[interval[1]]])
        LIA_duration = (is_LIA.index[interval[1]] - is_LIA.index[interval[0]]) / np.timedelta64(1, "s")
        LIA_durations.append(LIA_duration)
        
    ############################################################################################################
    #6, get the final sleep intervals
    print('Get the final sleep periods from nonREM periods and LIA periods...')
    #for each nonREM_interval, if inside the interval, the length of LIA interval is more than 5 seconds, then keep this REM interval
    #otherwise, set is_sleep_flags to False for that interval

    #copy is_nonREM to is_sleep_flags
    is_sleep_flags = is_nonREM.copy()
    for nonREM_interval in nonREM_intervals:
        
        nonREM_duration = (nonREM_interval[1] - nonREM_interval[0]) / np.timedelta64(1, "s")
        
        if nonREM_duration < sleep_duration:
            #set is_sleep_flags to False for that interval
            is_sleep_flags.loc[nonREM_interval[0]:nonREM_interval[1]] = False
        else:
            #check if LIA duration is more than 5 seconds in this nonREM interval
            flag = False
            for LIA_interval in LIA_intervals:
                duration =  (LIA_interval[1] - LIA_interval[0]) / np.timedelta64(1, "s")
                #if the LIA interval is within any of the nonREM_intervals and the duration is more than 5 seconds, then keep it
                if (LIA_interval[0] >= nonREM_interval[0]) and (LIA_interval[1] <= nonREM_interval[1]) and (duration > LIA_duration):
                    flag = True
                    break
            if flag == False:
                #set is_sleep_flags to False for that interval
                is_sleep_flags.loc[nonREM_interval[0]:nonREM_interval[1]] = False

    #get the sleep intervals
    sleep_int = find_ones_intervals(is_sleep_flags)
    sleep_intervals = []
    sleep_durations = []
    for interval in sleep_int:
        #store valid durations if it is more than duration seconds
        sleep_intervals.append([is_sleep_flags.index[interval[0]], is_sleep_flags.index[interval[1]]])
        sleep_duration = (is_sleep_flags.index[interval[1]] - is_sleep_flags.index[interval[0]]) / np.timedelta64(1, "s")
        sleep_durations.append(sleep_duration)
    
    ############################################################################################################
    #7, plot the results
    if plot==True:
        print('Plot the results...')
        fig = plt.figure(figsize=figsize, dpi=300, facecolor='white') 

        #make subplots height as 1:1:1:3
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2.5, 1])

        #set label font size
        plt.rcParams.update({'font.size': 20})

        #1, plot the aggregate hippocampal LFP signal
        ax0 = plt.subplot(gs[0, 0])
        ax0.plot(aggregate_hpc_lfp.index/np.timedelta64(1, "s"), aggregate_hpc_lfp, 'k', alpha=1)
        #add SIAThreshold as a horizontal line
        ax0.axhline(y=SIA_threshold, color='r', linestyle='--', label='SIA threshold')

        ax0.set_ylabel('LFP\nAmp.\n(z)')
        ax0.set_xticks([]); ax0.set_yticks([-2,0,2])
        ax0.spines['bottom'].set_visible(False)

        #2, plot the theta/alpha ratio
        ax1 = plt.subplot(gs[1, 0])
        ax1.plot(mean_theta2alpha_ratio.index/np.timedelta64(1, "s"), mean_theta2alpha_ratio, 'k', alpha=1)
        ax1.set_ylabel('Theta/Alpha\nRatio')
        #add REMthreshold as a horizontal line
        ax1.axhline(y=theta2alpha_thres, color='r', linestyle='--', label='REM threshold')

        ax1.set_xticks([]); ax1.set_yticks([0,2,4])
        ax1.spines['bottom'].set_visible(False)

        #3, plot the speed
        ax2 = plt.subplot(gs[2, 0])
        ax2.plot(speed.index/np.timedelta64(1, "s"), speed, 'k', alpha=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed\n(cm/s)')   

        #4 add shaded area for different sleep stages to all
        times = aggregate_hpc_lfp.index/np.timedelta64(1, "s")
        for ax in [ax0, ax1, ax2]:
            ax.set_xlim([times[0], times[-1]])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #set y label horizontal direction
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_horizontalalignment('center')
            #move the y label to the left
            ax.yaxis.set_label_coords(-0.15,0.3)
            
            #a, background color
            ax.axvspan(times[0], times[-1], alpha=0.5, color='lightgrey', label='Moving')
            
            #b, lowspeed sleep candidate as purple
            for i in range(len(lowspeed_intervals)):
                ax.axvspan(lowspeed_intervals[i][0]/np.timedelta64(1, "s"), lowspeed_intervals[i][1]/np.timedelta64(1, "s"), alpha=0.5, color='purple', label='REM')
            
            #c, nonREM intervals as green
            for i in range(len(nonREM_intervals)):
                ax.axvspan(nonREM_intervals[i][0]/np.timedelta64(1, "s"), nonREM_intervals[i][1]/np.timedelta64(1, "s"), alpha=0.5, color='green', label='LIA')

            
            #d, SIA intervals as orange
            for i in range(len(SIA_intervals)):
                SIA_interval = SIA_intervals[i]
                #if the SIA interval is within any of the nonREM_intervals, the plot it as orange
                for j in range(len(nonREM_intervals)):
                    nonREM_interval = nonREM_intervals[j]
                    if (SIA_interval[0] >= nonREM_interval[0]) and (SIA_interval[1] <= nonREM_interval[1]):
                        ax.axvspan(SIA_interval[0]/np.timedelta64(1, "s"), SIA_interval[1]/np.timedelta64(1, "s"), alpha=0.5, color='orange', label='SIA')
                        break

            #show the legend but without repeating the labels, put the legend outside the plot on the right side
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=False)

        #e, add sleep_interval as thin black bars above axvspan (not overlapping with axvspan) at 1.1*ylim upper limit to the first plot
        #get y lim for the first plot
        y_upperlim = ax0.get_ylim()[1]
        for i in range(len(sleep_intervals)):
            sleep_interval = sleep_intervals[i]
            ax0.plot([sleep_interval[0]/np.timedelta64(1, "s")+5, sleep_interval[1]/np.timedelta64(1, "s")-5], [y_upperlim, y_upperlim], 'r', alpha=1, linewidth=5)

        #5, plot the histogram of the aggregate hippocampal LFP signal by merging gs[0,1] and gs[1,1]

        ax3 = plt.subplot(gs[0:2, 1])
        # Get the histogram values and bin edges
        hist_values, bin_edges, _ = ax3.hist(aggregate_hpc_lfp, bins=100, density=True, color='k')
        
        try:
            #fitting an bimodal distribution to the histogram
            # Define the function to fit
            def bimodal(x, mu1, sigma1, mu2, sigma2, p):
                return p * norm.pdf(x, mu1, sigma1) + (1 - p) * norm.pdf(x, mu2, sigma2)

            # Fit the data using scipy.optimize.curve_fit
            p0 = [0, 1, 0, 1, 0.5]
            params, _ = curve_fit(bimodal, bin_edges[:-1], hist_values, p0=p0)

            # Plot the fitted curve on top of the histogram
            x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
            ax3.plot(x, bimodal(x, *params), 'orange', linewidth=3)

            #find the local minima of the fitted curve
            local_minima = argrelextrema(bimodal(x, *params), np.less)[0]
            #add vertical lines for the local minima of y lim 
            #get y lim
            ax3.plot([x[local_minima][0], x[local_minima][0]], ax3.get_ylim(), 'r--', alpha=1, linewidth=4)
        except:
            pass
            
        ax3.set_xlabel('LFP Amp. (z-score)')
        ax3.set_ylabel('Probability density')
        #up and right spines are not visible
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        plt.tight_layout()
        
        #save the figure
        animal, day, epoch = epoch_key
        fig.savefig(os.path.join(figdir, f'{animal}_{day:02d}_{epoch:02d}_sleep_periods.pdf'), dpi=300)
    
    #get the sleep information
    print('Get the sleep information and save it...')
    sleep_info = {'SIA_threshold': SIA_threshold,
                  'SIA_durations': SIA_durations,
                  'LIA_durations': LIA_durations,
                  'REM_durations': REM_durations,
                  'Sleep_durations': lowspeed_durations}
    
    #save to pickle file
    animal, day, epoch = epoch_key
    with open(os.path.join(figdir, f'{animal}_{day:02d}_{epoch:02d}_sleep_info.pickle'), 'wb') as f:
        pickle.dump(sleep_info, f)
        
    return is_sleep_flags, sleep_durations, sleep_intervals
