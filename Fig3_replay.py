import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from cann import CANN1D
import scipy
#set default ramndom seed for reproducibility
np.random.seed(0)
#set backend to cpu
bm.set_platform('cpu')

#build and run the network
cann = CANN1D(tau=1., tau_v=48., num=128, mbar=2.5)

dt = bm.get_dt()
dur=5000
init_dur = 500
init_index = int(init_dur/dt)

cann.reset_state()
Iext, length = bp.inputs.section_input(
    values=[cann.get_stimulus_by_pos(-2/3*np.pi), 0.],
    durations=[init_dur, dur],
    return_length=True
)

noise = 0.05* np.random.randn(len(Iext), cann.num)
#noise[init_index:-1] = 0
Iext = Iext + noise
Iext = bm.as_numpy(Iext)
runner = bp.DSRunner(cann,
                     inputs=('input', Iext, 'iter'),
                     monitors=['r'])
runner.run(length)

fr = runner.mon.r.T

pos = np.linspace(-np.pi,np.pi,cann.num)
num = int((dur+init_dur) / dt)
time = np.linspace(0, dur+init_dur, num)

#visualize bump sweeps
fig, ax = plt.subplots(figsize=(4,2))
#set parameters
labelsize = 10
ticksize = 8

plt.pcolormesh(time[init_index:-1:50]-time[0]-1200, pos, fr[:,init_index:-1:50]*1e3, cmap='inferno')
plt.xlim((0,200))
clb = plt.colorbar(ticklocation='right', ticks=[0,1,2])
clb.set_label('Firing rate (Hz)', fontsize=labelsize)
clb.ax.tick_params(labelsize=ticksize)
fig.savefig('Figures/Fig3_replay.pdf', dpi=300)
plt.show()
'''
probe_num = int( 1.9*bm.pi / v_ext/dt)
time=time[probe_num:-1]
index = np.linspace(1, cann.num, cann.num)
pos = np.linspace(-np.pi,np.pi,cann.num+1)
pos = pos[0:-1]

fr = runner.mon.r.T[:,probe_num:-1]
cI = runner.mon.centerI[probe_num:-1]
cU = runner.mon.center[probe_num:-1]
##Theta oscillations
relative_pos = runner.mon.center[probe_num:-1] - runner.mon.centerI[probe_num:-1]
relative_pos = bm.where(relative_pos > np.pi, relative_pos - 2*np.pi,  relative_pos)
relative_pos = bm.where(relative_pos < -np.pi, relative_pos + 2*np.pi,  relative_pos)
relative_pos = np.squeeze(relative_pos)
Peaks,_ = scipy.signal.find_peaks(relative_pos, width=300)
Trough,_ = scipy.signal.find_peaks(-relative_pos, width=300)

#visualize bump sweeps
fig, ax = plt.subplots(figsize=(4,2))
#set parameters
labelsize = 10
ticksize = 8
position_color = 'darkturquoise'

plt.pcolormesh(time[0:-1:50]-time[0], pos, fr[:,0:-1:50]*1e3, cmap='inferno')
clb = plt.colorbar(ticklocation='right', ticks=[0,1,2])
clb.set_label('Firing rate (Hz)', fontsize=labelsize)
clb.ax.tick_params(labelsize=ticksize)
#add animal position to the plot 
plt.plot(time-time[0], cI, color=position_color, linewidth=1)  

#add separate lines to theta sweeps
for peaks in Peaks:
    plt.plot([time[peaks]-time[0], time[peaks]-time[0]],[-np.pi,np.pi],'w--', linewidth=1)

plt.xlim(0, 1e3)
plt.ylim([-2.5, 2.5])
plt.xlabel('Time (s)', fontsize=labelsize)
plt.ylabel('Position (cm)', fontsize=labelsize)
# set x and y ticks
xticks = np.linspace(0, 1e3, 3)
yticks = np.linspace(-2.5,2.5,3)
ax.set_xticks(xticks)
ax.set_yticks(yticks)

#set tick labels
xticklabels = [0, 0.5, 1]
yticklabels = [0, int(2.5*100), int(5*100)]
ax.set_yticklabels(yticklabels)
ax.set_xticklabels(xticklabels)
ax.tick_params(axis='x', labelsize=ticksize)
ax.tick_params(axis='y', labelsize=ticksize)
plt.tight_layout()
fig.savefig('Figures/Fig3_replay.pdf', dpi=300)
'''